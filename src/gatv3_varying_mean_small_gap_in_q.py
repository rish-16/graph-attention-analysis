print ("Importing ...")
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric as tg
import torch_geometric.nn as tgnn
from torch_geometric.utils import get_laplacian, to_dense_adj
from torch_geometric.typing import Adj, OptTensor, PairTensor

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union

class Eigen(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
    
    def forward(self, edge_idx):
        lap_idx, lap_wt = get_laplacian(edge_idx, normalization="sym")
        lap_adj = to_dense_adj(lap_idx)
        eigenvals, eigenvecs = torch.linalg.eig(lap_adj)
        top_eig = eigenvecs.squeeze(0)[:, 1:self.k+1]
        top_eig = torch.real(top_eig)
        new_edge_features = torch.Tensor(edge_idx.size(1), 2 * self.k).to(edge_idx.device)
        new_edge_idx = edge_idx.T

        for idx, pair in enumerate(new_edge_idx):
            i, j = pair
            x_i_prime = top_eig[i]
            x_j_prime = top_eig[j]
            new_feat = torch.cat([x_i_prime, x_j_prime], dim=0)
            new_edge_features[idx] = new_feat

        return new_edge_features

class GATv3Layer(tgnn.MessagePassing):
    def __init__(self, indim, eigendim, outdim):
        super().__init__(aggr="add")
        self.original_mlp = nn.Sequential(
                nn.Linear(2 * indim, outdim), # account for extra Wx_i || Wx_j from GATv1
                nn.Linear(outdim, outdim),
                nn.LeakyReLU(0.02),
                nn.Linear(outdim, outdim)
            )
        
        self.eigen_mlp = nn.Sequential(
                nn.Linear(eigendim, outdim), # account for the fact that edge attributes are already concatenated
                nn.Linear(outdim, outdim),
                nn.LeakyReLU(0.02),
                nn.Linear(outdim, outdim)
            )
        self.W = nn.Linear(indim, indim)
        self.project = nn.Linear(outdim, 1)
        self.out = nn.Linear(indim, outdim)

        self.alpha = nn.Parameter(torch.rand(1, 1))

        self.beta = nn.Parameter(torch.rand(1, 1))
        
        self.all_gammas = None
        
        
    def forward(self, x, edge_idx, edge_attr):
        num_nodes = x.size(0)
        edge_idx, edge_attr = tg.utils.remove_self_loops(edge_idx, edge_attr)
        edge_idx, edge_attr = tg.utils.add_self_loops(edge_idx, edge_attr, num_nodes=num_nodes)
        
        return self.propagate(edge_idx, x=x, edge_attr=edge_attr)

    def message(self, x_j: torch.Tensor, x_i: torch.Tensor,
                edge_attr: torch.Tensor,
                index: torch.Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> torch.Tensor:
    
        cat = torch.cat([x_i, x_j], dim=1)
        
        node_attr = self.alpha * self.original_mlp(cat) # [E, d]
        edge_attr = self.beta * self.eigen_mlp(edge_attr) # [E, d]
        
        temp = F.leaky_relu(node_attr + edge_attr) # [E, d]
        project = self.project(temp)
        gamma = tg.utils.softmax(project, index, ptr, size_i) # [E, d]
        msg = gamma * self.out(x_j) # [E, d]
        
        self.all_gammas = gamma
        
        return msg

class GATv3(nn.Module):
    def __init__(self, indim, eigendim, hidden, outdim, k):
        super().__init__()

        self.eigen = Eigen(k)
        self.gat1 = GATv3Layer(indim, eigendim, hidden)
        self.relu = nn.ReLU()
        self.gat2 = GATv3Layer(hidden, eigendim, outdim)

    def forward(self, x, edge_idx):
#         with torch.no_grad():
        eigen_x = self.eigen(edge_idx)
        x = self.relu(self.gat1(x, edge_idx, eigen_x))
        out = self.gat2(x, edge_idx, eigen_x)

        return out        

torch.cuda.empty_cache()        

def get_gammas(Xw, ground_truth, gat_layer, edge_idx):
    all_gammas = gat_layer.all_gammas
    gamma_matrix = [[0 for j in range(Xw.size(0))] for i in range(Xw.size(0))]
    for idx, pair in enumerate(edge_idx.T):
        i, j = pair
        gamma = all_gammas[idx]
        
        gamma_matrix[i][j] = gamma.item()
        
    return gamma_matrix
        
def get_intra_inter_avg_gamma(gamma_matrix):
    """
    intra-edges are nodes with class 0
    inter-edges are nodes with class 1
    """
    
    d = len(gamma_matrix) // 2
    all_node_ids = list(range(len(gamma_matrix)))
    intra_edges = all_node_ids[:d]
    inter_edges = all_node_ids[d:]
    
    intra_edge_gammas = []
    inter_edge_gammas = []
    
    for i in range(len(gamma_matrix)):
        for j in range(len(gamma_matrix[i])):
            if j in intra_edges:
                intra_edge_gammas.append(gamma_matrix[i][j])
            elif j in inter_edges:
                inter_edge_gammas.append(gamma_matrix[i][j])
            else:
                pass
                
    return np.array(intra_edge_gammas), np.array(inter_edge_gammas)

print ("Building graph ...")
n = 400
d = int(np.ceil(n/(np.log(n)**2)))
p = 0.5
q = 0.1

sizes = [int(n/2), int(n/2)]
probs = [[p,q], [q,p]]

std_ = 0.1
mu_up = 20*std_*np.sqrt(np.log(n**2))/(2*np.sqrt(d))
mu_lb = 0.01*std_/(2*np.sqrt(d))

mus = np.geomspace(mu_lb, mu_up, 30, endpoint=True)
ground_truth = np.concatenate((np.zeros(int(n/2)), np.ones(int(n/2))))

avg_intra_edge_gamma_1 = []
avg_inter_edge_gamma_1 = []
avg_intra_edge_gamma_2 = []
avg_inter_edge_gamma_2 = []
std_intra_edge_gamma_1 = []
std_inter_edge_gamma_1 = []
std_intra_edge_gamma_2 = []
std_inter_edge_gamma_2 = []    
all_train_losses = []

print ("Starting training ...")
for mu in mus:
    print ("Building SBM ...")
    g = nx.stochastic_block_model(sizes, probs)
    adjlist = [[v for v in g.neighbors(i)] for i in range(n)]

    adj_matrix = [[0 for i in range(n)] for j in range(n)]
    for i in range(n):
        nbors = g.neighbors(i)
        for j in nbors:
            adj_matrix[i][j] = 1

    edge_idx, _ = tg.utils.dense_to_sparse(torch.from_numpy(np.array(adj_matrix)))
    edge_idx = edge_idx.cuda()

    for i in range(len(adjlist)):
        adjlist[i].append(i) # self-loops

    X = np.zeros((n,d))
    X[:int(n/2)] = -mu
    X[int(n/2):] = mu
    noise = std_*np.random.randn(n,d)
    X = X + noise

    R = 1
    mu_ = mu*np.ones(d)
    w = (R/np.linalg.norm(mu_))*mu_
    Xw = X@w


    HIDDEN = 16
    eigenK = 10 # take top 10 eigen vector features
    EPOCHS = 1000

    print ("Instantiating model ...")
    gat = GATv3(
        indim=1, 
        eigendim=eigenK*2,
        hidden=HIDDEN, 
        outdim=1, 
        k=eigenK
    ).cuda()
    crit = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.Adam(gat.parameters())

    Xw_tensor = torch.from_numpy(Xw).unsqueeze(-1).float().cuda()
    ground_truth_tensor = torch.from_numpy(ground_truth).unsqueeze(-1).float().cuda()

    print (f"Training with Mu: {mu}")

    train_losses = []

    for epoch in range(EPOCHS):
        optimiser.step()

        pred = gat(Xw_tensor, edge_idx)
        loss = crit(pred, ground_truth_tensor)

        loss.backward()
        optimiser.step()

        train_losses.append(loss.cpu().item())

        if epoch % 200 == 0:
            print (f"Epoch: {epoch} | Train BCE: {loss.cpu().item()}")

    print ("------------------------------------------\n\n")
    all_train_losses.append(np.array(train_losses))
    
    
    torch.cuda.empty_cache()

    print ("Analysing gamma values ...")
    gamma_matrix1 = get_gammas(Xw_tensor.cpu(), ground_truth_tensor.cpu(), gat.gat1.cpu(), edge_idx.cpu())
    gamma_matrix2 = get_gammas(Xw_tensor.cpu(), ground_truth_tensor.cpu(), gat.gat2.cpu(), edge_idx.cpu())

    intra1, inter1 = get_intra_inter_avg_gamma(gamma_matrix1)
    intra2, inter2 = get_intra_inter_avg_gamma(gamma_matrix2)

    avg_intra_gamma_1 = intra1.mean()
    avg_inter_gamma_1 = inter1.mean()

    avg_intra_gamma_2 = intra2.mean()
    avg_inter_gamma_2 = inter2.mean()

    std_intra_gamma_1 = intra1.std()
    std_inter_gamma_1 = inter1.std()

    std_intra_gamma_2 = intra2.std()
    std_inter_gamma_2 = inter2.std()

    avg_intra_edge_gamma_1.append(avg_intra_gamma_1)
    avg_inter_edge_gamma_1.append(avg_inter_gamma_1)
    avg_intra_edge_gamma_2.append(avg_intra_gamma_2)
    avg_inter_edge_gamma_2.append(avg_inter_gamma_2)
    std_intra_edge_gamma_1.append(std_intra_gamma_1)
    std_inter_edge_gamma_1.append(std_inter_gamma_1)
    std_intra_edge_gamma_2.append(std_intra_gamma_2)
    std_inter_edge_gamma_2.append(std_inter_gamma_2)
    
    print (f"Mu: {mu}")
    print (f"Intra1: {avg_intra_gamma_1} | Inter1: {avg_inter_gamma_1} | Intra2: {avg_intra_gamma_2} | Inter2: {avg_inter_gamma_2}")
    
    print ("-----------------------------------------------------------\n\n")
    
    torch.cuda.empty_cache()
avg_intra_edge_gamma_1 = np.array(avg_intra_edge_gamma_1)
avg_inter_edge_gamma_1 = np.array(avg_inter_edge_gamma_1)
avg_intra_edge_gamma_2 = np.array(avg_intra_edge_gamma_2)
avg_inter_edge_gamma_2 = np.array(avg_inter_edge_gamma_2)
std_intra_edge_gamma_1 = np.array(std_intra_edge_gamma_1)
std_inter_edge_gamma_1 = np.array(std_inter_edge_gamma_1)
std_intra_edge_gamma_2 = np.array(std_intra_edge_gamma_2)
std_inter_edge_gamma_2 = np.array(std_inter_edge_gamma_2)
all_train_losses = np.array(all_train_losses)

np.save("mus.npy", mus)
np.save("avg_intra_edge_gamma_1.npy", avg_intra_edge_gamma_1)
np.save("avg_inter_edge_gamma_1.npy", avg_inter_edge_gamma_1)
np.save("avg_intra_edge_gamma_2.npy", avg_intra_edge_gamma_2)
np.save("avg_inter_edge_gamma_2.npy", avg_inter_edge_gamma_2)
np.save("std_intra_edge_gamma_1.npy", std_intra_edge_gamma_1)
np.save("std_inter_edge_gamma_1.npy", std_inter_edge_gamma_1)
np.save("std_intra_edge_gamma_2.npy", std_intra_edge_gamma_2)
np.save("std_inter_edge_gamma_2.npy", std_inter_edge_gamma_2)
np.save("all_train_losses.npy", all_train_losses)

plt.plot(mus, avg_intra_edge_gamma_1, color="blue", label="Intra Edge Gamma")
plt.plot(mus, avg_inter_edge_gamma_1, color="green", label="Inter Edge Gamma")
plt.title("GATv3 Layer 1")
plt.xlabel("difference")
plt.ylabel("gamma")
plt.legend()
plt.savefig("layer1plot.png")

plt.plot(mus, avg_intra_edge_gamma_2, color="blue", label="Intra Edge Gamma")
plt.plot(mus, avg_inter_edge_gamma_2, color="green", label="Inter Edge Gamma")
plt.title("GATv3 Layer 2")
plt.xlabel("difference")
plt.ylabel("gamma")
plt.legend()
plt.savefig("layer2plot.png")