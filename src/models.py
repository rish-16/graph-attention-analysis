import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric as tg
import torch_geometric.nn as tgnn
from torch_geometric.utils import get_laplacian, to_dense_adj

import igraph
from igraph import Graph
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
import random

random.seed(0)

def plot_graph(graph):
    fig, axs = plt.subplots(1, 1)
    igraph.plot(
        graph,
        target=axs,
        layout="circle",
        vertex_color="lightblue",
        vertex_size=15
    )
    plt.show()

"""
source code for the GAT v3

1. Compute laplacian of adjacency matrix
2. Perform eigendecomp of laplacian
3. Get top-k smallest eigenvectors
4. Stack horizontally and create new node features X'
5. Use new features:
    - in conjunction with original node features X
    - ignore original node features X
6. Perform Message Passing
    - calculate alpha scores using two mlps
    - Get new node embeddings with weighted node features aX'
"""

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
        new_edge_features = torch.Tensor(edge_idx.size(1), 2 * self.k)
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
        self.out = nn.Linear(indim, outdim)

        self.alpha = nn.Parameter(torch.rand(1, 1))
        nn.init.xavier_uniform_(self.alpha.data, gain=1.414)

        self.beta = nn.Parameter(torch.rand(1, 1))
        nn.init.xavier_uniform_(self.beta.data, gain=1.414)
        
    def forward(self, x, edge_attr, edge_idx):
        edge_idx, edge_attr = tg.utils.add_self_loops(edge_idx, edge_attr)
        return self.propagate(edge_idx, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        cat = torch.cat([x_i, x_j], dim=1)
        
        node_attr = self.alpha * self.original_mlp(cat)
        edge_attr = self.beta * self.eigen_mlp(edge_attr)
        
        gamma = torch.softmax(F.leaky_relu(node_attr + edge_attr), 1)
        msg = gamma * self.out(x_j)
        
        return msg

class GATv3(nn.Module):
    def __init__(self, indim, eigendim, hidden, outdim, k):
        super().__init__()

        self.eigen = Eigen(k)
        self.gat1 = GATv3Layer(indim, eigendim, hidden)
        self.gat2 = GATv3Layer(hidden, eigendim, outdim)

    def forward(self, x, edge_idx, edge_attr):
        with torch.no_grad():
            eigen_x = self.eigen(edge_idx)
        x = torch.relu(self.gat1(x, eigen_x, edge_idx))
        out = torch.softmax(self.gat2(x, eigen_x, edge_idx), 1)

        return out

graph = Graph.Erdos_Renyi(n=15, p=0.2, directed=False, loops=False)
adj = torch.from_numpy(np.array(list(graph.get_adjacency())))
edge_idx, _ = tg.utils.dense_to_sparse(adj)
n_edges = graph.ecount()
edge_attr = torch.rand(n_edges, 64)
x = torch.rand(15, 128)
y = torch.rand(size=(15, 1))
y[y > 0.5] = 1
y[y <= 0.5] = 0

gat = GATv3(128, 8, 32, 10, 4)
pred = gat(x, edge_idx, edge_attr)
print (pred.shape)
# criterion = nn.CrossEntropyLoss()
# optim = torch.optim.Adam(gat.parameters())

# for i in range(10):
#     optim.zero_grad()
#     pred = gat(x, edge_idx, edge_attr).unsqueeze(0)
#     print (pred.shape)
#     loss = criterion(pred, y.unsqueeze(0))
#     loss.backward()
#     optim.step()

#     print (loss)
