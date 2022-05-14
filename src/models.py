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
import random

random.seed(0)

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
        top_eig = torch.linalg.eig(lap_adj)[1:self.k+1, :]
        return top_eig

class GATv3Layer(tgnn.MessagePassing):
    def __init__(self, indim, eigendim, outdim):
        super().__init__(aggr="add")
        self.original_mlp = nn.Sequential(
                nn.Linear(2 * indim, outdim), # account for extra Wx_i || W_j from GATv1
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
        return self.propagate(edge_idx, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        cat = torch.cat([x_i, x_j], dim=1)
        
        node_attr = self.alpha * self.original_mlp(cat)
        edge_attr = self.beta * self.eigen_mlp(edge_attr)
        
        gamma = torch.softmax(F.leaky_relu(node_attr + edge_attr), 1)
        msg = gamma * self.out(x_j)
        
        return msg

# model = GATv3Layer(256, 64, 10)
# edge_idx = torch.randint(0, 10, (2, 200)).long()
# edge_attr = torch.rand(200, 64)
# x = torch.rand(100, 256)
# y = model(x, edge_attr, edge_idx)
# print (y.shape)

graph = Graph.Erdos_Renyi(n=15, p=0.2, directed=False, loops=False)

# fig, axs = plt.subplots(1, 1)
# igraph.plot(
#     graph,
#     target=axs,
#     layout="circle",
#     vertex_color="lightblue",
#     vertex_size=15
# )
# plt.show()

adj = torch.from_numpy(np.array(list(graph.get_adjacency())))
edge_idx = tg.utils.dense_to_sparse(adj)
print (edge_idx)
eigen = Eigen(4)
eig = eigen(edge_idx)
print (eig.shape)