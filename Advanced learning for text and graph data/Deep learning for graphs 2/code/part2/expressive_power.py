"""
Deep Learning on Graphs - ALTEGRAD - Nov 2023
"""

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from models import GNN
from utils import sparse_mx_to_torch_sparse_tensor

# Initializes device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Hyperparameters
hidden_dim = 32
output_dim = 4
dropout = 0.0
neighbor_aggr = "mean"
readout = "mean"


############## Task 4

cycle_graphs = [nx.cycle_graph(k) for k in range(10, 20)]


############## Task 5

adj_batch = sparse_mx_to_torch_sparse_tensor(
    sp.block_diag([nx.adjacency_matrix(graph) for graph in cycle_graphs])
).to(device)

features_batch = torch.ones(adj_batch.size(0), 1).to(device)

idx_batch = torch.cat(
    [
        torch.tensor([j] * graph.number_of_nodes())
        for j, graph in enumerate(cycle_graphs)
    ]
).to(device)

############## Task 8

combinations = [("mean", "mean"), ("sum", "mean"), ("mean", "sum"), ("sum", "sum")]

models = [
    GNN(1, hidden_dim, output_dim, neighbor_aggr, readout, dropout).to(device)
    for neighbor_aggr, readout in combinations
]

for model, combinations in zip(models, combinations):
    print("Neighbor aggregation: {}, Readout: {} :".format(*combinations))
    print(model(features_batch, adj_batch, idx_batch).cpu().detach().numpy())


############## Task 9

G1 = nx.Graph()
G1.add_nodes_from(range(6))
G2 = nx.Graph()
G2.add_nodes_from(range(6))
for i in range(3):
    G1.add_edge(i, (i + 1) % 3)
    G1.add_edge(i + 3, 3 + (i + 1) % 3)
for i in range(6):
    G2.add_edge(i, (i + 1) % 6)

############## Task 10

graphs = [G1, G2]

adj = sparse_mx_to_torch_sparse_tensor(
    sp.block_diag([nx.adjacency_matrix(graph) for graph in graphs])
).to(device)

features = torch.ones(adj.size(0), 1).to(device)

idx = torch.cat(
    [torch.tensor([j] * graph.number_of_nodes()) for j, graph in enumerate(graphs)]
).to(device)


############## Task 11

model = GNN(1, hidden_dim, output_dim, "sum", "sum", dropout).to(device)
representations = model(features, adj, idx)

g1_representation = representations[0, :].cpu().detach().numpy()
g2_representation = representations[1, :].cpu().detach().numpy()
print(f"G1 representation : {g1_representation}")
print(f"G2 representation : {g2_representation}")
if (g1_representation != g2_representation).any():
    print("The GNN can distinguish between the two graphs.")
else:
    print("The GNN cannot distinguish between the two graphs.")
