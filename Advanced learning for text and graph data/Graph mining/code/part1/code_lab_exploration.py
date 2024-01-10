"""
Graph Mining - ALTEGRAD - Oct 2023
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

############## Task 1

G = nx.read_edgelist("datasets/CA-HepTh.txt", comments="#", delimiter="\t")
print("Number of nodes :", nx.number_of_nodes(G))
print("Number of edges :", nx.number_of_edges(G))


############## Task 2

print("Number of connected components :", nx.number_connected_components(G))

largest_cc = max(nx.connected_components(G), key=len)
print("The largest connected component has", len(largest_cc), "nodes")

subG = G.subgraph(largest_cc)
print(
    "The largest connected component in the graph has",
    nx.number_of_edges(subG),
    "edges",
)

############## Task 3
# Degree
degree_sequence = [G.degree(node) for node in G.nodes()]

print("Minimum degree of the nodes :", np.min(degree_sequence))
print("Maximum degree of the nodes :", np.max(degree_sequence))
print("Median degree of the nodes :", np.median(degree_sequence))
print("Mean degree of the nodes :", np.mean(degree_sequence))


############## Task 4

plt.hist(nx.degree_histogram(G), bins=50)
plt.savefig("degree_histogram.png")

plt.hist(nx.degree_histogram(G), bins=50)
plt.xscale("log")
plt.yscale("log")
plt.savefig("degree_histogram_log.png")


############## Task 5

# transitivity = 3 n_triangles / n_triads
# global_clustering_coefficient = n_triangles / (n_triangles + n_triads)
transitivity = nx.transitivity(G)
n_triangles = len(nx.triangles(G))
n_triads = 3 * n_triangles / transitivity
print("Global clustering coefficent :", n_triangles / (n_triangles + n_triads))
