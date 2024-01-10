"""
Graph Mining - ALTEGRAD - Oct 2023
"""

from random import randint

import networkx as nx
from scipy.sparse import diags, eye
from scipy.sparse.linalg import eigs
from sklearn.cluster import KMeans


############## Task 6
# Perform spectral clustering to partition graph G into k clusters
def spectral_clustering(G, k):
    # We arbitrarily choose the number of eigenvectors to take to be the same as the number of clusters
    # That is motivated by the fact that a more complicated clustering would require more features in the kmeans algorithm
    d = k

    adjacency_matrix = nx.adjacency_matrix(G)
    degree_matrix = diags(list(dict(G.degree()).values()), 0)
    laplacian_matrix = (
        eye(G.number_of_nodes()) - degree_matrix.power(-1) @ adjacency_matrix
    )  # The inverse of the degree matrix is the diagonal matrix with the inverse of the degrees on the diagonal
    _, eigenvectors = eigs(laplacian_matrix, k=d, which="SM")
    eigenvectors = eigenvectors.real

    clustering = KMeans(n_clusters=k, n_init=10).fit_predict(eigenvectors)
    nodes = list(G.nodes())
    return {nodes[i]: clustering[i] for i in range(len(clustering))}


############## Task 7

G = nx.read_edgelist("datasets/CA-HepTh.txt", comments="#", delimiter="\t")
subG = G.subgraph(max(nx.connected_components(G), key=len))
clusters = spectral_clustering(subG, 50)

############## Task 8
# Compute modularity value from graph G based on clustering


def modularity(G, clustering):
    Q = 0
    m = G.number_of_edges()
    clusters = set(clustering.values())
    for cluster in clusters:
        community = G.subgraph(
            [node for node in clustering.keys() if clustering[node] == cluster]
        )
        lc = community.number_of_edges()
        dc = sum([G.degree(node) for node in community.nodes()])
        Q += lc / m - (dc / (2 * m)) ** 2

    return Q


############## Task 9

spectral_clusters = spectral_clustering(subG, 50)
print("Spectral clustering modularity :", modularity(subG, spectral_clusters))

random_clusters = {node: randint(0, 49) for node in subG.nodes()}
print("Random clustering modularity :", modularity(subG, random_clusters))
