"""
Deep Learning on Graphs - ALTEGRAD - Nov 2023
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from deepwalk import deepwalk
from scipy.sparse import diags, eye, identity
from scipy.sparse.linalg import eigs, inv
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import accuracy_score

# Loads the karate network
G = nx.read_weighted_edgelist(
    "../data/karate.edgelist", delimiter=" ", nodetype=int, create_using=nx.Graph()
)
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

n = G.number_of_nodes()

# Loads the class labels
class_labels = np.loadtxt("../data/karate_labels.txt", delimiter=",", dtype=np.int32)
idx_to_class_label = dict()
for i in range(class_labels.shape[0]):
    idx_to_class_label[class_labels[i, 0]] = class_labels[i, 1]

y = list()
for node in G.nodes():
    y.append(idx_to_class_label[node])

y = np.array(y)


############## Task 5
# Visualizes the karate network

nx.draw(G, node_color=y)
plt.show()


############## Task 6
# Extracts a set of random walks from the karate network and feeds them to the Skipgram model
n_dim = 128
n_walks = 10
walk_length = 20
model = deepwalk(G, n_walks, walk_length, n_dim)

embeddings = np.zeros((n, n_dim))
for i, node in enumerate(G.nodes()):
    embeddings[i, :] = model.wv[str(node)]

idx = np.random.RandomState(seed=42).permutation(n)
idx_train = idx[: int(0.8 * n)]
idx_test = idx[int(0.8 * n) :]

X_train = embeddings[idx_train, :]
X_test = embeddings[idx_test, :]

y_train = y[idx_train]
y_test = y[idx_test]


############## Task 7
# Trains a logistic regression classifier and use it to make predictions

print("Using DeepWalk embeddings")

classifier = LogisticRegression().fit(X_train, y_train)

print(
    "Accuracy on the training data :",
    accuracy_score(y_train, classifier.predict(X_train)),
)

prediction = classifier.predict(X_test)

print("Accuracy on the test data :", accuracy_score(prediction, y_test))


############## Task 8
# Generates spectral embeddings

print("Using spectral embeddings")

adjacency_matrix = nx.adjacency_matrix(G)
degree_matrix = diags(list(dict(G.degree()).values()), 0)
laplacian_matrix = eye(G.number_of_nodes()) - degree_matrix.power(-1) @ adjacency_matrix
_, eigenvectors = eigs(laplacian_matrix, k=2, which="SM")
embeddings = eigenvectors.real

X_train = embeddings[idx_train, :]
X_test = embeddings[idx_test, :]

classifier = LogisticRegression().fit(X_train, y_train)

print(
    "Accuracy on the training data :",
    accuracy_score(y_train, classifier.predict(X_train)),
)

prediction = classifier.predict(X_test)

print("Accuracy on the test data :", accuracy_score(prediction, y_test))
