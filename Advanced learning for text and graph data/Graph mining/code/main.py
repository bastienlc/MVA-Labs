import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

G = nx.read_edgelist("datasets/CA-HepTh.txt", comments="#", delimiter="\t")

from part2.code_lab_community_detection import spectral_clustering

clustering = spectral_clustering(G, 50)
print(clustering)
