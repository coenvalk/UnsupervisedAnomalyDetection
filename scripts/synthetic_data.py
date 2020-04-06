# creates synthetic financial transaction data

import os
import pickle
import argparse
import numpy as np
import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create synthetic graph data.')
    parser.add_argument('N', type=int,
                   help='Number of nodes in graph')
    parser.add_argument('d', type=int,
                    help="data dimensionality")
    parser.add_argument('outfile', type=str,
                    help='Output pickle file')
    args = parser.parse_args()

    m = 2
    G = nx.barabasi_albert_graph(args.N, m)
    data = np.random.pareto(100, size=(G.size(), args.d))
    labels = {}
    for i, (u, v) in enumerate(G.edges):
        labels[(u, v)] = {"D": data[i]}
    nx.set_edge_attributes(G, labels)
    with open(args.outfile, 'wb') as f:
        pickle.dump(G, f)