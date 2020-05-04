import matplotlib
import numpy as np
import networkx as nx
from matplotlib import cm
from matplotlib import pyplot as plt
from sklearn import neighbors, preprocessing

def DNODA(G):
    node_data = nx.get_node_attributes(G, "D")
    scores = []
    for v in G:
        p = node_data[v]
        neighbors = np.array([node_data[u] for u in G.neighbors(v)])
        distances = np.sqrt( np.sum(np.square( p - neighbors ), axis=1) )
        scores.append(np.mean(distances))
    return scores

def GLODA(G):
    clf = neighbors.LocalOutlierFactor()
    node_data = nx.get_node_attributes(G, "D")
    dataset = [node_data[v] for v in G]
    clf.fit(dataset)
    scores = clf.negative_outlier_factor_ * -1
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 100))
    scores = scaler.fit_transform(np.reshape(scores, (-1, 1)))[:, 0]
    return scores

def CNA(G):
    node_data = nx.get_node_attributes(G, "D")
    communities = nx.algorithms.community.label_propagation_communities(G)
    scores = dict.fromkeys(G.nodes())
    for c in communities:
        for v in c:
            p = node_data[v]
            neighbors = np.array([node_data[u] for u in c])
            distances = np.sqrt( np.sum(np.square( p - neighbors ), axis=1) )
            scores[v] = np.mean(distances)
    ret = []
    for v in G:
        ret.append(scores[v])
    return ret

if __name__ == "__main__":
    algs = [
        (DNODA, "DNODA"),
        (GLODA, "GLODA"),
        (CNA, "CNA")
    ]
    N = 25
    n = 5 # n**2 == N
    minima = 0
    maxima = 100
    node_attribs = dict((v, {
        "D": [minima]
    }) for v in range(N + 1))
    node_attribs[0]["D"] = [maxima]
    norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.RdYlGn_r)

    star = nx.star_graph(N)
    nx.set_node_attributes(star, node_attribs)

    circle = nx.cycle_graph(N)
    nx.set_node_attributes(circle, node_attribs)

    for k in node_attribs:
        if k % 5 == 0:
            node_attribs[k]["D"] = [maxima]
    clique = nx.ring_of_cliques(n, n)
    nx.set_node_attributes(clique, node_attribs)

    for alg, name in algs:
        fig, axes = plt.subplots(ncols=3, figsize=(13, 4), tight_layout=True)

        axes[0].set_title("Star")
        labels = nx.get_node_attributes(star, "D")
        scores = alg(star)
        nx.draw_kamada_kawai(star, labels=labels, ax=axes[0], node_color=scores, cmap=cm.RdYlGn_r, vmin=minima, vmax=maxima)

        axes[1].set_title("Cycle")
        labels = nx.get_node_attributes(circle, "D")
        scores = alg(circle)
        nx.draw_kamada_kawai(circle, labels=labels, ax=axes[1], node_color=scores, cmap=cm.RdYlGn_r, vmin=minima, vmax=maxima)

        axes[2].set_title("Ring of Cliques")
        labels = nx.get_node_attributes(clique, "D")
        scores = alg(clique)
        nx.draw_kamada_kawai(clique, labels=labels, ax=axes[2], node_color=scores, cmap=cm.RdYlGn_r, vmin=minima, vmax=maxima)

        cbar = fig.colorbar(mapper, label="Outlier Score")
        plt.savefig(name + "Example")