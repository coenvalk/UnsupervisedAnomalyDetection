import matplotlib
import numpy as np
import networkx as nx
from matplotlib import cm
from operator import itemgetter
from matplotlib import pyplot as plt
from sklearn import preprocessing, neighbors

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

def oddball(G):
    node_data = nx.get_node_attributes(G, "D")
    x = []
    y = []
    for v in G:
        e = nx.ego_graph(G, v)
        p = node_data[v]
        ego_points = np.array([node_data[u] for u in e])
        distances = np.sqrt( np.sum(np.square( p - ego_points ), axis=1) )
        x.append(e.size())
        y.append(np.sum(distances))
    
    x_log = np.log(x)
    y_log = np.log(y)

    m, b = np.polyfit(x_log, y_log, 1)
    C = np.exp(b)
    theta = m

    y_fit = C * (x ** theta)
    
    scores = []
    for fit, gt in zip(y, y_fit):
        s = (max(fit, gt) / min(fit, gt)) * np.log(np.abs(fit - gt) + 1)
        scores.append(s)

    return x, y, C, theta, scores

        

if __name__ == "__main__":
    N = 25
    theta = 4
    G = nx.generators.random_graphs.powerlaw_cluster_graph(N, 4, 0.3)
    _, max_degree = sorted(G.degree(), key=itemgetter(1))[-1]
    node_attribs = dict.fromkeys(G)
    for k in G:
        Y = [G.degree(k)]
        node_attribs[k] = {"D": Y}
    nx.set_node_attributes(G, node_attribs)

    fig, axes = plt.subplots(ncols=4, tight_layout=True, figsize=(16, 4))

    axes[0].set_title("OddBall")
    x, y, C, theta, scores = oddball(G)
    print(C, theta)
    labels = nx.get_node_attributes(G, "D")
    print('Done')
    nx.draw_kamada_kawai(G, labels=labels, node_color=scores, cmap=cm.RdYlGn_r, ax=axes[0], node_size=150)

    axes[1].set_title("GLODA")
    scores = GLODA(G)
    nx.draw_kamada_kawai(G, labels=labels, node_color=scores, cmap=cm.RdYlGn_r, ax=axes[1], node_size=150)

    axes[2].set_title("DNODA")
    scores = DNODA(G)
    nx.draw_kamada_kawai(G, labels=labels, node_color=scores, cmap=cm.RdYlGn_r, ax=axes[2], node_size=150)

    axes[3].set_title("CNA")
    scores = CNA(G)
    nx.draw_kamada_kawai(G, labels=labels, node_color=scores, cmap=cm.RdYlGn_r, ax=axes[3], node_size=150)

    norm = matplotlib.colors.Normalize(vmin=0, vmax=100, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.RdYlGn_r)
    cbar = fig.colorbar(mapper, label="Outlier Score")
    fig.savefig("OddballExample")
    fig.clf()
    plt.clf()

    x_fit = np.unique(x)
    y_fit = C * (x_fit ** theta)
    plt.plot(x, y, 'o', label="data")
    plt.plot(x_fit, y_fit, label="fit")
    plt.xlabel("Number of Edges")
    plt.ylabel("Sum of Weights")
    plt.xscale('log')
    plt.yscale('log')
    plt.show()