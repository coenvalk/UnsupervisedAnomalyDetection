import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx
from datetime import datetime
from sklearn import neighbors, decomposition

def DNODA(G):
    node_data = nx.get_node_attributes(G, "D")
    scores = []
    for v in tqdm(G, leave=False):
        p = node_data[v]
        neighbors = np.array([node_data[u] for u in G.neighbors(v)])
        if len(neighbors):
            distances = np.sqrt( np.sum(np.square( p - neighbors ), axis=1) )
            scores.append(np.mean(distances))
        else:
            scores.append(0)
    return np.array(scores)

def GLODA(G):
    clf = neighbors.LocalOutlierFactor(n_jobs=-1)
    node_data = nx.get_node_attributes(G, "D")
    dataset = [node_data[v] for v in G]
    clf.fit(dataset)
    scores = clf.negative_outlier_factor_ * -1
    return scores

def CNA(G):
    node_data = nx.get_node_attributes(G, "D")
    communities = nx.algorithms.community.label_propagation_communities(G)
    scores = dict.fromkeys(G.nodes())
    for c in tqdm(communities, leave=False):
        neighbors = np.array([node_data[u] for u in c])
        for v in c:
            p = node_data[v]
            distances = np.sqrt( np.sum(np.square( p - neighbors ), axis=1) )
            scores[v] = np.mean(distances)
    ret = []
    for v in G:
        ret.append(scores[v])
    return np.array(ret)

def oddball(G, alg_type="EDPL"):
    x = []
    y = []
    for v in tqdm(G, leave=False):
        egonet = set(G.neighbors(v))
        egonet.add(v)
        if alg_type == "EWPL":
            distances = [np.linalg.norm(d['amount']) for _, _, d in G.edges(nbunch=egonet, data=True) ]
            x.append(len(G.edges(nbunch=egonet)) + 1)
            y.append(np.sum(distances) + 1)
        else:
            x.append(len(egonet))
            y.append(len(G.edges(nbunch=egonet)) + 1)
    
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

    return np.array(scores)

if __name__ == "__main__":
    worst_offenders = 20
    data_root = "./data"
    print("Loading data...")
    print("  Loading Enron...")
    enron = nx.read_edgelist(os.path.join(data_root, "Enron", "Enron.txt"))
    node_attribs = dict((v, {"D": [enron.degree(v)]}) for v in enron)
    nx.set_node_attributes(enron, node_attribs)
    print("  Done.")

    print("  Loading FEC data...")
    print("    Loading Graph...")
    donations = nx.read_edgelist(
        os.path.join(data_root, "donations", "donations.csv"),
        delimiter=",", data=[("amount", float)])
    print("    Done.")
    node_attribs = dict.fromkeys(donations.nodes())
    print("    Setting node attributes...")
    for v in tqdm(donations, leave=False):
        D = [d["amount"] for d in donations[v].values()]
        mean = np.mean(D)
        median = np.median(D)
        node_attribs[v] = {"D": [mean, median]}
    nx.set_node_attributes(donations, node_attribs)
    print("    Done.")
    print("    Generating Subset Graph...")
    subset = np.random.choice(
        list(donations.nodes()),
        int(0.05 * donations.order()), replace=False)
    donations = nx.subgraph(donations, subset)
    print("    Done.")
    print("  Done.")
    print("  Loading Bitcoin data...")
    edgelist = pd.read_csv(
        os.path.join(data_root,
            "elliptic_bitcoin_dataset",
            "elliptic_txs_edgelist.csv"))
    bitcoin = nx.from_pandas_edgelist(edgelist, source="txId1", target="txId2")
    d = pd.read_csv(
        os.path.join(data_root,
            "elliptic_bitcoin_dataset",
            "elliptic_txs_features.csv"),
        header=None)
    features = d.drop(columns=[0]).values
    pca = decomposition.PCA(n_components=0.8)
    features = pca.fit_transform(features)
    print(features.shape)
    node_attribs = dict((v, {"D": f}) for v, f in zip(d[0].values, features))
    nx.set_node_attributes(bitcoin, node_attribs)
    edge_attribs = dict.fromkeys(bitcoin.edges())
    for u, v in edge_attribs:
        u_data = node_attribs[u]["D"]
        v_data = node_attribs[v]["D"]
        edge_attribs[(u, v)] = {"amount": u_data - v_data}
    nx.set_edge_attributes(bitcoin, edge_attribs)
    print("  Done.")
    print("Done.")

    print("Enron Worst Offenders:")
    start = datetime.now()
    gloda_scores = GLODA(enron)
    print("  GLODA:", np.argsort(gloda_scores)[-worst_offenders:], datetime.now() - start)
    start = datetime.now()
    dnoda_scores = DNODA(enron)
    print("  DNODA:", np.argsort(dnoda_scores)[-worst_offenders:], datetime.now() - start)
    start = datetime.now()
    cna_scores = CNA(enron)
    print("  CNA:", np.argsort(cna_scores)[-worst_offenders:], datetime.now() - start)
    start = datetime.now()
    oddball_scores = oddball(enron)
    print("  OddBall:", np.argsort(oddball_scores)[-worst_offenders:], datetime.now() - start)
    gloda_scores = np.argsort(gloda_scores)
    dnoda_scores = np.argsort(dnoda_scores)
    cna_scores = np.argsort(cna_scores)
    oddball_scores = np.argsort(oddball_scores)
    borda = dict.fromkeys(np.unique(gloda_scores), 0)
    for i in range(len(gloda_scores)):
        borda[gloda_scores[i]] += i
        borda[dnoda_scores[i]] += i
        borda[cna_scores[i]] += i
        borda[oddball_scores[i]] += i
    borda_scores = sorted(borda.items(), key=lambda x: x[1])
    borda_final = np.array([x[0] for x in borda_scores])
    print("  Borda:", borda_final[-10:])
    print("Done.")
    print()

    print("FEC Donation Data Worst Offenders:")
    start = datetime.now()
    gloda_scores = GLODA(donations)
    print("  GLODA:", np.argsort(gloda_scores)[-worst_offenders:], datetime.now() - start)
    start = datetime.now()
    dnoda_scores = DNODA(donations)
    print("  DNODA:", np.argsort(dnoda_scores)[-worst_offenders:], datetime.now() - start)
    start = datetime.now()
    cna_scores = CNA(donations)
    print("  CNA:", np.argsort(cna_scores)[-worst_offenders:], datetime.now() - start)
    start = datetime.now()
    oddball_scores = oddball(donations, alg_type="EWPL")
    print("  OddBall:", np.argsort(oddball_scores)[-worst_offenders:], datetime.now() - start)
    gloda_scores = np.argsort(gloda_scores)
    dnoda_scores = np.argsort(dnoda_scores)
    cna_scores = np.argsort(cna_scores)
    oddball_scores = np.argsort(oddball_scores)
    borda = dict.fromkeys(np.unique(gloda_scores), 0)
    for i in range(len(gloda_scores)):
        borda[gloda_scores[i]] += i
        borda[dnoda_scores[i]] += i
        borda[cna_scores[i]] += i
        borda[oddball_scores[i]] += i
    borda_scores = sorted(borda.items(), key=lambda x: x[1])
    borda_final = np.array([x[0] for x in borda_scores])
    print("  Borda:", borda_final[-10:])
    print("Done.")
    print()


    print("Bitcoin Data Worst Offenders:")
    start = datetime.now()
    gloda_scores = GLODA(bitcoin)
    print("  GLODA:", np.argsort(gloda_scores)[-worst_offenders:], datetime.now() - start)
    start = datetime.now()
    dnoda_scores = DNODA(bitcoin)
    print("  DNODA:", np.argsort(dnoda_scores)[-worst_offenders:], datetime.now() - start)
    start = datetime.now()
    cna_scores = CNA(bitcoin)
    print("  CNA:", np.argsort(cna_scores)[-worst_offenders:], datetime.now() - start)
    start = datetime.now()
    oddball_scores = oddball(bitcoin, alg_type="EWPL")
    print("  OddBall:", np.argsort(oddball_scores)[-worst_offenders:], datetime.now() - start)
    gloda_scores = np.argsort(gloda_scores)
    dnoda_scores = np.argsort(dnoda_scores)
    cna_scores = np.argsort(cna_scores)
    oddball_scores = np.argsort(oddball_scores)
    borda = dict.fromkeys(np.unique(gloda_scores), 0)
    for i in range(len(gloda_scores)):
        borda[gloda_scores[i]] += i
        borda[dnoda_scores[i]] += i
        borda[cna_scores[i]] += i
        borda[oddball_scores[i]] += i
    borda_scores = sorted(borda.items(), key=lambda x: x[1])
    borda_final = np.array([x[0] for x in borda_scores])
    print("  Borda:", borda_final[-10:])
    print("Done.")
    print()

