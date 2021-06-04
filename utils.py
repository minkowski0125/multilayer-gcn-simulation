import torch
import random
import numpy as np
from torch import nn
from torch.nn import functional as F

import dgl
import dgl.function as fn
from dgl import DGLGraph
from dgl.data import citation_graph
import networkx as nx
import scipy.sparse as sp

import matplotlib.pyplot as plt

def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)

def load_pubmed_data(params):
    g = citation_graph.load_pubmed()[0]
    # g = g.remove_self_loop()

    deg = params['deg_num']
    sample_num = params['sample_num']
    assert deg % 2 == 0
    label_points = [torch.nonzero(g.ndata['label'] == 0).squeeze(1), torch.nonzero(g.ndata['label'] == 1).squeeze(1)]

    cnt = 0
    graphs, features, adjs, labels = [], [], [], []

    while cnt < sample_num:
        shufflers = [list(range(deg // 2)), list(range(deg // 2))]
        random.shuffle(shufflers[0])
        random.shuffle(shufflers[1])
        idx_pick = torch.cat([idx[shufflers[i]] for i, idx in enumerate(label_points)])

        sub_graph = g.subgraph(idx_pick)
        feature = sub_graph.ndata['feat']

        graphs.append(sub_graph)
        features.append(sub_graph.ndata['feat'])
        adjs.append(sub_graph.adjacency_matrix())
        labels.append([-1] * (deg // 2) + [1] * (deg //2))
        cnt += 1
    labels = torch.tensor(labels)

    return graphs, features, adjs, labels

def random_init(params):
    cnt = 0
    graphs, features, adjs, labels = [], [], [], []

    while cnt < sample_num:
        spmat = sp.rand(deg, deg, density=args.density)
        g = dgl.from_scipy(spmat)
        graphs.append(g)
        cnt += 1

    features = [torch.FloatTensor(torch.rand(params['deg_num'], params['feat_dim'])) for g in graphs]
    adjs = [g.adjacency_matrix() for g in graphs]
    labels = torch.randint(0, 2, (params['deg'], )) * 2 - 1.
    labels = torch.tensor(labels)

    return graphs, features, adjs, labels

def set_data(params):
    if params['name'] == 'pubmed':
        graphs, features, adjs, labels = load_pubmed_data(params)
        return graphs, features, adjs, labels
    elif params['name'] == 'random':
        graphs = []
        for i in range(params['sample_num']):
            g, num_classes = random_init(params['deg'], params['feat_dim'], params['density'])
            graphs.append(g)
    else:
        return None

    
    adjs = [g.adjacency_matrix() for g in graphs]
    labels = torch.randint(0, 2, (params['sample_num'], )) * 2 - 1.
    features = [ feature * 2 if labels[i] == 1 else feature * 2 - 2 for i, feature in enumerate(features)]

    return graphs, features, adjs, labels

def preprocess_features(features):
    r_inv = np.power((features**2).sum(1), -0.5).flatten()
    r_inv[torch.nonzero(r_inv == float('inf'))] = 0
    r_mat_inv = torch.abs(torch.diag(r_inv))
    return torch.mm(r_mat_inv, features)

def normalize_support(supports):
    d_inv_sqrt = np.power(supports.sum(1), -0.5).flatten()
    d_inv_sqrt[torch.nonzero(d_inv_sqrt == float('inf'))] = 0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    return torch.mm(torch.mm(d_mat_inv_sqrt, supports), d_mat_inv_sqrt)

def build_support(adj):
    return normalize_support(adj.to_dense()) + torch.eye(adj.shape[0])

def visualize(loss_series, enume, mode):
    y1, y2 = [], []
    for k, hidden in enumerate(enume):
        series = loss_series[k]
        y1.append(series[5000 - 1])
        start = series[0]
        for i, loss in enumerate(series):
            if loss < 0.01 * start:
                y2.append(i)
                break

    ax1 = plt.figure().add_subplot(111)

    ax1.set_ylabel('Iterations', fontsize=15)
    ax1.set_xlabel('Hidden dimension', fontsize=15)

    l1 = ax1.plot(hiddens, y2, linestyle='-.', marker='x', color='red', label='loss')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Loss', fontsize=15)
    l2 = ax2.plot(hiddens, y1, linestyle='-', marker='.', label='epoch')

    lns = l1 + l2
    lbs = [item.get_label() for item in lns]
    ax1.legend(lns, lbs, loc='best', fontsize=15)

    plt.show()