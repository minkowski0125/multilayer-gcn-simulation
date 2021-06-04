import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from model import GCN, CrossEntropy
from utils import *
from config import args

import dgl
import dgl.function as fn
from dgl import DGLGraph


def train(data = None, deg = None, feat_dim = None, hidden_dim = None, layer_num = None, writer = None, o = 0):
    device = torch.device('cpu')
    set_seed(args.seed)

    deg = deg if not deg is None else args.deg
    feat_dim = feat_dim if not feat_dim is None else args.feat_dim
    layer_num = layer_num if not layer_num is None else args.layer_num
    sample_num = args.sample_num
    hidden_dim = hidden_dim if not hidden_dim is None else args.hidden_dim

    graphs, features, adjs, labels = data

    features = [preprocess_features(feature) for feature in features]
    supports = [build_support(adj) for adj in adjs]

    deg_dim = features[0].shape[0]
    feat_dim = features[0].shape[1]

    test_gcn = GCN(deg_dim, feat_dim, hidden_dim, layer_num)
    if args.optim == 'gd':
        pass
    elif args.optim == 'sgd':
        optimizer = torch.optim.SGD(test_gcn.parameters(), lr=args.lr)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(test_gcn.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.Adam(test_gcn.parameters(), lr=args.lr)

    cnt = o
    ori_loss = 0
    loss_series = []
    test_gcn.train()
    cross_entropy_loss = CrossEntropy()
    for epoch in range(args.epochs):
        out = test_gcn((features[cnt], supports[cnt]))

        loss = cross_entropy_loss(out, labels[cnt])

        if writer is not None:
            writer.add_scalar(f'loss_deg{deg}_hidden{hidden_dim}_layer{layer_num}', loss, epoch)
        if args.optim == 'gd':
            test_gcn.zero_grad()
            loss.backward()
            with torch.no_grad():
                for param in test_gcn.parameters():
                    param -= args.lr * param.grad
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_series.append(loss.item())
        if epoch % 1 == 0:
            print('training:', epoch, loss.item())
        
        if loss.item() < 5e-4 and epoch > 5000:
            return loss_series

    return loss_series

if __name__ == '__main__':
    train()