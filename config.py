import argparse

args = argparse.ArgumentParser()
args.add_argument('--device', default='cpu')
args.add_argument('--seed', default=123, type=int)

args.add_argument('--dataset', default='pubmed')
args.add_argument('--sample_num', default=1, type=int)
args.add_argument('--deg', default=100, type=int)
args.add_argument('--feat_dim', default=200, type=int)
args.add_argument('--layer_num', default=5, type=int)
args.add_argument('--density', default=0.5, type=float)

args.add_argument('--hidden_dim', default=100, type=int)
args.add_argument('--optim', default='gd')
args.add_argument('--epochs', default=10000, type=int)
args.add_argument('--lr', default=1e-2, type=float)

args = args.parse_args()