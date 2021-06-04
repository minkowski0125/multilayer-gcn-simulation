import json
from utils import *
from config import args
from train import train
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    set_seed(args.seed)

    series = []
    if args.dataset == 'pubmed':
        graphs, features, adjs, labels = load_pubmed_data({
            'deg_num': args.deg,
            'sample_num': 1,
        })
    elif args.dataset == 'random':
        graphs, features, adjs, labels = load_pubmed_data({
            'deg_num': args.deg,
            'feat_dim': args.feat_dim,
            'sample_num': 1,
        })

    writer = SummaryWriter(f'./log_pubmed')

    hiddens = [50, 100, 200, 500, 1000, 1500, 2000, 3000]
    for hidden in hiddens:
        series.append(train(data = (graphs, features, adjs, labels), deg = args.deg, feat_dim = args.feat_dim, hidden_dim = hidden, layer_num = args.layer_num, o = 0, writer=writer))
        print()
        
    visualize(series, hiddens, 'hidden')
    # print(series)
