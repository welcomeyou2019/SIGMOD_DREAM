import os.path as osp
import argparse
import torch
# import torch.nn.functional as F
# from torch.nn import Linear
# from shortest_paths import ShortestPathTransform
# from torch_geometric.datasets import TUDataset, GNNBenchmarkDataset
from torch_geometric.loader import DataLoader
# from torch_geometric.nn import GCNConv, GraphMultisetTransformer
import os
from tqdm import tqdm
import random
import numpy as np
from models.GNN import GNN
# import torch_geometric.transforms as T
# from model_loader import get_model
from utils import query_data, create_model
import json
from pre_training import GMT_pretraining, GraphMix
from sklearn.model_selection import StratifiedShuffleSplit
current_path = os.getcwd()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def train(model, train_loader, optimizer, loss_func):
    model.train()

    total_loss = 0
    total_correct = 0
    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(args.device)
        # out = model(data.x, data.batch, data.edge_index)
        out, feature = model(data,return_feature=True, normalize=True)
        loss = loss_func(out, data.y)
        loss.backward()
        total_loss += data.num_graphs * float(loss)
        total_correct += int((out.argmax(dim=-1) == data.y).sum())
        optimizer.step()
    return total_loss / len(train_loader.dataset), total_correct/ len(train_loader.dataset), feature


@torch.no_grad()
def test(model, loader):
    model.eval()

    total_correct = 0
    for data in loader:
        data = data.to(args.device)
        out = model(data)
        total_correct += int((out.argmax(dim=-1) == data.y).sum())

    return total_correct / len(loader.dataset)


parser = argparse.ArgumentParser()

# model params
parser.add_argument('--method', type=str, choices=['new', 'graphcl', 'baseline', 'gla', 'causal',
                                                   'GraphMLPMixer','GraphViT','MPGNN','GraphMLPMixer4TreeNeighbour',
                                                   'MPGNN4TreeNeighbour'], default='baseline')
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--conv_type', type=str, choices=['GCN', 'SAGE', 'GAT', 'GIN','GMT'], default='GMT')
parser.add_argument('--pool_type', type=str, choices=['TopK', 'Edge', 'SAG', 'ASA','GMT'], default='GMT')
parser.add_argument('--layer-norm', type=bool, default=True)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--seed', type=int, default=123456789)
parser.add_argument('--dataset_name', type=str, default="Letter-high")

parser.add_argument('--gnn_type', type=str, default="GINEConv")
parser.add_argument('--nlayer_gnn', type=int, default=4)
parser.add_argument('--nlayer_mlpmixer', type=int, default=4)
parser.add_argument('--pool', type=str, default="mean")
parser.add_argument('--residual', type=bool, default=True)
parser.add_argument('--use_patch_pe', type=bool, default=True)
parser.add_argument('--lap_dim', type=int, default=0)
parser.add_argument('--rw_dim', type=int, default=0)
parser.add_argument('--n_patches', type=int, default=32)
parser.add_argument('--enable', type=bool, default=True)
parser.add_argument('--online', type=bool, default=True)
parser.add_argument('--num_hops', type=int, default=1)
parser.add_argument('--mlpmixer_dropout', type=float, default=0.)
parser.add_argument('--unknown_ratio', type=float, default=0.8)
parser.add_argument('--devide_ration', type=float, default=0.5)
# second model params

args = parser.parse_args()

setup_seed(args.seed)
args.device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

with open(f'config/{args.dataset_name}.json', 'r') as f:
    idx_dict = json.load(f)

args.source_idx = idx_dict['source']
args.target_idx = idx_dict['target']
args.unknown = []#idx_dict['unknown']
GMT_pretraining(args)
GraphMix(args)

