
import os.path as osp
import argparse
import torch
import os
from pre_training import setup_seed, GMT_pretraining, GraphMix, test, mnn, train
from tqdm import tqdm
import random
from torch_geometric.loader import DataLoader
import numpy as np
from models.GNN import GNN, MNN_GNN
# import torch_geometric.transforms as T
# from model_loader import get_model
from utils import query_data, create_model, query_index
import json
from copy import deepcopy
from random import sample
from sklearn.model_selection import StratifiedShuffleSplit
current_path = os.getcwd()

@torch.no_grad()
def inference(args, model, loader,stage='test'):
    model.eval()
    all_feature = None
    pred = None
    all_label = None
    for data in loader:
        data = data.to(args.device)
        feature = model(data)
        fc = model.readout(feature,stage=stage)
        out = model.predict(fc)
        label = data.y
        if args.dataset_name == 'Tox21':
            label = label[:,4]

        if all_feature == None:
            all_feature = feature
            pred = out
            all_label = label
        else:
            all_feature = torch.cat((all_feature, feature), dim=0)
            pred = torch.cat((pred, out), dim=0)
            all_label = torch.cat((all_label, label), dim=0)

    total_correct = (pred.argmax(dim=-1)==all_label).sum()
    return all_feature, pred, all_label, total_correct / len(loader.dataset)

def EM_training(args):
    args.method = 'first'

    if '->' in args.dataset_name:
        dataset_name = args.dataset_name.split('->')
        args.dataset_name = dataset_name[0]
        train_first = query_data(args)
        args.dataset_name = dataset_name[1]
        target_first = query_data(args)
        source_first = train_first[args.source_idx]
        val_first = train_first[args.val_idx]
        total_source_first = train_first

    else:
        dataset = query_data(args)
        source_first = dataset[args.source_idx]
        val_first = dataset[args.val_idx]
        target_first = dataset[args.target_idx]
        total_source_first = dataset[args.source_idx + args.val_idx]

    # source_first_loader_order并不参与训练，只是用来得到顺序的训练loader的特征
    source_first_loader_order = DataLoader(total_source_first, args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_first_loader = DataLoader(val_first, args.batch_size, shuffle=False, num_workers=args.num_workers)
    target_first_loader = DataLoader(target_first, args.batch_size, shuffle=False, num_workers=args.num_workers)

    model_first = GNN(args, num_features=source_first.num_node_features,num_classes=source_first.num_classes,
                    conv_type=args.conv_type, pool_type=args.pool_type).to(args.device)
    model_first.load_state_dict(torch.load(f'pretraining/first_{args.dataset_name}_E.pth'))
    optimizer_first = torch.optim.Adam(model_first.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_func = torch.nn.CrossEntropyLoss()

    target_first_feature, target_first_pred, target_first_label, _ = inference(args, model_first, target_first_loader)
    # target_first_pred = torch.nn.Softmax(-1)(target_first_pred)

    target_acc, _, _, _, _ = test(args, model_first, target_first_loader)
    print('Direct predict first acc:', target_acc)

    args.method = 'second'
    if '->' in args.dataset_name:
        dataset_name = args.dataset.split('->')
        args.dataset_name = dataset_name[0]
        train_second = query_data(args)
        source_second = train_second[args.source_idx]
        val_second = train_second[args.val_idx]
        args.dataset_name = dataset_name[1]
        target_second = query_data(args)
        total_source_second = train_second
    else:
        dataset = query_data(args)
        source_second = dataset[args.source_idx]
        val_second = dataset[args.val_idx]
        target_second = dataset[args.target_idx]
        total_source_second = dataset[args.source_idx+args.val_idx]

    # source_second_loader = DataLoader(source_second, args.batch_size, shuffle=True, num_workers=args.num_workers)
    target_second_loader = DataLoader(target_second, args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_second_loader = DataLoader(val_second, args.batch_size, shuffle=False, num_workers=args.num_workers)
    source_second_loader_order = DataLoader(total_source_second, args.batch_size, shuffle=False, num_workers=args.num_workers)

    model_second = create_model(args).to(args.device)
    model_second.load_state_dict(torch.load(f'pretraining/second_{args.dataset_name}_M.pth'))
    optimizer_second = torch.optim.Adam(model_second.parameters(), lr=args.lr, weight_decay=1e-4)

    target_acc, _, _, _, _ = test(args, model_second, target_second_loader)
    print('Direct predict second acc:', target_acc)

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
parser.add_argument('--epochs', type=int, default=50)
# parser.add_argument('--e_epochs', type=int, default=100)
parser.add_argument('--device', type=int, default=3)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--seed', type=int, default=123456789)
parser.add_argument('--dataset_name', type=str, default="Tox21")
parser.add_argument('--source', type=int, default=1)
parser.add_argument('--target', type=int, default=0)

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
parser.add_argument('--topk', type=int, default=20)
parser.add_argument('--gama', type=float, default=0.1)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--K', type=int, default=1024)
parser.add_argument('--e_threshold', type=float, default=0.7)
parser.add_argument('--m_threshold', type=float, default=0.7)
# second model params

args = parser.parse_args()

setup_seed(args.seed)
# print(torch.cuda.is_available())
if args.device >= 0:
    args.device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
else:
    args.device = torch.device("cpu")
# print(args.device)

# query_index(args)

# with open(f'config/{args.dataset_name}.json', 'r') as f:
#     idx_dict = json.load(f)
#
# args.source_idx = idx_dict['target']
# args.target_idx = idx_dict['source']
# args.unknown = idx_dict['unknown']
# if '->' in args.dataset_name:
#     print(args.dataset)
#     dataset_name = args.dataset.split('->')
#     # args.source = dataset_name[0]
#     # args.target = dataset_name[1]
#     args.dataset_name = dataset_name[0]
#     source_dataset = query_data(args)
#     args.dataset_name = dataset_name[1]
#     target_dataset = query_data(args)
# else:
#     dataset = query_data(args)
#     source_idx = np.load('idx/idx_%s_%d.npy' % (args.dataset_name, args.source))
#     target_idx = np.load('idx/idx_%s_%d.npy' % (args.dataset_name, args.target))
#     # print(source_idx,target_idx)
#     source_dataset = dataset[source_idx]
#     target_dataset = dataset[target_idx]

# if args.dataset_name == 'COIL-DEL':
#     args.num_class = int(100 * args.unknown_ratio)
# elif args.dataset_name == 'Letter-high':
#     args.num_class = int(15 * args.unknown_ratio)
# else:
#     args.num_class = int(10 * args.unknown_ratio)
# 懒得改，对性能影响不大
if '->' in args.dataset_name:
    name = args.dataset_name
    dataset_name = args.dataset_name.split('->')
    args.dataset_name = dataset_name[0]
    source_dataset = query_data(args)
    args.dataset_name = dataset_name[1]
    target_dataset = query_data(args)
    val_index = sample(list(range(len(source_dataset))), int(0.05 * len(source_dataset)))
    source_idx = list(range(len(source_dataset)))
    train_index = [item for item in source_idx if item not in val_index]
    args.source_idx = train_index
    args.val_idx = val_index
    args.target_idx = list(range(len(target_dataset)))

    args.dataset_name = name
    # pretraining(args)
    EM_training(args)
else:
    dataset = query_data(args)
    source_idx = np.load('idx/idx_%s_%d.npy' % (args.dataset_name, args.source))
    target_idx = np.load('idx/idx_%s_%d.npy' % (args.dataset_name, args.target))
    # print(source_idx)
    val_index = sample(list(source_idx), int(0.05 * len(source_idx)))
    train_index = [item for item in source_idx if item not in val_index]
    args.source_idx = train_index
    args.val_idx = val_index
    args.target_idx = target_idx

    for args.source in [1]:
        for args.target in [2]:
            print(f'source {args.source} -> target {args.target}')
            if args.source == args.target:
                continue
            else:
                # pretraining(args)
                EM_training(args)