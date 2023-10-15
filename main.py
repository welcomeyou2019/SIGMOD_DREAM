import os.path as osp
import argparse
import torch
import os
from pre_training import setup_seed, first_brunch, second_brunch, test, mnn, train,first_brunch_wo_val, second_brunch_wo_val
from tqdm import tqdm
import random
from torch_geometric.loader import DataLoader
import numpy as np
from models.GNN import GNN, MNN_GNN
from utils import query_data, create_model, query_index
import json
from models.model import PathNN
# from copy import deepcopy
from random import sample
from sklearn.model_selection import StratifiedShuffleSplit
current_path = os.getcwd()
# set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:6144

def pretraining(args):
    first_brunch_wo_val(args)
    second_brunch_wo_val(args)


@torch.no_grad()
def inference(args, model, loader,loss_func=torch.nn.CrossEntropyLoss()):
    model.eval()
    all_feature = None
    pred = None
    all_label = None
    total_loss = 0
    for data in loader:
        data = data.to(args.device)
        feature = model(data)
        fc = model.readout(feature)
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

        loss = loss_func(out, label.long())
        total_loss += data.num_graphs * float(loss)
    total_correct = (pred.argmax(dim=-1)==all_label).sum()
    return all_feature, pred, all_label, total_correct / len(loader.dataset),total_loss / len(loader.dataset)

def MNN_training(args, model, edge_index, feature, label, size):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_func = torch.nn.CrossEntropyLoss()
    for epoch in range(10):
        optimizer.zero_grad()
        pred = model(feature, edge_index)

        loss = loss_func(pred[:size], label[:size].long())
        loss.backward()
        optimizer.step()

def EM_training_wo_val(args):

    e_step_label_acc = []
    m_step_label_acc = []
    args.method = 'first'

    if '->' in args.dataset_name:
        name = args.dataset_name
        dataset_name = args.dataset_name.split('->')
        args.dataset_name = dataset_name[0]
        train_first = query_data(args)
        args.dataset_name = dataset_name[1]
        target_first = query_data(args)
        source_first = train_first
        # val_first = train_first[args.val_idx]
        # total_source_first = train_first
        args.dataset_name = name
    else:
        dataset = query_data(args)
        source_first = dataset[args.source_idx+args.val_idx]
        # val_first = dataset[args.val_idx]
        target_first = dataset[args.target_idx]
        # total_source_first = dataset[args.source_idx + args.val_idx]

    # source_first_loader_order并不参与训练，只是用来得到顺序的训练loader的特征
    source_first_loader = DataLoader(source_first, args.batch_size, shuffle=False, num_workers=args.num_workers)
    # source_first_loader_order = DataLoader(source_first, args.batch_size, shuffle=False, num_workers=args.num_workers)
    # val_first_loader = DataLoader(val_first, args.batch_size, shuffle=False, num_workers=args.num_workers)
    target_first_loader = DataLoader(target_first, args.batch_size, shuffle=False, num_workers=args.num_workers)

    model_first = GNN(args, num_features=max(source_first.num_node_features, target_first.num_node_features),num_classes=source_first.num_classes,
                    conv_type=args.conv_type, pool_type=args.pool_type).to(args.device)
    model_first.load_state_dict(torch.load(f'pretraining/first_{args.dataset_name}.pth'))
    optimizer_first = torch.optim.Adam(model_first.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_func = torch.nn.CrossEntropyLoss()

    target_acc, _, _, _, _ = test(args, model_first, target_first_loader)
    print('Direct predict first acc:', target_acc)

    args.method = 'second'
    if '->' in args.dataset_name:
        name = args.dataset_name
        dataset_name = args.dataset_name.split('->')
        args.dataset_name = dataset_name[0]
        train_second = query_data(args)
        source_second = train_second[args.source_idx]
        val_second = train_second[args.val_idx]
        args.dataset_name = dataset_name[1]
        target_second = query_data(args)
        total_source_second = train_second
        args.dataset_name = name
    else:
        dataset = query_data(args)
        source_second = dataset[args.source_idx]
        val_second = dataset[args.val_idx]
        target_second = dataset[args.target_idx]
        total_source_second = dataset[args.source_idx+args.val_idx]

    # source_second = deepcopy(source_first)
    # target_second = deepcopy(source_second)

    args.feat_num = max(source_second.num_node_features, target_second.num_node_features)
    source_second_loader = DataLoader(source_second, args.batch_size, shuffle=False, num_workers=args.num_workers)
    target_second_loader = DataLoader(target_second, args.batch_size, shuffle=False, num_workers=args.num_workers)
    # val_second_loader = DataLoader(val_second, args.batch_size, shuffle=False, num_workers=args.num_workers)
    # source_second_loader_order = DataLoader(total_source_second, args.batch_size, shuffle=False, num_workers=args.num_workers)

    # model_second = create_model(args).to(args.device)
    if args.path_type == 'all_simple_paths' :
        encode_distances = True
    else :
        encode_distances = False
    model_second = PathNN(args.feat_num, args.hidden_dim, args.cutoff, source_first.num_classes, args.dropout, args.device,
                        residuals = args.residuals, encode_distances=encode_distances
                        ).to(args.device)

    model_second.load_state_dict(torch.load(f'pretraining/second_{args.dataset_name}.pth'))
    optimizer_second = torch.optim.Adam(model_second.parameters(), lr=args.lr, weight_decay=1e-4)

    target_acc, _, _, _, _ = test(args, model_second, target_second_loader)
    print('Direct predict second acc:', target_acc)

    total_best_train_loss = 10000
    for em_step in range(10):
        if os.path.exists(f'pretraining/M_second_{args.dataset_name}.pth'):
            model_second.load_state_dict(torch.load(f'pretraining/M_second_{args.dataset_name}.pth'))
        source_second_feature, source_second_pred, source_second_label, _, _ = inference(args, model_second, source_second_loader)
        target_second_feature, target_second_pred, target_second_label, _, _ = inference(args, model_second, target_second_loader)
        target_second_pred = torch.nn.Softmax(-1)(target_second_pred)

        pesudo_second_idx = torch.where(target_second_pred > args.e_threshold)[0]
        pesudo_second_label = target_second_pred[pesudo_second_idx].argmax(dim=-1)
        ture_second_label = target_second_label[pesudo_second_idx]
        e_acc = (pesudo_second_label==ture_second_label).sum()/len(pesudo_second_idx)
        print('e_acc:', (pesudo_second_label==ture_second_label).sum()/len(pesudo_second_idx))
        e_step_label_acc.append(e_acc.item())

        # M step, generate mnn adj matrix for E step
        edge_index = mnn(source_second_feature, target_second_feature)

        # E step
        if '->' in args.dataset_name:
            name = args.dataset_name
            dataset_name = args.dataset_name.split('->')
            args.dataset_name = dataset_name[0]
            train_first = query_data(args)
            args.dataset_name = dataset_name[1]
            target_first_copy = query_data(args)
            source_first = train_first
            # val_first = train_first[args.val_idx]
            # total_source_first = train_first
            args.dataset_name = name
        else:
            dataset = query_data(args)
            source_first = dataset[args.source_idx + args.val_idx]
            # val_first = dataset[args.val_idx]
            target_first_copy = dataset[args.target_idx]
            # total_source_first = dataset[args.source_idx + args.val_idx]
        # target_first_copy = target_first
        if args.dataset_name == 'Tox21':
            target_first_copy.y[pesudo_second_idx,4] = pesudo_second_label.float().cpu().numpy()
        else:
            # print(pesudo_second_idx.dtype)
            target_first_copy.y[pesudo_second_idx.cpu()] = pesudo_second_label.cpu().numpy()

        if '->' in args.dataset_name:
            if source_first.num_node_features > target_first.num_node_features:
                for i in pesudo_second_idx:
                    target_first_copy[i].x = torch.cat((target_first_copy[i].x, torch.zeros(target_first_copy[i].x.shape[0], args.feat_num-target_first_copy[i].x.shape[-1]).to(target_first_copy[i].x.device)),dim=-1).to(target_first_copy[i].x.device)
            else:
                for i in source_first:
                    i.x = torch.cat((i.x, torch.zeros(i.x.shape[0], args.feat_num-i.x.shape[-1]).to(i.x.device)),dim=-1).to(i.x.device)
        E_training_data = source_first + target_first_copy[pesudo_second_idx]
        print('E step:',len(E_training_data),len(source_first),len(pesudo_second_idx),len(target_first))
        # print(source_first[0])
        # print(E_training_data[0])
        E_train_loader = DataLoader(E_training_data, args.batch_size, shuffle=True, num_workers=args.num_workers)
        # E_best_train_loss = 0
        for i in range(1):
            E_source_first_feature, E_source_pred, E_source_first_label, E_source_acc, E_source_loss = inference(args, model_first, source_first_loader)
            E_target_first_feature, E_target_pred, E_target_first_label, E_target_acc, _ = inference(args, model_first, target_first_loader)
            # _, _, _, source_acc = inference(args, model_first, source_first_loader)
            if total_best_train_loss > E_source_loss:
                best_E_source_first_feature = E_source_first_feature
                best_E_target_first_feature = E_target_first_feature
                print(f'Epoch: {em_step}, E step train acc: {E_source_acc}, target acc: {E_target_acc.item()}')
                torch.save(model_first.state_dict(),
                           os.path.join(current_path, f'pretraining/E_first_{args.dataset_name}_E.pth'))

        mnn_model = MNN_GNN(args, num_features=source_first.num_node_features, num_classes=source_first.num_classes,
                            conv_type='GCN', pool_type=args.pool_type).to(args.device)

        E_feature = torch.cat((best_E_source_first_feature, best_E_target_first_feature), dim=0)
        E_label = torch.cat((E_source_first_label, E_target_first_label), dim=0)

        MNN_training(args, mnn_model, edge_index, E_feature, E_label, best_E_source_first_feature.shape[0])

        E_pred = mnn_model(E_feature, edge_index)
        E_target_pred = E_pred[E_source_first_feature.shape[0]:]
        E_target_pred = torch.nn.Softmax(-1)(E_target_pred)
        # print(E_target_pred)

        # M setp
        pesudo_first_idx = torch.where(E_target_pred > args.m_threshold)[0]
        pesudo_first_label = E_target_pred[pesudo_first_idx].argmax(dim=-1)

        ture_first_label = E_target_first_label[pesudo_first_idx]
        m_acc = (pesudo_first_label == ture_first_label).sum() / len(pesudo_first_idx)
        print('m_acc:', (pesudo_first_label == ture_first_label).sum() / len(pesudo_first_idx))
        # print(len(pesudo_second_idx))
        m_step_label_acc.append(m_acc.item())

        if '->' in args.dataset_name:
            name = args.dataset_name
            dataset_name = args.dataset_name.split('->')
            args.dataset_name = dataset_name[0]
            train_second = query_data(args)
            source_second = train_second[args.source_idx]
            val_second = train_second[args.val_idx]
            args.dataset_name = dataset_name[1]
            target_second_copy = query_data(args)
            total_source_second = train_second
            args.dataset_name = name
        else:
            dataset = query_data(args)
            source_second = dataset[args.source_idx]
            val_second = dataset[args.val_idx]
            target_second_copy = dataset[args.target_idx]
            total_source_second = dataset[args.source_idx + args.val_idx]
        # target_second_copy = deepcopy(target_second)

        if args.dataset_name == 'Tox21':
            target_second_copy.y[pesudo_first_idx,4] = pesudo_first_label.float().cpu().numpy()
        else:
            target_second_copy.y[pesudo_first_idx.cpu()] = pesudo_first_label.cpu().numpy()

        if '->' in args.dataset_name:
            if source_second.num_node_features > target_second_copy.num_node_features:
                for i in pesudo_first_idx:
                    target_second_copy[i].x = torch.cat((target_second_copy[i].x, torch.zeros(target_second_copy[i].x.shape[0], args.feat_num-target_second_copy[i].x.shape[-1]).to(target_second_copy[i].x.device)),dim=-1).to(target_second_copy[i].x.device)
            else:
                for i in source_second:
                    i.x = torch.cat((i.x, torch.zeros(i.x.shape[0], args.feat_num-i.x.shape[-1]).to(i.x.device)),dim=-1).to(i.x.device)
        # print(len(target_second_copy))
        M_training_data = source_second + target_second_copy[pesudo_first_idx]
        print('M step',len(M_training_data), len(source_second), len(pesudo_first_idx), len(target_second))
        M_train_loader = DataLoader(M_training_data, args.batch_size, shuffle=True, num_workers=args.num_workers)

        # M_val_best = 0
        for i in range(1):
            train_loss, train_acc, all_feature, y, pred = train(args, model_second, M_train_loader, optimizer_second, loss_func)
            M_source_second_feature, M_source_pred, M_source_second_label, M_source_acc, M_source_loss = inference(args, model_second,
                                                                                                  source_second_loader)
            M_target_second_feature, M_target_pred, M_target_second_label, M_target_acc, _ = inference(args, model_second,
                                                                                            target_second_loader)
            # _, _, _, source_acc = inference(args, model_second,source_second_loader)
            if total_best_train_loss > M_source_loss:

                print(f'Epoch: {em_step}, M step train acc: {M_source_acc}, target acc: {M_target_acc.item()}')
                torch.save(model_second.state_dict(),
                           os.path.join(current_path, f'pretraining/M_second_{args.dataset_name}_M.pth'))
    print(f'e_step_select_pseudo_label_acc:{e_step_label_acc}')
    print(f'm_step_select_pseudo_label_acc:{m_step_label_acc}')

parser = argparse.ArgumentParser()

# model params
parser.add_argument('--method', type=str, choices=['new', 'graphcl', 'baseline', 'gla', 'causal',
                                                   'GraphMLPMixer','GraphViT','MPGNN','GraphMLPMixer4TreeNeighbour',
                                                   'MPGNN4TreeNeighbour'], default='first')
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--conv_type', type=str, choices=['GCN', 'SAGE', 'GAT', 'GIN','GMT'], default='GMT')
parser.add_argument('--pool_type', type=str, choices=['TopK', 'Edge', 'SAG', 'ASA','GMT'], default='GMT')
parser.add_argument('--layer-norm', type=bool, default=True)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=50)
# parser.add_argument('--e_epochs', type=int, default=100)
parser.add_argument('--device', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--seed', type=int, default=123456789)
parser.add_argument('--dataset_name', type=str, default="Mutagenicity")
parser.add_argument('--source', type=int, default=1)
parser.add_argument('--target', type=int, default=0)
parser.add_argument('--cutoff', type=int, default=3, metavar='N', help='Max number of nodes in paths (path length +1)')

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
parser.add_argument('--e_threshold', type=float, default=0.9)#0.6 cox
parser.add_argument('--m_threshold', type=float, default=0.9)
parser.add_argument('--path-type', default='all_shortest_paths', help='Type of extracted path')
parser.add_argument('--residuals', default = False, action = 'store_true', help='Whether to use residual connection in the update equation.')
# second model params

args = parser.parse_args()

setup_seed(args.seed)
# print(torch.cuda.is_available())
if args.device >= 0:
    args.device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
else:
    args.device = torch.device("cpu")

# for args.dataset_name in ['COX2->COX2_MD','COX2_MD->COX2','BZR->BZR_MD','BZR_MD->BZR','PROTEINS->DD','DD->PROTEINS']:
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
    pretraining(args)
    EM_training_wo_val(args)
else:
    dataset = query_data(args)
    source_idx = np.load('idx/node_%s_%d.npy' % (args.dataset_name, args.source))
    target_idx = np.load('idx/node_%s_%d.npy' % (args.dataset_name, args.target))
    # print(source_idx)
    val_index = sample(list(source_idx), int(0.05 * len(source_idx)))
    train_index = [item for item in source_idx if item not in val_index]
    args.source_idx = train_index
    args.val_idx = val_index
    args.target_idx = target_idx

    for args.source in [0,1,2,3]:
        for args.target in [0,1,2,3]:
            print(f'source {args.source} -> target {args.target}')
            if args.source == args.target:
                continue
            else:
                # try:
                pretraining(args)
                EM_training_wo_val(args)
                # except:
                #     continue



