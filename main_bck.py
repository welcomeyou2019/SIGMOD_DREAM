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
from random import sample
from sklearn.model_selection import StratifiedShuffleSplit
current_path = os.getcwd()
# set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:6144

def pretraining(args):
    GMT_pretraining(args)
    GraphMix(args)

@torch.no_grad()
def inference(args, model, loader,stage='test'):
    model.eval()
    all_feature = None
    pred = None
    all_label = None
    total_correct = 0
    for data in loader:
        data = data.to(args.device)
        feature = model(data)
        fc = model.readout(feature,stage=stage)
        out = model.predict(fc)
        label = data.y

        if all_feature == None:
            all_feature = feature
            pred = out
            all_label = label
        else:
            all_feature = torch.cat((all_feature, feature), dim=0)
            pred = torch.cat((pred, out), dim=0)
            all_label = torch.cat((all_label, label), dim=0)

        # total_correct += int((out.argmax(dim=-1) == label).sum())
    total_correct = (pred.argmax(dim=-1)==all_label).sum()
    return all_feature, pred, all_label, total_correct / len(loader.dataset)

def E_step_training(args, model, edge_index, feature, label, train_idx, test_idx):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_func = torch.nn.CrossEntropyLoss()
    test_idx = test_idx.long()
    # for i in model.named_parameters():
    #     print(i)
    for epoch in range(args.e_epochs):
        optimizer.zero_grad()
        pred_feature = model(feature, edge_index)
        readout = model.readout(pred_feature)
        pred = model.predict(readout)

        loss = loss_func(pred[train_idx], label[train_idx])
        loss.backward()
        optimizer.step()

        test = pred[test_idx].argmax(dim=-1)
        correct = (test == label[test_idx]).sum()
        test_acc = correct / len(test_idx)

        org_test = pred[int(feature.shape[0]/2):].argmax(dim=-1)
        org_label = label[int(feature.shape[0]/2):]
        org_acc = (org_test == org_label).sum() / feature.shape[0]/2
        print(f'epoch: {epoch}, test acc: {test_acc}, org acc: {org_acc}')


def EM_training(args):
    args.method = 'first'

    if '->' in args.dataset_name:
        dataset_name = args.dataset.split('->')
        args.dataset_name = dataset_name[0]
        source_first = query_data(args)
        args.dataset_name = dataset_name[1]
        target_first = query_data(args)
    else:
        dataset = query_data(args)
        source_first = dataset[args.source_idx+args.val_idx]
        target_first = dataset[args.target_idx]

    source_first_loader = DataLoader(source_first, args.batch_size, shuffle=True, num_workers=args.num_workers)
    source_first_loader_order = DataLoader(source_first, args.batch_size, shuffle=False, num_workers=args.num_workers)
    target_first_loader = DataLoader(target_first, args.batch_size, shuffle=False, num_workers=args.num_workers)

    model_first = GNN(args, num_features=source_first.num_node_features,num_classes=source_first.num_classes,
                    conv_type=args.conv_type, pool_type=args.pool_type).to(args.device)
    model_first.load_state_dict(torch.load(f'pretraining/first_{args.dataset_name}.pth'))
    optimizer_first = torch.optim.Adam(model_first.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_func = torch.nn.CrossEntropyLoss()

    source_first_feature, source_first_pred, source_first_label, _ = inference(args, model_first, source_first_loader_order)
    target_first_feature, target_first_pred, target_first_label, _ = inference(args, model_first, target_first_loader)
    target_first_pred = torch.nn.Softmax(-1)(target_first_pred)

    # generate pesudo label for first brunch
    # threshold controls the balance of categories, large threshod increase the pesudo label acc, but would cause large number of one category
    pesudo_first_idx = torch.where(target_first_pred>args.e_threshold)[0]
    pesudo_first_label = target_first_pred[pesudo_first_idx].argmax(dim=-1)
    true_first_label = target_first_label[pesudo_first_idx]

    target_acc, _, _, _, _ = test(args, model_first, target_first_loader)
    print('Direct predict first acc:', target_acc)

    # print('first brunch pesudo label accuracy:',(true_first_label==pesudo_first_label).sum()/len(pesudo_first_idx))
    # print('ratio of catgories:', pesudo_first_label.sum()/len(pesudo_first_idx))

    args.method = 'second'
    if '->' in args.dataset_name:
        dataset_name = args.dataset.split('->')
        args.dataset_name = dataset_name[0]
        source_second = query_data(args)
        args.dataset_name = dataset_name[1]
        target_second = query_data(args)
    else:
        dataset = query_data(args)
        source_second = dataset[args.source_idx + args.val_idx]
        target_second = dataset[args.target_idx]

    # source_second_loader = DataLoader(source_second, args.batch_size, shuffle=True, num_workers=args.num_workers)
    target_second_loader = DataLoader(target_second, args.batch_size, shuffle=False, num_workers=args.num_workers)
    source_second_loader_order = DataLoader(source_second, args.batch_size, shuffle=False, num_workers=args.num_workers)

    model_second = create_model(args).to(args.device)
    model_second.load_state_dict(torch.load(f'pretraining/second_{args.dataset_name}.pth'))
    optimizer_second = torch.optim.Adam(model_second.parameters(), lr=args.lr, weight_decay=1e-4)

    target_acc, _, _, _, _ = test(args, model_second, target_second_loader)
    print('Direct predict second acc:', target_acc)

    source_second_feature, source_second_pred, source_second_label, _ = inference(args, model_second, source_second_loader_order)
    target_second_feature, target_second_pred, target_second_label, _ = inference(args, model_second, target_second_loader)

    pesudo_second_idx = torch.where(target_second_pred > args.e_threshold)[0]
    pesudo_second_label = target_second_pred[pesudo_second_idx].argmax(dim=-1)
    true_second_label = target_second_label[pesudo_second_idx]
    # print('second brunch pesudo label accuracy:', (true_second_label == pesudo_second_label).sum() / len(pesudo_second_idx))
    # print('ratio of catgories:', pesudo_second_label.sum() / len(pesudo_second_idx))

    # M step, generate mnn adj matrix for E step
    edge_index = mnn(source_second_feature, target_second_feature)

    # E step

    target_first[pesudo_second_idx].data.y = pesudo_second_label
    E_training_data = source_first + target_first[pesudo_second_idx]

    E_train_loader = DataLoader(E_training_data, args.batch_size, shuffle=True, num_workers=args.num_workers)
    # E_test_loader = DataLoader(E_test_data, args.batch_size, shuffle=True, num_workers=args.num_workers)

    for i in range(10):
        train_loss, train_acc, all_feature, y, pred = train(args, model_first, E_train_loader, optimizer_first, loss_func)
        E_source_first_feature, E_source_pred, E_source_first_label, E_source_acc = inference(args, model_first, source_first_loader_order)
        E_target_first_feature, E_target_pred, E_target_first_label, E_target_acc = inference(args, model_first, target_first_loader)
        # E_target_pred = torch.nn.Softmax(-1)(E_target_pred)
        print('E step target acc:', E_target_acc)

    mnn_model = MNN_GNN(args, num_features=source_first.num_node_features, num_classes=source_first.num_classes,
                        conv_type='GCN', pool_type=args.pool_type).to(args.device)

    E_feature = torch.cat((E_source_first_feature, E_target_first_feature), dim=0)
    # E_label = torch.cat((source_first_label, target_first_label), dim=0)
    # E_train_idx = torch.cat((torch.arange(len(source_first)), (pesudo_first_idx + len(source_first))), dim=0)
    # print(len(E_train_idx))
    # rest_first_idx = []
    # for i in range(len(target_first)):
    #     if i not in pesudo_first_idx:
    #         rest_first_idx.append(i)
    # rest_first_idx = torch.Tensor(rest_first_idx) + len(source_first)

    E_pred = mnn_model(E_feature, edge_index)
    E_target_pred = E_pred[:E_target_first_feature.shape[0]].argmax(-1)

    # M setp
    pesudo_first_idx = torch.where(E_target_pred > args.m_threshold)[0]
    pesudo_first_label = E_target_pred[pesudo_first_idx].argmax(dim=-1)
    # true_first_label = target_second_label[pesudo_first_idx]
    # print('acc:', (pesudo_first_label==true_first_label).sum()/len(pesudo_first_idx))
    target_second[pesudo_first_idx].data.y = pesudo_first_label
    M_training_data = source_second + target_second[pesudo_first_idx]
    M_train_loader = DataLoader(M_training_data, args.batch_size, shuffle=True, num_workers=args.num_workers)

    for i in range(10):
        train_loss, train_acc, all_feature, y, pred = train(args, model_second, M_train_loader, optimizer_second, loss_func)
        # target_acc, M_target_feature, _, M_target_pred, _ = test(args, model_second, target_second_loader)
        M_source_second_feature, M_source_pred, M_source_second_label, M_source_acc = inference(args, model_second,
                                                                                              source_second_loader_order)
        M_target_second_feature, M_target_pred, M_target_second_label, M_target_acc = inference(args, model_second,
                                                                                        target_second_loader)
        print('M step target acc:', M_target_acc)

# def fine_tune(args):
#     GMT_fine_tune(args)
#     GraphMix_fine_tune(args)

# def run_train(args):
#     print('start training main!')
#     args.method = 'baseline'
#     dataset_GMT = query_data(args)
#
#     # val_index = sample(args.source_idx, int(0.05 * len(args.source_idx)))
#     # train_index = [item for item in args.source_idx if item not in val_index]
#     # train_dataset = dataset_GMT[train_index]
#     # val_dataset = dataset_GMT[val_index]
#     # test_dataset = dataset_GMT[args.target_idx] + dataset_GMT[args.unknown]
#
#     source_GMT = dataset_GMT[args.source_idx]
#
#     target_GMT = dataset_GMT[args.target_idx] + dataset_GMT[args.unknown]
#     model_GMT = GNN(args, num_features=dataset_GMT.num_node_features, num_classes=int(dataset_GMT.num_classes*args.unknown_ratio)+1,
#             conv_type=args.conv_type, pool_type=args.pool_type,K=args.K).to(args.device)
#     model_GMT.load_state_dict(torch.load(f'pretraining/GMT_{args.dataset_name}_{args.device}.pth'))
#     source_GMT_loader = DataLoader(source_GMT, args.batch_size, shuffle=True, num_workers=args.num_workers)
#     target_GMT_loader = DataLoader(target_GMT, args.batch_size, shuffle=True, num_workers=args.num_workers)
#
#     target_acc,_,_,_ = test(args, model_GMT, target_GMT_loader)
#     print('Direct predict GMT acc:', target_acc)
#
#     args.method = 'GraphMLPMixer'
#     dataset_Mix = query_data(args)
#     if args.dataset_name == 'MNIST' or args.dataset_name == 'CIFAR10':
#         args.lap_dim = 8
#     model_MIX = create_model(args).to(args.device)
#     model_MIX.load_state_dict(torch.load(f'pretraining/GraphMix_{args.dataset_name}_{args.device}.pth'))
#     source_MIX = dataset_Mix[args.source_idx]
#     target_MIX = dataset_Mix[args.target_idx] + dataset_Mix[args.unknown]
#     source_MIX_loader = DataLoader(source_MIX, args.batch_size, shuffle=True, num_workers=args.num_workers)
#     target_MIX_loader = DataLoader(target_MIX, args.batch_size, shuffle=True, num_workers=args.num_workers)
#
#     target_MIX_acc, _, _, _ = test(args, model_MIX, target_MIX_loader)
#     print('Direct predict MIX acc:', target_MIX_acc)
#
#     opt = torch.optim.Adam([
#         # {'params': model_GMT.parameters(), 'lr': args.lr},
#         {'params': model_MIX.parameters(), 'lr': args.lr},
#         ])
#     loss_func = torch.nn.CrossEntropyLoss()
#     # all_acc = []
#     # for i in range(10):
#     test_acc = train_ood(args, model_GMT, model_MIX, source_GMT_loader, target_GMT_loader, source_MIX_loader, target_MIX_loader,opt,loss_func)
#     return test_acc


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
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--e_epochs', type=int, default=100)
parser.add_argument('--device', type=int, default=3)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--seed', type=int, default=123456789)
parser.add_argument('--dataset_name', type=str, default="Mutagenicity")
parser.add_argument('--source', type=int, default=0)
parser.add_argument('--target', type=int, default=1)

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
print(torch.cuda.is_available())
if args.device >= 0:
    args.device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
else:
    args.device = torch.device("cpu")
print(args.device)

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

if '->' in args.dataset_name:
    print(args.dataset)
    dataset_name = args.dataset.split('->')
    args.dataset_name = dataset_name[0]
    source_dataset = query_data(args)
    args.dataset_name = dataset_name[1]
    target_dataset = query_data(args)
    val_index = sample(list(range(len(source_dataset))), int(0.05 * len(source_dataset)))
    source_idx = list(range(len(source_dataset)))
    train_index = [item for item in source_idx if item not in val_index]
    args.source_idx = source_idx
    args.val_idx = val_index
    args.test_idx = list(range(len(target_dataset)))
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


EM_training(args)
# pretraining(args)
# edge_index = mnn(f"tmp/MIX_{args.dataset_name}_train_feature.txt",f"tmp/MIX_{args.dataset_name}_val_feature.txt",f"tmp/MIX_{args.dataset_name}_test_feature.txt")

# fine_tune(args)

# all_acc = []
# # for i in range(5):
# test_acc = run_train(args)
# print(test_acc)
# print(max(test_acc))
#
# with open(f"{args.dataset_name}_{args.device}.txt", "w") as log_file:
#     np.savetxt(log_file, np.array(test_acc))
    # all_acc.append(torch.tensor(test_acc))
# all_acc = torch.stack(all_acc, dim=0)
# acc_mean = all_acc.mean(dim=0)
# best_epoch = acc_mean.argmax().item()
# print(f'---------------- Best Epoch: {best_epoch} ----------------')
# print('Mean: {:7f}, Std: {:7f}'.format(all_acc[:, best_epoch].mean(), all_acc[:, best_epoch].std()), flush=True)


