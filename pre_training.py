import os.path as osp
import argparse
import time

import torch
from torch_geometric.loader import DataLoader
import os
from tqdm import tqdm
from random import sample
import random
import numpy as np
from models.GNN import GNN, MNN_GNN
import scipy.sparse as sp
from utils import query_data, create_model
from models.model import PathNN
# from scipy.spatial import distance_matrix
import torch.nn.functional as F
import json
from scipy.stats import beta
from sklearn.model_selection import StratifiedShuffleSplit
current_path = os.getcwd()

def distance_matrix(source, target, threshold=1000000):

    m, k = source.shape
    n, _ = target.shape

    if m*n*k < threshold:
        source = source.unsqueeze(1)
        target = target.unsqueeze(0)
        result = torch.sum((source - target) ** 2, dim=-1) ** (0.5)
    else:
        result = torch.empty((m, n))
        if m < n:
            for i in range(m):
                result[i, :] = torch.sum((source[i] - target)**2,dim=-1)**(0.5)
        else:
            for j in range(n):
                result[:, j] = torch.sum((source - target[j])**2,dim=-1)**(0.5)
    return result

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def add_pose(data):
    data.x = torch.cat([data.x, data.pos], dim=-1)
    return data

def train(args, model, train_loader, optimizer, loss_func):
    model.train()

    total_loss = 0
    total_correct = 0
    all_feature = None
    all_y = None
    all_fc = None
    all_pred = None
    for data in train_loader:

        optimizer.zero_grad()
        data = data.to(args.device)
        feature = model(data)
        # break
        # print(feature.shape)
        if args.dataset_name == 'TRIANGLES':
            y = data.y - 1
        else:
            y = data.y
        # print(y.shape)
        if args.dataset_name == 'Tox21':
            y = y[:,4]
        fc = model.readout(feature, y)
        out = model.predict(fc)
        # print(out.shape)
        if all_feature == None:
            all_feature = feature
            all_fc = fc
            all_y = y
            all_pred = out
        else:
            all_feature = torch.cat((all_feature, feature), dim=0)
            all_fc = torch.cat((all_fc, fc), dim=0)
            all_y = torch.cat((all_y, y), dim=0)
            all_pred = torch.cat((all_pred, out),dim=0)
        # print(out, data.y)

        # print(y)
        loss = loss_func(out, y.long())
        # print('loss',loss)
        loss.backward()
        total_loss += data.num_graphs * float(loss)
        # print(total_loss)
        total_correct += int((out.argmax(dim=-1) == y).sum())
        optimizer.step()
    return total_loss / len(train_loader.dataset), total_correct/ len(train_loader.dataset), all_feature, all_y, all_pred


@torch.no_grad()
def test(args, model, loader,stage='test',feature=False):
    model.eval()

    total_correct = 0
    all_feature = None
    all_y = None
    all_fc = None
    all_label = None
    all_pred = None
    for data in loader:
        data = data.to(args.device)
        feature = model(data)
        fc = model.readout(feature)
        out = model.predict(fc)
        if args.dataset_name == 'TRIANGLES':
            label = data.y - 1
        else:
            label = data.y
        if args.dataset_name == 'Tox21':
            label = label[:,4]
        # label = data.y
        # label[label > args.num_class] = args.num_class

        if all_feature == None:
            all_feature = feature
            all_fc = fc
            all_label = label
            # all_y = label
            all_y = out.argmax(dim=-1)
            all_pred = out
        else:
            all_feature = torch.cat((all_feature, feature), dim=0)
            all_fc = torch.cat((all_fc, fc), dim=0)
            all_label = torch.cat((all_label, label), dim=0)
            all_y = torch.cat((all_y, out.argmax(dim=-1)), dim=0)
            all_pred = torch.cat((all_pred, out), dim=0)

        total_correct += int((out.argmax(dim=-1) == label).sum())
    # if feature == False:
    return total_correct / len(loader.dataset), all_feature, all_y, all_pred, all_label
    # else:
    #     return total_correct / len(loader.dataset), all_feature, all_y, all_fc

def first_brunch(args):
    print('Pre-training first brunch')
    args.method = 'first'

    if '->' in args.dataset_name:
        name = args.dataset_name
        dataset_name = args.dataset_name.split('->')
        args.dataset_name = dataset_name[0]
        source_dataset = query_data(args)
        args.dataset_name = dataset_name[1]
        target_dataset = query_data(args)
        train_dataset = source_dataset[args.source_idx]
        val_dataset = source_dataset[args.val_idx]
        test_dataset = target_dataset
        args.dataset_name = name
    else:
        dataset = query_data(args)
        train_dataset = dataset[args.source_idx]
        val_dataset = dataset[args.val_idx]
        test_dataset = dataset[args.target_idx]

    # if '->' in args.dataset_name:
    #     if train_dataset.num_node_features > test_dataset.num_node_features:
    #         print(train_dataset.num_node_features , test_dataset.num_node_features)
    #         for feature_idx in range(len(test_dataset.features)):
    #             test_dataset.features[feature_idx] = torch.cat((torch.from_numpy(test_dataset.features[feature_idx]), torch.zeros((test_dataset.features[feature_idx].shape[0],
    #                                                                           train_dataset.num_node_features - test_dataset.num_node_features))),
    #                                         dim=-1)
    #     else:
    #         for feature_idx in range(len(train_dataset.features)):
    #             train_dataset.features[feature_idx] = torch.cat((torch.from_numpy(train_dataset.features[feature_idx]), torch.zeros((train_dataset.features[feature_idx].shape[0],
    #                                                                              test_dataset.num_node_features - train_dataset.num_node_features))),
    #                                          dim=-1)
    #         for feature_idx_val in range(len(val_dataset.features)):
    #             val_dataset.features[feature_idx_val] = torch.cat((val_dataset.features[feature_idx_val], torch.zeros((val_dataset.features[feature_idx_val].shape[0],
    #                                                                          test_dataset.num_node_features - train_dataset.num_node_features))),
    #                                        dim=-1)

    model = GNN(args, num_features=max(train_dataset.num_node_features, test_dataset.num_node_features), num_classes=train_dataset.num_classes,
        conv_type=args.conv_type, pool_type=args.pool_type).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_func = torch.nn.CrossEntropyLoss()

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128)
    test_loader = DataLoader(test_dataset, batch_size=128)

    best_val = 0

    for epoch in tqdm(range(1, args.epochs+1)):
        train_loss, train_acc, feature, label, fc = train(args, model, train_loader, optimizer, loss_func)
        # print(model.feature)
        val_acc, val_feature, val_label, val_fc,_ = test(args, model, val_loader)
        test_acc, test_feature, test_label, _, _ = test(args, model, test_loader)
        if val_acc > best_val:
            best_val = val_acc
            best_gmt_feature = feature
            best_gmt_fc = fc
            best_gmt_val_fc = val_fc
            best_gmt_train_feature = feature
            best_gmt_val_feature = val_feature
            best_gmt_target_feature = test_feature
            torch.save(model.state_dict(),
                       os.path.join(current_path, f'pretraining/first_{args.dataset_name}.pth'))
        print(f'Epoch: {epoch:03d}, Loss: {train_loss:.4f}, train acc: {train_acc}'
                  f'Val Acc: {best_val:.4f}, Test Acc: {test_acc:.4f}')

    with open(f"tmp/first_{args.dataset_name}_train_feature.txt", "w") as log_file:
        np.savetxt(log_file, best_gmt_train_feature.cpu().detach().numpy())

    with open(f"tmp/first_{args.dataset_name}_train_label.txt", "w") as log_file:
        np.savetxt(log_file, label.cpu().detach().numpy())

    with open(f"tmp/first_{args.dataset_name}_val_feature.txt", "w") as log_file:
        np.savetxt(log_file, best_gmt_val_feature.cpu().detach().numpy())

    with open(f"tmp/first_{args.dataset_name}_val_label.txt", "w") as log_file:
        np.savetxt(log_file, val_label.cpu().detach().numpy())

    with open(f"tmp/first_{args.dataset_name}_test_feature.txt", "w") as log_file:
        np.savetxt(log_file, best_gmt_target_feature.cpu().detach().numpy())

    with open(f"tmp/first_{args.dataset_name}_test_label.txt", "w") as log_file:
        np.savetxt(log_file, test_label.cpu().detach().numpy())

    return best_gmt_train_feature, best_gmt_val_feature, best_gmt_target_feature, label, val_label, test_label

def second_brunch(args):
    print('Pre-training second brunch')
    args.method = 'second'
    if args.dataset_name == 'MNIST' or args.dataset_name == 'CIFAR10':
        args.lap_dim = 8

    # dataset = query_data(args)
    if '->' in args.dataset_name:
        name = args.dataset_name
        dataset_name = args.dataset_name.split('->')
        args.dataset_name = dataset_name[0]
        source_dataset = query_data(args)
        args.dataset_name = dataset_name[1]
        target_dataset = query_data(args)
        train_dataset = source_dataset[args.source_idx]
        val_dataset = source_dataset[args.val_idx]
        test_dataset = target_dataset
        args.dataset_name = name
    else:
        dataset = query_data(args)
        train_dataset = dataset[args.source_idx]
        val_dataset = dataset[args.val_idx]
        test_dataset = dataset[args.target_idx]
    # print(train_dataset[0])
    # if '->' in args.dataset_name:
    #     if train_dataset.num_node_features > test_dataset.num_node_features:
    #         test_dataset.features = torch.cat((test_dataset.features, torch.zeros((test_dataset.features.shape[0], train_dataset.num_node_features-test_dataset.num_node_features))),dim=-1)
    #     else:
    #         train_dataset.features = torch.cat((train_dataset.features, torch.zeros((train_dataset.features.shape[0],
    #                                                                           test_dataset.num_node_features - train_dataset.num_node_features))),
    #                                         dim=-1)
    #         val_dataset.features = torch.cat((val_dataset.features, torch.zeros((val_dataset.features.shape[0],
    #                                                                             test_dataset.num_node_features - train_dataset.num_node_features))),
    #                                          dim=-1)

        # if train_dataset.num_edge_features > test_dataset.num_edge_features:
        #     test_dataset.data.edge_attr = torch.cat((test_dataset.data.edge_attr, torch.zeros(test_dataset.data.edge_attr.shape[0], train_dataset.edge_attr-test_dataset.edge_attr)),dim=0)
        # else:
        #     train_dataset.data.edge_attr = torch.cat((train_dataset.data.edge_attr, torch.zeros((train_dataset.data.edge_attr.shape[0],
        #                                                                       test_dataset.edge_attr - train_dataset.edge_attr))),
        #                                     dim=-1)
        #     val_dataset.data.edge_attr = torch.cat((val_dataset.data.edge_attr, torch.zeros((val_dataset.data.edge_attr.shape[0],
        #                                                                         test_dataset.edge_attr - train_dataset.edge_attr))),
        #                                      dim=-1)

    args.feat_num = max(train_dataset.num_node_features, test_dataset.num_node_features)
    args.edge_num = max(train_dataset.num_edge_features, test_dataset.num_edge_features)
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, args.batch_size, num_workers=args.num_workers)
    # model = create_model(args).to(args.device)
    if args.path_type == 'all_simple_paths' :
        encode_distances = True
    else :
        encode_distances = False
    model = PathNN(args.feat_num, args.hidden_dim, args.cutoff, train_dataset.num_classes, args.dropout, args.device,
                        residuals = args.residuals, encode_distances=encode_distances
                        ).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_func = torch.nn.CrossEntropyLoss()

    best_val = 0
    for epoch in tqdm(range(args.epochs)):
        train_loss,train_acc,feature, label, fc = train(args, model, train_loader, optimizer, loss_func)
        val_acc, val_feature, val_label, val_fc,_ = test(args, model, val_loader)
        test_acc, test_feature, test_label, _,_ = test(args, model, test_loader)
        if val_acc > best_val:
            best_val = val_acc
            best_mix_val_feature = val_feature
            best_mix_train_feature = feature
            best_mix_test_feature = test_feature
            best_mix_fc = fc
            best_mix_val_fc = val_fc
            torch.save(model.state_dict(),
                       os.path.join(current_path, f'pretraining/second_{args.dataset_name}.pth'))

        print(f'Epoch: {epoch:03d}, Loss: {train_loss:.4f}, Train acc: {train_acc}, Val Acc: {best_val:.4f}, Test Acc: {test_acc:.4f}')

    with open(f"tmp/second_{args.dataset_name}_train_feature.txt", "w") as log_file:
        np.savetxt(log_file, best_mix_train_feature.cpu().detach().numpy())

    with open(f"tmp/second_{args.dataset_name}_train_label.txt", "w") as log_file:
        np.savetxt(log_file, label.cpu().detach().numpy())

    with open(f"tmp/second_{args.dataset_name}_val_feature.txt", "w") as log_file:
        np.savetxt(log_file, best_mix_val_feature.cpu().detach().numpy())

    with open(f"tmp/second_{args.dataset_name}_val_label.txt", "w") as log_file:
        np.savetxt(log_file, val_label.cpu().detach().numpy())

    with open(f"tmp/second_{args.dataset_name}_test_feature.txt", "w") as log_file:
        np.savetxt(log_file, best_mix_test_feature.cpu().detach().numpy())

    with open(f"tmp/second_{args.dataset_name}_test_label.txt", "w") as log_file:
        np.savetxt(log_file, test_label.cpu().detach().numpy())

    return best_mix_train_feature, best_mix_val_feature, best_mix_test_feature, label, val_label, test_label

def first_brunch_wo_val(args):
    print('Pre-training first brunch')
    args.method = 'first'

    if '->' in args.dataset_name:
        name = args.dataset_name
        dataset_name = args.dataset_name.split('->')
        args.dataset_name = dataset_name[0]
        source_dataset = query_data(args)
        args.dataset_name = dataset_name[1]
        target_dataset = query_data(args)
        train_dataset = source_dataset
        # val_dataset = source_dataset[args.val_idx]
        test_dataset = target_dataset
        args.dataset_name = name
    else:
        dataset = query_data(args)
        train_dataset = dataset[args.source_idx+args.val_idx]
        # val_dataset = dataset[args.val_idx]
        test_dataset = dataset[args.target_idx]

    # if '->' in args.dataset_name:
    #     if train_dataset.num_node_features > test_dataset.num_node_features:
    #         print(train_dataset.num_node_features , test_dataset.num_node_features)
    #         for feature_idx in range(len(test_dataset.features)):
    #             test_dataset.features[feature_idx] = torch.cat((torch.from_numpy(test_dataset.features[feature_idx]), torch.zeros((test_dataset.features[feature_idx].shape[0],
    #                                                                           train_dataset.num_node_features - test_dataset.num_node_features))),
    #                                         dim=-1)
    #     else:
    #         for feature_idx in range(len(train_dataset.features)):
    #             train_dataset.features[feature_idx] = torch.cat((torch.from_numpy(train_dataset.features[feature_idx]), torch.zeros((train_dataset.features[feature_idx].shape[0],
    #                                                                              test_dataset.num_node_features - train_dataset.num_node_features))),
    #                                          dim=-1)
    #         for feature_idx_val in range(len(val_dataset.features)):
    #             val_dataset.features[feature_idx_val] = torch.cat((val_dataset.features[feature_idx_val], torch.zeros((val_dataset.features[feature_idx_val].shape[0],
    #                                                                          test_dataset.num_node_features - train_dataset.num_node_features))),
    #                                        dim=-1)

    model = GNN(args, num_features=max(train_dataset.num_node_features, test_dataset.num_node_features), num_classes=train_dataset.num_classes,
        conv_type=args.conv_type, pool_type=args.pool_type).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_func = torch.nn.CrossEntropyLoss()

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=128)
    test_loader = DataLoader(test_dataset, batch_size=128)

    best_train_loss = 10000

    for epoch in tqdm(range(1, args.epochs+1)):
        train_loss, train_acc, feature, label, fc = train(args, model, train_loader, optimizer, loss_func)
        # print(model.feature)
        # val_acc, val_feature, val_label, val_fc,_ = test(args, model, val_loader)
        test_acc, test_feature, test_label, _, _ = test(args, model, test_loader)
        if best_train_loss > train_loss:
            best_train_loss = train_loss
            # best_val = val_acc
            # best_gmt_feature = feature
            # best_gmt_fc = fc
            # best_gmt_val_fc = val_fc
            best_gmt_train_feature = feature
            # best_gmt_val_feature = val_feature
            best_gmt_target_feature = test_feature
            torch.save(model.state_dict(),
                       os.path.join(current_path, f'pretraining/first_{args.dataset_name}.pth'))
            print(f'Epoch: {epoch:03d}, Loss: {train_loss:.4f}, train acc: {train_acc}'
                  f' Test Acc: {test_acc:.4f}')

    with open(f"tmp/first_{args.dataset_name}_train_feature.txt", "w") as log_file:
        np.savetxt(log_file, best_gmt_train_feature.cpu().detach().numpy())

    with open(f"tmp/first_{args.dataset_name}_train_label.txt", "w") as log_file:
        np.savetxt(log_file, label.cpu().detach().numpy())

    # with open(f"tmp/first_{args.dataset_name}_val_feature.txt", "w") as log_file:
    #     np.savetxt(log_file, best_gmt_val_feature.cpu().detach().numpy())

    # with open(f"tmp/first_{args.dataset_name}_val_label.txt", "w") as log_file:
    #     np.savetxt(log_file, val_label.cpu().detach().numpy())

    with open(f"tmp/first_{args.dataset_name}_test_feature.txt", "w") as log_file:
        np.savetxt(log_file, best_gmt_target_feature.cpu().detach().numpy())

    with open(f"tmp/first_{args.dataset_name}_test_label.txt", "w") as log_file:
        np.savetxt(log_file, test_label.cpu().detach().numpy())

    return best_gmt_train_feature, best_gmt_target_feature, label, test_label

def second_brunch_wo_val(args):
    print('Pre-training second brunch')
    args.method = 'second'
    if args.dataset_name == 'MNIST' or args.dataset_name == 'CIFAR10':
        args.lap_dim = 8

    # dataset = query_data(args)
    if '->' in args.dataset_name:
        name = args.dataset_name
        dataset_name = args.dataset_name.split('->')
        args.dataset_name = dataset_name[0]
        source_dataset = query_data(args)
        args.dataset_name = dataset_name[1]
        target_dataset = query_data(args)
        train_dataset = source_dataset
        # val_dataset = source_dataset[args.val_idx]
        test_dataset = target_dataset
        args.dataset_name = name
    else:
        dataset = query_data(args)
        train_dataset = dataset[args.source_idx+args.val_idx]
        # val_dataset = dataset[args.val_idx]
        test_dataset = dataset[args.target_idx]
    # print(train_dataset[0])
    # if '->' in args.dataset_name:
    #     if train_dataset.num_node_features > test_dataset.num_node_features:
    #         test_dataset.features = torch.cat((test_dataset.features, torch.zeros((test_dataset.features.shape[0], train_dataset.num_node_features-test_dataset.num_node_features))),dim=-1)
    #     else:
    #         train_dataset.features = torch.cat((train_dataset.features, torch.zeros((train_dataset.features.shape[0],
    #                                                                           test_dataset.num_node_features - train_dataset.num_node_features))),
    #                                         dim=-1)
    #         val_dataset.features = torch.cat((val_dataset.features, torch.zeros((val_dataset.features.shape[0],
    #                                                                             test_dataset.num_node_features - train_dataset.num_node_features))),
    #                                          dim=-1)

        # if train_dataset.num_edge_features > test_dataset.num_edge_features:
        #     test_dataset.data.edge_attr = torch.cat((test_dataset.data.edge_attr, torch.zeros(test_dataset.data.edge_attr.shape[0], train_dataset.edge_attr-test_dataset.edge_attr)),dim=0)
        # else:
        #     train_dataset.data.edge_attr = torch.cat((train_dataset.data.edge_attr, torch.zeros((train_dataset.data.edge_attr.shape[0],
        #                                                                       test_dataset.edge_attr - train_dataset.edge_attr))),
        #                                     dim=-1)
        #     val_dataset.data.edge_attr = torch.cat((val_dataset.data.edge_attr, torch.zeros((val_dataset.data.edge_attr.shape[0],
        #                                                                         test_dataset.edge_attr - train_dataset.edge_attr))),
        #                                      dim=-1)

    args.feat_num = max(train_dataset.num_node_features, test_dataset.num_node_features)
    args.edge_num = max(train_dataset.num_edge_features, test_dataset.num_edge_features)
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
    # val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, args.batch_size, num_workers=args.num_workers)
    # model = create_model(args).to(args.device)
    if args.path_type == 'all_simple_paths' :
        encode_distances = True
    else :
        encode_distances = False
    model = PathNN(args.feat_num, args.hidden_dim, args.cutoff, train_dataset.num_classes, args.dropout, args.device,
                        residuals = args.residuals, encode_distances=encode_distances
                        ).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_func = torch.nn.CrossEntropyLoss()

    best_train_loss = 10000
    for epoch in tqdm(range(args.epochs)):
        train_loss,train_acc,feature, label, fc = train(args, model, train_loader, optimizer, loss_func)
        # val_acc, val_feature, val_label, val_fc,_ = test(args, model, val_loader)
        test_acc, test_feature, test_label, _,_ = test(args, model, test_loader)
        if best_train_loss > train_loss:
            best_train_loss = train_loss
            # best_mix_val_feature = val_feature
            best_mix_train_feature = feature
            best_mix_test_feature = test_feature
            # best_mix_fc = fc
            # best_mix_val_fc = val_fc
            torch.save(model.state_dict(),
                       os.path.join(current_path, f'pretraining/second_{args.dataset_name}.pth'))

            print(f'Epoch: {epoch:03d}, Loss: {train_loss:.4f}, Train acc: {train_acc}, Test Acc: {test_acc:.4f}')

    with open(f"tmp/second_{args.dataset_name}_train_feature.txt", "w") as log_file:
        np.savetxt(log_file, best_mix_train_feature.cpu().detach().numpy())

    with open(f"tmp/second_{args.dataset_name}_train_label.txt", "w") as log_file:
        np.savetxt(log_file, label.cpu().detach().numpy())

    # with open(f"tmp/second_{args.dataset_name}_val_feature.txt", "w") as log_file:
    #     np.savetxt(log_file, best_mix_val_feature.cpu().detach().numpy())
    #
    # with open(f"tmp/second_{args.dataset_name}_val_label.txt", "w") as log_file:
    #     np.savetxt(log_file, val_label.cpu().detach().numpy())

    with open(f"tmp/second_{args.dataset_name}_test_feature.txt", "w") as log_file:
        np.savetxt(log_file, best_mix_test_feature.cpu().detach().numpy())

    with open(f"tmp/second_{args.dataset_name}_test_label.txt", "w") as log_file:
        np.savetxt(log_file, test_label.cpu().detach().numpy())

    return best_mix_train_feature, best_mix_test_feature, label, test_label

def mnn(source_feature, target_feature, topk=5):
    # if train_path or val_path or test_path:
    #     train_feature = torch.from_numpy(np.loadtxt(train_path))
    #     val_feature = torch.from_numpy(np.loadtxt(val_path))
    #     test_feature = torch.from_numpy(np.loadtxt(test_path))
    #
    #     train_label = torch.from_numpy(np.loadtxt('tmp/first_Mutagenicity_train_label.txt'))
    #     val_label = torch.from_numpy(np.loadtxt('tmp/first_Mutagenicity_val_label.txt'))
    #     test_label = torch.from_numpy(np.loadtxt('tmp/first_Mutagenicity_test_label.txt'))
    #     source_label = torch.cat((train_label, val_label), dim=0)
    '''
    第一种方法，计算所有特征的距离，不仅仅使用source->target和target->source的pair作为graph
    '''
    # total_feature = torch.cat((train_feature, val_feature, test_feature), dim=0)
    # d_s_t = -distance_matrix(total_feature, total_feature)
    # t_s_topk_index = d_s_t.topk(topk, dim=-1).indices
    # total_adj = torch.zeros((total_feature.shape[0], total_feature.shape[0]))
    #
    # for i in range(total_feature.shape[0]):
    #     total_adj[i, t_s_topk_index[i]] = 1
    # total_adj = total_adj + torch.eye(total_feature.shape[0])


    '''
    第二种方法，使用sorce->target和target->source 相同的连接作为连接矩阵
    '''
    # source_feature = torch.cat((train_feature, val_feature), dim=0)
    # target_feature = test_feature

    d_s_t = -distance_matrix(source_feature, target_feature)

    t_s_topk_index = d_s_t.topk(topk, dim=-1).indices
    s_t_topk_index = d_s_t.T.topk(topk, dim=-1).indices
    # print(t_s_topk_index)
    # for i in range(source_feature.shape[0]):
    #     print(source_label[i])
    #     print(test_label[t_s_topk_index[i]])


    t_s_adjacency = torch.zeros((source_feature.shape[0], target_feature.shape[0]))
    s_t_adjacency = torch.zeros((target_feature.shape[0], source_feature.shape[0]))

    for i in range(source_feature.shape[0]):
        t_s_adjacency[i, t_s_topk_index[i]] = 1
    for j in range(target_feature.shape[0]):
        s_t_adjacency[j, s_t_topk_index[j]] = 1

    mnn_adjacency = t_s_adjacency * s_t_adjacency.T
    "构建source+target的邻接矩阵，其中source和source部分均为0，target和target部分也为0"
    total_feature = torch.cat((source_feature, target_feature), dim=0)
    total_adj = torch.zeros((total_feature.shape[0], total_feature.shape[0]))
    total_adj[:source_feature.shape[0], source_feature.shape[0]:] = mnn_adjacency
    total_adj[source_feature.shape[0]:, :source_feature.shape[0]] = mnn_adjacency.T
    total_adj = total_adj + torch.eye(total_adj.shape[0])

    adj = sp.coo_matrix(total_adj)
    values = adj.data
    indices = np.vstack((adj.row, adj.col))  # 我们真正需要的coo形式
    adj = torch.LongTensor(indices)  # PyG框架需要的coo形式

    return adj

