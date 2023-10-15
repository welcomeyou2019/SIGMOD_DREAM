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
from models.GNN import GNN
from utils import query_data, create_model
# from scipy.spatial import distance_matrix
import torch.nn.functional as F
import json
from scipy.stats import beta
from sklearn.model_selection import StratifiedShuffleSplit
current_path = os.getcwd()

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
    for data in train_loader:
        if args.method == 'GraphMLPMixer' and model.use_lap:
            batch_pos_enc = data.lap_pos_enc
            sign_flip = torch.rand(batch_pos_enc.size(1))
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            data.lap_pos_enc = batch_pos_enc * sign_flip.unsqueeze(0)

        optimizer.zero_grad()
        data = data.to(args.device)
        feature = model(data)
        fc = model.readout(feature,data.y)
        out = model.predict(fc)
        if all_feature == None:
            all_feature = feature
            all_fc = fc
            all_y = data.y
        else:
            all_feature = torch.cat((all_feature, feature), dim=0)
            all_fc = torch.cat((all_fc, fc), dim=0)
            all_y = torch.cat((all_y, data.y), dim=0)

        loss = loss_func(out, data.y)
        loss.backward()
        total_loss += data.num_graphs * float(loss)
        total_correct += int((out.argmax(dim=-1) == data.y).sum())
        optimizer.step()
    return total_loss / len(train_loader.dataset), total_correct/ len(train_loader.dataset), all_feature, all_y, all_fc


@torch.no_grad()
def test(args, model, loader,stage='test'):
    model.eval()

    total_correct = 0
    all_feature = None
    all_y = None
    all_fc = None
    for data in loader:
        data = data.to(args.device)
        feature = model(data)
        fc = model.readout(feature,stage=stage)
        out = model.predict(fc)

        label = data.y
        label[label > args.num_class] = args.num_class

        if all_feature == None:
            all_feature = feature
            all_fc = fc
            # all_y = label
            all_y = out.argmax(dim=-1)
        else:
            all_feature = torch.cat((all_feature, feature), dim=0)
            all_fc = torch.cat((all_fc, fc), dim=0)
            all_y = torch.cat((all_y, out.argmax(dim=-1)), dim=0)

        total_correct += int((out.argmax(dim=-1) == label).sum())

    return total_correct / len(loader.dataset), all_feature, all_y, all_fc

def GMT_pretraining(args):
    args.method = 'baseline'
    dataset = query_data(args)
    # print(dataset)

    val_index = sample(args.source_idx, int(0.05*len(args.source_idx)))
    train_index = [item for item in args.source_idx if item not in val_index]
    train_dataset = dataset[train_index]
    val_dataset = dataset[val_index]
    test_dataset = dataset[args.target_idx] + dataset[args.unknown]

    model = GNN(args, num_features=dataset.num_node_features, num_classes=int(dataset.num_classes*args.unknown_ratio)+1,
        conv_type=args.conv_type, pool_type=args.pool_type).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_func = torch.nn.CrossEntropyLoss()

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128)
    test_loader = DataLoader(test_dataset, batch_size=128)

    best_val = 0

    for epoch in tqdm(range(1, args.epochs+1)):
        train_loss, _, feature, label, fc = train(args, model, train_loader, optimizer, loss_func)
        # print(model.feature)
        val_acc, val_feature, val_label, val_fc = test(args, model, val_loader)
        test_acc, _, _, _ = test(args, model, test_loader)
        if val_acc > best_val:
            best_val = val_acc
            best_gmt_feature = feature
            best_gmt_fc = fc
            best_gmt_val_fc = val_fc
            best_gmt_val_feature = val_feature
            torch.save(model.state_dict(),
                       os.path.join(current_path, f'pretraining/GMT_{args.dataset_name}.pth'))
            print(f'Epoch: {epoch:03d}, Loss: {train_loss:.4f}, '
                  f'Val Acc: {best_val:.4f}, Test Acc: {test_acc:.4f}')

    with open("tmp/GMT_feature.txt", "w") as log_file:
        np.savetxt(log_file, best_gmt_feature.cpu().detach().numpy())

    with open("tmp/GMT_label.txt", "w") as log_file:
        np.savetxt(log_file, label.cpu().detach().numpy())

    # with open("tmp/GMT_fc.txt", "w") as log_file:
    #     np.savetxt(log_file, best_gmt_fc.cpu().detach().numpy())
    #
    # with open("tmp/GMT_val_fc.txt", "w") as log_file:
    #     np.savetxt(log_file, best_gmt_val_fc.cpu().detach().numpy())

    with open("tmp/GMT_val_feature.txt", "w") as log_file:
        np.savetxt(log_file, best_gmt_val_feature.cpu().detach().numpy())

    with open("tmp/GMT_val_label.txt", "w") as log_file:
        np.savetxt(log_file, val_label.cpu().detach().numpy())

    return best_gmt_feature, label

def GraphMix(args):
    args.method = 'GraphMLPMixer'
    if args.dataset_name == 'MNIST' or args.dataset_name == 'CIFAR10':
        args.lap_dim = 8

    dataset = query_data(args)
    # print(dataset[0])
    val_index = sample(args.source_idx, int(0.05 * len(args.source_idx)))
    train_index = [item for item in args.source_idx if item not in val_index]
    train_dataset = dataset[train_index]
    val_dataset = dataset[val_index]
    test_dataset = dataset[args.target_idx] + dataset[args.unknown]

    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, args.batch_size, num_workers=args.num_workers)
    # print(args.device)
    model = create_model(args).to(args.device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr)
    loss_func = torch.nn.CrossEntropyLoss()

    best_val = 0
    for epoch in tqdm(range(args.epochs)):
        train_loss,_,feature, label, fc = train(args, model, train_loader, optimizer, loss_func)
        val_acc, val_feature, val_label, val_fc = test(args, model, val_loader)
        test_acc, _, _, _ = test(args, model, test_loader)
        if val_acc > best_val:
            best_val = val_acc
            best_mix_feature = feature
            best_mix_val_feature = val_feature
            best_mix_fc = fc
            best_mix_val_fc = val_fc
            torch.save(model.state_dict(),
                       os.path.join(current_path, f'pretraining/GraphMix_{args.dataset_name}.pth'))

            print(f'Epoch: {epoch:03d}, Loss: {train_loss:.4f}, Val Acc: {best_val:.4f}, Test Acc: {test_acc:.4f}')

    with open("tmp/MIX_feature.txt", "w") as log_file:
        np.savetxt(log_file, best_mix_feature.cpu().detach().numpy())

    with open("tmp/MIX_label.txt", "w") as log_file:
        np.savetxt(log_file, label.cpu().detach().numpy())

    with open("tmp/MIX_val_feature.txt", "w") as log_file:
        np.savetxt(log_file, best_mix_val_feature.cpu().detach().numpy())

    with open("tmp/MIX_val_label.txt", "w") as log_file:
        np.savetxt(log_file, val_label.cpu().detach().numpy())

    return best_mix_feature, label

def construct_ood(model):
    if model == 'GMT':
        feature = torch.from_numpy(np.loadtxt("tmp/GMT_feature.txt"))
        label = torch.from_numpy(np.loadtxt("tmp/GMT_label.txt"))
        val_feature = torch.from_numpy(np.loadtxt("tmp/GMT_val_feature.txt"))
        val_label = torch.from_numpy(np.loadtxt("tmp/GMT_val_label.txt"))
        # print(feature.shape, label.shape)
    elif model == 'GraphMix':
        feature = torch.from_numpy(np.loadtxt("tmp/MIX_feature.txt"))
        label = torch.from_numpy(np.loadtxt("tmp/MIX_label.txt"))
        val_feature = torch.from_numpy(np.loadtxt("tmp/MIX_val_feature.txt"))
        val_label = torch.from_numpy(np.loadtxt("tmp/MIX_val_label.txt"))
    # print(feature.shape,label.shape,val_label.shape, val_feature.shape)
    idx = np.arange(feature.shape[0])
    np.random.shuffle(idx)
    label_shuffle = label[idx]
    different_label_idx = torch.where(label!=label_shuffle)[0]
    weight = np.random.beta(2,2,1)[0]
    ood_feature = weight * feature[different_label_idx] + (1-weight) * feature[idx][different_label_idx]
    all_feature = torch.cat((feature, ood_feature[:int(feature.shape[0]/len(list(set(label.numpy())))),:]),dim=0)
    all_label = torch.cat((label, len(list(set(label.numpy())))*torch.ones(int(feature.shape[0]/len(list(set(label.numpy())))))),dim=0)
    # print(int(feature.shape[0]/len(list(set(label.numpy())))),int(feature.shape[0]/len(list(set(label.numpy())))*1.25))
    val_all_feature = torch.cat((val_feature, ood_feature[int(feature.shape[0]/len(list(set(label.numpy())))):int(feature.shape[0]/len(list(set(label.numpy())))*1.25),:]),dim=0)
    val_all_label = torch.cat((val_label, len(list(set(label.numpy())))*torch.ones(int(feature.shape[0]/len(list(set(label.numpy())))*1.25)-int(feature.shape[0]/len(list(set(label.numpy())))))),dim=0)
    # print(all_feature.shape, all_label.shape, val_all_label.shape, val_all_feature.shape)
    return all_feature, all_label, val_all_feature, val_all_label

def add_ood(args, feature, label):
    idx = np.arange(feature.shape[0])
    np.random.shuffle(idx)
    label_shuffle = label[idx]
    different_label_idx = torch.where(label != label_shuffle)[0]
    weight = np.random.beta(2, 2, 1)[0]
    ood_feature = weight * feature[different_label_idx] + (1 - weight) * feature[idx][different_label_idx]
    all_feature = torch.cat((feature, ood_feature), dim=0)[:int(feature.shape[0]*1.5),:]
    if args.dataset_name == 'COIL-DEL':
        all_label = torch.cat((label, int(100*args.unknown_ratio) * torch.ones(all_feature.shape[0]-feature.shape[0]).to(args.device)),dim=0)
    elif args.dataset_name == 'Letter-high':
        all_label = torch.cat((label, int(15 * args.unknown_ratio) * torch.ones(all_feature.shape[0]-feature.shape[0]).to(args.device)), dim=0)
    else:
        all_label = torch.cat((label, int(10 * args.unknown_ratio) * torch.ones(all_feature.shape[0]-feature.shape[0]).to(args.device)), dim=0)
    return all_feature, all_label

def GMT_fine_tune(args):
    if args.dataset_name == 'Letter-high':
        num_node_features = 2
        num_classes = 15
    elif args.dataset_name == 'COIL-DEL':
        num_node_features = 2
        num_classes = 100
    elif args.dataset_name == 'CIFAR10':
        num_node_features = 5
        num_classes = 10
    else:
        num_node_features = 3
        num_classes = 10

    model = GNN(args, num_features=num_node_features,
                num_classes=int(num_classes * args.unknown_ratio) + 1,
                conv_type=args.conv_type, pool_type=args.pool_type).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_func = torch.nn.CrossEntropyLoss()

    model.load_state_dict(torch.load(f'pretraining/GMT_{args.dataset_name}.pth'))
    for name, parameters in model.named_parameters():
        if 'linear' not in name:
            parameters.requires_grad = False
    feature, label, val_feature, val_label = construct_ood('GMT')
    feature = feature.float().to(args.device)
    label = label.long().to(args.device)
    val_feature = val_feature.float().to(args.device)
    val_label = val_label.long().to(args.device)

    total_feature = torch.cat((feature, val_feature),dim=0)
    total_label = torch.cat((label, val_label), dim=0)
    with open("tmp/GMT_ood_feature.txt", "w") as log_file:
        np.savetxt(log_file, total_feature.cpu().detach().numpy())

    with open("tmp/GMT_ood_label.txt", "w") as log_file:
        np.savetxt(log_file, total_label.cpu().detach().numpy())

    best_val = 0
    for epoch in tqdm(range(args.epochs)):
        optimizer.zero_grad()
        fc = model.readout(feature,label)
        out = model.predict(fc)
        loss = loss_func(out, label)
        loss.backward()
        optimizer.step()

        val_fc = model.readout(val_feature, val_label)
        val_out = model.predict(val_fc)
        total_correct = int((val_out.argmax(dim=-1) == val_label).sum())
        val_acc = total_correct / val_feature.shape[0]
        if best_val < val_acc:
            best_val = val_acc
            torch.save(model.state_dict(),
                       os.path.join(current_path, f'fine_tune/GMT_{args.dataset_name}.pth'))

        print(f'Epoch: {epoch:03d}, Val Acc: {best_val:.4f}')

def GraphMix_fine_tune(args):
    if args.dataset_name == 'Letter-high':
        num_node_features = 2
        num_classes = 15
    elif args.dataset_name == 'COIL-DEL':
        num_node_features = 2
        num_classes = 100
    elif args.dataset_name == 'CIFAR10':
        num_node_features = 5
        num_classes = 10
    else:
        num_node_features = 3
        num_classes = 10

    args.method = 'GraphMLPMixer'
    model = create_model(args).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_func = torch.nn.CrossEntropyLoss()

    model.load_state_dict(torch.load(f'pretraining/GraphMix_{args.dataset_name}.pth'))
    for name, parameters in model.named_parameters():
        if 'output_decoder' not in name:
            parameters.requires_grad = False
    feature, label, val_feature, val_label = construct_ood('GraphMix')
    feature = feature.float().to(args.device)
    label = label.long().to(args.device)
    val_feature = val_feature.float().to(args.device)
    val_label = val_label.long().to(args.device)

    total_feature = torch.cat((feature, val_feature), dim=0)
    total_label = torch.cat((label, val_label), dim=0)
    with open("tmp/MIX_ood_feature.txt", "w") as log_file:
        np.savetxt(log_file, total_feature.cpu().detach().numpy())

    with open("tmp/MIX_ood_label.txt", "w") as log_file:
        np.savetxt(log_file, total_label.cpu().detach().numpy())

    best_val = 0
    for epoch in tqdm(range(args.epochs)):
        optimizer.zero_grad()
        fc = model.readout(feature, label)
        out = model.predict(fc)
        loss = loss_func(out, label)
        loss.backward()
        optimizer.step()

        val_fc = model.readout(val_feature, val_label)
        val_out = model.predict(val_fc)
        total_correct = int((val_out.argmax(dim=-1) == val_label).sum())
        val_acc = total_correct / val_feature.shape[0]
        if best_val < val_acc:
            best_val = val_acc
            torch.save(model.state_dict(),
                       os.path.join(current_path, f'fine_tune/GraphMix_{args.dataset_name}.pth'))

        print(f'Epoch: {epoch:03d}, Val Acc: {best_val:.4f}')

def calculate_center(num_class, feature, label):
    unique_class = torch.unique(label)
    # print(unique_class)
    center = torch.zeros(num_class+1, feature.shape[-1])
    for j in unique_class:
        idx = torch.where(label==j)[0]
        # print(f'class {j} idx: {idx}, length: {len(idx)}, feature shape: {feature.shape}')
        center[j.long()] = torch.mean(feature[idx], dim=0)
    return center

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

def mirror_loss(num_class, topk, source_feature, target_feature, source_label, target_label):
    # t1 = time.time()
    d_t_s = distance_matrix(target_feature, source_feature)
    # t2 = time.time()
    # print('calculate distance:',t2-t1)
    t_s_topk_index = d_t_s.topk(topk, dim=-1).indices
    s_t_topk_index = d_t_s.T.topk(topk, dim=-1).indices
    target_map_feature = torch.zeros(source_feature.shape)
    source_map_feature = torch.zeros(target_feature.shape)
    for i in range(t_s_topk_index.shape[0]):
        idx_t_s = t_s_topk_index[i]
        idx_s_t = s_t_topk_index[i]
        source_map_feature[i] = torch.mean(source_feature[idx_t_s],dim=0)
        target_map_feature[i] = torch.mean(target_feature[idx_s_t], dim=0)

    # d_s_t = distance_matrix(source_feature.detach().cpu().numpy(), target_feature.detach().cpu().numpy())
    # s_t_topk_index = torch.from_numpy(d_s_t).topk(topk, dim=-1).indices
    # d_s_t = distance_matrix(source_feature, target_feature)

    # for i in range(s_t_topk_index.shape[0]):
    #     idx = s_t_topk_index[i]
    #     target_map_feature[i] = torch.mean(target_feature[idx],dim=0)

    source_center = calculate_center(num_class, source_feature, source_label).unsqueeze(1).to(target_feature.device)
    target_center = calculate_center(num_class, target_feature, target_label).unsqueeze(1).to(target_feature.device)
    target_feature_r = target_feature.unsqueeze(0).repeat(num_class+1,1,1)
    source_feature_r = source_feature.unsqueeze(0).repeat(num_class+1, 1, 1)
    source_map_feature_r = source_map_feature.unsqueeze(0).repeat(num_class+1,1,1).to(target_feature.device)
    target_map_feature_r = target_map_feature.unsqueeze(0).repeat(num_class+1, 1, 1).to(target_feature.device)

    diff_t = target_feature_r - target_center
    d_t = -torch.norm(diff_t, dim=-1).transpose(1,0)
    q_t = torch.softmax(d_t, dim=-1)

    diff_s = source_feature_r - source_center
    d_s = -torch.norm(diff_s, dim=-1).transpose(1,0)
    q_s = torch.softmax(d_s, dim=-1)

    diff_s_t = source_map_feature_r - source_center
    d_s_t = -torch.norm(diff_s_t, dim=-1).transpose(1, 0)
    q_s_t = torch.softmax(d_s_t, dim=-1)

    diff_t_s = target_map_feature_r - target_center
    d_t_s = -torch.norm(diff_t_s, dim=-1).transpose(1, 0)
    q_t_s = torch.softmax(d_t_s, dim=-1)
    # print(q_t,q_s_t,q_s,q_t_s)
    mirror_loss = F.kl_div(q_t.log(), q_s_t, reduction='mean') + F.kl_div(q_s.log(), q_t_s, reduction='mean')
    # mirror_loss = F.cross_entropy(q_t, q_s_t, reduction='mean') + F.cross_entropy(q_s, q_t_s, reduction='mean')

    return mirror_loss

def align_label(args, y):
    if args.dataset_name == 'COIL-DEL':
        y[y > int(args.unknown_ratio * 100)] = int(args.unknown_ratio * 100)
    elif args.dataset_name == 'Letter-high':
        y[y > int(args.unknown_ratio * 15)] = int(args.unknown_ratio * 15)
    else:
        y[y > int(args.unknown_ratio * 10)] = int(args.unknown_ratio * 10)
    return y

def train_ood(args, model_GMT, model_MIX, source_GMT_loader, target_GMT_loader, source_MIX_loader, target_MIX_loader,optimizer,loss_func,K=8192):
    total_acc = []
    for epoch in tqdm(range(200)):
        total_source_GMT_feature = None
        total_source_GMT_fc = None
        total_source_GMT_label = None
        total_target_GMT_feature = None
        total_target_GMT_fc = None
        total_target_GMT_label = None

        total_source_MIX_feature = None
        total_source_MIX_fc = None
        total_source_MIX_label = None
        total_target_MIX_feature = None
        total_target_MIX_fc = None
        total_target_MIX_label = None

        # source_ptr = 0
        # target_ptr = 0

        source_loss = 0
        source_loss_1 = 0
        total_GMT_correct = 0
        total_MIX_correct = 0
        simility_loss = 0

        optimizer.zero_grad()

        if args.dataset_name in ['MNIST','CIFAR10']:
            for idx, (source_GMT_data, target_GMT_data,source_MIX_data, target_MIX_data) in enumerate(zip(source_GMT_loader,target_GMT_loader,source_MIX_loader, target_MIX_loader)):
                source_GMT_data, target_GMT_data, source_MIX_data, target_MIX_data = source_GMT_data.to(args.device), target_GMT_data.to(args.device), source_MIX_data.to(args.device), target_MIX_data.to(args.device)
                source_GMT_feature_tmp = model_GMT(source_GMT_data)
                source_GMT_feature, source_GMT_label = add_ood(args, source_GMT_feature_tmp, source_GMT_data.y)
                source_GMT_fc = model_GMT.readout(source_GMT_feature,source_GMT_label, domain='source')
                source_GMT_out = model_GMT.predict(source_GMT_fc)

                target_GMT_feature = model_GMT(target_GMT_data)
                target_GMT_label = align_label(args, target_GMT_data.y)
                target_GMT_fc = model_GMT.readout(target_GMT_feature, target_GMT_label, domain='target')
                target_GMT_out = model_GMT.predict(target_GMT_fc)

                source_MIX_feature_tmp = model_MIX(source_MIX_data)
                source_MIX_feature, source_MIX_label = add_ood(args, source_MIX_feature_tmp, source_MIX_data.y)
                source_MIX_fc = model_MIX.readout(source_MIX_feature,source_MIX_label, domain='source')
                source_MIX_out = model_MIX.predict(source_MIX_fc)

                target_MIX_feature = model_MIX(target_MIX_data)
                target_MIX_label = align_label(args, target_MIX_data.y)
                target_MIX_fc = model_MIX.readout(target_MIX_feature, target_MIX_label, domain='target')
                target_MIX_out = model_MIX.predict(target_MIX_fc)

                # target_GMT_label = target_GMT_data.y
                # target_MIX_label = target_MIX_data.y
                # if args.dataset_name == 'COIL-DEL':
                #     target_GMT_label[target_GMT_label > int(args.unknown_ratio * 100)] = int(args.unknown_ratio * 100)
                #     target_MIX_label[target_MIX_label > int(args.unknown_ratio * 100)] = int(args.unknown_ratio * 100)
                # elif args.dataset_name == 'Letter-high':
                #     target_GMT_label[target_GMT_label > int(args.unknown_ratio * 15)] = int(args.unknown_ratio * 15)
                #     target_MIX_label[target_MIX_label > int(args.unknown_ratio * 15)] = int(args.unknown_ratio * 15)
                # else:
                #     target_GMT_label[target_GMT_label > int(args.unknown_ratio * 10)] = int(args.unknown_ratio * 10)
                #     target_MIX_label[target_MIX_label > int(args.unknown_ratio * 10)] = int(args.unknown_ratio * 10)
                total_GMT_correct += int((target_GMT_out.argmax(dim=-1) == target_GMT_label).sum())
                total_MIX_correct += int((target_MIX_out.argmax(dim=-1) == target_MIX_label).sum())

                source_loss = F.cross_entropy(source_MIX_out, source_MIX_label.long())
                # simility_loss += target_GMT_data.num_graphs * F.cross_entropy(target_MIX_out, target_MIX_label)
                simility_loss = F.kl_div(torch.log_softmax(target_MIX_out,dim=-1), torch.softmax(target_GMT_out,dim=-1))
                # print(idx)
                if idx < 16:
                    total_loss = source_loss + simility_loss
                else:
                    total_source_GMT_fc = model_GMT.source_fc.T
                    total_target_GMT_fc = model_GMT.target_fc.T
                    total_source_GMT_label = model_GMT.source_label
                    total_target_GMT_label = model_GMT.target_label
                    # print('label:',total_source_GMT_label)

                    total_source_MIX_feature = model_MIX.source_feature.T
                    total_target_MIX_feature = model_MIX.target_feature.T
                    total_source_MIX_fc = model_MIX.source_fc.T
                    total_target_MIX_fc = model_MIX.target_fc.T
                    total_source_MIX_label = model_MIX.source_label
                    total_target_MIX_label = model_MIX.target_label
                    t0 = time.time()
                    g_GMT_mirror_loss = mirror_loss(args.num_class, args.topk, total_source_GMT_fc, total_target_GMT_fc,
                                                    total_source_GMT_label, total_target_GMT_label)
                    # t1 = time.time()
                    # print(t1-t0)
                    # f_MIX_mirror_loss = mirror_loss(args.num_class, args.topk, total_source_MIX_feature, total_target_MIX_feature, total_source_MIX_label, total_target_MIX_label)
                    # t2 = time.time()
                    # print(t2-t1)
                    g_MIX_mirror_loss = mirror_loss(args.num_class, args.topk, total_source_MIX_fc, total_target_MIX_fc, total_source_MIX_label, total_target_MIX_label)
                    # t3 = time.time()
                    # print(t3-t2)
                    total_loss = source_loss + simility_loss + args.gama * (g_GMT_mirror_loss + g_MIX_mirror_loss)
                total_loss.backward()
                optimizer.step()
        else:
            for idx, (source_GMT_data, target_GMT_data, source_MIX_data, target_MIX_data) in enumerate(zip(source_GMT_loader, target_GMT_loader, source_MIX_loader, target_MIX_loader)):
                source_GMT_data, target_GMT_data, source_MIX_data, target_MIX_data = source_GMT_data.to(args.device), target_GMT_data.to(args.device), source_MIX_data.to(args.device), target_MIX_data.to(args.device)
                source_GMT_feature_tmp = model_GMT(source_GMT_data)
                source_GMT_feature, source_GMT_label = add_ood(args, source_GMT_feature_tmp, source_GMT_data.y)
                source_GMT_fc = model_GMT.readout(source_GMT_feature)
                source_GMT_out = model_GMT.predict(source_GMT_fc)

                target_GMT_feature = model_GMT(target_GMT_data)
                target_GMT_fc = model_GMT.readout(target_GMT_feature)
                target_GMT_out = model_GMT.predict(target_GMT_fc)

                source_MIX_feature_tmp = model_MIX(source_MIX_data)
                source_MIX_feature, source_MIX_label = add_ood(args, source_MIX_feature_tmp, source_MIX_data.y)
                source_MIX_fc = model_MIX.readout(source_MIX_feature)
                source_MIX_out = model_MIX.predict(source_MIX_fc)

                target_MIX_feature = model_MIX(target_MIX_data)
                target_MIX_fc = model_MIX.readout(target_MIX_feature)
                target_MIX_out = model_MIX.predict(target_MIX_fc)

                target_GMT_label = target_GMT_data.y
                target_MIX_label = target_MIX_data.y
                if args.dataset_name == 'COIL-DEL':
                    target_GMT_label[target_GMT_label > int(args.unknown_ratio * 100)] = int(args.unknown_ratio * 100)
                    target_MIX_label[target_MIX_label > int(args.unknown_ratio * 100)] = int(args.unknown_ratio * 100)
                elif args.dataset_name == 'Letter-high':
                    target_GMT_label[target_GMT_label > int(args.unknown_ratio * 15)] = int(args.unknown_ratio * 15)
                    target_MIX_label[target_MIX_label > int(args.unknown_ratio * 15)] = int(args.unknown_ratio * 15)
                else:
                    target_GMT_label[target_GMT_label > int(args.unknown_ratio * 10)] = int(args.unknown_ratio * 10)
                    target_MIX_label[target_MIX_label > int(args.unknown_ratio * 10)] = int(args.unknown_ratio * 10)
                total_GMT_correct += int((target_GMT_out.argmax(dim=-1) == target_GMT_label).sum())
                total_MIX_correct += int((target_MIX_out.argmax(dim=-1) == target_MIX_label).sum())

                source_loss += source_MIX_data.num_graphs * F.cross_entropy(source_MIX_out, source_MIX_label.long())
                # simility_loss += target_GMT_data.num_graphs * F.cross_entropy(target_MIX_out, target_MIX_label)
                simility_loss += target_GMT_data.num_graphs * F.kl_div(torch.log_softmax(target_MIX_out,dim=-1), torch.softmax(target_GMT_out,dim=-1))

                # batch_size = source_GMT_feature.shape[0]
                # # print(source_GMT_feature.shape,source_MIX_feature.shape)
                #
                # total_source_GMT_feature[source_ptr: min(source_ptr + batch_size, K), :] = source_GMT_feature[: min(batch_size, K-source_ptr), :]
                # total_source_GMT_fc[source_ptr: min(source_ptr + batch_size, K), :] = source_GMT_fc[: min(batch_size, K-source_ptr), :]
                # total_source_GMT_label[source_ptr: min(source_ptr + batch_size, K)] = source_GMT_label[: min(batch_size, K-source_ptr)]
                #
                # total_source_MIX_feature[source_ptr: min(source_ptr + batch_size, K), :] = source_MIX_feature[: min(batch_size, K-source_ptr), :]
                # total_source_MIX_fc[source_ptr: min(source_ptr + batch_size, K), :] = source_MIX_fc[: min(batch_size, K-source_ptr), :]
                # total_source_MIX_label[source_ptr: min(source_ptr + batch_size, K)] = source_MIX_label[: min(batch_size, K-source_ptr)]
                # source_ptr = (source_ptr + batch_size) % K
                #
                # batch_size = target_GMT_feature.shape[0]
                # total_target_GMT_feature[target_ptr: min(target_ptr + batch_size, K), :] = target_GMT_feature[: min(batch_size, K-target_ptr), :]
                # total_target_GMT_fc[target_ptr: min(target_ptr + batch_size, K), :] = target_GMT_fc[: min(batch_size, K-target_ptr), :]
                # total_target_GMT_label[target_ptr: min(target_ptr + batch_size, K)] = target_GMT_out.argmax(dim=-1)[: min(batch_size, K-target_ptr)]
                #
                # total_target_MIX_feature[target_ptr: min(target_ptr + batch_size, K), :] = target_MIX_feature[: min(batch_size, K-target_ptr), :]
                # total_target_MIX_fc[target_ptr: min(target_ptr + batch_size, K), :] = target_MIX_fc[: min(batch_size, K-target_ptr), :]
                # total_target_MIX_label[target_ptr: min(target_ptr + batch_size, K)] = target_MIX_out.argmax(dim=-1)[: min(batch_size, K-target_ptr)]
                # target_ptr = (target_ptr +batch_size) % K
                #
                # # f_GMT_mirror_loss = mirror_loss(args.num_class, args.topk, total_source_GMT_feature, total_target_GMT_feature, total_source_GMT_label, total_target_GMT_label)
                # g_GMT_mirror_loss = mirror_loss(args.num_class, args.topk, total_source_GMT_fc, total_target_GMT_fc, total_source_GMT_label, total_target_GMT_label)
                #
                # f_MIX_mirror_loss = mirror_loss(args.num_class, args.topk, total_source_MIX_feature, total_target_MIX_feature, total_source_MIX_label, total_target_MIX_label)
                # g_MIX_mirror_loss = mirror_loss(args.num_class, args.topk, total_source_MIX_fc, total_target_MIX_fc, total_source_MIX_label, total_target_MIX_label)
                # total_loss = source_loss + simility_loss  + args.gama * (g_GMT_mirror_loss + f_MIX_mirror_loss + g_MIX_mirror_loss)
                # total_loss.backward()
                # optimizer.step()

                if total_target_GMT_feature == None:
                    total_source_GMT_feature = source_GMT_feature
                    total_source_GMT_fc = source_GMT_fc
                    total_source_GMT_label = source_GMT_label
                    total_target_GMT_feature = target_GMT_feature
                    total_target_GMT_fc = target_GMT_fc
                    total_target_GMT_label = target_GMT_out.argmax(dim=-1) #pseudo label

                    total_source_MIX_feature = source_MIX_feature
                    total_source_MIX_fc = source_MIX_fc
                    total_source_MIX_label = source_MIX_label
                    total_target_MIX_feature = target_MIX_feature
                    total_target_MIX_fc = target_MIX_fc
                    total_target_MIX_label = target_MIX_out.argmax(dim=-1)
                else:
                    total_source_GMT_feature = torch.cat((total_source_GMT_feature, source_GMT_feature), dim=0)
                    total_source_GMT_fc = torch.cat((total_source_GMT_fc, source_GMT_fc), dim=0)
                    total_source_GMT_label = torch.cat((total_source_GMT_label, source_GMT_label), dim=0)
                    total_target_GMT_feature = torch.cat((total_target_GMT_feature, target_GMT_feature), dim=0)
                    total_target_GMT_fc = torch.cat((total_target_GMT_fc, target_GMT_fc), dim=0)
                    total_target_GMT_label = torch.cat((total_target_GMT_label, target_GMT_out.argmax(dim=-1)), dim=0)

                    total_source_MIX_feature = torch.cat((total_source_MIX_feature, source_MIX_feature), dim=0)
                    total_source_MIX_fc = torch.cat((total_source_MIX_fc, source_MIX_fc), dim=0)
                    total_source_MIX_label = torch.cat((total_source_MIX_label, source_MIX_label), dim=0)
                    total_target_MIX_feature = torch.cat((total_target_MIX_feature, target_MIX_feature), dim=0)
                    total_target_MIX_fc = torch.cat((total_target_MIX_fc, target_MIX_fc), dim=0)
                    total_target_MIX_label = torch.cat((total_target_MIX_label, target_MIX_out.argmax(dim=-1)), dim=0)
                    # print(total_target_MIX_feature.shape)
            if args.dataset_name == 'COIL-DEL':
                args.num_class = int(100*args.unknown_ratio)
            elif args.dataset_name == 'Letter-high':
                args.num_class = int(15 * args.unknown_ratio)
            else:
                args.num_class = int(10 * args.unknown_ratio)

            # f_GMT_mirror_loss = mirror_loss(args.num_class, args.topk, total_source_GMT_feature, total_target_GMT_feature, total_source_GMT_label, total_target_GMT_label)
            g_GMT_mirror_loss = mirror_loss(args.num_class, args.topk, total_source_GMT_fc, total_target_GMT_fc, total_source_GMT_label, total_target_GMT_label)

            f_MIX_mirror_loss = mirror_loss(args.num_class, args.topk, total_source_MIX_feature, total_target_MIX_feature, total_source_MIX_label, total_target_MIX_label)
            g_MIX_mirror_loss = mirror_loss(args.num_class, args.topk, total_source_MIX_fc, total_target_MIX_fc, total_source_MIX_label, total_target_MIX_label)
            #
            source_avg_loss = source_loss / len(source_GMT_loader.dataset)
            simility_avg = simility_loss / len(target_GMT_loader.dataset)
            #
            total_loss = source_avg_loss +simility_avg + args.gama * (g_GMT_mirror_loss + f_MIX_mirror_loss + g_MIX_mirror_loss)
            total_loss.backward()
            optimizer.step()

        target_MIX_acc, _, _, _ = test(args, model_MIX, target_MIX_loader)
        total_acc.append(target_MIX_acc)
        print(f'Epoch: {epoch:03d}, GMT Target Acc: {total_GMT_correct/len(target_GMT_loader.dataset):.4f}, MIX Target Acc: {target_MIX_acc:.4f}')
        # print(max(total_acc))
    return total_acc
