import os.path as osp
import torch.nn.functional as F
from torch_geometric.utils import degree
from torch_geometric.datasets import TUDataset,GNNBenchmarkDataset
import torch
import torch.utils.data
import os
# from tqdm import tqdm
import numpy as np
import networkx as nx
import tqdm
from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import Data
import json
from torch_geometric.data import Dataset, Data
from scipy.sparse import csr_matrix
import igraph as ig
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from core.transform import MetisPartitionTransform, PositionalEncodingTransform, RandomPartitionTransform
import torch_geometric.transforms as T
from torch_geometric.transforms import Compose
from torch_geometric.utils import to_undirected
from torch_geometric.data import collate
from core.model import GraphMLPMixer, MPGNN, GraphMLPMixer4TreeNeighbour, MPGNN4TreeNeighbour, GraphViT
from typing import Callable, List, Optional
import logging
from ogb.graphproppred import PygGraphPropPredDataset

class BenchmarkDataset(GNNBenchmarkDataset):
    names = ['PATTERN', 'CLUSTER', 'MNIST', 'CIFAR10', 'TSP', 'CSL']

    root_url = 'https://data.pyg.org/datasets/benchmarking-gnns'
    urls = {
        'PATTERN': f'{root_url}/PATTERN_v2.zip',
        'CLUSTER': f'{root_url}/CLUSTER_v2.zip',
        'MNIST': f'{root_url}/MNIST_v2.zip',
        'CIFAR10': f'{root_url}/CIFAR10_v2.zip',
        'TSP': f'{root_url}/TSP_v2.zip',
        'CSL': 'https://www.dropbox.com/s/rnbkp5ubgk82ocu/CSL.zip?dl=1',
    }

    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        self.name = name
        assert self.name in self.names

        if self.name == 'CSL' :
            split = 'train'
            logging.warning(
                ("Dataset 'CSL' does not provide a standardized splitting. "
                 "Instead, it is recommended to perform 5-fold cross "
                 "validation with stratifed sampling"))

        super().__init__(root, name, transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)

        path = self.processed_paths[0]
        self.data, self.slices = torch.load(path)

    @property
    def processed_file_names(self) -> List[str]:
        if self.name == 'CSL':
            return ['data.pt']
        else:
            return ['all_data.pt']

    def process(self):
        if self.name == 'CSL':
            data_list = self.process_CSL()
            torch.save(self.collate(data_list), self.processed_paths[0])
        else:
            inputs = torch.load(self.raw_paths[0])
            data_list = []
            for i in range(len(inputs)):
                for data_dict in inputs[i]:
                    data_list.append(Data(**data_dict))

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            torch.save(self.collate(data_list), self.processed_paths[0])

class SuperpixelTransform(object):
    # combine position and intensity feature, ignore edge value
    def __call__(self, data):
        data.x = torch.cat([data.x, data.pos], dim=-1)
        data.edge_attr = None  # remove edge_attr
        data.edge_index = to_undirected(data.edge_index)
        return data


def query_data(args):
    current_path = os.getcwd()
    path = osp.join(current_path, '..', 'data', args.dataset_name)
    if args.dataset_name in ['COX2','COX2_MD','BZR','BZR_MD','Mutagenicity','PROTEINS_full','DD','FRANKENSTEIN','NCI1']:
#         dataset = TUDataset(path, name=args.dataset_name)
#     elif args.dataset_name == 'Tox21':
#         dataset = PygGraphPropPredDataset(root=path,
#                                           name='ogbg-moltox21',
#                                           )
#     elif args.dataset_name in ['MNIST', 'CIFAR10']:
#         dataset = BenchmarkDataset(path, args.dataset_name, transform=add_pose)
#
#     else:
#         print("Dataset not supported.")
#
#     return dataset
#
# else:
    # pre_transform = PositionalEncodingTransform(
    #     rw_dim=args.rw_dim, lap_dim=args.lap_dim)
    #
    # if args.dataset_name == 'MNIST' or args.dataset_name == 'CIFAR10':
    #     transform_train = transform_eval = SuperpixelTransform()
    # else:
    #     transform_train = transform_eval = None
    #
    # if args.n_patches > 0:
    #     # metis partition
    #     if args.enable:
    #         _transform_train = MetisPartitionTransform(n_patches=args.n_patches,
    #                                                    drop_rate=args.dropout,
    #                                                    num_hops=args.num_hops,
    #                                                    is_directed=False)
    #
    #         _transform_eval = MetisPartitionTransform(n_patches=args.n_patches,
    #                                                   drop_rate=0.0,
    #                                                   num_hops=args.num_hops,
    #                                                   is_directed=False)
    #     # random partition
    #     else:
    #         _transform_train = RandomPartitionTransform(
    #             n_patches=args.n_patches, num_hops=args.num_hops)
    #         _transform_eval = RandomPartitionTransform(
    #             n_patches=args.n_patches, num_hops=args.num_hops)
    #     if args.dataset_name == 'MNIST' or args.dataset_name == 'CIFAR10':
    #         transform_train = Compose([transform_train, _transform_train])
    #         transform_eval = Compose([transform_eval, _transform_eval])
    #     else:
    #         transform_train = _transform_train
    #         transform_eval = _transform_eval
    #
    #
    # if args.dataset_name == 'MNIST' or args.dataset_name == 'CIFAR10':
    #     path = osp.join(current_path, '..', 'data1', args.dataset_name)
    #     dataset = BenchmarkDataset(path, args.dataset_name, pre_transform=pre_transform, transform=transform_train)
    #
    # elif args.dataset_name in ['COX2','COX2_MD','BZR','BZR_MD','COIL-RAG','Mutagenicity','PROTEINS','DD','FRANKENSTEIN']:
    #     path = osp.join(current_path, '..', 'data1', args.dataset_name)
    #     dataset = TUDataset(root=path, name=args.dataset_name, use_node_attr=True,
    #                         pre_transform=pre_transform, transform=transform_train)
    # elif args.dataset_name == 'Tox21':
    #     path = osp.join(current_path, '..', 'data1', args.dataset_name)
    #     dataset = PygGraphPropPredDataset(root=path,
    #                                       name='ogbg-moltox21',
    #                         pre_transform=pre_transform, transform=transform_train
    #                                       )
    #
    # else:
    #     print("Dataset not supported.")
        path = osp.join(current_path, 'config','dataset_config.json')
        with open(path, 'r') as f:
            config = json.load(f)[args.dataset_name]

        Gs, features, y = load_data(args.dataset_name, config['use_node_labels'], config['use_node_attributes'],
                                            config['degree_as_tag'])
        dataset = PathDataset(Gs, features, y, args.cutoff, args.path_type)

        return dataset
    elif args.dataset_name == 'Tox21':
        dataset = PygGraphPropPredDataset(root=path,
                                          name='ogbg-moltox21',
                                          )
        return dataset
    else:
        print('no dataset')


def load_data(ds_name, use_node_labels, use_node_attributes, degree_as_tag=False):
    """Read a text file containing adjacency matrix and node initial representations and converts it to
    a list of networkx graph objects."""
    current_path = os.getcwd()
    path = osp.join(current_path, '..', 'data/')
    graph_indicator = np.loadtxt(path+"%s/%s/raw/%s_graph_indicator.txt" % (ds_name, ds_name, ds_name), dtype=np.int64)
    _, graph_size = np.unique(graph_indicator, return_counts=True)

    edges = np.loadtxt(path+"%s/%s/raw/%s_A.txt" % (ds_name, ds_name, ds_name), dtype=np.int64, delimiter=",")
    edges -= 1
    A = csr_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                   shape=(graph_indicator.size, graph_indicator.size))

    xx = []
    if use_node_labels:
        x = np.loadtxt(path+"%s/%s/raw/%s_node_labels.txt" % (ds_name, ds_name, ds_name), dtype=np.int64).reshape(-1, 1)
        enc = OneHotEncoder(sparse=False)
        x = enc.fit_transform(x)
        xx.append(x)
    if use_node_attributes:
        x = np.loadtxt(path+"%s/%s/raw/%s_node_attributes.txt" % (ds_name, ds_name,ds_name), dtype=np.float64, delimiter=',')
        xx.append(x)
    if degree_as_tag:
        x = A.sum(axis=1)
        enc = OneHotEncoder(sparse=False)
        x = enc.fit_transform(x)
        xx.append(x)
    elif not use_node_labels and not use_node_attributes and not degree_as_tag:
        x = np.ones((A.shape[0], 1))
        xx.append(x)

    x = np.hstack(xx)
    adj = []
    features = []
    idx = 0
    for i in range(graph_size.size):
        adj.append(A[idx:idx + graph_size[i], idx:idx + graph_size[i]])
        features.append(x[idx:idx + graph_size[i], :])
        idx += graph_size[i]

    class_labels = np.loadtxt(path+"%s/%s/raw/%s_graph_labels.txt" % (ds_name, ds_name,ds_name), dtype=np.int64)

    enc = LabelEncoder()
    class_labels = enc.fit_transform(class_labels)
    y = np.array([class_labels[i] for i in range(class_labels.size)])

    Gs = list()
    for i in range(len(adj)):
        Gs.append(nx.from_scipy_sparse_array(adj[i]))

    # with open(f"datasets/{ds_name}/{ds_name}_splits.json", "r") as f :
    #     splits = json.load(f)
    # splits = generate_split(Gs, ds_name)
    return Gs, features, y

class PathDataset(Dataset):
    """
    Computes paths for all nodes in graphs and convert it to pytorch dataset object.
    """

    def __init__(self, Gs, features, y, cutoff, path_type, min_length=0, undirected=True):
        super().__init__()
        self.Gs = Gs
        self.features = features
        self.y = y
        self.cutoff = cutoff
        self.path_type = path_type
        self.undirected = undirected

        if all([self.path_type is not None, cutoff > 2]):
            self.gs = [ig.Graph.from_networkx(g) for g in Gs]
            self.graph_info = list()
            for g in tqdm.tqdm(self.gs):
                self.graph_info.append(generate_paths(g, cutoff, path_type, undirected=undirected))
            self.diameter = max([i[1] for i in self.graph_info])
        else:
            self.diameter = cutoff
        self.min_length = min_length
        self.datalist = [self._create_data(i) for i in range(self.len())]

    def len(self):
        return len(self.Gs)

    def num_nodes(self):
        return sum([G.number_of_nodes() for G in self.Gs])

    def _create_data(self, index):
        data = ModifData(**from_networkx(self.Gs[index]).stores[0])
        data.x = torch.FloatTensor(self.features[index])
        data.y = torch.LongTensor([self.y[index]])
        setattr(data, f'path_2', data.edge_index.T.flip(1))

        if self.path_type != None:
            if self.path_type == 'all_simple_paths':
                setattr(data, f"sp_dists_2", torch.LongTensor(self.graph_info[index][2][0]).flip(1))
            # setattr(data, f'distances_2', torch.cat([torch.zeros(data.edge_index.size(0), 1), torch.ones(data.edge_index.size(0),1)], dim = 1))
            for jj in range(1, self.cutoff - 1):

                paths = torch.LongTensor(self.graph_info[index][0][jj]).view(-1, jj + 2)
                if paths.size(0) > 0:
                    setattr(data, f'path_{jj + 2}', paths.flip(1))
                    if self.path_type == 'all_simple_paths':
                        setattr(data, f"sp_dists_{jj + 2}", torch.LongTensor(self.graph_info[index][2][jj]).flip(1))
                else:
                    setattr(data, f'path_{jj + 2}', torch.empty(0, jj + 2).long())

                    if self.path_type == 'all_simple_paths':
                        setattr(data, f"sp_dists_{jj + 2}", torch.empty(0, jj + 2).long())
        return data

    def get(self, index):
        return self.datalist[index]


class ModifData(Data):
    def __init__(self, edge_index=None, x=None, *args, **kwargs):
        super().__init__(x=x, edge_index=edge_index, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):

        if 'index' in key or 'face' in key or "path" in key:
            return self.num_nodes
        else:
            return 0

    def __cat_dim__(self, key, value, *args, **kwargs):
        if 'index' in key or 'face' in key:  # or "path" in key or "indicator" in key:
            return 1
        else:
            return 0

def generate_paths(g, cutoff, path_type, weights=None, undirected=True):
    """
    Generates paths for all nodes in the graph, based on specified path type. This function uses igraph rather than networkx
    to generate paths as it gives a more than 10x speedup.
    """
    if undirected and g.is_directed():
        g.to_undirected()

    path_length = np.array(g.distances())
    if path_type != "all_simple_paths":
        diameter = g.diameter(directed=False)
        diameter = diameter + 1 if diameter + 1 < cutoff else cutoff

    else:
        diameter = cutoff

    X = [[] for i in range(cutoff - 1)]
    sp_dists = [[] for i in range(cutoff - 1)]

    for n1 in range(g.vcount()):

        if path_type == "all_simple_paths":
            paths_ = g.get_all_simple_paths(n1, cutoff=cutoff - 1)

            for path in paths_:
                idx = len(path) - 2
                if len(path) > 0:
                    X[idx].append(path)
                    # Adding geodesic distance
                    sp_dist = []
                    for node in path:
                        sp_dist.append(path_length[n1, node])
                    sp_dists[idx].append(sp_dist)

        else:
            valid_ngb = [i for i in np.where((path_length[n1] <= cutoff - 1) & (path_length[n1] > 0))[0] if i > n1]
            for n2 in valid_ngb:
                if path_type == "shortest_path":
                    paths_ = g.get_shortest_paths(n1, n2, weights=weights)
                elif path_type == "all_shortest_paths":
                    paths_ = g.get_all_shortest_paths(n1, n2, weights=weights)

                for path in paths_:
                    idx = len(path) - 2
                    X[idx].append(path)
                    X[idx].append(list(reversed(path)))

    return X, diameter, sp_dists

def add_pose(data):
    data.x = torch.cat([data.x, data.pos], dim=-1)
    return data

def create_model(args):

    if args.dataset_name == 'CIFAR10':
        node_type = 'Linear'
        edge_type = 'Discrete'
        nfeat_node = 5
        nfeat_edge = 1
        nout = int(10*args.unknown_ratio)+1

    elif args.dataset_name == 'MNIST':
        node_type = 'Linear'
        edge_type = 'Discrete'
        nfeat_node = 3
        nfeat_edge = 1
        nout = int(10*args.unknown_ratio)+1

    elif args.dataset_name == 'Letter-high':
        node_type = 'Linear'
        edge_type = 'Discrete'
        nfeat_node = 2
        nfeat_edge = 1
        nout = int(15*args.unknown_ratio)+1
    elif args.dataset_name == 'COLORS-3':
        node_type = 'Linear'
        edge_type = 'Discrete'
        nfeat_node = 5
        nfeat_edge = 1
        nout = int(11*args.unknown_ratio)+1
    elif args.dataset_name == 'MSRC_21':
        node_type = 'Linear'
        edge_type = 'Discrete'
        nfeat_node = 24
        nfeat_edge = 1
        nout = int(20*args.unknown_ratio)+1
    elif args.dataset_name == 'TRIANGLES':
        node_type = 'Discrete'
        edge_type = 'Discrete'
        nfeat_node = 2539
        nfeat_edge = 1
        nout = int(10*args.unknown_ratio)+1
    elif args.dataset_name == 'COIL-RAG':
        node_type = 'Linear'
        edge_type = 'Discrete'
        nfeat_node = 64
        nfeat_edge = 1
        nout = int(100*args.unknown_ratio)+1
    elif args.dataset_name == 'COIL-DEL':
        node_type = 'Linear'
        edge_type = 'Discrete'
        nfeat_node = 2
        nfeat_edge = 1
        nout = int(100*args.unknown_ratio)+1
    elif args.dataset_name == 'MSRC_9':
        node_type = 'Linear'
        edge_type = 'Discrete'
        nfeat_node = 10
        nfeat_edge = 1
        nout = int(8*args.unknown_ratio)+1
    elif args.dataset_name == 'Mutagenicity':
        node_type = 'Linear'
        edge_type = 'Linear'
        nfeat_node = 14
        nfeat_edge = 3
        nout = 2
    elif args.dataset_name == 'Tox21':
        node_type = 'Linear'
        edge_type = 'Linear'
        nfeat_node = 9
        nfeat_edge = 3
        nout = 2
    elif args.dataset_name == 'FRANKENSTEIN':
        node_type = 'Linear'
        edge_type = 'Discrete'
        nfeat_node = 780
        nfeat_edge = 1
        nout = 2
    elif '->' in args.dataset_name:
        node_type = 'Linear'
        edge_type = 'Discrete'
        nfeat_node = args.feat_num
        nfeat_edge = 1
        nout = 2

    # print(nfeat_node)
    if args.method == 'second':
        return GraphMLPMixer(nfeat_node=nfeat_node,
                             nfeat_edge=nfeat_edge,
                             nhid=args.hidden_dim,
                             nout=nout,
                             nlayer_gnn=args.nlayer_gnn,
                             node_type=node_type,
                             edge_type=edge_type,
                             nlayer_mlpmixer=args.nlayer_mlpmixer,
                             gnn_type=args.gnn_type,
                             rw_dim=args.rw_dim,
                             lap_dim=args.lap_dim,
                             pooling=args.pool,
                             dropout=args.dropout,
                             mlpmixer_dropout=args.mlpmixer_dropout,
                             n_patches=args.n_patches,
                             use_patch_pe=args.use_patch_pe,
                             K=args.K)
    elif args.method == 'GraphViT':
        return GraphViT(nfeat_node=nfeat_node,
                        nfeat_edge=nfeat_edge,
                        nhid=args.hidden_dim,
                        nout=nout,
                        nlayer_gnn=args.nlayer_gnn,
                        node_type=node_type,
                        edge_type=edge_type,
                        nlayer_mlpmixer=args.nlayer_mlpmixer,
                        gnn_type=args.gnn_type,
                        rw_dim=args.rw_dim,
                        lap_dim=args.lap_dim,
                        pooling=args.pool,
                        dropout=args.dropout,
                        mlpmixer_dropout=args.mlpmixer_dropout,
                        n_patches=args.n_patches,
                        use_patch_pe=args.use_patch_pe)
    elif args.method == 'MPGNN':
        return MPGNN(
            nfeat_node=nfeat_node,
            nfeat_edge=nfeat_edge,
            nhid=args.hidden_dim,
            nout=nout,
            nlayer_gnn=args.nlayer_gnn,
            node_type=node_type,
            edge_type=edge_type,
            gnn_type=args.gnn_type,
            rw_dim=args.rw_dim,
            lap_dim=args.lap_dim,
            pooling=args.pool,
            dropout=args.dropout)
    elif args.method == 'GraphMLPMixer4TreeNeighbour':
        return GraphMLPMixer4TreeNeighbour(
            nfeat_node=nfeat_node,
            nfeat_edge=nfeat_edge,
            nhid=args.hidden_dim,
            nout=nout,
            nlayer_gnn=args.nlayer_gnn,
            node_type=node_type,
            edge_type=edge_type,
            nlayer_mlpmixer=args.nlayer_mlpmixer,
            gnn_type=args.gnn_type,
            rw_dim=args.rw_dim,
            lap_dim=args.lap_dim,
            pooling=args.pool,
            dropout=args.dropout,
            mlpmixer_dropout=args.mlpmixer_dropout,
            n_patches=args.n_patches,
            use_patch_pe=args.use_patch_pe
        )
    elif args.method == 'MPGNN4TreeNeighbour':
        return MPGNN4TreeNeighbour(
            nfeat_node=nfeat_node,
            nfeat_edge=nfeat_edge,
            nhid=args.hidden_dim,
            nout=nout,
            nlayer_gnn=args.nlayer_gnn,
            node_type=node_type,
            edge_type=edge_type,
            gnn_type=args.gnn_type,
            rw_dim=args.rw_dim,
            lap_dim=args.lap_dim,
            pooling=args.pool,
            dropout=args.dropout)

def query_index(args):
    num_class = {'MNIST': 10, 'CIFAR10': 10, 'Letter-high': 15, 'COIL-DEL': 100, 'COLORS-3':11, 'MSRC_21':20, 'TRIANGLES':10, 'REDDIT-MULTI-12K':11,'Fingerprint':15,'COIL-RAG':100,'MSRC_9':8}
    unknown_class = list(
        range(int(args.unknown_ratio * num_class[args.dataset_name]), num_class[args.dataset_name]))
    print(unknown_class)

    args.method = 'baseline'
    dataset_gmt = query_data(args)
    # print(dataset_gmt)
    # args.method = 'graphmix'
    # dataset_mix = query_data(args)
    source_idx = []
    target_idx = []
    # node_no = []
    known = []
    unknown = []
    idx_dict = {}
    class_idx = {}
    class_node_no = {}
    for index,data in tqdm(enumerate(dataset_gmt)):
        # print(data.x)
        if args.dataset_name == 'TRIANGLES':
            y = data.y - 1
        else:
            y = data.y
        if y.item() in unknown_class:
            unknown.append(index)
        else:
            if y.item() not in class_idx:
                class_idx[y.item()] = [index]
                class_node_no[y.item()] = [data.x.shape[0]]
            else:
                class_idx[y.item()].append(index)
                class_node_no[y.item()].append(data.edge_index.shape[-1]/(data.x.shape[0]+0.0000001))
            # known.append(index)
            # node_no.append(data.x.shape[0])
    for class_node in class_node_no:
        node_no = sorted(range(len(class_node_no[class_node])), key=lambda k: class_node_no[class_node][k])

        # print(node_no)
        for i in node_no[:int(len(node_no)/2)]:
            source_idx.append(class_idx[class_node][i])
        for j in node_no[int(len(node_no)/2):]:
            target_idx.append(class_idx[class_node][j])

    idx_dict['source'] = source_idx
    idx_dict['target'] = target_idx
    idx_dict['unknown'] = unknown

    info_dict = json.dumps(idx_dict)
    f = open(f'config/{args.dataset_name}.json', 'w')
    f.write(info_dict)

    print(len(source_idx),len(target_idx),len(unknown))

def query_da_index(args):
    num_class = {'MNIST': 10, 'CIFAR10': 10, 'Letter-high': 15, 'COIL-DEL': 100, 'COLORS-3':11, 'MSRC_21':20, 'TRIANGLES':10, 'REDDIT-MULTI-12K':11,'Fingerprint':15,'COIL-RAG':100,'MSRC_9':8}
    unknown_class = list(
        range(int(args.unknown_ratio * num_class[args.dataset_name]), num_class[args.dataset_name]))
    print(unknown_class)

    args.method = 'baseline'
    dataset_gmt = query_data(args)
    # print(dataset_gmt)
    # args.method = 'graphmix'
    # dataset_mix = query_data(args)
    source_idx = []
    target_idx = []
    # node_no = []
    known = []
    unknown = []
    idx_dict = {}
    class_idx = {}
    class_node_no = {}
    for index,data in tqdm(enumerate(dataset_gmt)):
        # print(data.x)
        if args.dataset_name == 'TRIANGLES':
            y = data.y - 1
        else:
            y = data.y
        if y.item() in unknown_class:
            unknown.append(index)
        else:
            if y.item() not in class_idx:
                class_idx[y.item()] = [index]
                class_node_no[y.item()] = [data.x.shape[0]]
            else:
                class_idx[y.item()].append(index)
                class_node_no[y.item()].append(data.edge_index.shape[-1]/(data.x.shape[0]+0.0000001))
            # known.append(index)
            # node_no.append(data.x.shape[0])
    for class_node in class_node_no:
        node_no = sorted(range(len(class_node_no[class_node])), key=lambda k: class_node_no[class_node][k])

        # print(node_no)
        for i in node_no[:int(len(node_no)/2)]:
            source_idx.append(class_idx[class_node][i])
        for j in node_no[int(len(node_no)/2):]:
            target_idx.append(class_idx[class_node][j])

    idx_dict['source'] = source_idx
    idx_dict['target'] = target_idx
    idx_dict['unknown'] = unknown

    info_dict = json.dumps(idx_dict)
    f = open(f'config/{args.dataset_name}.json', 'w')
    f.write(info_dict)

    print(len(source_idx),len(target_idx),len(unknown))