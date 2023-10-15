import networkx as nx
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
from utils import query_data
from utils import BenchmarkDataset
import os.path as osp
import os

current_path = os.getcwd()
path = osp.join(current_path, '..', 'data', 'CIFAR10')
dataset = BenchmarkDataset(path, 'CIFAR10')
graph_list = []
for data in dataset:
    # print(data.edge_index.shape)
    if data.edge_index.shape[-1] > 1100:
        graph_list.append(data)
print(len(graph_list))
# graph = torch.load('../temp/causal_%d'%1,map_location=torch.device('cpu'))
# print(graph)
# dat = torch.load('vis_data/causal.pt')
# data = Batch()
# data.x = dat['x']
# data.edge_index = dat['edge_index']
# data.batch = dat['batch']
# data.causal = dat['causal']

# def load_pyg_graph(index):
#     mask = data.batch == index
#     causal = data.causal[mask]
#     graph = torch.load('causal_%d'%index)
#     graph.detach().cpu()
#     graph.causal = causal.detach().cpu()
#     return graph

def show(idx, save=None, attn=True):
    graph = graph_list[idx]
    print(graph.y)
    nx_graph = to_networkx(graph, to_undirected=True)
    pos = {}
    ymax = graph.pos[:, 1].max()
    xmax = graph.pos[:, 0].max()
    for i in range(graph.num_nodes):
        position = graph.pos[i].numpy()
        x = position[1]
        y = xmax - position[0]
        pos[i] = (x, y)
    if not attn:
        color = graph.x[:, 0]
    # else:
    #     color = graph.causal.numpy()
    fig, ax = plt.subplots(figsize=(5, 5))
    nx.draw(nx_graph, node_color=color, cmap='Blues', node_size=100, ax=ax)
    if save is not None:
        plt.savefig(save, dpi=600, bbox_inches='tight')
    plt.draw()
    plt.show()

idx = 2
show(idx, attn=False,save=f'{idx}.png')
# show(3, save='vis_data/causal_4.png')