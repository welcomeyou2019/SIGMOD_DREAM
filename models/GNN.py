import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
# from ours.utils import kmeans_clustering


class GNN(nn.Module):
    def __init__(self, args, num_features, num_classes, conv_type='GIN', pool_type='TopK', emb=True) -> None:
        super().__init__()
        self.args = args
        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_dim = args.hidden_dim
        self.pooling_ratio = 0.5
        self.conv_type = conv_type
        self.pool_type = pool_type

        self.embedding = nn.Sequential(nn.Linear(self.num_features, self.hidden_dim), nn.ReLU()) if emb else None

        # Define convolutional layers
        if self.conv_type == 'GCN':
            self.conv1 = gnn.GCNConv(self.hidden_dim, self.hidden_dim)
            self.conv2 = gnn.GCNConv(self.hidden_dim, self.hidden_dim)
            self.conv3 = gnn.GCNConv(self.hidden_dim, self.hidden_dim)
        elif conv_type == 'SAGE':
            self.conv1 = gnn.SAGEConv(self.hidden_dim, self.hidden_dim)
            self.conv2 = gnn.SAGEConv(self.hidden_dim, self.hidden_dim)
            self.conv3 = gnn.SAGEConv(self.hidden_dim, self.hidden_dim)
        elif conv_type == 'GAT':
            self.conv1 = gnn.GATConv(self.hidden_dim, self.hidden_dim, heads=4, concat=False)
            self.conv2 = gnn.GATConv(self.hidden_dim, self.hidden_dim, heads=4, concat=False)
            self.conv3 = gnn.GATConv(self.hidden_dim, self.hidden_dim, heads=4, concat=False)
        elif conv_type == 'GIN':
            self.conv1 = gnn.GINConv(gnn.MLP([self.hidden_dim, self.hidden_dim, self.hidden_dim]))
            self.conv2 = gnn.GINConv(gnn.MLP([self.hidden_dim, self.hidden_dim, self.hidden_dim]))
            self.conv3 = gnn.GINConv(gnn.MLP([self.hidden_dim, self.hidden_dim, self.hidden_dim]))
        elif conv_type == 'GMT':
            self.conv1 = gnn.GCNConv(self.hidden_dim, self.hidden_dim)
            self.conv2 = gnn.GCNConv(self.hidden_dim, self.hidden_dim)
            self.conv3 = gnn.GCNConv(self.hidden_dim, self.hidden_dim)
        else:
            raise ValueError("Invalid conv_type: %s" % conv_type)

        # Define Pooling layers
        if pool_type == 'TopK':
            self.pool1 = gnn.TopKPooling(self.hidden_dim, self.pooling_ratio)
            self.pool2 = gnn.TopKPooling(self.hidden_dim, self.pooling_ratio)
            self.pool3 = gnn.TopKPooling(self.hidden_dim, self.pooling_ratio)
        elif pool_type == 'SAG':
            self.pool1 = gnn.SAGPooling(self.hidden_dim, self.pooling_ratio)
            self.pool2 = gnn.SAGPooling(self.hidden_dim, self.pooling_ratio)
            self.pool3 = gnn.SAGPooling(self.hidden_dim, self.pooling_ratio)
        elif pool_type == 'Edge':
            self.pool1 = gnn.EdgePooling(self.hidden_dim)
            self.pool2 = gnn.EdgePooling(self.hidden_dim)
            self.pool3 = gnn.EdgePooling(self.hidden_dim)
        elif pool_type == 'ASA':
            self.pool1 = gnn.ASAPooling(self.hidden_dim)
            self.pool2 = gnn.ASAPooling(self.hidden_dim)
            self.pool3 = gnn.ASAPooling(self.hidden_dim)
        elif pool_type == 'GMT':
            self.pool1 = gnn.GraphMultisetTransformer(self.hidden_dim,self.hidden_dim,2*self.hidden_dim)
        else:
            raise ValueError("Invalid pool_type %s" % pool_type)

        self.bn1 = nn.BatchNorm1d(self.hidden_dim)
        self.bn2 = nn.BatchNorm1d(self.hidden_dim)
        self.bn3 = nn.BatchNorm1d(self.hidden_dim)
        self.bn4 = nn.BatchNorm1d(self.hidden_dim)

        # Define Linear Layers
        self.linear1 = nn.Linear(self.hidden_dim*2, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.linear3 = nn.Linear(self.hidden_dim // 2, self.num_classes)

        # Define activation function
        self.relu = F.leaky_relu
        self.dropout = nn.Dropout(args.dropout)

        self.novel_buffer = []

    def forward(self, data, return_feature=False):
        x = data.x
        if x.shape[-1] < self.num_features:
            x = torch.cat((x, torch.zeros(x.shape[0], self.num_features-x.shape[-1]).to(self.args.device)),dim=-1).to(self.args.device)
        # print(x.shape)
        edge_index = data.edge_index
        batch = data.batch
        edge_attr = data.edge_attr

        if self.embedding is not None:
            x = self.embedding(x)

        x = self.dropout(self.relu(self.bn1(self.conv1(x, edge_index, edge_attr)), negative_slope=0.1))
        if self.pool_type == 'Edge':
            x, edge_index, batch, _ = self.pool1(x, edge_index, batch=batch)
        elif self.pool_type == 'ASA':
            x, edge_index, _, batch, _ = self.pool1(x, edge_index, batch=batch)
        elif self.pool_type == 'GMT':
            pass
        else:
            x, edge_index, edge_attr, batch, _, _ = self.pool1(x, edge_index, batch=batch)

        if self.pool_type == 'GMT':
            # print(self.pool1)
            x1 = self.pool1(x, index=batch, edge_index=edge_index)
        else:
            x1 = torch.cat([gnn.global_max_pool(x, batch), gnn.global_mean_pool(x, batch)], dim=1)

        x = self.dropout(self.relu(self.bn2(self.conv2(x, edge_index, edge_attr)), negative_slope=0.1))
        if self.pool_type == 'Edge':
            x, edge_index, batch, _ = self.pool1(x, edge_index, batch=batch)
        elif self.pool_type == 'ASA':
            x, edge_index, _, batch, _ = self.pool1(x, edge_index, batch=batch)
        elif self.pool_type == 'GMT':
            pass
        else:
            x, edge_index, edge_attr, batch, _, _ = self.pool2(x, edge_index, batch=batch)

        if self.pool_type == 'GMT':
            x2 = self.pool1(x, index=batch, edge_index=edge_index)
        else:
            x2 = torch.cat([gnn.global_max_pool(x, batch), gnn.global_mean_pool(x, batch)], dim=1)

        x = self.dropout(self.relu(self.bn3(self.conv3(x, edge_index, edge_attr)), negative_slope=0.1))
        if self.pool_type == 'Edge':
            x, edge_index, batch, _ = self.pool1(x, edge_index, batch=batch)
        elif self.pool_type == 'ASA':
            x, edge_index, _, batch, _ = self.pool1(x, edge_index, batch=batch)
        elif self.pool_type == 'GMT':
            pass
        else:
            x, edge_index, edge_attr, batch, _, _ = self.pool3(x, edge_index, batch=batch)
        if self.pool_type == 'GMT':
            x3 = self.pool1(x, index=batch, edge_index=edge_index)
        else:
            x3 = torch.cat([gnn.global_max_pool(x, batch), gnn.global_mean_pool(x, batch)], dim=1)

        x = self.relu(x1, negative_slope=0.1) + \
            self.relu(x2, negative_slope=0.1) + \
            self.relu(x3, negative_slope=0.1)

        feature = self.linear1(x)



        return feature

    def readout(self, feature,label=None, return_feature=False, normalize=True, stage='train',domain='source'):
        x = self.dropout(self.relu(feature, negative_slope=0.1))
        x = self.relu(self.linear2(x), negative_slope=0.1)

        # x = self.linear3(x)

        if return_feature:
            if normalize:
                return x, F.normalize(feature, p=2, dim=-1)
            else:
                return x, feature

        # if self.K != 0:
        #     if label != None:
        #         if stage == 'train':
        #             self._dequeue_and_enqueue(feature, x, label, domain)

        return x

    def predict(self, feature):

        x = self.linear3(feature)
        return x


class MNN_GNN(nn.Module):
    def __init__(self, args, num_features, num_classes, conv_type='GIN', pool_type='TopK', emb=True) -> None:
        super().__init__()
        self.args = args
        self.num_features = num_features
        # print(num_classes)
        self.num_classes = num_classes
        self.hidden_dim = args.hidden_dim
        self.pooling_ratio = 0.5
        self.conv_type = conv_type
        self.pool_type = pool_type
        # self.K = K

        # self.embedding = nn.Sequential(nn.Linear(self.num_features, self.hidden_dim), nn.ReLU()) if emb else None

        if self.conv_type == 'GCN':
            self.conv1 = gnn.GCNConv(self.hidden_dim, self.hidden_dim)
            self.conv2 = gnn.GCNConv(self.hidden_dim, self.hidden_dim)
            self.conv3 = gnn.GCNConv(self.hidden_dim, self.hidden_dim)
        elif conv_type == 'SAGE':
            self.conv1 = gnn.SAGEConv(self.hidden_dim, self.hidden_dim)
            self.conv2 = gnn.SAGEConv(self.hidden_dim, self.hidden_dim)
            self.conv3 = gnn.SAGEConv(self.hidden_dim, self.hidden_dim)
        elif conv_type == 'GAT':
            self.conv1 = gnn.GATConv(self.hidden_dim, self.hidden_dim, heads=4, concat=False)
            self.conv2 = gnn.GATConv(self.hidden_dim, self.hidden_dim, heads=4, concat=False)
            self.conv3 = gnn.GATConv(self.hidden_dim, self.hidden_dim, heads=4, concat=False)
        elif conv_type == 'GIN':
            self.conv1 = gnn.GINConv(gnn.MLP([self.hidden_dim, self.hidden_dim, self.hidden_dim]))
            self.conv2 = gnn.GINConv(gnn.MLP([self.hidden_dim, self.hidden_dim, self.hidden_dim]))
            self.conv3 = gnn.GINConv(gnn.MLP([self.hidden_dim, self.hidden_dim, self.hidden_dim]))
        elif conv_type == 'GMT':
            self.conv1 = gnn.GCNConv(self.hidden_dim, self.hidden_dim)
            self.conv2 = gnn.GCNConv(self.hidden_dim, self.hidden_dim)
            self.conv3 = gnn.GCNConv(self.hidden_dim, self.hidden_dim)
        else:
            raise ValueError("Invalid conv_type: %s" % conv_type)

        # Define Pooling layers
        # if pool_type == 'TopK':
        #     self.pool1 = gnn.TopKPooling(self.hidden_dim, self.pooling_ratio)
        #     self.pool2 = gnn.TopKPooling(self.hidden_dim, self.pooling_ratio)
        #     self.pool3 = gnn.TopKPooling(self.hidden_dim, self.pooling_ratio)
        # elif pool_type == 'SAG':
        #     self.pool1 = gnn.SAGPooling(self.hidden_dim, self.pooling_ratio)
        #     self.pool2 = gnn.SAGPooling(self.hidden_dim, self.pooling_ratio)
        #     self.pool3 = gnn.SAGPooling(self.hidden_dim, self.pooling_ratio)
        # elif pool_type == 'Edge':
        #     self.pool1 = gnn.EdgePooling(self.hidden_dim)
        #     self.pool2 = gnn.EdgePooling(self.hidden_dim)
        #     self.pool3 = gnn.EdgePooling(self.hidden_dim)
        # elif pool_type == 'ASA':
        #     self.pool1 = gnn.ASAPooling(self.hidden_dim)
        #     self.pool2 = gnn.ASAPooling(self.hidden_dim)
        #     self.pool3 = gnn.ASAPooling(self.hidden_dim)
        # elif pool_type == 'GMT':
        #     self.pool1 = gnn.GraphMultisetTransformer(self.hidden_dim,self.hidden_dim,2*self.hidden_dim)
        # else:
        #     raise ValueError("Invalid pool_type %s" % pool_type)

        self.bn1 = nn.BatchNorm1d(self.hidden_dim)
        self.bn2 = nn.BatchNorm1d(self.hidden_dim)
        self.bn3 = nn.BatchNorm1d(self.hidden_dim)
        self.bn4 = nn.BatchNorm1d(self.hidden_dim)

        # Define Linear Layers
        self.linear1 = nn.Linear(self.hidden_dim, self.hidden_dim//2)
        # self.linear2 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.linear3 = nn.Linear(self.hidden_dim//2, self.num_classes)

        # Define activation function
        self.relu = F.leaky_relu
        self.dropout = nn.Dropout(args.dropout)

    #     self.novel_buffer = []
    #
    #     if self.K != 0:
    #         self.register_buffer("source_feature", torch.randn(self.hidden_dim, K))
    #         self.register_buffer("source_fc", torch.randn(self.hidden_dim // 2, K))
    #         self.register_buffer("source_label", torch.randn(K))
    #         self.register_buffer("source_queue_ptr", torch.zeros(1, dtype=torch.long))
    #
    #         self.register_buffer("target_feature", torch.randn(self.hidden_dim, K))
    #         self.register_buffer("target_fc", torch.randn(self.hidden_dim // 2, K))
    #         self.register_buffer("target_label", torch.randn(K))
    #         self.register_buffer("target_queue_ptr", torch.zeros(1, dtype=torch.long))
    #
    # @torch.no_grad()
    # def _dequeue_and_enqueue(self, feature, fc, label,domain):
    #     # gather keys before updating queue
    #     batch_size = feature.shape[0]
    #     # print(feature.shape)
    #     # assert self.K % batch_size == 0  # for simplicity
    #
    #     # replace the keys at ptr (dequeue and enqueue)
    #     if domain == 'source':
    #         ptr = int(self.source_queue_ptr)
    #         # print(min(ptr + batch_size, self.K),min(batch_size, self.K-ptr))
    #         # print(ptr,batch_size)
    #         self.source_feature[:, ptr: min(ptr + batch_size, self.K)] = feature.T[:,:min(batch_size, self.K-ptr)]
    #         self.source_fc[:, ptr: min(ptr + batch_size, self.K)] = fc.T[:,:min(batch_size, self.K-ptr)]
    #         self.source_label[ptr: min(ptr + batch_size, self.K)] = label[:min(batch_size, self.K-ptr)]
    #         ptr = (ptr + batch_size) % self.K  # move pointer
    #
    #         self.source_queue_ptr[0] = ptr
    #     else:
    #         ptr = int(self.target_queue_ptr)
    #
    #         self.target_feature[:, ptr: min(ptr + batch_size, self.K)] = feature.T[:,:min(batch_size, self.K-ptr)]
    #         self.target_fc[:, ptr: min(ptr + batch_size, self.K)] = fc.T[:,:min(batch_size, self.K-ptr)]
    #         self.target_label[ptr: min(ptr + batch_size, self.K)] = label[:min(batch_size, self.K-ptr)]
    #         ptr = (ptr + batch_size) % self.K  # move pointer
    #
    #         self.target_queue_ptr[0] = ptr

    def forward(self, x, edge_index):
        edge_attr = None
        feature = x
        if torch.cuda.is_available():
            feature = feature.to(self.args.device)
            edge_index = edge_index.to(self.args.device)

        # if self.embedding is not None:
        #     x = self.embedding(x)

        x1 = self.dropout(self.relu(self.conv1(x, edge_index, edge_attr), negative_slope=0.1))

        # x2 = self.dropout(self.relu(self.bn2(self.conv2(x1, edge_index, edge_attr)), negative_slope=0.1))
        #
        # x3 = self.dropout(self.relu(self.bn3(self.conv3(x2, edge_index, edge_attr)), negative_slope=0.1))

        x = self.relu(x1, negative_slope=0.1)
        x = self.dropout(self.relu(self.bn4(x), negative_slope=0.1))
        x = feature + 0.01 * x
        x = self.relu(self.linear1(x), negative_slope=0.1)

        x = self.linear3(x)

        # feature = self.linear1(x)

        return x

    # def readout(self, feature,label=None, return_feature=False, normalize=True, stage='train',domain='source'):
    #     x = self.dropout(self.relu(self.bn4(feature), negative_slope=0.1))
    #     x = self.relu(self.linear2(x), negative_slope=0.1)
    #
    #     # x = self.linear3(x)
    #
    #     if return_feature:
    #         if normalize:
    #             return x, F.normalize(feature, p=2, dim=-1)
    #         else:
    #             return x, feature
    #
    #     return x
    #
    # def predict(self, feature):
    #
    #     x = self.linear3(feature)
    #     return x