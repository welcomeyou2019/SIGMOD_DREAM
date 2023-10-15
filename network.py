import numpy as np
import math
from torch_geometric.nn import GINConv
from torch_scatter import scatter
# from gnn import *
import torch
from torch import nn
import torch.nn.functional as F
import copy
from functools import wraps
from sklearn.cluster import KMeans

class MLP(nn.Module):
    def __init__(self, dim, hidden_size, projection_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            #nn.PReLU(),
            nn.Linear(hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)

def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy

class DEAL(nn.Module):
    def __init__(self, net, emb_dim=300, projection_hidden_size=2048, projection_size=512, prediction_size = 2):
        super().__init__()

        self.projection_hidden_size = projection_hidden_size

        self.online_encoder = net
        self.target_encoder = None
        self.prediction_size = prediction_size
        self.online_projector = MLP(emb_dim, projection_hidden_size, projection_size)
        self.predictor = MLP(projection_size, projection_hidden_size, prediction_size)  # predict_size)

    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def loss_fn(self, x, y):
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        return 2 - 2 * (x * y).sum(dim=-1)

    def forward(self, source_batch, target_batch, ad_net,i, entropy=None, coeff=None, random_layer=None, device='cpu'):
        features_source = self.online_encoder(source_batch)
        features_target = self.online_encoder(target_batch)

        online_pred_one = self.online_encoder.readout(features_source)  # self.online_predictor(online_proj_one)
        online_pred_two = self.online_encoder.readout(features_target)  # self.online_predictor(online_proj_two)

        outputs_source = self.online_encoder.predict(online_pred_one)  # self.online_predictor(online_proj_one)
        outputs_target = self.online_encoder.predict(online_pred_two)  # self.online_predictor(online_proj_two)

        features = torch.cat((features_source, features_target), dim=0)
        outputs = torch.cat((outputs_source, outputs_target), dim=0)
        softmax_out = nn.Softmax(dim=1)(outputs)

        entropy = Entropy(softmax_out)
        # print(features.shape, softmax_out.shape)
        transfer_loss = CDAN([features, softmax_out], ad_net, entropy, calc_coeff(i), random_layer,
                                  device, features_source.shape[0], features_target.shape[0])
        classifier_loss = nn.CrossEntropyLoss()(outputs_source, source_batch.y)
        return transfer_loss, classifier_loss

    def embed(self, target):
        # online_l_one = self.online_encoder(source, None)
        _, features_target = self.online_encoder(target, None)
        online_pred_two = self.online_projector(features_target)
        outputs_target = self.predictor(online_pred_two)
        return outputs_target.detach()


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

class RandomLayer(nn.Module):
    def __init__(self, input_dim_list=[], output_dim=1024, device='cpu'):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim).to(device) for i in range(self.input_num)] #512,2

    def forward(self, input_list):
        # for i in range(self.input_num):
        #     print(input_list[i].shape, self.random_matrix[i].shape)
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0/len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]


def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

class AdversarialNetwork(nn.Module):
  def __init__(self, in_feature, hidden_size):
    super(AdversarialNetwork, self).__init__()
    self.ad_layer1 = nn.Linear(in_feature, hidden_size)
    self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
    self.ad_layer3 = nn.Linear(hidden_size, 1)
    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()
    self.dropout1 = nn.Dropout(0.5)
    self.dropout2 = nn.Dropout(0.5)
    self.sigmoid = nn.Sigmoid()
    self.apply(init_weights)
    self.iter_num = 0
    self.alpha = 10
    self.low = 0.0
    self.high = 1.0
    self.max_iter = 10000.0

  def forward(self, x):
    if self.training:
        self.iter_num += 1
    coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
    x = x * 1.0
    x.register_hook(grl_hook(coeff))
    x = self.ad_layer1(x)
    x = self.relu1(x)
    x = self.dropout1(x)
    x = self.ad_layer2(x)
    x = self.relu2(x)
    x = self.dropout2(x)
    y = self.ad_layer3(x)
    y = self.sigmoid(y)
    return y

  def output_num(self):
    return 1
  def get_parameters(self):
    return [{"params":self.parameters(), "lr_mult":10, 'decay_mult':2}]

def global_add_pool(x, batch, size = None, dim=0):
    size = int(batch.max().item() + 1) if size is None else size
    return scatter(x, batch, dim=dim, dim_size=size, reduce='add')

def CDAN(input_list, ad_net, entropy=None, coeff=None, random_layer=None, device='cpu', source_dim=0, target_dim=0):
    softmax_output = input_list[1].detach()
    feature = input_list[0]
    # print(softmax_output, feature.shape)
    if random_layer is None:
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
    else:
        # print(feature.shape, softmax_output.shape)
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1)))
    # batch_size = softmax_output.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * source_dim + [[0]] * target_dim)).float().to(device)
    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0+torch.exp(-entropy)
        source_mask = torch.ones_like(entropy)
        source_mask[feature.size(0)//2:] = 0
        source_weight = entropy*source_mask
        target_mask = torch.ones_like(entropy)
        target_mask[0:feature.size(0)//2] = 0
        target_weight = entropy*target_mask
        weight = source_weight / torch.sum(source_weight).detach().item() + \
                 target_weight / torch.sum(target_weight).detach().item()
        return torch.sum(weight.view(-1, 1) * nn.BCELoss(reduction='none')(ad_out, dc_target)) / torch.sum(weight).detach().item()
    else:
        return nn.BCELoss()(ad_out, dc_target)
