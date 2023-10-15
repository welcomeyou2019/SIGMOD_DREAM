import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, Embedding
from torch_scatter import scatter_max, scatter_mean, scatter_sum
from models.hsp_gin_layer import instantiate_mlp
from models.hsp_gin_layer import GIN_HSP_Layer


# Modes: GC: Graph Classification.
GRAPH_CLASS = "gc"


class NetHSP_GIN(torch.nn.Module):
    def __init__(
        self,
        num_features,
        num_classes,
        emb_sizes=None,
        max_distance=5,
        mode=GRAPH_CLASS,
        eps=0,
        drpt_prob=0.5,
        scatter="max",
        inside_aggr="sum",
        outside_aggr="weight",
        device="cpu",
        batch_norm=True,
        layer_norm=False,
        pool_gc=False,
        residual_frequency=-1,
        dataset=None,
        learnable_emb=False,
        use_feat=False,
        nb_edge_types=1,
    ):
        super(NetHSP_GIN, self).__init__()
        if emb_sizes is None:  # Python default handling for mutable input
            emb_sizes = [64, 64, 64]  # The 0th entry is the input feature size.
        self.num_features = num_features
        self.max_distance = max_distance
        self.emb_sizes = emb_sizes
        self.num_layers = len(self.emb_sizes) - 1
        self.eps = eps
        self.drpt_prob = drpt_prob
        self.scatter = scatter
        self.device = device
        self.mode = mode
        self.dataset = dataset

        self.inside_aggr = inside_aggr
        self.outside_aggr = outside_aggr
        # self.ogb_gc = ogb_gc
        self.use_feat = use_feat  # The OGB feature use
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.pool_gc = pool_gc
        self.residual_freq = residual_frequency
        self.learnable_emb = learnable_emb
        self.nb_edge_types = nb_edge_types

        additional_kwargs = {"edgesum_relu": True}
        self.initial_mlp = instantiate_mlp(
            in_channels=num_features,
            out_channels=emb_sizes[0],
            device=device,
            batch_norm=batch_norm,
            final_activation=True,
        )
        self.initial_linear = Linear(emb_sizes[0], num_classes).to(device)

        hsp_layers = []
        linears = []
        if self.layer_norm:
            layer_norms = []
        for i in range(self.num_layers):
            hsp_layer = GIN_HSP_Layer(
                in_channels=emb_sizes[i],
                out_channels=emb_sizes[i + 1],
                eps=self.eps,
                max_distance=self.max_distance,
                inside_aggr=inside_aggr,
                batch_norm=batch_norm,
                outside_aggr=outside_aggr,
                dataset=dataset,
                nb_edge_types=self.nb_edge_types,
                device=device,
                **additional_kwargs
            ).to(device)
            hsp_layers.append(hsp_layer)
            if self.layer_norm:
                layer_norms.append(torch.nn.LayerNorm(emb_sizes[i + 1]))
            linears.append(Linear(emb_sizes[i + 1], num_classes).to(device))

        self.hsp_modules = ModuleList(hsp_layers)
        self.linear_modules = ModuleList(linears)
        if self.layer_norm:
            self.layer_norms = ModuleList(layer_norms)

    def reset_parameters(self):
        if self.layer_norm:
            for x in self.layer_norms:
                x.reset_parameters()
        if hasattr(self, "initial_mlp"):
            for module in self.initial_mlp:
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()
        for (name, module) in self._modules.items():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        for module in self.hsp_modules:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        for module in self.linear_modules:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def pooling(self, x_feat, batch):
        if self.scatter == "max":
            return scatter_max(x_feat, batch, dim=0)[0].to(self.device)
        elif self.scatter == "mean":
            return scatter_mean(x_feat, batch, dim=0).to(self.device)
        else:
            pass

    def forward(self, data):
        x_feat = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        edge_weights = data.edge_weights.to(self.device)

        if self.inside_aggr[0] == "r":  # 'Relational' variant (for QM9)
            edge_attr = data.edge_attr.to(self.device)
        else:
            edge_attr = None

        batch = data.batch.to(self.device)

        # Input encoding
        x_feat = self.initial_mlp(x_feat)

        out = F.dropout(
            self.pooling(self.initial_linear(x_feat), batch), p=self.drpt_prob
        )

        if self.residual_freq > 0:
            last_state_list = [x_feat]  # If skip connections are being used
        for idx, value in enumerate(zip(self.hsp_modules, self.linear_modules)):
            hsp_layer, linear_layer = value
            if (
                self.inside_aggr == "edgesum"
            ):  # For OGBG (Only load direct edges for memory footprint reduction)
                edge_embeddings = None
                edge_attr = data.edge_attr.to(self.device)
            else:
                edge_embeddings = None
            x_feat = hsp_layer(
                node_embeddings=x_feat,
                edge_index=edge_index,
                edge_weights=edge_weights,
                batch=batch,
                edge_attr=edge_attr,
                direct_edge_embs=edge_embeddings,
            ).to(self.device)
            if self.residual_freq > 0:  # Time to introduce a residual
                if self.residual_freq <= idx + 1:
                    x_feat = (
                        x_feat + last_state_list[-self.residual_freq]
                    )  # Residual connection
                last_state_list.append(
                    x_feat
                )  # Add the new state to the list for easy referencing

            # if self.mode == GRAPH_CLASS or (self.mode == GRAPH_REG and self.pool_gc):
            if not self.layer_norm:
                out += F.dropout(
                    linear_layer(self.pooling(x_feat, batch)),
                    p=self.drpt_prob,
                    training=self.training,
                )
            else:
                out += F.dropout(
                    linear_layer(
                        self.layer_norms[idx](self.pooling(x_feat, batch))
                    ),
                    p=self.drpt_prob,
                    training=self.training,
                )

        return out

    def log_hop_weights(self, neptune_client, exp_dir):
        if self.outside_aggr in ["weight"]:
            for i, layer in enumerate(self.hsp_modules):
                data = layer.hop_coef.data
                soft_data = F.softmax(data, dim=0)
                for d, (v, sv) in enumerate(zip(data, soft_data), 1):
                    log_dir = exp_dir + "/conv_" + str(i) + "/" + "weight_" + str(d)
                    neptune_client[log_dir].log(v)
                    soft_log_dir = (
                        exp_dir + "/conv_" + str(i) + "/" + "soft_weight_" + str(d)
                    )
                    neptune_client[soft_log_dir].log(sv)
