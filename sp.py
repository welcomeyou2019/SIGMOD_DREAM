import time
import configparser
import torch
import argparse
import os.path as osp
from model_loader import get_model
# from experiments.run_gc import run_model_gc
# from experiments.run_gc_ogb import run_model_gc_ogb
import neptune.new as neptune
from torch_geometric.datasets import TUDataset
import os
import torch_geometric.transforms as T

def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


# Neptune configuration
config = configparser.ConfigParser()
config.read("config.ini")

if config["DEFAULT"]["neptune_token"] and config["DEFAULT"]["neptune_token"] != "...":
    neptune_client = neptune.init(
        project=config["DEFAULT"]["neptune_project"],
        api_token=config["DEFAULT"]["neptune_token"],
    )
else:
    neptune_client = None

# CLI configuration
parser = argparse.ArgumentParser()
parser.add_argument(
    "-d", "--dataset_name", help="Dataset to test the model on.", default='NCI1'
)
parser.add_argument("-b", "--batch_size", help="Batch size.", default=32, type=int)
parser.add_argument("-m", "--model", help="The model we will use.", default="GIN")

# Training arguments
parser.add_argument("--lr", help="Learning rate.", default=0.001, type=float)

# Model specific arguments
parser.add_argument(
    "--max_distance", help="Maximal distance in HSP model (K)", default=5, type=int
)
parser.add_argument(
    "--num_layers", help="Number of HSP layers in the model.", default=1, type=int
)
parser.add_argument(
    "--emb_dim", help="Size of the emb dimension.", default=64, type=int
)
parser.add_argument("--scatter", help="Max or Mean pooling.", default="max")
parser.add_argument("--dropout", help="Dropout probability.", default=0.5, type=float)
parser.add_argument("--eps", help="Epsilon in GIN.", default=0.0, type=float)
parser.add_argument("--epochs", help="Number of epochs.", default=300, type=int)
parser.add_argument("--mode", help="Model mode - gc/gr.", default="gc")
parser.add_argument(
    "--pool_gc",
    help="Choose the mode-specific pool (default) or use GC pooling",
    type=str2bool,
    default=False,
)
parser.add_argument(
    "--batch_norm",
    help="Use batch norm within layer MLPs (default True)",
    type=str2bool,
    default=True,
)
parser.add_argument(
    "--layer_norm",
    help="Use layer norm after every message passing iteration (default False)",
    type=str2bool,
    default=True,
)
parser.add_argument(
    "--learnable_emb",
    help="(For synthetic experiments). Whether to set feature embeddings to be "
    "learnable (Default False)",
    type=str2bool,
    default=False,
)
parser.add_argument(
    "--specific_task",
    help="(For QM9) Run all tasks (-1, default) or a specific task by index",
    type=int,
    default=-1,
)
parser.add_argument(
    "--nb_reruns", help="(For QM9) Repeats per task (default 5)", type=int, default=5
)
parser.add_argument(
    "--res_freq",
    help="The layer interval for residual connections (default: -1, i.e., no residual)",
    type=int,
    default=-1,
)
parser.add_argument(
    "--use_feat",
    help="(OGBG). Whether to use all features (Default False)",
    type=str2bool,
    default=False,
)

args = parser.parse_args()

# Add arguments to neptune
if neptune_client:
    neptune_client["parameters"] = vars(args)


BATCH = args.batch_size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root_dir = osp.join(osp.dirname(osp.realpath(__file__)), "..")


# dataset, second_return, ogb_metric = get_dataset(args, root_dir)
current_path = os.getcwd()
path = osp.join(current_path, '..', 'data', args.dataset)
if args.dataset_name in ['Letter-high', 'COIL-DEL', 'MUTAG']:
    dataset = TUDataset(root=path, name=args.dataset_name, use_node_attr=True, transform=T.AddSelfLoops()).shuffle()
    model = get_model(args, num_features=dataset.num_features,num_classes=dataset.num_tasks)
    print(model)

    # model = get_model(
    #     args,
    #     device,
    #     num_features=dataset.num_features,
    #     num_classes=dataset.num_classes,
    # )
    # splits = second_return
    # run_model_gc(
    #     model,
    #     dataset,
    #     splits,
    #     lr=args.lr,
    #     batch_size=BATCH,
    #     epochs=args.epochs,
    #     neptune_client=neptune_client,
    # )
