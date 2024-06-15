import torch
from stealing_link.attack_eraser import StealingAttack
from stealing_link.partial_graph_generation import PartialGraphGeneration
import config
import numpy as np
from torch_geometric.datasets import Planetoid, Coauthor
import torch_geometric.transforms as T
from sklearn.decomposition import PCA
import argparse

args = argparse.ArgumentParser()

args.add_argument(
    "--partial_graph_path", type=str, default="data/partial_graph_with_id/"
)
# args.add_argument("--attack_partial_graph_ratio", type=float, default=0.5)
args.add_argument("--attack_operator", type=str, default="concate_all")
args.add_argument("--attack_metric_type", type=str, default="kl_divergence")
args.add_argument("--dataset_name", type=str, default="CS")
args.add_argument("--ratio_unlearned", type=float, default=0.05)
args.add_argument("--exp", type=str, default="edge")
args.add_argument(
    "--file_path",
    type=str,
    default="/IDEA/attack_materials/seed_20221012_IDEA_CS_edge_0.05_GCN.pth",
)
args = args.parse_args()

# file_path = "/IDEA/attack_materials/seed_20221012_IDEA_CS_edge_0.05_GCN.pth"
model = torch.load(args["file_path"])

if args["dataset_name"] in ["cora", "pubmed", "citeseer"]:
    dataset = Planetoid(
        "raw_data", args["dataset_name"], transform=T.NormalizeFeatures()
    )
    labels = np.unique(dataset.data.y.numpy())
    data = dataset[0]
elif args["dataset_name"] in ["CS", "Phys"]:
    if args["dataset_name"] == "Phys":
        dataset = Coauthor(
            "raw_data",
            name="Physics",
        )
        data = dataset[0]
        feature = data.x
        pca = PCA(n_components=500)
        reduced_feature = pca.fit_transform(feature.cpu().detach().numpy())
        data.x = torch.from_numpy(reduced_feature)
    else:
        dataset = Coauthor(
            "raw_data",
            name="CS",
        )
        data = dataset[0]
else:
    raise Exception("unsupported dataset")


PartialGraphGeneration(args, data, model["removed_edges"], model["predicted_prob"])
StealingAttack(args)
