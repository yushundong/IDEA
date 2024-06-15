import torch
import torch.nn as nn
from stealing_link.attack_eraser import StealingAttack
from stealing_link.partial_graph_generation import PartialGraphGeneration
import config
import numpy as np
import pdb
from torch_geometric.datasets import Planetoid, Coauthor
import torch_geometric.transforms as T
from sklearn.decomposition import PCA

import argparse
import csv
import os


# file_path = "../IDEA/attack_materials/IDEA_cora_feature_0.05_GCN.pth"
# model = torch.load(file_path)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--partial_graph_path", type=str, default="data/partial_graph_with_id/"
)
parser.add_argument("--attack_partial_graph_ratio", type=float, default=0.5)
parser.add_argument("--attack_operator", type=str, default="concate_all")
parser.add_argument("--attack_metric_type", type=str, default="kl_divergence")
parser.add_argument("--dataset_name", type=str, default="CS")

parser.add_argument("--remove_feature_ratio", type=float, default=0.2)
parser.add_argument("--run_seed_feature", type=int, default=2)
parser.add_argument("--model_name", type=str, default="GCN")

parser.add_argument("--ratio_unlearned", type=float, default=0.05)
parser.add_argument("--exp", type=str, default="edge")
args = parser.parse_args()

if args.dataset_name in ["cora", "pubmed", "citeseer"]:
    dataset = Planetoid("raw_data", args.dataset_name, transform=T.NormalizeFeatures())
    labels = np.unique(dataset.data.y.numpy())
    data = dataset[0]
elif args.dataset_name in ["CS", "Phys"]:
    if args.dataset_name == "Coauthor_Phys":
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


if args.dataset_name == "CS":
    model = torch.load(
        f"../IDEA/attack_materials/seed_{args.run_seed_feature}_IDEA_CS_partial_feature_0.05_{args.model_name}_{args.remove_feature_ratio}.pth"
    )
elif args.dataset_name == "cora":
    model = torch.load(
        f"../IDEA/attack_materials/seed_{args.run_seed_feature}_IDEA_cora_partial_feature_0.05_{args.model_name}_{args.remove_feature_ratio}.pth"
    )
elif args.dataset_name == "citeseer":
    model = torch.load(
        f"../IDEA/attack_materials/seed_{args.run_seed_feature}_IDEA_citeseer_partial_feature_0.05_{args.model_name}_{args.remove_feature_ratio}.pth"
    )
elif args.dataset_name == "pubmed":
    model = torch.load(
        f"../IDEA/attack_materials/seed_{args.run_seed_feature}_IDEA_pubmed_partial_feature_0.05_{args.model_name}_{args.remove_feature_ratio}.pth"
    )
# model = torch.load(file_path)


# Define your predicted probabilities and target labels
predicted_probs = (model["unlearned_feature_pre"])[
    model["unlearned_feature_node_idx"]
]  # Example predicted probabilities
target_labels = data.y[model["unlearned_feature_node_idx"]]  # Example target labels

# Calculate the cross-entropy loss
loss_fn = nn.CrossEntropyLoss()
loss_w = loss_fn(predicted_probs, target_labels)


# Define your predicted probabilities and target labels
predicted_probs = (model["predicted_prob"].cpu())[
    model["unlearned_feature_node_idx"]
]  # Example predicted probabilities
target_labels = data.y[model["unlearned_feature_node_idx"]]  # Example target labels

# Calculate the cross-entropy loss
loss_fn = nn.CrossEntropyLoss()
loss_wo = loss_fn(predicted_probs, target_labels)

print(
    f"{args.run_seed_feature}_IDEA_{args.dataset_name}_partial_feature_0.05_{args.model_name}_{args.remove_feature_ratio} loss_w: {loss_w:.4f}, loss_wo: {loss_wo:.4f}",
    file=open("result/loss_feature.txt", "a"),
)

csv_file_path = "result/loss_feature.csv"

# Check if the file already exists to decide on writing headers
file_exists = os.path.isfile(csv_file_path)

# Open the file in append mode
with open(csv_file_path, "a", newline="") as csvfile:
    # Define the fieldnames
    fieldnames = [
        "run_seed_feature",
        "dataset_name",
        "partial_feature",
        "model_name",
        "remove_feature_ratio",
        "loss_w",
        "loss_wo",
    ]
    # Create a DictWriter object
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # Write the header only if the file is being created
    if not file_exists:
        writer.writeheader()

    # Write the data row
    writer.writerow(
        {
            "run_seed_feature": args.run_seed_feature,
            "dataset_name": f"IDEA_{args.dataset_name}",
            "partial_feature": "0.05",
            "model_name": args.model_name,
            "remove_feature_ratio": args.remove_feature_ratio,
            "loss_w": f"{loss_w:.4f}",
            "loss_wo": f"{loss_wo:.4f}",
        }
    )
