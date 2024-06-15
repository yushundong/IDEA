import torch
from mia.MLG_TSTF import MIA
import config
import numpy as np
from torch_geometric.datasets import Planetoid, Coauthor
import torch_geometric.transforms as T
from sklearn.decomposition import PCA
import pdb
import csv
import os
import argparse


NUM_RUNS = 1
auc_list = []
acc_list = []
acc_pos_list = []
f1_pos_list = []

parser = argparse.ArgumentParser()
parser.add_argument("--attack_partial_graph_ratio", type=float, default=0.5)
parser.add_argument("--attack_operator", type=str, default="concate_all")
parser.add_argument("--attack_metric_type", type=str, default="kl_divergence")
parser.add_argument("--dataset_name", type=str, default="CS")
parser.add_argument("--exp", type=str, default="node")
parser.add_argument("--ratio_unlearned", type=float, default=0.05)
parser.add_argument(
    "--file_path",
    type=str,
    default="../IDEA/attack_materials/seed_20221012_IDEA_CS_node_0.05_GCN.pth",
)
args = parser.parse_args()
args = vars(args)


args["dataset_name"] = "CS"
args["ratio_unlearned"] = 0.05
args["exp"] = "node"
seed = ["20221012", "20230202", "20230203", "20230204", "20230205"]

for i in range(NUM_RUNS):

    # file_path = f"../IDEA/attack_materials/seed_{seed[i]}_IDEA_CS_node_0.05_GCN.pth"
    # file_path = "../IDEA/attack_materials/IDEA_cora_node_0.05_SGC.pth"
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

    train_indices = model["train_indices"].nonzero().view(-1)
    test_indices = model["test_indices"].nonzero().view(-1)
    mia = MIA(
        args,
        model["predicted_prob"],  # unlearned_feature_pre, predicted_prob
        model["removed_nodes"],
        train_indices,
        test_indices,
    )
    acc_pos, f1_pos, auc, acc = mia.get_results()
    auc_list.append(auc)
    acc_list.append(acc)
    acc_pos_list.append(acc_pos)
    f1_pos_list.append(f1_pos)

auc_mean = np.mean(auc_list)
acc_mean = np.mean(acc_list)
auc_std = np.std(auc_list)
acc_std = np.std(acc_list)
acc_pos_mean = np.mean(acc_pos_list)
f1_pos_mean = np.mean(f1_pos_list)
acc_pos_std = np.std(acc_pos_list)
f1_pos_std = np.std(f1_pos_list)

if not os.path.exists("result/node_attack_results.csv"):
    with open("result/node_attack_results.csv", "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "Dataset",
                "Exp",
                "Unlearn Ratio",
                "AUC Mean",
                "AUC Std",
                "ACC Mean",
                "ACC Std",
            ]
        )
with open("result/node_attack_results.csv", "a", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(
        [args["dataset_name"], "node", 0.05, auc_mean, auc_std, acc_mean, acc_std]
    )
