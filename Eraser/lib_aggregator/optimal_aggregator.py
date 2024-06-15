import copy
import logging

import numpy as np
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from lib_aggregator.opt_dataset import OptDataset
from lib_dataset.data_store import DataStore
from lib_utils import utils


class OptimalAggregator:
    def __init__(self, run, target_model, data, args, removed_nodes=None):
        self.logger = logging.getLogger("optimal_aggregator")
        self.args = args

        self.run = run
        self.target_model = target_model
        self.data = data
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.removed_nodes = removed_nodes

        self.num_shards = args["num_shards"]

    def generate_train_data(self):
        data_store = DataStore(self.args)
        train_indices, _ = data_store.load_train_test_split()

        # sample a set of nodes from train_indices
        if self.args["num_opt_samples"] == 1000:
            train_indices = np.random.choice(train_indices, size=1000, replace=False)
        elif self.args["num_opt_samples"] == 10000:
            train_indices = np.random.choice(
                train_indices, size=int(train_indices.shape[0] * 0.1), replace=False
            )
        elif self.args["num_opt_samples"] == 1:
            train_indices = np.random.choice(
                train_indices, size=int(train_indices.shape[0]), replace=False
            )
        elif self.args["num_opt_samples"] == 0:
            train_indices = train_indices

        train_indices = np.sort(train_indices)
        # self.logger.info(
        #     "Using %s samples for optimization" % (int(train_indices.shape[0]))
        # )

        x = self.data.x[train_indices]
        y = self.data.y[train_indices]
        edge_index = utils.filter_edge_index(self.data.edge_index, train_indices)

        train_data = Data(x=x, edge_index=torch.from_numpy(edge_index), y=y)
        train_data.train_mask = torch.zeros(train_indices.shape[0], dtype=torch.bool)
        train_data.test_mask = torch.ones(train_indices.shape[0], dtype=torch.bool)
        self.true_labels = y

        # generate attack data
        attack_x = self.data.x
        attack_y = self.data.y
        attack_edge_index = self.data.edge_index

        attack_data = Data(x=attack_x, edge_index=attack_edge_index, y=attack_y)
        attack_data.train_mask = torch.zeros(attack_y.shape[0], dtype=torch.bool)
        attack_data.test_mask = torch.ones(attack_y.shape[0], dtype=torch.bool)
        self.attack_labels = attack_y

        self.posteriors = {}
        for shard in range(self.num_shards):
            self.target_model.data = train_data
            data_store.load_target_model(self.run, self.target_model, shard)
            self.posteriors[shard] = self.target_model.posterior().to(self.device)
        self.attack_posteriors = {}
        if self.args["is_attack_feature"]:
            if self.args["dataset_name"] == "Coauthor_CS":
                self.remove_dim_idx = torch.load(
                    f"../IDEA/attack_materials/seed_{self.args['run_seed_feature']}_IDEA_CS_partial_feature_0.05_{self.args['target_model']}_{self.args['remove_feature_ratio']}.pth"
                )["unlearned_feature_dim_idx"]
            elif self.args["dataset_name"] == "cora":
                self.remove_dim_idx = torch.load(
                    f"../IDEA/attack_materials/seed_{self.args['run_seed_feature']}_IDEA_cora_partial_feature_0.05_{self.args['target_model']}_{self.args['remove_feature_ratio']}.pth"
                )["unlearned_feature_dim_idx"]
            elif self.args["dataset_name"] == "citeseer":
                self.remove_dim_idx = torch.load(
                    f"../IDEA/attack_materials/seed_{self.args['run_seed_feature']}_IDEA_citeseer_partial_feature_0.05_{self.args['target_model']}_{self.args['remove_feature_ratio']}.pth"
                )["unlearned_feature_dim_idx"]
            elif self.args["dataset_name"] == "pubmed":
                self.remove_dim_idx = torch.load(
                    f"../IDEA/attack_materials/seed_{self.args['run_seed_feature']}_IDEA_pubmed_partial_feature_0.05_{self.args['target_model']}_{self.args['remove_feature_ratio']}.pth"
                )["unlearned_feature_dim_idx"]
            if self.removed_nodes is not None and self.args["is_feature_removed"]:
                for row in self.removed_nodes:
                    for col in self.remove_dim_idx:
                        attack_data.x[row][col] = 0
        for shard in range(self.num_shards):
            self.target_model.data = attack_data
            data_store.load_target_model(self.run, self.target_model, shard)
            self.attack_posteriors[shard] = self.target_model.posterior().to(
                self.device
            )

        # self.logger.info("Saving posteriors.")
        data_store.save_attack_posteriors(self.attack_posteriors, self.run)

    def optimization(self):
        weight_para = nn.Parameter(
            torch.full((self.num_shards,), fill_value=1.0 / self.num_shards),
            requires_grad=True,
        )
        optimizer = optim.Adam([weight_para], lr=self.args["opt_lr"])
        scheduler = MultiStepLR(
            optimizer, milestones=[500, 1000], gamma=self.args["opt_lr"]
        )

        train_dset = OptDataset(self.posteriors, self.true_labels)
        train_loader = DataLoader(
            train_dset, batch_size=32, shuffle=True, num_workers=0
        )

        min_loss = 1000.0
        for epoch in tqdm(range(self.args["opt_num_epochs"])):
            loss_all = 0.0

            a = 0
            import time

            time1 = time.time()
            for posteriors, labels in train_loader:
                labels = labels.to(self.device)
                optimizer.zero_grad()
                loss = self._loss_fn(posteriors, labels, weight_para)

                loss.backward()
                loss_all += loss

                optimizer.step()
                a += posteriors[0].shape[0]
                with torch.no_grad():
                    weight_para[:] = torch.clamp(weight_para, min=0.0)
            scheduler.step()
            # print("time: ", time.time() - time1)
            # pdb.set_trace()

            if loss_all < min_loss:
                ret_weight_para = copy.deepcopy(weight_para)
                min_loss = loss_all

            # self.logger.info("epoch: %s, loss: %s" % (epoch, loss_all))

        return ret_weight_para / torch.sum(ret_weight_para)

    def _loss_fn(self, posteriors, labels, weight_para):
        aggregate_posteriors = torch.zeros_like(posteriors[0])
        for shard in range(self.num_shards):
            aggregate_posteriors += weight_para[shard] * posteriors[shard]

        aggregate_posteriors = F.softmax(aggregate_posteriors, dim=1)
        loss_1 = F.cross_entropy(aggregate_posteriors, labels)
        loss_2 = torch.sqrt(torch.sum(weight_para**2) + 1e-8)

        return loss_1 + loss_2
