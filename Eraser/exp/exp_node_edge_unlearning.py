import logging

import pickle
import pdb
import time
from collections import defaultdict

import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.data import Data

import config
from exp.exp import Exp
from lib_gnn_model.graphsage.graphsage import SAGE
from lib_gnn_model.gat.gat import GAT
from lib_gnn_model.gin.gin import GIN
from lib_gnn_model.gcn.gcn import GCN
from lib_gnn_model.mlp.mlp import MLP
from lib_gnn_model.sgc.sgc import SGC
from lib_gnn_model.node_classifier import NodeClassifier
from lib_aggregator.aggregator import Aggregator
from stealing_link.partial_graph_generation import PartialGraphGeneration
from stealing_link.attack_eraser import StealingAttack
from lib_utils import utils


class ExpNodeEdgeUnlearning(Exp):
    def __init__(self, args):
        super(ExpNodeEdgeUnlearning, self).__init__(args)
        self.logger = logging.getLogger("exp_node_edge_unlearning")
        self.target_model_name = self.args["target_model"]

        self.load_data()

        # return self.run_exp()

    def run_exp(self):
        # unlearning efficiency
        run_f1 = np.empty((0))
        unlearning_time = np.empty((0))
        self.f1_all = []
        self.unlearning_time_all = []
        for run in range(self.args["num_runs"]):
            self.run = run

            self.determine_target_model()
            time1 = time.time()
            self.time_edge_unlearning = 0

            # self.logger.info("Run %f" % run)
            self.train_target_models(run)
            time2 = time.time()
            aggregate_f1_score = self.aggregate(run)
            time3 = time.time()
            if (
                self.args["num_unlearned_edges"] != 0
                or self.args["ratio_deleted_edges"] != 0
            ):
                node_unlearning_time = self.unlearning_time_statistic()
            else:
                node_unlearning_time = 0
            run_f1 = np.append(run_f1, aggregate_f1_score)
            unlearning_time = np.append(unlearning_time, node_unlearning_time)
            # self.num_unlearned_edges = 0
            # model utility
            if (
                self.args["num_unlearned_edges"] != 0
                or self.args["ratio_deleted_edges"] != 0
            ):
                # unlearning_time_rough = (time2 - time1) / self.args["num_shards"] + (
                #     time3 - time2
                # )
                unlearning_time_rough = time3 - time1
                unlearning_time_edge_unlearning = (
                    self.time_edge_unlearning + time3 - time2
                )
            else:
                unlearning_time_rough = time3 - time1
                unlearning_time_edge_unlearning = (
                    self.time_edge_unlearning + time3 - time2
                )

            if self.args["edge_unlearning"]:
                unlearning_time_rough = unlearning_time_edge_unlearning

            self.f1_all.append(aggregate_f1_score)
            self.unlearning_time_all.append(unlearning_time_rough)

            if (
                not self.args["edge_unlearning"]
                and self.args["is_attack"]  # is edge attack
            ):  # delta attack
                self.logger.info("attack")
                # load posterior data
                self.posterior_optimal = self.data_store.load_attack_posteriors(
                    self.run, "opt"
                )
                # self.edge_removes = self.data_store.load_removed_edges()
                self.edge_removes = None
                # attack
                PartialGraphGeneration(
                    self.args,
                    self.raw_data,
                    self.edge_removes,
                    self.posterior_optimal,
                )

                StealingAttack(self.args)

        self.f1_score_avg = np.average(self.f1_all)
        self.f1_score_std = np.std(self.f1_all)
        self.unlearning_time_avg = (
            np.average(self.unlearning_time_all)
            if len(self.unlearning_time_all) != 0
            else 0
        )
        self.unlearning_time_std = (
            np.std(self.unlearning_time_all)
            if len(self.unlearning_time_all) != 0
            else 0
        )

        # print(
        #     f'setting: {self.args["dataset_name"]}_{self.args["target_model"]}_{self.args["partition_method"]}_{self.args["aggregator"]}_',
        #     "edge_node_num_unlearned_edges: ",
        #     self.args["num_unlearned_edges"],
        #     "ratio_deleted_edges: ",
        #     self.args["ratio_deleted_edges"],
        #     "num_unlearned_nodes: ",
        #     self.args["num_unlearned_nodes"],
        #     "ratio_unlearned_nodes: ",
        #     self.args["ratio_unlearned_nodes"],
        #     "F1 score: ",
        #     self.f1_all,
        #     "unlearning time: ",
        #     self.unlearning_time_all,
        #     file=open(self.args["file_name"], "a"),
        # )
        self.logger.info(
            "f1_avg: %s f1_std: %s time_avg: %s time_std: %s"
            % (
                self.f1_score_avg,
                self.f1_score_std,
                self.unlearning_time_avg,
                self.unlearning_time_std,
            )
        )

        return (
            self.f1_score_avg,
            self.f1_score_std,
            self.unlearning_time_avg,
            self.unlearning_time_std,
        )

    def attack(self):
        self.logger.info("attack")
        self.posterior_optimal = self.data_store.load_attack_posteriors(self.run, "opt")

    def load_data(self):
        self.shard_data = self.data_store.load_shard_data()
        self.raw_data = self.data_store.load_raw_data()
        self.train_data = self.data_store.load_train_data()

        self.unlearned_shard_data = self.shard_data

    def determine_target_model(self):
        num_feats = self.train_data.num_features
        num_classes = len(self.train_data.y.unique())
        if not self.args["is_use_batch"]:
            if self.target_model_name == "SAGE":
                self.target_model = SAGE(num_feats, num_classes)
            elif self.target_model_name == "GCN":
                self.target_model = GCN(num_feats, num_classes)
            elif self.target_model_name == "GAT":
                self.target_model = GAT(num_feats, num_classes, args=self.args)
            elif self.target_model_name == "GIN":
                self.target_model = GIN(num_feats, num_classes)
            elif self.target_model_name == "SGC":
                self.target_model = SGC(num_feats, num_classes)
            else:
                raise Exception("unsupported target model")
        else:
            if self.target_model_name == "MLP":
                self.target_model = MLP(num_feats, num_classes)
            else:
                self.target_model = NodeClassifier(num_feats, num_classes, self.args)

    def train_target_models(self, run):
        if self.args["is_train_target_model"]:
            if self.args["edge_unlearning"]:
                self.logger.info("training unlearned models")
            else:
                self.logger.info("training target models")

            self.time = {}
            for shard in tqdm(range(self.args["num_shards"])):
                self.time[shard] = self._train_model(run, shard)
            if self.args["edge_unlearning"]:
                is_exist = self.data_store.load_is_exist()
                self.time_edge_unlearning = (
                    is_exist * np.array(list(self.time.values()))
                ).sum()

    def aggregate(self, run):
        self.logger.info("aggregating submodels")

        # posteriors, true_label = self.generate_posterior()
        aggregator = Aggregator(
            run,
            self.target_model,
            self.train_data,
            self.unlearned_shard_data,
            self.args,
        )
        aggregator.generate_posterior()
        self.aggregate_f1_score = aggregator.aggregate()

        # self.logger.info("Final Test F1: %s" % (self.aggregate_f1_score,))
        return self.aggregate_f1_score

    def _generate_unlearning_request(self, num_unlearned="assign"):
        node_list = []
        for key, value in self.community_to_node.items():
            # node_list.extend(value.tolist())
            node_list.extend(value)
        if num_unlearned == "assign":
            num_of_unlearned_nodes = self.args["num_unlearned_nodes"]
        elif num_unlearned == "ratio":
            num_of_unlearned_nodes = int(
                self.args["ratio_unlearned_nodes"] * len(node_list)
            )

        if self.args["unlearning_request"] == "random":
            unlearned_nodes_indices = np.random.choice(
                node_list, num_of_unlearned_nodes, replace=False
            )

        elif self.args["unlearning_request"] == "top1":
            sorted_shards = sorted(
                self.community_to_node.items(), key=lambda x: len(x[1]), reverse=True
            )
            unlearned_nodes_indices = np.random.choice(
                sorted_shards[0][1], num_of_unlearned_nodes, replace=False
            )

        elif self.args["unlearning_request"] == "adaptive":
            sorted_shards = sorted(
                self.community_to_node.items(), key=lambda x: len(x[1]), reverse=True
            )
            candidate_list = np.concatenate(
                [
                    sorted_shards[i][1]
                    for i in range(int(self.args["num_shards"] / 2) + 1)
                ],
                axis=0,
            )
            unlearned_nodes_indices = np.random.choice(
                candidate_list, num_of_unlearned_nodes, replace=False
            )

        elif self.args["unlearning_request"] == "last5":
            sorted_shards = sorted(
                self.community_to_node.items(), key=lambda x: len(x[1]), reverse=False
            )
            candidate_list = np.concatenate(
                [
                    sorted_shards[i][1]
                    for i in range(int(self.args["num_shards"] / 2) + 1)
                ],
                axis=0,
            )
            unlearned_nodes_indices = np.random.choice(
                candidate_list, num_of_unlearned_nodes, replace=False
            )

        return unlearned_nodes_indices

    def unlearning_time_statistic(self):
        self.community_to_node = self.data_store.load_community_data()
        if self.args["is_train_target_model"] and self.args["num_shards"] != 1:
            # random sample 5% nodes, find their belonging communities
            unlearned_nodes = self._generate_unlearning_request(num_unlearned="ratio")
            belong_community = []
            unlearned_edges = np.load(
                "../Graph-Unlearning/temp_data/unlearned_indices.npy"
            )
            if (
                self.args["num_unlearned_edges"] != 0
                or self.args["ratio_deleted_edges"] != 0
            ):
                for sample_edge in range(len(unlearned_edges[0])):
                    for community, node in self.community_to_node.items():
                        if (
                            np.in1d(unlearned_edges[0][sample_edge], node).any()
                            or np.in1d(unlearned_edges[1][sample_edge], node).any()
                        ):
                            belong_community.append(community)
            # for sample_node in range(len(unlearned_nodes)):
            #     for community, node in self.community_to_node.items():
            #         if np.in1d(unlearned_nodes[sample_node], node).any():
            #             belong_community.append(community)

            # calculate the total unlearning time and group unlearning time
            group_unlearning_time = []
            node_unlearning_time = []
            for shard in range(self.args["num_shards"]):
                if belong_community.count(shard) != 0:
                    group_unlearning_time.append(self.time[shard])
                    node_unlearning_time.extend(
                        [
                            float(self.time[shard])
                            for j in range(belong_community.count(shard))
                        ]
                    )
            return node_unlearning_time

        elif self.args["is_train_target_model"] and self.args["num_shards"] == 1:
            return self.time[0]

        else:
            return 0

    def _train_model(self, run, shard):
        # self.logger.info("training target models, run %s, shard %s" % (run, shard))

        start_time = time.time()
        self.target_model.data = self.unlearned_shard_data[shard]
        self.target_model.train_model()
        train_time = time.time() - start_time

        self.data_store.save_target_model(run, self.target_model, shard)

        return train_time
