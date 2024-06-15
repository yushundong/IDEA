import logging
import torch
import pdb

torch.cuda.empty_cache()

from sklearn.metrics import f1_score
import numpy as np

from lib_aggregator.optimal_aggregator import OptimalAggregator
from lib_dataset.data_store import DataStore
import torch.nn as nn


class Aggregator:
    def __init__(self, run, target_model, data, shard_data, args):
        self.logger = logging.getLogger("Aggregator")
        self.args = args

        self.data_store = DataStore(self.args)

        self.run = run
        self.target_model = target_model
        self.data = data
        self.shard_data = shard_data

        self.num_shards = args["num_shards"]

    def generate_posterior(self, suffix=""):
        self.true_label = (
            self.shard_data[0].y[self.shard_data[0]["test_mask"]].detach().cpu().numpy()
        )
        self.posteriors = {}

        for shard in range(self.args["num_shards"]):
            self.target_model.data = self.shard_data[shard]
            self.data_store.load_target_model(
                self.run, self.target_model, shard, suffix
            )
            self.posteriors[shard] = self.target_model.posterior()
        # self.logger.info("Saving posteriors.")

        self.data_store.save_posteriors(self.posteriors, self.run, suffix)

    def aggregate(self, removed_nodes=None):
        self.removed_nodes = removed_nodes
        if self.args["aggregator"] == "mean":
            aggregate_f1_score = self._mean_aggregator()
        elif self.args["aggregator"] == "optimal":
            aggregate_f1_score = self._optimal_aggregator()
        elif self.args["aggregator"] == "majority":
            aggregate_f1_score = self._majority_aggregator()
        else:
            raise Exception("unsupported aggregator.")

        return aggregate_f1_score

    def _mean_aggregator(self):
        posterior = self.posteriors[0]
        for shard in range(1, self.num_shards):
            posterior += self.posteriors[shard]

        posterior = posterior / self.num_shards
        ##### save posterior for attack#####
        self.data_store.save_posteriors(posterior, self.run, suffix="_mean")
        return f1_score(
            self.true_label, posterior.argmax(axis=1).cpu().numpy(), average="micro"
        )

    def _majority_aggregator(self):
        pred_labels = []

        for shard in range(self.num_shards):
            pred_labels.append(self.posteriors[shard].argmax(axis=1).cpu().numpy())

        pred_labels = np.stack(pred_labels)
        pred_label = np.argmax(
            np.apply_along_axis(
                np.bincount,
                axis=0,
                arr=pred_labels,
                minlength=self.posteriors[0].shape[1],
            ),
            axis=0,
        )

        return f1_score(self.true_label, pred_label, average="micro")

    def _optimal_aggregator(self):
        optimal = OptimalAggregator(
            self.run, self.target_model, self.data, self.args, self.removed_nodes
        )
        optimal.generate_train_data()
        weight_para = optimal.optimization()
        self.data_store.save_optimal_weight(weight_para, run=self.run)

        posterior = self.posteriors[0] * weight_para[0]
        for shard in range(1, self.num_shards):
            posterior += self.posteriors[shard] * weight_para[shard]

        self.attack_posteriors = self.data_store.load_attack_posteriors(self.run)
        attack_posterior = self.attack_posteriors[0] * weight_para[0]
        for shard in range(1, self.num_shards):
            attack_posterior += self.attack_posteriors[shard] * weight_para[shard]

        if self.removed_nodes is not None and self.args["is_attack_feature"]:
            loss_fn = nn.CrossEntropyLoss()
            loss0 = loss_fn(
                attack_posterior[self.removed_nodes].cpu(),
                self.data.y[self.removed_nodes],
            )
            print(
                f"{self.args['target_model']} {self.args['partition_method']} {self.args['dataset_name']} feature_removed {self.args['is_feature_removed']} run_seed_feature {self.args['run_seed_feature']} remove_feature_ratio {self.args['remove_feature_ratio']} loss0: {loss0}",
                file=open("result/loss_feature.txt", "a"),
            )
            if self.args["is_attack_feature"]:
                assert False
        self.data_store.save_attack_posteriors(attack_posterior, self.run, suffix="opt")

        return f1_score(
            self.true_label, posterior.argmax(axis=1).cpu().numpy(), average="micro"
        )
