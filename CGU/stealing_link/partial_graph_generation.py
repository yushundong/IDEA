# from utils import *
import pickle as pkl
import json
import random
import time
import pdb
import argparse
import numpy as np
import os
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import sys
import torch
from scipy.spatial import distance
import torch.nn.functional as F


class PartialGraphGeneration:
    def __init__(self, args, data, edge_removes, attack_posteriors):
        self.args = vars(args)
        self.saving_path = self.args["partial_graph_path"]
        edge_removes = torch.tensor(edge_removes).T
        self.load_data(data, edge_removes, attack_posteriors)
        self.generate()

    def load_data(self, data, edge_removes, attack_posteriors):
        self.data = data
        self.edge_removes = edge_removes
        # self.gcn_pred = attack_posteriors
        self.gcn_pred = F.softmax(attack_posteriors, dim=1)

    # def load_prediction(self):
    # self.dense_pred = load(self.args.dense_pred)

    def generate(self):
        adj = self.data.edge_index
        features = self.data.x.cpu()
        edge_removes = self.edge_removes

        if isinstance(features, np.ndarray):
            feature_arr = features
        elif isinstance(features, torch.Tensor):
            feature_arr = features.numpy()
        else:
            feature_arr = features.toarray()
        self.feature_arr = feature_arr.tolist()

        gcn_pred = self.gcn_pred.tolist()

        node_num = len(gcn_pred)
        link, unlink = self.get_link(adj, node_num, edge_removes)
        random.shuffle(link)
        random.shuffle(unlink)
        label = []
        for row in link:
            label.append(1)
        for row in unlink:
            label.append(0)
        # generate 10% to 100% of known edges
        t_start = time.time()
        for i in range(1, 11):
            # print("generating: %d percent" % (i * 10), time.time() - t_start)
            self.generate_train_test(link, unlink, gcn_pred, i / 10.0)

    def generate_train_test(self, link, unlink, gcn_pred, train_ratio):
        """
        train test are with respect to the links not nodes
        """
        train = []
        test = []

        train_len = len(link) * train_ratio
        for i in range(len(link)):
            # print(i)
            link_id0 = link[i][0]
            link_id1 = link[i][1]

            line_link = {
                "label": 1,
                "gcn_pred0": gcn_pred[link_id0],
                "gcn_pred1": gcn_pred[link_id1],
                # "dense_pred0": dense_pred[link_id0],
                # "dense_pred1": dense_pred[link_id1],
                "feature_arr0": self.feature_arr[link_id0],
                "feature_arr1": self.feature_arr[link_id1],
                "id_pair": [int(link_id0), int(link_id1)],
            }  # generate corresponding link data

            unlink_id0 = unlink[i][0]
            unlink_id1 = unlink[i][1]

            line_unlink = {
                "label": 0,
                "gcn_pred0": gcn_pred[unlink_id0],
                "gcn_pred1": gcn_pred[unlink_id1],
                # "dense_pred0": dense_pred[unlink_id0],
                # "dense_pred1": dense_pred[unlink_id1],
                "feature_arr0": self.feature_arr[unlink_id0],
                "feature_arr1": self.feature_arr[unlink_id1],
                "id_pair": [int(unlink_id0), int(unlink_id1)],
            }

            if i < train_len:
                train.append(line_link)
                train.append(line_unlink)
            else:
                test.append(line_link)
                test.append(line_unlink)

        # test if directory exists
        if not os.path.exists(self.saving_path):
            os.makedirs(self.saving_path)

        with open(
            self.saving_path
            + "%s_train_ratio_%0.1f_train.json" % (self.args["dataset"], train_ratio),
            "w",
        ) as wf1, open(
            self.saving_path
            + "%s_train_ratio_%0.1f_test.json" % (self.args["dataset"], train_ratio),
            "w",
        ) as wf2:
            for row in train:
                wf1.write("%s\n" % json.dumps(row))
            for row in test:
                wf2.write("%s\n" % json.dumps(row))

    def get_link(self, adj, node_num, edge_removes):
        unlink = []
        link = []
        existing_set = set([])
        # rows, cols = adj.nonzero()
        rows_remove = edge_removes[0]
        cols_remove = edge_removes[1]
        # rows_remove = adj[0][: len(rows_remove)]
        # cols_remove = adj[1][: len(cols_remove)]

        # print("There are %d edges in this dataset" % len(rows_remove))
        for i in range(len(rows_remove)):
            r_index = rows_remove[i]
            c_index = cols_remove[i]
            if r_index < c_index:
                link.append([r_index, c_index])
                existing_set.add(",".join([str(r_index), str(c_index)]))

        rows = adj[0]
        cols = adj[1]
        for i in range(len(rows)):
            r_index = rows[i].item()
            c_index = cols[i].item()
            if r_index < c_index:
                if ",".join([str(r_index), str(c_index)]) not in existing_set:
                    existing_set.add(",".join([str(r_index), str(c_index)]))

        random.seed(1)
        t_start = time.time()
        while len(unlink) < len(link):
            # if len(unlink) % 1000 == 0:
            #     print(len(unlink), time.time() - t_start)

            row = random.randint(0, node_num - 1)
            col = random.randint(0, node_num - 1)
            if row > col:
                row, col = col, row
            edge_str = ",".join([str(row), str(col)])
            if (row != col) and (edge_str not in existing_set):
                unlink.append([row, col])
                existing_set.add(edge_str)
        return link, unlink
