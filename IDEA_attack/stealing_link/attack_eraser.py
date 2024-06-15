from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from .keras_utils import *
from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import (
    cosine,
    euclidean,
    correlation,
    chebyshev,
    braycurtis,
    canberra,
    cityblock,
    sqeuclidean,
)
from .utils import kl_divergence, js_divergence, entropy
import argparse
import csv
import os

batch_size = 128
num_classes = 2
epochs = 50


def average(a, b):
    return (a + b) / 2


def hadamard(a, b):
    return a * b


def weighted_l1(a, b):
    return abs(a - b)


def weighted_l2(a, b):
    return abs((a - b) * (a - b))


def concate_all(a, b):
    return np.concatenate(
        (average(a, b), hadamard(a, b), weighted_l1(a, b), weighted_l2(a, b))
    )


class StealingAttack:
    def __init__(
        self,
        args,
    ):
        self.args = args
        self.dataset = args["dataset_name"]
        self.partial_graph_path = args["partial_graph_path"]
        self.ratio = args["attack_partial_graph_ratio"]
        self.operator = args["attack_operator"]
        self.metric_type = args["attack_metric_type"]
        self.attack()

    def attack(self):
        # the data, split between train and test sets
        train_path = self.partial_graph_path + "%s_train_ratio_%s_train.json" % (
            self.dataset,
            self.ratio,
        )
        test_path = self.partial_graph_path + "%s_train_ratio_%s_test.json" % (
            self.dataset,
            self.ratio,
        )

        x_train, x_test, y_train, y_test = self.load_data(train_path, test_path)
        x_train = x_train.astype("float32")
        x_test = x_test.astype("float32")
        # print(x_train.shape[0], "train samples")
        # print(x_test.shape[0], "test samples")

        ss = StandardScaler()
        x_train = ss.fit_transform(x_train)
        x_test = ss.fit_transform(x_test)
        x_train_shape = x_train.shape[-1]
        x_test_shape = x_train_shape

        # convert class vectors to binary class matrices
        y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)

        model = Sequential()
        model.add(Dense(32, activation="relu", input_shape=(x_train_shape,)))
        model.add(Dropout(0.5))
        model.add(Dense(32, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation="softmax"))
        # model.summary()

        model.compile(loss="categorical_crossentropy", optimizer=Adam())
        model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=0,
            validation_data=(x_test, y_test),
        )

        y_pred = model.predict(x_test)

        # add precision recall score
        y_test_label = [row[1] for row in y_test]
        y_pred_label = [round(row[1]) for row in y_pred]

        test_acc = accuracy_score(y_test_label, y_pred_label)
        test_precision = precision_score(y_test_label, y_pred_label)
        test_recall = recall_score(y_test_label, y_pred_label)
        test_auc = roc_auc_score(y_test, y_pred)
        print(
            "Test accuracy:",
            test_acc,
            "Test Precision",
            test_precision,
            "Test Recall",
            test_recall,
            "Test auc:",
            test_auc,
            "-->attack perfomance!",
        )

        # with open(f"result/attack_3.txt", "a") as wf:
        #     wf.write(
        #         "%s,%s,%d,%0.5f,%0.5f,%0.5f,%0.5f,%s\n"
        #         % (
        #             self.dataset,
        #             "attack3_metrics_%s" % (self.operator),
        #             epochs,
        #             test_acc,
        #             test_precision,
        #             test_recall,
        #             test_auc,
        #             self.ratio,
        #         )
        #     )

        if not os.path.exists("result/attack_results.csv"):
            with open("result/attack_results.csv", "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(
                    [
                        "Dataset",
                        "Exp",
                        "Unlearn Ratio",
                        "Attack Metrics",
                        "Epochs",
                        "Test Accuracy",
                        "Test Precision",
                        "Test Recall",
                        "Test AUC",
                        "Ratio",
                    ]
                )
        with open("result/attack_results.csv", "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    self.dataset,
                    self.args["exp"],
                    self.args["ratio_unlearned"],
                    "attack3_metrics_%s" % (self.operator),
                    epochs,
                    test_acc,
                    test_precision,
                    test_recall,
                    test_auc,
                    self.ratio,
                ]
            )

    def operator_func(self, operator, a, b):
        if operator == "average":
            return average(a, b)
        elif operator == "hadamard":
            return hadamard(a, b)
        elif operator == "weighted_l1":
            return weighted_l1(a, b)
        elif operator == "weighted_l2":
            return weighted_l2(a, b)
        elif operator == "concate_all":
            return concate_all(a, b)

    def get_metrics(self, a, b, metric_type, operator_func):
        if metric_type == "kl_divergence":
            s1 = np.array([kl_divergence(a, b)])
            s2 = np.array(kl_divergence(b, a))

        elif metric_type == "js_divergence":
            s1 = np.array([js_divergence(a, b)])
            s2 = np.array(js_divergence(b, a))

        elif metric_type == "entropy":
            s1 = np.array([entropy(a)])
            s2 = np.array([entropy(b)])
        return operator_func(self.operator, s1, s2)

    def load_data(self, train_path, test_path):
        similarity_list = [
            cosine,
            euclidean,
            correlation,
            chebyshev,
            braycurtis,
            canberra,
            cityblock,
            sqeuclidean,
        ]
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        train_data = open(train_path).readlines()
        test_data = open(test_path).readlines()
        for row in train_data:
            row = json.loads(row)
            # x_train.append(row["gcn_pred"]+row["dense_pred"])
            a = np.array(row["gcn_pred0"])
            b = np.array(row["gcn_pred1"])
            feature_vec1 = self.operator_func(self.operator, a, b)  # posterior poerator

            t0 = np.array(row["gcn_pred0"])
            t1 = np.array(row["gcn_pred1"])
            target_similarity = np.array([row(t0, t1) for row in similarity_list])
            target_metric_vec = self.get_metrics(
                t0, t1, self.metric_type, self.operator_func
            )
            line = np.concatenate((feature_vec1, target_similarity, target_metric_vec))
            # line = np.concatenate((target_similarity, reference_similarity,feature_similarity,target_metric_vec,reference_metric_vec,feature_metric_vec))

            line = np.nan_to_num(line)
            x_train.append(line)  # concatenate target and reference output, []
            y_train.append(row["label"])
        for row in test_data:
            row = json.loads(row)
            a = np.array(row["gcn_pred0"])
            b = np.array(row["gcn_pred1"])
            feature_vec1 = self.operator_func(self.operator, a, b)  # posterior poerator

            t0 = np.array(row["gcn_pred0"])
            t1 = np.array(row["gcn_pred1"])
            target_similarity = np.array([row(t0, t1) for row in similarity_list])
            target_metric_vec = self.get_metrics(
                t0, t1, self.metric_type, self.operator_func
            )
            line = np.concatenate((feature_vec1, target_similarity, target_metric_vec))
            line = np.nan_to_num(line)
            x_test.append(line)
            y_test.append(row["label"])
        return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)
