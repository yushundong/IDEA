from __future__ import print_function
import argparse
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import argparse
from tqdm import tqdm
import os
from sklearn.decomposition import PCA
import pdb
from sklearn.linear_model import LogisticRegression

# from program.sgc_unlearn.mia.MLG_TSTF_with_ill_precition import MIA
from mia.MLG_TSTF import MIA
import csv

# from mia.MLG_TSTF_w_original_shadow_model import MIA

# Below is for graph learning part
from torch_geometric.nn.conv import MessagePassing
from typing import Optional

from torch import Tensor
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import degree

from torch_scatter import scatter_add
from torch_sparse import SparseTensor, fill_diag, matmul, mul
from torch_sparse import sum as sparsesum

from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes

from torch_geometric.datasets import Planetoid, Coauthor, Amazon, CitationFull
from ogb.nodeproppred import PygNodePropPredDataset
import os.path as osp

from torch.nn import init
from utils import *

from sklearn import preprocessing
from numpy.linalg import norm
import warnings

warnings.filterwarnings("ignore")
from silence_tensorflow import silence_tensorflow


silence_tensorflow()


def main(args):
    ######
    # Load the data
    print("=" * 10 + "Loading data" + "=" * 10)
    print("Dataset:", args.dataset)
    # read data from PyG datasets (cora, citeseer, pubmed)
    if args.dataset in ["cora", "citeseer", "pubmed"]:
        path = osp.join(args.data_dir, "data", args.dataset)
        dataset = Planetoid(path, args.dataset)
        data = dataset[0].to(device)
        data = random_planetoid_splits(
            data,
            num_classes=dataset.num_classes,
            val_lb=int(data.x.shape[0] * 0),
            test_lb=int(data.x.shape[0] * 0.1),
            Flag=1,
        ).to(device)
    elif args.dataset == "Physics":
        dataset = Coauthor(args.data_dir, args.dataset)
        data = dataset[0].to(device)

        feature = data.x
        pca = PCA(n_components=500)
        reduced_feature = pca.fit_transform(feature.cpu().detach().numpy())
        data.x = torch.from_numpy(reduced_feature).to(device)

        data = random_planetoid_splits(
            data,
            num_classes=dataset.num_classes,
            val_lb=int(data.x.shape[0] * 0),
            test_lb=int(data.x.shape[0] * 0.1),
            Flag=1,
        ).to(device)
        # data.y = data.y.squeeze(-1)
    elif args.dataset == "CS":
        dataset = Coauthor(args.data_dir, args.dataset)
        data = dataset[0].to(device)
        data = random_planetoid_splits(
            data,
            num_classes=dataset.num_classes,
            val_lb=int(data.x.shape[0] * 0),
            test_lb=int(data.x.shape[0] * 0.1),
            Flag=1,
        ).to(device)
    elif args.dataset in ["ogbn-arxiv", "ogbn-products"]:
        dataset = PygNodePropPredDataset(name=args.dataset, root=args.data_dir)
        data = dataset[0].to(device)
        split_idx = dataset.get_idx_split()
        data.train_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
        data.train_mask[split_idx["train"]] = True
        data.val_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
        data.val_mask[split_idx["valid"]] = True
        data.test_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
        data.test_mask[split_idx["test"]] = True
        data.y = data.y.squeeze(-1)
    elif args.dataset in ["computers", "photo"]:
        path = osp.join(args.data_dir, "data", args.dataset)
        dataset = Amazon(path, args.dataset)
        data = dataset[0]
        data = random_planetoid_splits(
            data, num_classes=dataset.num_classes, val_lb=500, test_lb=1000, Flag=1
        ).to(device)
    else:
        raise ("Error: Not supported dataset yet.")
    # save the degree of each node for later use
    test_indices = data.test_mask.nonzero().squeeze(-1)
    train_indices = data.train_mask.nonzero().squeeze(-1)
    row = data.edge_index[0]
    deg = degree(row)

    # ratio to remove
    if args.is_ratio:
        args.num_removes = int(args.ratio_removes * data.x.shape[0])

    ##########
    # our removal
    # all norm here is NOT accumulated, need to use np.cumsum in plots
    grad_norm_approx = torch.zeros((args.num_removes, args.trails)).float()
    removal_times = torch.zeros(
        (args.num_removes, args.trails)
    ).float()  # record the time of each removal
    acc_removal = torch.zeros(
        (2, args.num_removes, args.trails)
    ).float()  # record the acc after removal, 0 for val, 1 for test
    grad_norm_worst = torch.zeros(
        (args.num_removes, args.trails)
    ).float()  # worst case norm bound
    grad_norm_real = torch.zeros((args.num_removes, args.trails)).float()  # true norm
    # graph retrain
    removal_times_graph_retrain = torch.zeros((args.num_removes, args.trails)).float()
    acc_graph_retrain = torch.zeros((2, args.num_removes, args.trails)).float()
    # guo removal
    grad_norm_approx_guo = torch.zeros((args.num_removes, args.trails)).float()
    removal_times_guo = torch.zeros(
        (args.num_removes, args.trails)
    ).float()  # record the time of each removal
    acc_guo = torch.zeros(
        (2, args.num_removes, args.trails)
    ).float()  # first row for val acc, second row for test acc
    grad_norm_real_guo = torch.zeros(
        (args.num_removes, args.trails)
    ).float()  # true norm
    # guo retrain
    removal_times_guo_retrain = torch.zeros(
        (args.num_removes, args.trails)
    ).float()  # record the time of each removal
    acc_guo_retrain = torch.zeros(
        (2, args.num_removes, args.trails)
    ).float()  # first row for val acc, second row for test acc
    acc_train_test = torch.zeros(
        args.trails
    ).float()  # first row for val acc, second row for test acc
    time_train_test = torch.zeros(args.trails).float()
    attack_acc_all = torch.zeros(args.trails).float()
    attack_auc_all = torch.zeros(args.trails).float()
    acc_pos_all = torch.zeros(args.trails).float()
    f1_pos_all = torch.zeros(args.trails).float()

    for trail_iter in range(args.trails):
        print("*" * 10, trail_iter, "*" * 10)
        if args.fix_random_seed:
            # fix the random seed for perm
            np.random.seed(trail_iter)

        ##########

        # process features
        if args.featNorm:
            X = preprocess_data(data.x).to(device)
        else:
            X = data.x.to(device)
        # save a copy of X for removal
        X_scaled_copy_guo = X.clone().detach().float()

        # process labels
        if args.train_mode == "binary":
            if "+" in args.Y_binary:
                # two classes are specified
                class1 = int(args.Y_binary.split("+")[0])
                class2 = int(args.Y_binary.split("+")[1])
                Y = data.y.clone().detach().float()
                Y[data.y == class1] = 1
                Y[data.y == class2] = -1
                interested_data_mask = (data.y == class1) + (data.y == class2)
                train_mask = data.train_mask * interested_data_mask
                val_mask = data.val_mask * interested_data_mask
                test_mask = data.test_mask * interested_data_mask
            else:
                # one vs rest
                class1 = int(args.Y_binary)
                Y = data.y.clone().detach().float()
                Y[data.y == class1] = 1
                Y[data.y != class1] = -1
                train_mask = data.train_mask
                val_mask = data.val_mask
                test_mask = data.test_mask
            y_train, y_val, y_test = (
                Y[train_mask].to(device),
                Y[val_mask].to(device),
                Y[test_mask].to(device),
            )
        else:
            # multiclass classification

            train_mask = data.train_mask
            val_mask = data.val_mask
            test_mask = data.test_mask
            y_train = F.one_hot(data.y[data.train_mask]) * 2 - 1
            y_train_ = F.one_hot(data.y[data.train_mask]).float()
            y_train = y_train.float().to(device)
            y_val = data.y[data.val_mask].to(device)
            y_test = data.y[data.test_mask].to(device)

        assert args.noise_mode == "data"

        if args.compare_gnorm:
            # if we want to compare the residual gradient norm of three cases, we should not add noise
            # and make budget very large
            b_std = 0
        else:
            if args.noise_mode == "data":
                b_std = args.std
            elif args.noise_mode == "worst":
                b_std = args.std  # change to worst case sigma
            else:
                raise ("Error: Not supported noise model.")

        #############
        # initial training with graph
        print("=" * 10 + "Training on full dataset with graph" + "=" * 10)
        start_trail = time.time()
        Propagation = MyGraphConv(
            K=args.prop_step,
            add_self_loops=args.add_self_loops,
            alpha=args.alpha,
            XdegNorm=args.XdegNorm,
            GPR=args.GPR,
        ).to(device)
        # pdb.set_trace()
        if args.prop_step > 0:
            X = Propagation(X, data.edge_index)

        X = X.float()
        X_train = X[train_mask].to(device)
        X_val = X[val_mask].to(device)
        X_test = X[test_mask].to(device)

        print(
            "Train node:{}, Val node:{}, Test node:{}, Edges:{}, Feature dim:{}".format(
                X_train.shape[0],
                X_val.shape[0],
                X_test.shape[0],
                data.edge_index.shape[1],
                X_train.shape[1],
            )
        )
        ############
        # train removal-enabled linear model
        print(
            "With graph, train mode:", args.train_mode, ", optimizer:", args.optimizer
        )

        # reserved for future extension
        weight = None
        # in our case weight should always be None
        assert weight is None
        # record the optimal gradient norm wrt the whole training set
        opt_grad_norm = 0

        if args.train_mode == "ovr":
            b = b_std * torch.randn(X_train.size(1), y_train.size(1)).float().to(device)
            if args.train_sep:
                # train K binary LR models separately
                w = torch.zeros(b.size()).float().to(device)
                for k in range(y_train.size(1)):
                    if weight is None:
                        w[:, k] = lr_optimize(
                            X_train,
                            y_train[:, k],
                            args.lam,
                            b=b[:, k],
                            num_steps=args.num_steps,
                            verbose=args.verbose,
                            opt_choice=args.optimizer,
                            lr=args.lr,
                            wd=args.wd,
                        )
                    else:
                        w[:, k] = lr_optimize(
                            X_train[weight[:, k].gt(0)],
                            y_train[:, k][weight[:, k].gt(0)],
                            args.lam,
                            b=b[:, k],
                            num_steps=args.num_steps,
                            verbose=args.verbose,
                            opt_choice=args.optimizer,
                            lr=args.lr,
                            wd=args.wd,
                        )
            else:
                # train K binary LR models jointly
                w = ovr_lr_optimize_entropy(
                    X_train,
                    y_train_,
                    args.lam,
                    weight,
                    b=b,
                    num_steps=args.num_steps,
                    verbose=args.verbose,
                    opt_choice=args.optimizer,
                    lr=args.lr,
                    wd=args.wd,
                    y_=y_train,
                )

            # record the opt_grad_norm
            for k in range(y_train.size(1)):
                opt_grad_norm += (
                    lr_grad(w[:, k], X_train, y_train[:, k], args.lam).norm().cpu()
                )
        else:
            b = b_std * torch.randn(X_train.size(1)).float().to(device)
            w = lr_optimize(
                X_train,
                y_train,
                args.lam,
                b=b,
                num_steps=args.num_steps,
                verbose=args.verbose,
                opt_choice=args.optimizer,
                lr=args.lr,
                wd=args.wd,
            )
            opt_grad_norm = lr_grad(w, X_train, y_train, args.lam).norm().cpu()
        train_time = time.time() - start_trail
        # print("Time elapsed: %.2fs" % (train_time))
        if args.train_mode == "ovr":
            test_acc = ovr_lr_eval(w, X_test, y_test)
            # print("Val accuracy = %.4f" % ovr_lr_eval(w, X_val, y_val))
            # print("Test accuracy = %.4f" % ovr_lr_eval(w, X_test, y_test))
        else:
            test_acc = lr_eval(w, X_test, y_test)
            # print("Val accuracy = %.4f" % lr_eval(w, X_val, y_val))
            # print("Test accuracy = %.4f" % lr_eval(w, X_test, y_test))

        time_train_test[trail_iter] = train_time
        acc_train_test[trail_iter] = test_acc

        ###########
        if args.compare_guo:
            # initial training without graph
            print("=" * 10 + "Training on full dataset without graph" + "=" * 10)
            start = time.time()

            # only the data preparation part is different
            X_train = X_scaled_copy_guo[train_mask].to(device)
            X_val = X_scaled_copy_guo[val_mask].to(device)
            X_test = X_scaled_copy_guo[test_mask].to(device)

            print(
                "Train node:{}, Val node:{}, Test node:{}, Feature dim:{}".format(
                    X_train.shape[0], X_val.shape[0], X_test.shape[0], X_train.shape[1]
                )
            )
            ######
            # train removal-enabled linear model without graph
            print(
                "Without graph, train mode:",
                args.train_mode,
                ", optimizer:",
                args.optimizer,
            )

            weight = None
            # in our case weight should always be None
            assert weight is None
            opt_grad_norm_guo = 0

            if args.train_mode == "ovr":
                b = b_std * torch.randn(X_train.size(1), y_train.size(1)).float().to(
                    device
                )
                if args.train_sep:
                    # train K binary LR models separately
                    w_guo = torch.zeros(b.size()).float().to(device)
                    for k in range(y_train.size(1)):
                        if weight is None:
                            w_guo[:, k] = lr_optimize(
                                X_train,
                                y_train[:, k],
                                args.lam,
                                b=b[:, k],
                                num_steps=args.num_steps,
                                verbose=args.verbose,
                                opt_choice=args.optimizer,
                                lr=args.lr,
                                wd=args.wd,
                            )
                        else:
                            w_guo[:, k] = lr_optimize(
                                X_train[weight[:, k].gt(0)],
                                y_train[:, k][weight[:, k].gt(0)],
                                args.lam,
                                b=b[:, k],
                                num_steps=args.num_steps,
                                verbose=args.verbose,
                                opt_choice=args.optimizer,
                                lr=args.lr,
                                wd=args.wd,
                            )
                else:
                    # train K binary LR models jointly
                    w_guo = ovr_lr_optimize(
                        X_train,
                        y_train,
                        args.lam,
                        weight,
                        b=b,
                        num_steps=args.num_steps,
                        verbose=args.verbose,
                        opt_choice=args.optimizer,
                        lr=args.lr,
                        wd=args.wd,
                    )
                # record the opt_grad_norm
                for k in range(y_train.size(1)):
                    opt_grad_norm_guo += (
                        lr_grad(w_guo[:, k], X_train, y_train[:, k], args.lam)
                        .norm()
                        .cpu()
                    )
            else:
                b = b_std * torch.randn(X_train.size(1)).float().to(device)
                w_guo = lr_optimize(
                    X_train,
                    y_train,
                    args.lam,
                    b=b,
                    num_steps=args.num_steps,
                    verbose=args.verbose,
                    opt_choice=args.optimizer,
                    lr=args.lr,
                    wd=args.wd,
                )
                opt_grad_norm_guo = (
                    lr_grad(w_guo, X_train, y_train, args.lam).norm().cpu()
                )

            print("Time elapsed: %.2fs" % (time.time() - start))
            if args.train_mode == "ovr":
                print("Val accuracy = %.4f" % ovr_lr_eval(w_guo, X_val, y_val))
                print("Test accuracy = %.4f" % ovr_lr_eval(w_guo, X_test, y_test))
            else:
                print("Val accuracy = %.4f" % lr_eval(w_guo, X_val, y_val))
                print("Test accuracy = %.4f" % lr_eval(w_guo, X_test, y_test))

        ###########
        # budget for removal
        c_val = get_c(args.delta)
        # if we need to compute the norms, we should not retrain at all
        if args.compare_gnorm:
            budget = 1e5
        else:
            if args.train_mode == "ovr":
                budget = get_budget(b_std, args.eps, c_val) * y_train.size(1)
            else:
                budget = get_budget(b_std, args.eps, c_val)
        gamma = 1 / 4  # pre-computed for -logsigmoid loss
        print("Budget:", budget)
        ##########

        train_id = torch.arange(data.x.shape[0]).to(device)[train_mask]
        perm = torch.from_numpy(np.random.permutation(train_id.shape[0]))
        removal_queue = train_id[perm]
        edge_mask = torch.ones(data.edge_index.shape[1], dtype=torch.bool)

        X_scaled_copy = X_scaled_copy_guo.clone().detach().float()
        w_approx = w.clone().detach()  # copy the parameters to modify
        X_old = X.clone().detach().to(device)

        num_retrain = 0
        grad_norm_approx_sum = 0
        node_removes = []
        # start the removal process
        print("=" * 10 + "Testing our removal" + "=" * 10)
        for i in tqdm(range(args.num_removes)):
            node_removes.append(removal_queue[i].item())
            # First, replace removal features with 0 vector
            X_scaled_copy[removal_queue[i]] = 0
            if args.removal_mode == "node":
                # Then remove the correpsonding edges
                edge_mask[data.edge_index[0] == removal_queue[i]] = False
                edge_mask[data.edge_index[1] == removal_queue[i]] = False
                # make sure we do not remove self-loops
                self_loop_idx = (
                    torch.logical_and(
                        data.edge_index[0] == removal_queue[i],
                        data.edge_index[1] == removal_queue[i],
                    )
                    .nonzero()
                    .squeeze(-1)
                )
                if self_loop_idx.size(0) > 0:
                    edge_mask[self_loop_idx] = True

            start = time.time()
            # Get propagated features
            if args.prop_step > 0:
                X_new = Propagation(X_scaled_copy, data.edge_index[:, edge_mask]).to(
                    device
                )
            else:
                X_new = X_scaled_copy.to(device)

            X_val_new = X_new[val_mask]
            X_test_new = X_new[test_mask]

            # note that the removed data point should still not be used in computing K or H
            # removal_queue[(i+1):] are the remaining training idx
            K = get_K_matrix(X_new[removal_queue[(i + 1) :]]).to(device)
            spec_norm = sqrt_spectral_norm(K)

            if args.train_mode == "ovr":
                # removal from all one-vs-rest models
                X_rem = X_new[removal_queue[(i + 1) :]]
                for k in range(y_train.size(1)):
                    y_rem = y_train[perm[(i + 1) :], k]
                    H_inv = lr_hessian_inv(w_approx[:, k], X_rem, y_rem, args.lam)
                    # grad_i is the difference
                    grad_old = lr_grad(
                        w_approx[:, k],
                        X_old[removal_queue[i:]],
                        y_train[perm[i:], k],
                        args.lam,
                    )
                    grad_new = lr_grad(w_approx[:, k], X_rem, y_rem, args.lam)
                    grad_i = grad_old - grad_new
                    Delta = H_inv.mv(grad_i)
                    Delta_p = X_rem.mv(Delta)
                    # update w here. If beta exceed the budget, w_approx will be retrained
                    w_approx[:, k] += Delta
                    # data dependent norm
                    grad_norm_approx[i, trail_iter] += (
                        Delta.norm() * Delta_p.norm() * spec_norm * gamma
                    ).cpu()
                    if args.compare_gnorm:
                        grad_norm_real[i, trail_iter] += (
                            lr_grad(w_approx[:, k], X_rem, y_rem, args.lam).norm().cpu()
                        )
                        if args.removal_mode == "node":
                            grad_norm_worst[i, trail_iter] += get_worst_Gbound_node(
                                args.lam,
                                X_rem.shape[0],
                                args.prop_step,
                                deg[removal_queue[i]],
                            ).cpu()
                        elif args.removal_mode == "feature":
                            grad_norm_worst[i, trail_iter] += get_worst_Gbound_feature(
                                args.lam, X_rem.shape[0], deg[removal_queue[i]]
                            ).cpu()
                # decide after all classes
                if grad_norm_approx_sum + grad_norm_approx[i, trail_iter] > budget:
                    # retrain the model
                    grad_norm_approx_sum = 0
                    b = b_std * torch.randn(
                        X_train.size(1), y_train.size(1)
                    ).float().to(device)
                    w_approx = ovr_lr_optimize(
                        X_rem,
                        y_train[perm[(i + 1) :]],
                        args.lam,
                        weight,
                        b=b,
                        num_steps=args.num_steps,
                        verbose=args.verbose,
                        opt_choice=args.optimizer,
                        lr=args.lr,
                        wd=args.wd,
                    )
                    num_retrain += 1
                else:
                    grad_norm_approx_sum += grad_norm_approx[i, trail_iter]
                # record acc each round
                acc_removal[0, i, trail_iter] = ovr_lr_eval(w_approx, X_val_new, y_val)
                acc_removal[1, i, trail_iter] = ovr_lr_eval(
                    w_approx, X_test_new, y_test
                )
                # posterior = X.mm(w_approx

            else:
                # removal from a single binary logistic regression model
                X_rem = X_new[removal_queue[(i + 1) :]]
                y_rem = y_train[perm[(i + 1) :]]
                H_inv = lr_hessian_inv(w_approx, X_rem, y_rem, args.lam)
                # grad_i should be the difference
                grad_old = lr_grad(
                    w_approx, X_old[removal_queue[i:]], y_train[perm[i:]], args.lam
                )
                grad_new = lr_grad(w_approx, X_rem, y_rem, args.lam)
                grad_i = grad_old - grad_new
                Delta = H_inv.mv(grad_i)
                Delta_p = X_rem.mv(Delta)
                w_approx += Delta
                grad_norm_approx[i, trail_iter] += (
                    Delta.norm() * Delta_p.norm() * spec_norm * gamma
                ).cpu()
                if args.compare_gnorm:
                    grad_norm_real[i, trail_iter] += (
                        lr_grad(w_approx, X_rem, y_rem, args.lam).norm().cpu()
                    )
                    if args.removal_mode == "node":
                        grad_norm_worst[i, trail_iter] += get_worst_Gbound_node(
                            args.lam,
                            X_rem.shape[0],
                            args.prop_step,
                            deg[removal_queue[i]],
                        ).cpu()
                    elif args.removal_mode == "feature":
                        grad_norm_worst[i, trail_iter] += get_worst_Gbound_feature(
                            args.lam, X_rem.shape[0], deg[removal_queue[i]]
                        ).cpu()

                if grad_norm_approx_sum + grad_norm_approx[i, trail_iter] > budget:
                    # retrain the model
                    grad_norm_approx_sum = 0
                    b = b_std * torch.randn(X_train.size(1)).float().to(device)
                    w_approx = lr_optimize(
                        X_rem,
                        y_rem,
                        args.lam,
                        b=b,
                        num_steps=args.num_steps,
                        verbose=args.verbose,
                        opt_choice=args.optimizer,
                        lr=args.lr,
                        wd=args.wd,
                    )
                    num_retrain += 1
                else:
                    grad_norm_approx_sum += grad_norm_approx[i, trail_iter]
                # record acc each round
                acc_removal[0, i, trail_iter] = lr_eval(w_approx, X_val_new, y_val)
                acc_removal[1, i, trail_iter] = lr_eval(w_approx, X_test_new, y_test)

            removal_times[i, trail_iter] = time.time() - start
            # Remember to replace X_old with X_new
            X_old = X_new.clone().detach()
            # if i % args.disp == 0:
            #     print(
            #         "Iteration %d: time = %.2fs, number of retrain = %d"
            #         % (i + 1, removal_times[i, trail_iter], num_retrain)
            #     )
            #     print(
            #         "Val acc = %.4f, Test acc = %.4f"
            #         % (acc_removal[0, i, trail_iter], acc_removal[1, i, trail_iter])
            #     )

        if args.is_attack_feauture:
            try:
                remove_dim_idx = torch.load(
                    f"/IDEA/attack_materials/seed{args.run_seed_feature}_IDEA_cora_partial_feature_0.05_SGC_{args.remove_feature_ratio}.pth"
                )["unlearned_feature_dim_idx"]
            except:
                remove_dim_idx = [0]

            one_hot_y = F.one_hot(data.y, data.y.max().item() + 1).float()
            posterior0 = (F.softmax(X.mm(w_approx), 1))[node_removes]
            for row in node_removes:
                for col in remove_dim_idx:
                    X[row][col] = 0  # w/o
            posterior = X.mm(w_approx)
            posterior1 = (F.softmax(X.mm(w_approx), 1))[node_removes]
            loss_fn = nn.CrossEntropyLoss()
            loss0 = loss_fn(posterior0, data.y[node_removes])
            loss1 = loss_fn(posterior1, data.y[node_removes])
            if not os.path.exists("result/loss_feature.csv"):
                with open("result/loss_feature.csv", "w") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            "dataset",
                            "run_seed_feature",
                            "remove_feature_ratio",
                            "loss_w",
                            "loss_wo",
                        ]
                    )
            with open("result/loss_feature.csv", "a") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        args.dataset,
                        args.run_seed_feature,
                        args.remove_feature_ratio,
                        loss0.item(),
                        loss1.item(),
                    ]
                )

        #######
        # retrain each round with graph
        if args.compare_retrain:
            X_scaled_copy = X_scaled_copy_guo.clone().detach()
            edge_mask = torch.ones(data.edge_index.shape[1], dtype=torch.bool)
            # start the removal process
            print("=" * 10 + "Testing with graph retrain" + "=" * 10)
            for i in range(args.num_removes):
                # First, replace removal features with 0 vector
                X_scaled_copy[removal_queue[i]] = 0
                # Then remove the correpsonding edges
                if args.removal_mode == "node":
                    edge_mask[data.edge_index[0] == removal_queue[i]] = False
                    edge_mask[data.edge_index[1] == removal_queue[i]] = False
                    # make sure we do not remove self-loops
                    self_loop_idx = (
                        torch.logical_and(
                            data.edge_index[0] == removal_queue[i],
                            data.edge_index[1] == removal_queue[i],
                        )
                        .nonzero()
                        .squeeze(-1)
                    )
                    if self_loop_idx.size(0) > 0:
                        edge_mask[self_loop_idx] = True

                start = time.time()
                # Get propagated features
                if args.prop_step > 0:
                    X_new = Propagation(
                        X_scaled_copy, data.edge_index[:, edge_mask]
                    ).to(device)
                else:
                    X_new = X_scaled_copy.to(device)

                X_val_new = X_new[val_mask]
                X_test_new = X_new[test_mask]

                if args.train_mode == "ovr":
                    # removal from all one-vs-rest models
                    X_rem = X_new[removal_queue[(i + 1) :]]
                    y_rem = y_train[perm[(i + 1) :]]
                    # retrain the model
                    # we do not need to add noise if we are retraining every time
                    # b = b_std * torch.randn(X_train.size(1), y_train.size(1)).float().to(device)
                    w_graph_retrain = ovr_lr_optimize(
                        X_rem,
                        y_rem,
                        args.lam,
                        weight,
                        b=None,
                        num_steps=args.num_steps,
                        verbose=args.verbose,
                        opt_choice=args.optimizer,
                        lr=args.lr,
                        wd=args.wd,
                    )
                    acc_graph_retrain[0, i, trail_iter] = ovr_lr_eval(
                        w_graph_retrain, X_val_new, y_val
                    )
                    acc_graph_retrain[1, i, trail_iter] = ovr_lr_eval(
                        w_graph_retrain, X_test_new, y_test
                    )
                else:
                    # removal from a single binary logistic regression model
                    X_rem = X_new[removal_queue[(i + 1) :]]
                    y_rem = y_train[perm[(i + 1) :]]
                    # retrain the model
                    # b = b_std * torch.randn(X_train.size(1)).float().to(device)
                    w_graph_retrain = lr_optimize(
                        X_rem,
                        y_rem,
                        args.lam,
                        b=None,
                        num_steps=args.num_steps,
                        verbose=args.verbose,
                        opt_choice=args.optimizer,
                        lr=args.lr,
                        wd=args.wd,
                    )
                    acc_graph_retrain[0, i, trail_iter] = lr_eval(
                        w_graph_retrain, X_val_new, y_val
                    )
                    acc_graph_retrain[1, i, trail_iter] = lr_eval(
                        w_graph_retrain, X_test_new, y_test
                    )

                removal_times_graph_retrain[i, trail_iter] = time.time() - start
                if i % args.disp == 0:
                    print(
                        "Iteration %d, time = %.2fs, val acc = %.4f, test acc = %.4f"
                        % (
                            i + 1,
                            removal_times_graph_retrain[i, trail_iter],
                            acc_graph_retrain[0, i, trail_iter],
                            acc_graph_retrain[1, i, trail_iter],
                        )
                    )

        #######
        # guo removal
        if args.compare_guo and args.removal_mode != "edge":
            w_approx_guo = w_guo.clone().detach()  # copy the parameters to modify
            num_retrain = 0
            grad_norm_approx_sum_guo = 0
            # prepare the train/val/test sets
            X_train = X_scaled_copy_guo[train_mask].to(device)
            X_train_perm = X_train[perm]
            y_train_perm = y_train[perm]
            K = get_K_matrix(X_train_perm).to(device)
            X_val = X_scaled_copy_guo[val_mask].to(device)
            X_test = X_scaled_copy_guo[test_mask].to(device)
            # start the removal process
            print("=" * 10 + "Testing Guo et al. removal" + "=" * 10)
            for i in range(args.num_removes):
                start = time.time()
                if args.train_mode == "ovr":
                    # removal from all one-vs-rest models
                    X_rem = X_train_perm[(i + 1) :]
                    # update matrix K
                    K -= torch.outer(X_train_perm[i], X_train_perm[i])
                    spec_norm = sqrt_spectral_norm(K)
                    for k in range(y_train_perm.size(1)):
                        y_rem = y_train_perm[(i + 1) :, k]
                        H_inv = lr_hessian_inv(
                            w_approx_guo[:, k], X_rem, y_rem, args.lam
                        )
                        # grad_i is the difference
                        grad_i = lr_grad(
                            w_approx_guo[:, k],
                            X_train_perm[i].unsqueeze(0),
                            y_train_perm[i, k].unsqueeze(0),
                            args.lam,
                        )
                        Delta = H_inv.mv(grad_i)
                        Delta_p = X_rem.mv(Delta)
                        # update w here. If beta exceed the budget, w_approx_guo will be retrained
                        w_approx_guo[:, k] += Delta
                        grad_norm_approx_guo[i, trail_iter] += (
                            Delta.norm() * Delta_p.norm() * spec_norm * gamma
                        ).cpu()
                        if args.compare_gnorm:
                            grad_norm_real_guo[i, trail_iter] += (
                                lr_grad(w_approx_guo[:, k], X_rem, y_rem, args.lam)
                                .norm()
                                .cpu()
                            )
                    # decide after all classes
                    if (
                        grad_norm_approx_sum_guo + grad_norm_approx_guo[i, trail_iter]
                        > budget
                    ):
                        # retrain the model
                        grad_norm_approx_sum_guo = 0
                        b = b_std * torch.randn(
                            X_train_perm.size(1), y_train_perm.size(1)
                        ).float().to(device)
                        w_approx_guo = ovr_lr_optimize(
                            X_rem,
                            y_train_perm[(i + 1) :],
                            args.lam,
                            weight,
                            b=b,
                            num_steps=args.num_steps,
                            verbose=args.verbose,
                            opt_choice=args.optimizer,
                            lr=args.lr,
                            wd=args.wd,
                        )
                        num_retrain += 1
                    else:
                        grad_norm_approx_sum_guo += grad_norm_approx_guo[i, trail_iter]
                    # record the acc each round
                    acc_guo[0, i, trail_iter] = ovr_lr_eval(w_approx_guo, X_val, y_val)
                    acc_guo[1, i, trail_iter] = ovr_lr_eval(
                        w_approx_guo, X_test, y_test
                    )
                else:
                    # removal from a single binary logistic regression model
                    X_rem = X_train_perm[(i + 1) :]
                    y_rem = y_train_perm[(i + 1) :]
                    H_inv = lr_hessian_inv(w_approx_guo, X_rem, y_rem, args.lam)
                    grad_i = lr_grad(
                        w_approx_guo,
                        X_train_perm[i].unsqueeze(0),
                        y_train_perm[i].unsqueeze(0),
                        args.lam,
                    )
                    K -= torch.outer(X_train_perm[i], X_train_perm[i])
                    spec_norm = sqrt_spectral_norm(K)
                    Delta = H_inv.mv(grad_i)
                    Delta_p = X_rem.mv(Delta)
                    w_approx_guo += Delta
                    grad_norm_approx_guo[i, trail_iter] += (
                        Delta.norm() * Delta_p.norm() * spec_norm * gamma
                    ).cpu()
                    if args.compare_gnorm:
                        grad_norm_real_guo[i, trail_iter] += (
                            lr_grad(w_approx_guo, X_rem, y_rem, args.lam).norm().cpu()
                        )
                    if (
                        grad_norm_approx_sum_guo + grad_norm_approx_guo[i, trail_iter]
                        > budget
                    ):
                        # retrain the model
                        grad_norm_approx_sum_guo = 0
                        b = b_std * torch.randn(X_train_perm.size(1)).float().to(device)
                        w_approx_guo = lr_optimize(
                            X_rem,
                            y_rem,
                            args.lam,
                            b=b,
                            num_steps=args.num_steps,
                            verbose=args.verbose,
                            opt_choice=args.optimizer,
                            lr=args.lr,
                            wd=args.wd,
                        )
                        num_retrain += 1
                    else:
                        grad_norm_approx_sum_guo += grad_norm_approx_guo[i, trail_iter]
                    # record the acc each round
                    acc_guo[0, i, trail_iter] = lr_eval(w_approx_guo, X_val, y_val)
                    acc_guo[1, i, trail_iter] = lr_eval(w_approx_guo, X_test, y_test)

                removal_times_guo[i, trail_iter] = time.time() - start
                if i % args.disp == 0:
                    print(
                        "Iteration %d: time = %.2fs, number of retrain = %d"
                        % (i + 1, removal_times_guo[i, trail_iter], num_retrain)
                    )
                    print(
                        "Val acc = %.4f, Test acc = %.4f"
                        % (acc_guo[0, i, trail_iter], acc_guo[1, i, trail_iter])
                    )

        #######
        # retrain each round without graph
        if args.removal_mode != "edge" and args.compare_retrain and args.compare_guo:
            X_train = X_scaled_copy_guo[train_mask].to(device)
            X_train_perm = X_train[perm]
            y_train_perm = y_train[perm]
            X_val = X_scaled_copy_guo[val_mask].to(device)
            X_test = X_scaled_copy_guo[test_mask].to(device)

            # start the removal process
            print("=" * 10 + "Testing without graph retrain" + "=" * 10)
            for i in range(args.num_removes):
                start = time.time()
                if args.train_mode == "ovr":
                    # removal from all one-vs-rest models
                    X_rem = X_train_perm[(i + 1) :]
                    y_rem = y_train_perm[(i + 1) :]
                    # retrain the model
                    # b = b_std * torch.randn(X_train_perm.size(1), y_train_perm.size(1)).float().to(device)
                    w_guo_retrain = ovr_lr_optimize(
                        X_rem,
                        y_rem,
                        args.lam,
                        weight,
                        b=None,
                        num_steps=args.num_steps,
                        verbose=args.verbose,
                        opt_choice=args.optimizer,
                        lr=args.lr,
                        wd=args.wd,
                    )
                    acc_guo_retrain[0, i, trail_iter] = ovr_lr_eval(
                        w_guo_retrain, X_val, y_val
                    )
                    acc_guo_retrain[1, i, trail_iter] = ovr_lr_eval(
                        w_guo_retrain, X_test, y_test
                    )
                else:
                    # removal from a single binary logistic regression model
                    X_rem = X_train_perm[(i + 1) :]
                    y_rem = y_train_perm[(i + 1) :]
                    # retrain the model
                    # b = b_std * torch.randn(X_train_perm.size(1)).float().to(device)
                    w_guo_retrain = lr_optimize(
                        X_rem,
                        y_rem,
                        args.lam,
                        b=None,
                        num_steps=args.num_steps,
                        verbose=args.verbose,
                        opt_choice=args.optimizer,
                        lr=args.lr,
                        wd=args.wd,
                    )
                    acc_guo_retrain[0, i, trail_iter] = lr_eval(
                        w_guo_retrain, X_val, y_val
                    )
                    acc_guo_retrain[1, i, trail_iter] = lr_eval(
                        w_guo_retrain, X_test, y_test
                    )

                removal_times_guo_retrain[i, trail_iter] = time.time() - start
                if i % args.disp == 0:
                    print(
                        "Iteration %d, time = %.2fs, val acc = %.4f, test acc = %.4f"
                        % (
                            i + 1,
                            removal_times_guo_retrain[i, trail_iter],
                            acc_guo_retrain[0, i, trail_iter],
                            acc_guo_retrain[1, i, trail_iter],
                        )
                    )
        # attack
        if args.is_attack_node:
            mia = MIA(args, posterior, node_removes, train_indices, test_indices)
            acc_pos, f1_pos, attack_acc, attack_auc = mia.get_results()
            attack_acc_all[trail_iter] = attack_acc
            attack_auc_all[trail_iter] = attack_auc
            acc_pos_all[trail_iter] = acc_pos
            f1_pos_all[trail_iter] = f1_pos

    attack_acc_mean = attack_acc_all.mean().numpy()
    attack_acc_std = attack_acc_all.std().numpy()
    attack_auc_mean = attack_auc_all.mean().numpy()
    attack_auc_std = attack_auc_all.std().numpy()
    acc_pos_mean = acc_pos_all.mean().numpy()
    acc_pos_std = acc_pos_all.std().numpy()
    f1_pos_mean = f1_pos_all.mean().numpy()
    f1_pos_std = f1_pos_all.std().numpy()

    # save attack result with csv
    if not os.path.exists("result/node_attack_results.csv"):
        with open("result/node_attack_results.csv", "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "Dataset",
                    "Exp",
                    "Unlearn Ratio",
                    "ACC Pos Mean",
                    "ACC Pos Std",
                    "F1 Pos Mean",
                    "F1 Pos Std",
                ]
            )
    with open("result/node_attack_results.csv", "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                args.dataset,
                "node",
                args.ratio_removes,
                acc_pos_mean,
                acc_pos_std,
                f1_pos_mean,
                f1_pos_std,
            ]
        )

    # save all results
    if not osp.exists(args.result_dir):
        os.makedirs(args.result_dir)
    save_path = (
        "%s/%s_std_%.0e_lam_%.0e_nr_%d_K_%d_opt_%s_mode_%s_eps_%.1f_delta_%.0e"
        % (
            args.result_dir,
            args.dataset,
            b_std,
            args.lam,
            args.num_removes,
            args.prop_step,
            args.optimizer,
            args.removal_mode,
            args.eps,
            args.delta,
        )
    )
    if args.train_mode == "binary":
        save_path += "_bin_%s" % args.Y_binary
    if args.GPR:
        save_path += "_gpr"
    if args.compare_gnorm:
        save_path += "_gnorm"
    if args.compare_retrain:
        save_path += "_retrain"
    if args.compare_guo:
        save_path += "_withguo"

    save_path += ".pth"
    torch.save(
        {
            "grad_norm_approx": grad_norm_approx,
            "removal_times": removal_times,
            "acc_removal": acc_removal,  # 2 * num_removes * trails
            "grad_norm_worst": grad_norm_worst,
            "grad_norm_real": grad_norm_real,
            "removal_times_graph_retrain": removal_times_graph_retrain,
            "acc_graph_retrain": acc_graph_retrain,
            "grad_norm_approx_guo": grad_norm_approx_guo,
            "removal_times_guo": removal_times_guo,
            "acc_guo": acc_guo,
            "removal_times_guo_retrain": removal_times_guo_retrain,
            "acc_guo_retrain": acc_guo_retrain,
            "grad_norm_real_guo": grad_norm_real_guo,
        },
        save_path,
    )

    acc_removal_test_mean = acc_removal[-1, -1].mean().numpy()
    acc_removal_test_std = acc_removal[-1, -1].std().numpy()
    acc_test_mean = acc_train_test.mean().numpy()
    acc_test_std = acc_train_test.std().numpy()

    time_train_test_mean = time_train_test.mean().numpy()
    time_train_test_std = time_train_test.std().numpy()
    removal_time_mean = removal_times.sum(0).mean().numpy()
    removal_time_std = removal_times.sum(0).std().numpy()

    writer = [
        (acc_test_mean, acc_test_std),
        (time_train_test_mean, time_train_test_std),
        (acc_removal_test_mean, acc_removal_test_std),
        (removal_time_mean, removal_time_std),
    ]
    print(
        f"Training results: f1_avg: {acc_test_mean}, f1_std: {acc_test_std} time_avg: {time_train_test_mean}, time_std: {time_train_test_std} "
    )
    print(
        f"Unlearning results: f1_avg: {acc_removal_test_mean}, f1_std: {acc_removal_test_std} removal_time_avg: {removal_time_mean}, removal_time_std: {removal_time_std}"
    )

    return writer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training a removal-enabled linear model [node/feature]"
    )
    parser.add_argument(
        "--data_dir", type=str, default="./PyG_datasets", help="data directory"
    )
    parser.add_argument(
        "--result_dir", type=str, default="result", help="directory for saving results"
    )
    parser.add_argument("--dataset", type=str, default="cora", help="dataset")
    parser.add_argument("--lam", type=float, default=1e-2, help="L2 regularization")
    parser.add_argument(
        "--std",
        type=float,
        default=1e-2,
        help="standard deviation for objective perturbation",
    )
    parser.add_argument(
        "--num_removes", type=int, default=500, help="number of data points to remove"
    )
    parser.add_argument(
        "--num_steps", type=int, default=100, help="number of optimization steps"
    )
    parser.add_argument(
        "--train_mode", type=str, default="ovr", help="train mode [ovr/binary]"
    )
    parser.add_argument(
        "--train_sep",
        action="store_true",
        default=False,
        help="train binary classifiers separately",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False, help="verbosity in optimizer"
    )
    # New arguments below
    parser.add_argument(
        "--device", type=int, default=1, help="nonnegative int for cuda id, -1 for cpu"
    )
    parser.add_argument(
        "--prop_step",
        type=int,
        default=2,
        help="number of steps of graph propagation/convolution",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="we use D^{-a}AD^{-(1-a)} as propagation matrix",
    )
    parser.add_argument(
        "--XdegNorm",
        type=bool,
        default=False,
        help="Apply our degree normaliztion trick",
    )
    parser.add_argument(
        "--add_self_loops",
        type=bool,
        default=True,
        help="Add self loops in propagation matrix",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="LBFGS",
        help="Choice of optimizer. [LBFGS/Adam]",
    )
    parser.add_argument("--lr", type=float, default=1, help="Learning rate")
    parser.add_argument(
        "--wd", type=float, default=5e-4, help="Weight decay factor for Adam"
    )
    parser.add_argument(
        "--featNorm", type=bool, default=False, help="Row normalize feature to norm 1."
    )
    parser.add_argument(
        "--GPR", action="store_true", default=False, help="Use GPR model"
    )
    parser.add_argument(
        "--balance_train",
        action="store_true",
        default=False,
        help="Subsample training set to make it balance in class size.",
    )
    parser.add_argument(
        "--Y_binary",
        type=str,
        default="0",
        help="In binary mode, is Y_binary class or Y_binary_1 vs Y_binary_2 (i.e., 0+1).",
    )
    parser.add_argument(
        "--noise_mode",
        type=str,
        default="data",
        help="Data dependent noise or worst case noise [data/worst].",
    )
    parser.add_argument(
        "--removal_mode", type=str, default="node", help="[feature/edge/node]."
    )
    parser.add_argument(
        "--eps", type=float, default=1.0, help="Eps coefficient for certified removal."
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-4,
        help="Delta coefficient for certified removal.",
    )
    parser.add_argument("--disp", type=int, default=10, help="Display frequency.")
    parser.add_argument(
        "--trails", type=int, default=5, help="Number of repeated trails."
    )
    parser.add_argument(
        "--fix_random_seed",
        action="store_true",
        default=False,
        help="Use fixed random seed for removal queue.",
    )
    parser.add_argument(
        "--compare_gnorm",
        action="store_true",
        default=False,
        help="Compute norm of worst case and real gradient each round.",
    )
    parser.add_argument(
        "--compare_retrain",
        action="store_true",
        default=False,
        help="Compare acc with retraining each round.",
    )
    parser.add_argument(
        "--compare_guo",
        action="store_true",
        default=False,
        help="Compare performance with Guo et al.",
    )
    parser.add_argument(
        "--write",
        default=True,
        help="Write results to csv file.",
    )
    parser.add_argument(
        "--csv_file_name",
        type=str,
        default="result/unlearning_results",
        help="csv file name.",
    )
    parser.add_argument(
        "--ratio_removes",
        type=float,
        default=0.0,
        help="Ratio of nodes/edges to remove.",
    )

    parser.add_argument(
        "--is_ratio",
        action="store_true",
        default=False,
        help="Whether to use ratio of nodes/edges to remove.",
    )

    parser.add_argument(
        "--remove_feature_ratio",
        type=float,
        default=0.2,
        help="Ratio of features to remove.",
    )

    parser.add_argument(
        "--run_seed_feature", type=int, default=2, help="Seed for feature removal"
    )
    parser.add_argument(
        "--is_attack_node",
        action="store_true",
        default=False,
        help="Whether to attack node.",
    )
    parser.add_argument(
        "--is_attack_feauture",
        action="store_true",
        default=False,
        help="Whether to attack feature.",
    )

    # Use this if turning into .py code

    args = parser.parse_args()

    # Use this if running using notebook
    # args = parser.parse_args([])

    # this script is only for feature/node removal
    assert args.removal_mode in ["feature", "node"]
    # dont compute norm together with retrain
    assert not (args.compare_gnorm and args.compare_retrain)

    if args.device > -1:
        # device = torch.device("cuda:" + str(args.device))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    writer = main(args)

    def writer_to_csv(writing_list, name="unlearning_results"):
        import os
        import pandas as pd

        # 指定CSV文件的路径
        csv_file_path = args.csv_file_name + ".csv"

        # 检查CSV文件是否存在，如果不存在则创建
        if not os.path.exists(csv_file_path):
            # 创建一个空的DataFrame，并保存为CSV文件
            df = pd.DataFrame(
                columns=[
                    "dataset",
                    "model",
                    "unlearn_task",
                    "is_ratio",
                    "unlearn_ratio",
                    "f1_score_avg",
                    "f1_score_std",
                    "training_time_avg",
                    "training_time_std",
                    "f1_score_unlearn_avg",
                    "f1_score_unlearn_std",
                    "unlearning_time_avg",
                    "unlearning_time_std",
                ]
            )
            df.to_csv(csv_file_path, index=False)

        # # 读取CSV文件
        df = pd.read_csv(csv_file_path)

        # # 检查表头是否存在，如果不存在则输入表头
        # if df.empty:
        #     df.columns = ["dataset", "model", "unlearn_task", "unlearn_ratio", \
        #                                "f1_score_avg", "f1_score_std", "training_time_avg", \
        #                                 "training_time_std", "f1_score_unlearn_avg", "f1_score_unlearn_std", \
        #                                     "unlearning_time_avg", "unlearning_time_std", "my_bound_avg", \
        #                                         "my_bound_std", "certified_edge_bound_avg", "certified_edge_std", \
        #                                             "certified_edge_worst_bound_avg", "certified_edge_worst_bound_std", \
        #                                                 "actual_diff_avg", "actual_diff_std"]

        # 将输入的数据添加到DataFrame

        if args.is_ratio == True:
            args_unlearn_num = args.ratio_removes
            is_ratio = True
        else:
            args_unlearn_num = args.num_removes
            is_ratio = False

        new_row = {
            "dataset": args.dataset,
            "model": "SGC" if args.GPR == False else "GPR",
            "unlearn_task": args.removal_mode,
            "is_ratio": is_ratio,
            "unlearn_ratio": args_unlearn_num,
            "f1_score_avg": writing_list[0][0],
            "f1_score_std": writing_list[0][1],
            "training_time_avg": writing_list[1][0],
            "training_time_std": writing_list[1][1],
            "f1_score_unlearn_avg": writing_list[2][0],
            "f1_score_unlearn_std": writing_list[2][1],
            "unlearning_time_avg": writing_list[3][0],
            "unlearning_time_std": writing_list[3][1],
        }
        # df = df.append(new_row, ignore_index=True)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        # 保存更新后的DataFrame到CSV文件
        df.to_csv(csv_file_path, index=False)

    if args.write == True:
        writer_to_csv(writer)
