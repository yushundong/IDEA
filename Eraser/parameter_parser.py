import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parameter_parser():
    """
    A method to parse up command line parameters.
    The default hyper-parameters give a good quality representation without grid search.
    """
    parser = argparse.ArgumentParser()

    ######################### general parameters ################################
    parser.add_argument(
        "--is_vary",
        type=bool,
        default=False,
        help="control whether to use multiprocess",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="citeseer",
        choices=["cora", "citeseer", "pubmed", "Coauthor_CS", "Coauthor_Phys"],
    )

    parser.add_argument(
        "--exp",
        type=str,
        default="attack_unlearning",
        choices=[
            "partition",
            "unlearning",
            "node_edge_unlearning",
            "attack_unlearning",
            "node",
            "edge",
        ],
    )
    parser.add_argument("--cuda", type=int, default=0, help="specify gpu")
    parser.add_argument("--num_threads", type=int, default=1)

    parser.add_argument("--is_upload", type=str2bool, default=True)
    parser.add_argument(
        "--database_name",
        type=str,
        default="unlearning_dependant",
        choices=[
            "unlearning_dependant",
            "unlearning_adaptive",
            "unlearning_graph_structure",
            "gnn_unlearning_shards",
            "unlearning_delta_plot",
            "gnn_unlearning_utility",
            "unlearning_ratio",
            "unlearning_partition_baseline",
            "unlearning_ratio",
            "attack_unlearning",
        ],
    )

    ########################## graph partition parameters ######################
    parser.add_argument("--is_split", type=str2bool, default=True)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--use_test_neighbors", type=str2bool, default=True)
    parser.add_argument("--is_partition", type=str2bool, default=True)
    parser.add_argument("--is_prune", type=str2bool, default=False)
    parser.add_argument("--num_shards", type=int, default=20)
    parser.add_argument("--is_constrained", type=str2bool, default=True)
    parser.add_argument("--is_gen_embedding", type=str2bool, default=True)
    parser.add_argument("--write", type=str2bool, default=True)

    parser.add_argument(
        "--partition_method",
        type=str,
        default="sage_km",
        choices=["sage_km", "random", "lpa", "metis", "lpa_base", "sage_km_base"],
    )
    parser.add_argument("--terminate_delta", type=int, default=0)
    parser.add_argument("--shard_size_delta", type=float, default=0.005)

    ########################## unlearning parameters ###########################
    parser.add_argument("--repartition", type=str2bool, default=False)

    ########################## training parameters ###########################
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--is_train_target_model", type=str2bool, default=True)
    parser.add_argument("--is_use_node_feature", type=str2bool, default=True)
    parser.add_argument(
        "--is_use_batch",
        type=str2bool,
        default=False,
        help="Use batch train GNN models.",
    )
    parser.add_argument(
        "--target_model",
        type=str,
        default="GAT",
        choices=["SAGE", "GAT", "MLP", "GCN", "GIN", "SGC"],
    )
    parser.add_argument("--train_lr", type=float, default=0.01)
    parser.add_argument("--train_weight_decay", type=float, default=0.001)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--num_runs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--test_batch_size", type=int, default=64)
    parser.add_argument(
        "--aggregator",
        type=str,
        default="mean",
        choices=["mean", "majority", "optimal"],
    )

    parser.add_argument("--opt_lr", type=float, default=0.001)
    parser.add_argument("--opt_decay", type=float, default=0.0001)
    parser.add_argument("--opt_num_epochs", type=int, default=10)
    parser.add_argument(
        "--unlearning_request",
        type=str,
        default="random",
        choices=["random", "adaptive", "dependant", "top1", "last5"],
    )

    ########################## analysis parameters ###################################
    parser.add_argument("--num_unlearned_nodes", type=int, default=0)
    parser.add_argument("--ratio_unlearned_nodes", type=float, default=0)
    parser.add_argument("--num_unlearned_edges", type=int, default=0)
    parser.add_argument("--ratio_deleted_edges", type=float, default=0)
    parser.add_argument("--num_opt_samples", type=int, default=1000)

    parser.add_argument(
        "--file_name",
        type=str,
        default="../Graph-Unlearning/temp_data/unlearning_result.txt",
    )
    parser.add_argument(
        "--csv_file_name",
        type=str,
        default="../Graph-Unlearning/temp_data/unlearning_result",
    )
    parser.add_argument("--is_ratio", type=str2bool, default=False, required=True)
    parser.add_argument("--num_unlearned", type=int, default=0, help="unlearn_num")
    parser.add_argument("--ratio_unlearned", type=float, default=0)

    ########################## attack parameters ###################################

    parser.add_argument("--edge_unlearning", type=str2bool, default=False)
    parser.add_argument("--is_attack", type=str2bool, default=False)
    parser.add_argument("--is_attack_node", type=str2bool, default=False)
    parser.add_argument(
        "--partial_graph_path", type=str, default="data/partial_graph_with_id/"
    )
    parser.add_argument(
        "--attack_partial_graph_ratio",
        type=float,
        default=0.5,
        help="ratio of partial graph",
    )

    parser.add_argument(
        "--attack_operator",
        type=str,
        default="concate_all",
        choices=["average", "hadamard", "weighted_l1", "weighted_l2", "concate_all"],
        help="number of partial graph",
    )

    parser.add_argument(
        "--attack_metric_type",
        type=str,
        default="kl_divergence",
        choices=["kl_divergence", "js_divergence", "entropy"],
    )

    parser.add_argument("--is_feature_removed", type=str2bool, default=False)
    parser.add_argument("--run_seed_feature", type=int, default=20230202)
    parser.add_argument("--remove_feature_ratio", type=float, default=0.2)
    parser.add_argument("--is_attack_feature", type=str2bool, default=False)

    ########################## final process ###################################

    args = vars(parser.parse_args())
    if args["dataset_name"] == "cora":
        args["num_shards"] = 10
    elif args["dataset_name"] == "citeseer":
        args["num_shards"] = 10
    elif args["dataset_name"] == "pubmed":
        args["num_shards"] = 30
    elif args["dataset_name"] == "Coauthor_CS":
        args["num_shards"] = 30
    elif args["dataset_name"] == "Coauthor_Phys":
        args["num_shards"] = 100

    if args["num_unlearned"] != 0:
        args["num_unlearned_nodes"] = args["num_unlearned"]
        args["num_unlearned_edges"] = args["num_unlearned"]
    elif args["ratio_unlearned"] != 0:
        args["ratio_unlearned_nodes"] = args["ratio_unlearned"]
        args["ratio_deleted_edges"] = args["ratio_unlearned"]

    if args["is_ratio"]:
        args["num_unlearned_nodes"] = 0
        args["num_unlearned_edges"] = 0
    else:
        args["ratio_unlearned_nodes"] = 0
        args["ratio_deleted_edges"] = 0

    if args["exp"] == "node":
        args["num_unlearned_edges"] = 0
        args["ratio_deleted_edges"] = 0
    elif args["exp"] == "edge":
        args["num_unlearned_nodes"] = 0
        args["ratio_unlearned_nodes"] = 0

    return args
