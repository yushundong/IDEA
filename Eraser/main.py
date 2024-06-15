import logging

import torch
import warnings

warnings.filterwarnings("ignore")
from silence_tensorflow import silence_tensorflow


silence_tensorflow()
import os

from exp.exp_graph_partition import ExpGraphPartition
from exp.exp_node_edge_unlearning import ExpNodeEdgeUnlearning
from exp.exp_edge_unlearning import ExpUnlearning

from exp.exp_attack_unlearning import ExpAttackUnlearning
from exp.exp_attack_unlearning_attack import ExpAttackUnlearning_
from parameter_parser import parameter_parser


def config_logger(save_name):
    # create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(levelname)s:%(asctime)s: - %(name)s - : %(message)s"
    )

    # create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def main(args, exp):
    # config the logger
    logger_name = "_".join(
        (
            exp,
            args["dataset_name"],
            args["partition_method"],
            str(args["num_shards"]),
            str(args["test_ratio"]),
        )
    )
    config_logger(logger_name)
    # logging.info(logger_name)

    torch.set_num_threads(args["num_threads"])
    torch.cuda.set_device(args["cuda"])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args["cuda"])

    # subroutine entry for different methods
    if exp == "partition":
        ExpGraphPartition(args)
    elif exp == "unlearning":
        ExpUnlearning(args)
    elif exp == "node_edge_unlearning":
        train_model = ExpNodeEdgeUnlearning(args)
        (
            train_f1_avg,
            train_f1_std,
            train_time_avg,
            train_time_std,
        ) = train_model.run_exp()
    elif exp == "attack_unlearning":
        ExpAttackUnlearning_(args)
    elif exp == "node":
        assert args["num_unlearned_nodes"] != 0 or args["ratio_unlearned_nodes"] != 0
        assert args["num_unlearned_edges"] == 0 and args["ratio_deleted_edges"] == 0
        ExpGraphPartition(args)
        train_model = ExpNodeEdgeUnlearning(args)
        # tmp_num_run = args["num_runs"]
        # args["num_runs"] = 1
        (
            train_f1_avg,
            train_f1_std,
            train_time_avg,
            train_time_std,
        ) = train_model.run_exp()
        # args["num_runs"] = tmp_num_run
        unlearn_model = ExpAttackUnlearning(args)
        (
            unlearn_f1_avg,
            unlearn_f1_std,
            unlearn_time_avg,
            unlearn_time_std,
        ) = unlearn_model.run_exp()

        # the socket
    elif exp == "edge":
        assert args["num_unlearned_nodes"] == 0 and args["ratio_unlearned_nodes"] == 0
        assert args["num_unlearned_edges"] != 0 or args["ratio_deleted_edges"] != 0
        # get a full train one
        args_num = args["num_unlearned_edges"]
        args_ratio = args["ratio_deleted_edges"]
        args["num_unlearned_edges"] = 0
        args["ratio_deleted_edges"] = 0
        # tmp_num_run = args["num_runs"]
        # args["num_runs"] = 1
        ExpGraphPartition(args)
        train_model = ExpNodeEdgeUnlearning(args)
        (
            train_f1_avg,
            train_f1_std,
            train_time_avg,
            train_time_std,
        ) = train_model.run_exp()

        args["num_unlearned_edges"] = args_num
        args["ratio_deleted_edges"] = args_ratio
        # args["num_runs"] = tmp_num_run
        ExpGraphPartition(args)
        unlearn_model = ExpNodeEdgeUnlearning(args)
        (
            unlearn_f1_avg,
            unlearn_f1_std,
            unlearn_time_avg,
            unlearn_time_std,
        ) = unlearn_model.run_exp()
    else:
        raise Exception("unsupported attack")

    # write to csv
    writer = [
        (train_f1_avg, train_f1_std),
        (train_time_avg, train_time_std),
        (unlearn_f1_avg, unlearn_f1_std),
        (unlearn_time_avg, unlearn_time_std),
    ]

    def writer_to_csv(writing_list, name="unlearning_results"):
        import os
        import pandas as pd

        # 指定CSV文件的路径
        csv_file_path = args["csv_file_name"] + ".csv"

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
                    "partition_method",
                    "aggregator",
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
        is_ratio = False
        if (
            args["num_unlearned_nodes"] != 0
            and args["ratio_unlearned_nodes"] == 0
            and args["num_unlearned_edges"] == 0
            and args["ratio_deleted_edges"] == 0
        ):
            args_unlearn_num = args["num_unlearned_nodes"]
            is_ratio = False
        elif (
            (args["num_unlearned_nodes"] == 0)
            and (args["ratio_unlearned_nodes"] != 0)
            and (args["num_unlearned_edges"] == 0)
            and (args["ratio_deleted_edges"] == 0)
        ):
            args_unlearn_num = args["ratio_unlearned_nodes"]
            is_ratio = True
        elif (
            (args["num_unlearned_nodes"] == 0)
            and (args["ratio_unlearned_nodes"] == 0)
            and (args["num_unlearned_edges"] != 0)
            and (args["ratio_deleted_edges"] == 0)
        ):
            args_unlearn_num = args["num_unlearned_edges"]
            is_ratio = False
        elif (
            (args["num_unlearned_nodes"] == 0)
            and (args["ratio_unlearned_nodes"] == 0)
            and (args["num_unlearned_edges"] == 0)
            and (args["ratio_deleted_edges"] != 0)
        ):
            args_unlearn_num = args["ratio_deleted_edges"]
            is_ratio = True

        new_row = {
            "dataset": args["dataset_name"],
            "model": args["target_model"],
            "unlearn_task": args["exp"],
            "is_ratio": is_ratio,
            "unlearn_ratio": args_unlearn_num,
            "partition_method": args["partition_method"],
            "aggregator": args["aggregator"],
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

    if args["write"] == True:
        writer_to_csv(writer)


if __name__ == "__main__":
    args = parameter_parser()

    main(args, args["exp"])
