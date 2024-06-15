import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parameter_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=20221012, help='control whether to use multiprocess')

    ######################### general parameters ################################
    parser.add_argument('--is_vary', type=bool, default=False, help='control whether to use multiprocess')
    parser.add_argument('--cuda', type=int, default=0, help='specify gpu')
    parser.add_argument('--num_threads', type=int, default=1)
    
    ########################## unlearning task parameters ######################
    parser.add_argument('--dataset_name', type=str, default='citeseer',
                        choices=["cora", "citeseer", "pubmed", "CS", "Physics"])
    parser.add_argument('--unlearn_task', type=str, default='edge', choices=["edge", "node", 'feature', 'partial_feature'])
    parser.add_argument('--unlearn_ratio', type=float, default=0.1)

    ########################## training parameters ###########################
    parser.add_argument('--is_split', type=str2bool, default=True, help='splitting train/test data')
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--use_test_neighbors', type=str2bool, default=True)
    parser.add_argument('--is_train_target_model', type=str2bool, default=True)
    parser.add_argument('--is_retrain', type=str2bool, default=True)
    parser.add_argument('--is_use_node_feature', type=str2bool, default=False)
    parser.add_argument('--is_use_batch', type=str2bool, default=True, help="Use batch train GNN models.")
    parser.add_argument('--target_model', type=str, default='GAT', choices=["GAT", 'MLP', "GCN", "GIN","SGC"])
    parser.add_argument('--train_lr', type=float, default=0.01)
    parser.add_argument('--train_weight_decay', type=float, default=0)
    parser.add_argument('--num_epochs', type=int, default=3000)  # 3000
    parser.add_argument('--num_runs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--test_batch_size', type=int, default=64)

    ########################## GIF parameters ###########################
    parser.add_argument('--iteration', type=int, default=5)
    parser.add_argument('--scale', type=int, default=50)
    parser.add_argument('--damp', type=float, default=0.0)


    ########################## unlearning certification parameters ###########################
    parser.add_argument('--unlearn_feature_partial_ratio', type=float, default=0.5)
    parser.add_argument('--gaussian_mean', type=float, default=0.0)
    parser.add_argument('--gaussian_std', type=float, default=0.0)


    ########################## unlearning certification bound setting ###########################
    parser.add_argument('--l', type=float, default=0.25, help="lipschitz constant of the loss.")
    parser.add_argument('--lambda', type=float, default=1.0, help="(original) loss function is lambda-strongly convex.")  # 0.05 1.0 
    parser.add_argument('--c', type=float, default=0.5, help="numerical bound of the training loss regarding each sample.")  # 3.0 0.5

    ########################## unlearning baselines certification bound setting ###########################
    parser.add_argument('--M', type=float, default=0.25, help="the loss is M - Lipschitz Hessian in terms of w, i.e., gamma_1 in certified edge unlearning.")
    parser.add_argument('--c1', type=float, default=1.0, help="value of the derivative of loss is c1 bounded.")
    parser.add_argument('--lambda_edge_unlearn', type=float, default=1.0, help="regularization term weight - edge unlearning.")
    parser.add_argument('--gamma_2', type=float, default=1.0, help="lipschitz constant of first-order derivative of the loss - edge unlearning.")
    parser.add_argument('--file_name', type=str, default="unlearning_results", help="file name for results.")
    parser.add_argument('--write', type=bool, default=True, help="write to keep results.")
    
    args = vars(parser.parse_args())

    return args
