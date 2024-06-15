import logging
import time
import os
import torch
from torch.autograd import grad
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pandas as pd

torch.cuda.empty_cache()
from torch_geometric.data import NeighborSampler
import numpy as np
from src.data_zoo import DataProcessing
from src.node_classifier import NodeClassifier


class Exp:
    def __init__(self, args):
        self.logger = logging.getLogger('exp')

        self.args = args
        self.data_zoo = DataProcessing(args)

    def load_data(self):
        pass


class IDEA(Exp):
    def __init__(self, args):
        super(IDEA, self).__init__(args)

        if self.args['target_model'] in ["GCN", "SGC"]:

            self.args['gaussian_std'] = 0.6
            if self.args['dataset_name'] in ["cora", "citeseer"]:
                self.args['iteration'] = 100
                self.args['scale'] = 500
            elif self.args['dataset_name'] in ["pubmed"]:
                self.args['iteration'] = 1e4  # 1000
                self.args['scale'] = 5e4
            elif self.args['dataset_name'] in ["CS", "Physics"]:
                self.args['gaussian_std'] = 0.1
                self.args['iteration'] = 1000
                self.args['scale'] = 5e6

        elif self.args['target_model'] in ["GIN"]:
            self.args['gaussian_std'] = 0.005
            if self.args['dataset_name'] in ["cora", "citeseer"]:
                self.args['iteration'] = 100
                self.args['scale'] = 1000000
            elif self.args['dataset_name'] in ["pubmed"]:
                self.args['gaussian_std'] = 0.001
                self.args['iteration'] = 1000
                self.args['scale'] = 50000000
            elif self.args['dataset_name'] in ["CS", "Physics"]:
                self.args['iteration'] = 1000
                self.args['scale'] = 100000000

        elif self.args['target_model'] in ["GAT"]:
            self.args['gaussian_std'] = 0.005
            if self.args['dataset_name'] in ["cora", "citeseer"]:
                self.args['iteration'] = 100
                self.args['scale'] = 500000
            elif self.args['dataset_name'] in ["pubmed"]:
                self.args['iteration'] = 1000
                self.args['scale'] = 500000
            elif self.args['dataset_name'] in ["CS", "Physics"]:
                self.args['iteration'] = 1000
                self.args['scale'] = 100000000

        if self.args['unlearn_task'] in ["partial_feature"]:
            self.args['gaussian_std'] = 0.6
            if self.args['dataset_name'] in ["cora", "citeseer"]:
                self.args['iteration'] = 100
                self.args['scale'] = 500

        self.attack_preparations = {}

        self.logger = logging.getLogger('IDEA Initiated.')
        self.deleted_nodes = np.array([])     
        self.feature_nodes = np.array([])
        self.influence_nodes = np.array([])

        self.certification_alpha1 = 0.0
        self.certification_alpha2 = 0.0
        self.samples_to_be_unlearned = 0.0

        self.originally_trained_model_params = None

        self.load_data()
        self.num_feats = self.data.num_features
        self.train_test_split()
        self.unlearning_request()

        self.target_model_name = self.args['target_model']

        self.determine_target_model()

        run_f1 = np.empty((0))
        run_f1_unlearning = np.empty((0))
        run_f1_retrain = np.empty((0))

        unlearning_times = np.empty((0))
        training_times = np.empty((0))
        my_bounds = np.empty((0))
        certified_edge_bounds = np.empty((0))
        certified_edge_worst_bounds = np.empty((0))
        actual_diffs = np.empty((0))

        for run in range(self.args['num_runs']):
            self.logger.info("Run %f" % run)

            run_training_time, result_tuple = self._train_model(run)
            self.originally_trained_model_params = [p.clone() for p in self.target_model.model.parameters() if p.requires_grad]
            f1_score = self.evaluate(run)
 
            run_f1 = np.append(run_f1, f1_score)
            training_times = np.append(training_times, run_training_time)

            # unlearn and compute certification stats
            unlearning_time, f1_score_unlearning, params_change = self.approxi(result_tuple)
            my_bound, certified_edge_bound, certified_edge_worst_bound, actual_diff, f1_score_retrain = self.alpha_computation(params_change, result_tuple)
            self.attack_preparations["unlearned_feature_pre"] = self.target_model.attack_preparations["unlearned_feature_pre"]
            self.attack_preparations["predicted_prob"] = self.target_model.attack_preparations["predicted_prob"]
                       

            unlearning_times = np.append(unlearning_times, unlearning_time)
            run_f1_unlearning = np.append(run_f1_unlearning, f1_score_unlearning)

            run_f1_retrain = np.append(run_f1_retrain, f1_score_retrain)

            my_bounds = np.append(my_bounds, my_bound)
            certified_edge_bounds = np.append(certified_edge_bounds, certified_edge_bound)
            certified_edge_worst_bounds = np.append(certified_edge_worst_bounds, certified_edge_worst_bound)
            actual_diffs = np.append(actual_diffs, actual_diff)

            self.attack_preparations["train_indices"] = self.data.train_mask
            self.attack_preparations["test_indices"] = self.data.test_mask  
            exp_marker = [self.args['dataset_name'], self.args['unlearn_task'], str(self.args['unlearn_ratio']), self.args['target_model'], str(self.args['unlearn_feature_partial_ratio'])]
            exp_marker_string = "_".join(exp_marker)

            current_dir = os.getcwd()
            attack_materials_dir = os.path.join(current_dir, 'attack_materials')
            if not os.path.exists(attack_materials_dir):
                os.makedirs(attack_materials_dir)

            torch.save(self.attack_preparations, 'attack_materials/seed_' + str(self.args['seeds']) + '_IDEA_' + exp_marker_string + '.pth')

        
        self.logger.info("Completed. \n")
        f1_score_avg = np.average(run_f1)
        f1_score_std = np.std(run_f1)

        training_time_avg = np.average(training_times)
        training_time_std = np.std(training_times)

        f1_score_unlearning_avg = np.average(run_f1_unlearning)
        f1_score_unlearning_std = np.std(run_f1_unlearning)

        f1_retrain_avg = np.average(run_f1_retrain)
        f1_retrain_std = np.std(run_f1_retrain)

        unlearning_time_avg = np.average(unlearning_times)
        unlearning_time_std = np.std(unlearning_times)

        my_bounds_avg = np.average(my_bounds)
        my_bounds_std = np.std(my_bounds)

        certified_edge_bounds_avg = np.average(certified_edge_bounds)
        certified_edge_bounds_std = np.std(certified_edge_bounds)

        certified_edge_worst_bounds_avg = np.average(certified_edge_worst_bounds)
        certified_edge_worst_bounds_std = np.std(certified_edge_worst_bounds)

        actual_diffs_avg = np.average(actual_diffs)
        actual_diffs_std = np.std(actual_diffs)

        self.logger.info("f1_score: avg=%s, std=%s" % (f1_score_avg, f1_score_std))
        self.logger.info("model training time: avg=%s, std=%s seconds" % (training_time_avg, training_time_std))

        self.logger.info("retrain f1_score: avg=%s, std=%s" % (f1_retrain_avg, f1_retrain_std))

        self.logger.info("f1_score of GIF: avg=%s, std=%s" % (f1_score_unlearning_avg, f1_score_unlearning_std))
        self.logger.info("GIF unlearing time: avg=%s, std=%s " % (unlearning_time_avg, unlearning_time_std))


        self.logger.info("my_bound: avg=%s, std=%s " % (my_bounds_avg, my_bounds_std))
        self.logger.info("certified_edge_bound: avg=%s, std=%s " % (certified_edge_bounds_avg, certified_edge_bounds_std))
        self.logger.info("certified_edge_worst_bounds: avg=%s, std=%s " % (certified_edge_worst_bounds_avg, certified_edge_worst_bounds_std))
        self.logger.info("actual_diffs: avg=%s, std=%s " % (actual_diffs_avg, actual_diffs_std))

        writer = [(f1_score_avg, f1_score_std), (training_time_avg, training_time_std), (f1_score_unlearning_avg, f1_score_unlearning_std), (unlearning_time_avg, unlearning_time_std), (my_bounds_avg, my_bounds_std), (certified_edge_bounds_avg, certified_edge_bounds_std), (certified_edge_worst_bounds_avg, certified_edge_worst_bounds_std), (actual_diffs_avg, actual_diffs_std), (f1_retrain_avg, f1_retrain_std)]

        if self.args['write'] == True:
            self.writer_to_csv(writer)

    def load_data(self):
        self.data = self.data_zoo.load_raw_data()

    def train_test_split(self):
        if self.args['is_split']:
            self.logger.info('splitting train/test data')
            self.train_indices, self.test_indices = train_test_split(np.arange((self.data.num_nodes)), test_size=self.args['test_ratio'], random_state=100)
            self.data_zoo.save_train_test_split(self.train_indices, self.test_indices)

            self.data.train_mask = torch.from_numpy(np.isin(np.arange(self.data.num_nodes), self.train_indices))
            self.data.test_mask = torch.from_numpy(np.isin(np.arange(self.data.num_nodes), self.test_indices))
        else:
            self.train_indices, self.test_indices = self.data_zoo.load_train_test_split()

            self.data.train_mask = torch.from_numpy(np.isin(np.arange(self.data.num_nodes), self.train_indices))
            self.data.test_mask = torch.from_numpy(np.isin(np.arange(self.data.num_nodes), self.test_indices))

    def unlearning_request(self):
        self.logger.debug("Train data  #.Nodes: %f, #.Edges: %f" % (
            self.data.num_nodes, self.data.num_edges))

        self.data.x_unlearn = self.data.x.clone()
        self.data.edge_index_unlearn = self.data.edge_index.clone()
        edge_index = self.data.edge_index.numpy()
        unique_indices = np.where(edge_index[0] < edge_index[1])[0]

        if self.args["unlearn_task"] == 'node':
            unique_nodes = np.random.choice(len(self.train_indices),
                                            int(len(self.train_indices) * self.args['unlearn_ratio']),
                                            replace=False)
            self.data.edge_index_unlearn = self.update_edge_index_unlearn(unique_nodes)
            self.samples_to_be_unlearned = float(int(len(self.train_indices) * self.args['unlearn_ratio']))

            self.attack_preparations["removed_nodes"] = unique_nodes

        if self.args["unlearn_task"] == 'edge':

            # setting 1
            remove_indices = np.random.choice(
                unique_indices,
                int(unique_indices.shape[0] * self.args['unlearn_ratio']),
                replace=False)
            remove_edges = edge_index[:, remove_indices]
            unique_nodes = np.unique(remove_edges)

            self.attack_preparations["removed_edges"] = remove_edges
        
            self.data.edge_index_unlearn = self.update_edge_index_unlearn(unique_nodes, remove_indices)
            self.samples_to_be_unlearned = 0.0


            # # setting 2
            # # sample non-existed edges
            # remove_edges = negative_sampling(self.data.edge_index, num_neg_samples=int(unique_indices.shape[0] * self.args['unlearn_ratio']))

            # # change the original graph topology to add these edges
            # self.data.edge_index = torch.cat((self.data.edge_index, remove_edges), 1)
            # self.data.edge_index_unlearn = self.data.edge_index_unlearn  # no change
            # unique_nodes = np.unique(remove_edges.numpy())

            # self.attack_preparations["removed_edges"] = remove_edges
            # self.samples_to_be_unlearned = 0.0

        if self.args["unlearn_task"] == 'feature':
            unique_nodes = np.random.choice(len(self.train_indices),
                                            int(len(self.train_indices) * self.args['unlearn_ratio']),
                                            replace=False)
            self.data.x_unlearn[unique_nodes] = 0.
            self.samples_to_be_unlearned = 0.0
            self.attack_preparations["unlearned_feature_node_idx"] = unique_nodes

        if self.args["unlearn_task"] == 'partial_feature':
            unique_nodes = np.random.choice(len(self.train_indices),
                                            int(len(self.train_indices) * self.args['unlearn_ratio']),
                                            replace=False)

            unlearn_feature_dims = np.random.randint(0, self.data.x.shape[1], size=int(self.data.x.shape[1] * self.args['unlearn_feature_partial_ratio']))
            self.data.x_unlearn[unique_nodes[:, None], unlearn_feature_dims] = 0.0
            self.samples_to_be_unlearned = 0.0

            self.attack_preparations["unlearned_feature_node_idx"] = unique_nodes
            self.attack_preparations["unlearned_feature_dim_idx"] = unlearn_feature_dims

        self.find_k_hops(unique_nodes)

    def update_edge_index_unlearn(self, delete_nodes, delete_edge_index=None):
        edge_index = self.data.edge_index.numpy()

        unique_indices = np.where(edge_index[0] < edge_index[1])[0]
        unique_indices_not = np.where(edge_index[0] > edge_index[1])[0]

        if self.args["unlearn_task"] == 'edge':
            remain_indices = np.setdiff1d(unique_indices, delete_edge_index)
        else:
            unique_edge_index = edge_index[:, unique_indices]
            delete_edge_indices = np.logical_or(np.isin(unique_edge_index[0], delete_nodes),
                                                np.isin(unique_edge_index[1], delete_nodes))
            remain_indices = np.logical_not(delete_edge_indices)
            remain_indices = np.where(remain_indices == True)

        remain_encode = edge_index[0, remain_indices] * edge_index.shape[1] * 2 + edge_index[1, remain_indices]
        unique_encode_not = edge_index[1, unique_indices_not] * edge_index.shape[1] * 2 + edge_index[0, unique_indices_not]
        sort_indices = np.argsort(unique_encode_not)
        remain_indices_not = unique_indices_not[sort_indices[np.searchsorted(unique_encode_not, remain_encode, sorter=sort_indices)]]
        remain_indices = np.union1d(remain_indices, remain_indices_not)

        return torch.from_numpy(edge_index[:, remain_indices])

    def determine_target_model(self):
        self.logger.info('target model: %s' % (self.args['target_model'],))
        num_classes = len(self.data.y.unique())

        self.target_model = NodeClassifier(self.num_feats, num_classes, self.args)

    def evaluate(self, run):
        self.logger.info('model evaluation')

        start_time = time.time()
        posterior = self.target_model.posterior()
        test_f1 = f1_score(
            self.data.y[self.data['test_mask']].cpu().numpy(), 
            posterior.argmax(axis=1).cpu().numpy(), 
            average="micro"
        )

        evaluate_time = time.time() - start_time
        self.logger.info("Evaluation cost %s seconds." % evaluate_time)

        self.logger.info("Final Test F1: %s" % (test_f1,))
        return test_f1

    def _train_model(self, run):
        self.logger.info('training target models, run %s' % run)

        start_time = time.time()
        self.target_model.data = self.data
        res = self.target_model.train_model(
            (self.deleted_nodes, self.feature_nodes, self.influence_nodes))
        train_time = time.time() - start_time
        self.logger.info("Model training time: %s" % (train_time))

        return train_time, res
        
    def find_k_hops(self, unique_nodes):
        edge_index = self.data.edge_index.numpy()

        hops = 2
        if self.args["unlearn_task"] == 'node':
            hops = 3
        influenced_nodes = unique_nodes
        for _ in range(hops):
            target_nodes_location = np.isin(edge_index[0], influenced_nodes)
            neighbor_nodes = edge_index[1, target_nodes_location]
            influenced_nodes = np.append(influenced_nodes, neighbor_nodes)
            influenced_nodes = np.unique(influenced_nodes)
        neighbor_nodes = np.setdiff1d(influenced_nodes, unique_nodes)
        if self.args["unlearn_task"] == 'feature' or 'partial_feature':
            self.feature_nodes = unique_nodes
            self.influence_nodes = neighbor_nodes
        if self.args["unlearn_task"] == 'node':
            self.deleted_nodes = unique_nodes
            self.influence_nodes = neighbor_nodes
        if self.args["unlearn_task"] == 'edge':
            self.influence_nodes = influenced_nodes

    def approxi(self, res_tuple):
        '''
        res_tuple == (grad_all, grad1, grad2)
        '''

        start_time = time.time()
        iteration, damp, scale = self.args['iteration'], self.args['damp'], self.args['scale']

        v = tuple(grad1 - grad2 for grad1, grad2 in zip(res_tuple[1], res_tuple[2]))
        h_estimate = tuple(grad1 - grad2 for grad1, grad2 in zip(res_tuple[1], res_tuple[2]))
        for _ in range(iteration):

            model_params  = [p for p in self.target_model.model.parameters() if p.requires_grad]
            hv            = self.hvps(res_tuple[0], model_params, h_estimate)
            with torch.no_grad():
                h_estimate    = [ v1 + (1-damp)*h_estimate1 - hv1/scale
                            for v1, h_estimate1, hv1 in zip(v, h_estimate, hv)]

        params_change = [h_est / scale for h_est in h_estimate]
        params_esti   = [p1 + p2 for p1, p2 in zip(params_change, model_params)]

        self.params_esti = params_esti

        end_time = time.time()

        # add Gaussian Noise
        gaussian_noise = [(torch.randn(item.size()) * self.args['gaussian_std'] + self.args['gaussian_mean']).to(item.device) for item in params_esti]
        params_esti = [item1 + item2 for item1, item2 in zip(gaussian_noise, params_esti)]

        test_F1 = self.target_model.evaluate_unlearn_F1(params_esti)

        return end_time - start_time, test_F1, params_change

    def alpha_computation(self, params_change, result_tuple):

        # bound given by alpha 1 + alpha 2
        m = self.samples_to_be_unlearned
        t = self.influence_nodes.shape[0]

        self.certification_alpha1 = (m * self.args['l'] + (m ** 2 + self.args['l'] ** 2 + 4 * self.args['lambda'] * len(self.train_indices) * t * self.args['c']) ** 0.5) / (self.args['lambda'] * len(self.train_indices))
        params_change_flatten = [item.flatten() for item in params_change]
        self.certification_alpha2 = torch.norm(torch.cat(params_change_flatten), 2)
        self.logger.info("Certification related stats:  ")
        self.logger.info("certification_alpha1 (bound): %s" % self.certification_alpha1)
        self.logger.info("certification_alpha2 (l2 of modification): %s" % self.certification_alpha2)
        total_bound = self.certification_alpha1 + self.certification_alpha2
        self.logger.info("total bound given by alpha1 + alpha2: %s" % total_bound)

        # bound given by certified edge
        certified_edge_bound = self.certification_alpha2 ** 2 * self.args['M'] / self.args['l']
        self.logger.info("data-dependent bound given by certified edge: %s" % certified_edge_bound)

        # worset bound given by certified edge  
        certified_edge_worst_bound = self.args['M'] * (self.args['gamma_2'] ** 2) * (self.args['c1'] ** 2) * (t ** 2) / ((self.args['lambda_edge_unlearn'] ** 4) * len(self.train_indices))
        self.logger.info("worst bound given by certified edge: %s" % certified_edge_worst_bound)

        # recover the originally trained model
        idx = 0
        for p in self.target_model.model.parameters():
            p.data = self.originally_trained_model_params[idx].clone()
            idx = idx + 1

        # continue optimizing the model with data already updated
        self.target_model.train_model_continue((self.deleted_nodes, self.feature_nodes, self.influence_nodes))

        # actual difference
        original_params = [p.flatten() for p in self.params_esti]
        retraining_model_params = [p.flatten() for p in self.target_model.model.parameters() if p.requires_grad]
        actual_param_difference = torch.norm((torch.cat(original_params) - torch.cat(retraining_model_params)), 2).detach()
        self.logger.info("actual params difference: %s" % actual_param_difference)

        test_F1_retrain = self.target_model.evaluate_unlearn_F1([p for p in self.target_model.model.parameters() if p.requires_grad])
        self.logger.info("retrain f1 score: %s" % test_F1_retrain)

        return total_bound.cpu().numpy(), certified_edge_bound.cpu().numpy(), certified_edge_worst_bound, actual_param_difference.cpu().numpy(), test_F1_retrain

    # compute actual Hessian matrix
    def eval_hessian(self, loss_grad, model):
        cnt = 0
        for g in loss_grad:
            g_vector = g.contiguous().view(-1) if cnt == 0 else torch.cat([g_vector, g.contiguous().view(-1)])
            cnt = 1
        l = g_vector.size(0)
        hessian = torch.zeros(l, l).to(loss_grad[0].device)
        for idx in range(l):
            grad2rd = torch.autograd.grad(g_vector[idx], model.parameters(), create_graph=True)
            cnt = 0
            for g in grad2rd:
                g2 = g.contiguous().view(-1) if cnt == 0 else torch.cat([g2, g.contiguous().view(-1)])
                cnt = 1
            hessian[idx] = g2
        return hessian

    def hvps(self, grad_all, model_params, h_estimate):
        element_product = 0
        for grad_elem, v_elem in zip(grad_all, h_estimate):
            element_product += torch.sum(grad_elem * v_elem)
        
        return_grads = grad(element_product,model_params,create_graph=True)
        return return_grads
    
    def writer_to_csv(self, writing_list, name="unlearning_results"):

        csv_file_path = self.args['file_name'] + ".csv"
        if not os.path.exists(csv_file_path):
            df = pd.DataFrame(columns=["dataset", "model", "unlearn_task", "unlearn_ratio", \
                                       "f1_score_avg", "f1_score_std", "training_time_avg", \
                                        "training_time_std", "f1_score_unlearn_avg", "f1_score_unlearn_std", \
                                            "unlearning_time_avg", "unlearning_time_std", "my_bound_avg", \
                                                "my_bound_std", "certified_edge_bound_avg", "certified_edge_std", \
                                                    "certified_edge_worst_bound_avg", "certified_edge_worst_bound_std", \
                                                        "actual_diff_avg", "actual_diff_std", "f1_retrain_avg", "f1_retrain_std"])
            df.to_csv(csv_file_path, index=False)

        df = pd.read_csv(csv_file_path)
        new_row = {"dataset": self.args['dataset_name'], "model": self.args['target_model'], "unlearn_task": self.args['unlearn_task'], "unlearn_ratio": self.args['unlearn_ratio'], \
                                       "f1_score_avg": writing_list[0][0], "f1_score_std": writing_list[0][1], "training_time_avg": writing_list[1][0], \
                                        "training_time_std": writing_list[1][1], "f1_score_unlearn_avg": writing_list[2][0], "f1_score_unlearn_std": writing_list[2][1], \
                                            "unlearning_time_avg": writing_list[3][0], "unlearning_time_std": writing_list[3][1], "my_bound_avg": writing_list[4][0], \
                                                "my_bound_std": writing_list[4][1], "certified_edge_bound_avg": writing_list[5][0], "certified_edge_std": writing_list[5][1], \
                                                    "certified_edge_worst_bound_avg": writing_list[6][0], "certified_edge_worst_bound_std": writing_list[6][1], \
                                                        "actual_diff_avg": writing_list[7][0], "actual_diff_std": writing_list[7][1], "f1_retrain_avg": writing_list[8][0], "f1_retrain_std": writing_list[8][1]}
        df = df.append(new_row, ignore_index=True)
        df.to_csv(csv_file_path, index=False)
