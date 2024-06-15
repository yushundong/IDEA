## Usage

This folder is used to reproduce the code of GraphEraser and support implementing both node attack (GraphMIA) and edge attack (StealLink). To choose different running configurations (dataset/target_model/partition_method/etc), change the parameters in run_new_[node/edge/feature].sh. Then run the following command to train & unlearn & attack.

```
bash run_new_node.sh # for node unlearning
bash run_new_edge.sh # for edge unlearning
```
## Parameter explanation
```
--exp node # unlearning mode, within node/edge
--is_attack True # edge attack
--is_attack_node True # node attack
--is_ratio True # whether unlearn nodes by ratio/number
```

## Examples

### Train & unlearn

Here we consider setting as 

`dataset : cora | target_model : GCN | partition_method : random | aggregator : optimal | unlearning_mode : node`

Run
```
bash run_new_node.sh
```
to train and unlearn the GCN model. With running once, we present the sample log as follows.
```
INFO:2024-02-07 13:45:19,573: - lib_graph_partition.graph_partition - : graph partition, method: random
INFO:2024-02-07 13:45:19,574: - exp_graph_partition - : generating shard data
INFO:2024-02-07 13:45:19,583: - data_store - : saving shard data
INFO:2024-02-07 13:45:19,654: - exp_node_edge_unlearning - : training target models
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:04<00:00,  2.09it/s]
INFO:2024-02-07 13:45:24,430: - exp_node_edge_unlearning - : aggregating submodels
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 18.72it/s]
INFO:2024-02-07 13:45:25,021: - exp_node_edge_unlearning - : f1_avg: 0.6051660516605166 f1_std: 0.0 time_avg: 5.3660688400268555 time_std: 0.0
INFO:2024-02-07 13:45:25,121: - exp_attack_unlearning - : retraining the unlearned model
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:02<00:00,  3.62it/s]
INFO:2024-02-07 13:45:27,883: - exp_attack_unlearning - : aggregating submodels
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 21.56it/s]
INFO:2024-02-07 13:45:28,410: - exp_attack_unlearning - : f1_avg: 0.5424354243542435 f1_std: 0.0 time_avg: 3.2886276245117188 time_std: 0.0
```

The final unlearning result containing performance and unlearning time is also save into a CSV file as `temp_data/new_setting.csv`.

dataset | model | unlearn_task | is_ratio | unlearn_ratio | partition_method | aggregator | f1_score_avg | f1_score_std | training_time_avg | training_time_std | f1_score_unlearn_avg | f1_score_unlearn_std | unlearning_time_avg | unlearning_time_std 
:---: | :---: | :---: | :---: |:---: |:---: |:---: |:---: |:---: |:---: |:---: |:---: |:---: |:---: |:---:
cora | GCN | node | True | 0.05 | random | optimal | 0.4895 | 0.0191 | 12.0968 | 0.6121 | 0.4649 | 0.0197 | 11.3277 | 0.2202

### Attack

To attack the node unlearning, simply add the `--is_attack_node True` in the .sh script, then rerun.
```
bash run_new_node.sh
```
We present part of the sample log as follows
```
INFO:2024-02-07 13:51:54,139: - exp_attack_unlearning - : attack
shadow training
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 300/300 [00:13<00:00, 22.00it/s]
shadow training done
attack training
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 300/300 [00:18<00:00, 15.94it/s]
attack training done
Test accuracy: 0.503 AUROC: 0.500 precision: 0.499 recall 0.503 F1 score 0.472 ===> Attack Performance!
```

The attack result is also saved into a CSV file as `result/node_attack_results.csv`

Dataset | Partition Method | Aggregator | Exp | Unlearn Ratio | AUC | AUC Std | ACC | ACC Std
:---: |:---: |:---: |:---: |:---: |:---: |:---: |:---: |:---: 
cora | random | optimal | node | 0.05 | 0.4997| 0.0 | 0.4986 | 0.0