## Usage

This folder is used to reproduce the code of CGU and support implementing both node attack (GraphMIA) and edge attack (StealLink). To choose different running configurations (dataset/unlearning ratio/etc), change the parameters in run_new_[node/edge/feature].sh. Then run the following command to train & unlearn & attack.

```
bash run_new_node.sh # for node unlearning
bash run_new_edge.sh # for edge unlearning
bash run_new_feature.sh # for feature unlearning
```
## Parameter explanation
```
--removal_mode node # unlearning mode, within node/edge
--is_attack (store_true) # edge attack
--is_attack_node (store_true) # node attack
--is_attack_feature (store_true) # feauture attack (require removed feature_dimension data from IDEA)
--is_ratio (store_true) # whether unlearn instances by ratio/number
```

## Examples

### Train & unlearn

Here we consider setting as 

`dataset : cora | unlearning_mode : edge`

Run
```
bash run_new_edge.sh
```
to train and unlearn the GCN model. With running once, we present the sample log as follows.
```
==========Loading data==========
Dataset: cora
********** 0 **********
==========Training on full dataset with graph==========
Train node:2438, Val node:270, Test node:270, Edges:10661, Feature dim:1433
With graph, train mode: ovr , optimizer: LBFGS
==========Testing our edge removal==========
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 27/27 [00:03<00:00,  8.58it/s]
Training results: f1_avg: 0.8523, f1_std: 0.0 time_avg: 0.9792, time_std: 0.0 
Unlearning results: f1_avg: 0.8545, f1_std: 0.0 removal_time_avg: 3.1361, removal_time_std: 0.0
```

The final unlearning result containing performance and unlearning time is also save into a CSV file as `result/unlearning_results.csv`.

dataset | model | unlearn_task | is_ratio | unlearn_ratio | f1_score_avg | f1_score_std | training_time_avg | training_time_std | f1_score_unlearn_avg | f1_score_unlearn_std | unlearning_time_avg | unlearning_time_std
:---: | :---: | :---: | :---: |:---: |:---: |:---: |:---: |:---: |:---: |:---: |:---: |:---: 
cora | SGC | edge | True | 0.01 | 0.82740337 | 0.0 | 0.5913925 | 0.47879428 | 0.823591 | 0.0 | 2.897691 | 0.2393388

### Attack

To attack the node unlearning, simply add the `--is_attack` in the .sh script, then rerun.
```
bash run_new_edge.sh
```
We present part of the sample log as follows
```
attack starts
2/2 [==============================] - 0s 882us/step
Test accuracy: 0.5 Test Precision 0.5 Test Recall 0.5185185185185185 Test auc: 0.5692729766803841
attack ends
```

The attack result for each run is also saved into a CSV file as `result/attack_results.csv`

Dataset | Exp | Unlearn Ratio | Attack Metrics | Epochs | Test Accuracy | Test Precision | Test Recall | Test AUC | Ratio
:---: |:---: |:---: |:---: |:---: |:---: |:---: |:---: |:---: | :---: 
cora | edge | 0.01 | attack3_metrics_concate_all | 50 | 0.5 | 0.5 | 0.5185185185185185 | 0.5692729766803841 | 0.5