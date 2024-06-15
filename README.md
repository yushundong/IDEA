# IDEA
This the open-source code for KDD' 24 IDEA: A Flexible Framework of Certified Unlearning for Graph Neural Networks.



## Supplementary Results for Table 1

Here we present the performance of vanilla GNNs trained on different datasets as the supplementary results for Table 1 as a comparison. We can observe that they are either slightly better or comparable to the reported results derived from *Re-Training* (in Table 1 of our manuscript). This is consistent with other existing works, and it further reveals that IDEA only sacrifices limited model utility.

| Model | Cora   | CiteSeer    | PubMed     | CS     | Physics    |
|-------|--------------|---------------|--------------|--------------|--------------|
| GCN   | 78.23 $\pm$ 0.9  | 68.27 $\pm$ 4.0   | 76.98 $\pm$ 1.8  | 87.66 $\pm$ 1.1  | 94.81 $\pm$ 1.0  |
| SGC   | 80.07 $\pm$ 1.8  | 63.26 $\pm$ 10.4  | 76.81 $\pm$ 1.9  | 88.57 $\pm$ 1.2  | 93.43 $\pm$ 2.5  |
| GIN   | 83.52 $\pm$ 1.9  | 72.67 $\pm$ 0.9   | 86.90 $\pm$ 1.3  | 90.20 $\pm$ 0.2  | 94.73 $\pm$ 0.7  |
| GAT   | 84.50 $\pm$ 0.5  | 71.27 $\pm$ 1.4   | 84.40 $\pm$ 0.2  | 91.73 $\pm$ 0.2  | 95.79 $\pm$ 0.2  |


## Supplementary Results for Table 2

Here we present the performance of attacking vanilla SGC trained on Cora dataset as the supplementary results for Table 2 as a comparison. We can observe that they are larger in values than the reported results. This observation reveals the effectiveness of the adopted unlearning approaches, which also further solidates the unlearning effectiveness of IDEA.

| Model | Node Unlearning | Edge Unlearning |
|-------|-------|-------|
| SGC| 55.69 $\pm$ 1.1 | 71.10 $\pm$ 2.1|




## Usage of IDEA
The folder `IDEA` includes implementations of IDEA (folder `src`) and scripts supporting the reported experiments.

Run the following command to reproduce the reported experiments.

```
./1_bounds.sh # Reproducing Figure 3
./2_utilities.sh # Reproducing Table 1
./3_running_time.sh # Reproducing Figure 4
./4_node_edge_unlearning_for_attack.sh # Reproducing preparations for Table 2 and all other results in Appendix
./5_partial_attribute_unlearning.sh # Reproducing preparations for Table 3 and all other results in Appendix
```

### Examples
Here we take the unlearning $\ell_2$ bounds of IDEA as an example.

Run

```
./1_bounds.sh
```

Exemplary log
```
Selected GPU: 0  
Start Dataset = cora task unlearn = edge model = GCN unlearn ratio = 0.01  
Selected GPU: 1  
Start Dataset = cora task unlearn = edge model = GCN unlearn ratio = 0.02  
Selected GPU: 3  
Start Dataset = cora task unlearn = edge model = GCN unlearn ratio = 0.03  
Selected GPU: 1  
Start Dataset = cora task unlearn = edge model = GCN unlearn ratio = 0.04  
Selected GPU: 1  
Start Dataset = cora task unlearn = edge model = GCN unlearn ratio = 0.05  
Selected GPU: 1  
Start Dataset = cora task unlearn = edge model = GCN unlearn ratio = 0.06  
Selected GPU: 3  
Start Dataset = cora task unlearn = edge model = GCN unlearn ratio = 0.07  
Selected GPU: 1  
Start Dataset = cora task unlearn = edge model = GCN unlearn ratio = 0.08  
Selected GPU: 4  
Start Dataset = cora task unlearn = edge model = GCN unlearn ratio = 0.09  
Selected GPU: 0  
Start Dataset = cora task unlearn = edge model = GCN unlearn ratio = 0.1  
...   
```
This automatically generates a csv file named `unlearning_bounds.csv` including all numerical results. Detailed logs are stored in `unlearning_bounds_logs.txt`.

Exemplary results from `unlearning_bounds.csv` (associated with the logs shown above) is given below.

| dataset | model | unlearn_task | unlearn_ratio | f1_score_avg       | f1_score_std         | training_time_avg  | training_time_std  | f1_score_unlearn_avg | f1_score_unlearn_std | unlearning_time_avg | unlearning_time_std | my_bound_avg       | my_bound_std       | certified_edge_bound_avg | certified_edge_std | certified_edge_worst_bound_avg | certified_edge_worst_bound_std | actual_diff_avg    | actual_diff_std    |
|---------|-------|--------------|---------------|--------------------|----------------------|--------------------|--------------------|----------------------|----------------------|---------------------|---------------------|--------------------|--------------------|--------------------------|--------------------|--------------------------------|--------------------------------|--------------------|--------------------|
| cora    | GCN   | node         | 0.01          | 0.8511685116851170 | 0.004602284608578    | 5.07682474454244   | 0.007061816527726  | 0.7724477244772450   | 0.0046022846085779   | 0.1136529445648190  | 0.0222720360532479  | 18.36172930399580  | 0.0482265170820444 | 37.01830291748050        | 0.5854173804783640 | 236.3894132129670              | 2.8421709430404E-14            | 8.942059199015300  | 0.0290648921004638 |
| cora    | GCN   | node         | 0.02          | 0.8511685116851170 | 0.0017395000767196   | 5.093575716018680  | 0.0950053655426568 | 0.7724477244772450   | 0.0046022846085779   | 0.1126994291941320  | 0.0171324093761708  | 20.330174763997400 | 0.1186580022271320 | 47.27667363484700        | 1.6413476988635200 | 336.07919573245800             | 0.0                            | 9.315165519714360  | 0.0977979937353956 |
| cora    | GCN   | node         | 0.03          | 0.8523985239852400 | 0.005218500230159    | 5.174511194229130  | 0.0434043501364312 | 0.7761377613776140   | 0.0034790001534393   | 0.12740159034729    | 0.014840871850016   | 21.133344650268600 | 0.0467297363530789 | 51.61177062988280        | 0.6708882027824150 | 382.119409109561               | 5.6843418860808E-14            | 9.50989055633545   | 0.1148858657953290 |
| cora    | GCN   | node         | 0.04          | 0.8523985239852400 | 0.0030129025126483   | 5.254716714223230  | 0.0420685271731397 | 0.7859778597785980   | 0.005218500230159    | 0.1489001909891760  | 0.0082322127398724  | 22.23640823364260  | 0.1740132356555690 | 61.24066034952800        | 2.701514021922300  | 429.01364382437400             | 0.0                            | 9.983645757039390  | 0.0424001412460612 |
| cora    | GCN   | node         | 0.05          | 0.8511685116851170 | 0.004602284608578    | 5.234224637349450  | 0.1958559432284830 | 0.7650676506765070   | 0.004602284608578    | 0.1130084991455080  | 0.0213413254302411  | 22.848140080769900 | 0.1550712550128820 | 67.68686930338540        | 2.5446535149483200 | 447.24497332786200             | 5.6843418860808E-14            | 10.23299471537270  | 0.0975980988990892 |
| cora    | GCN   | node         | 0.06          | 0.8523985239852400 | 1.11022302462516E-16 | 4.980882962544760  | 0.0815385875183387 | 0.7601476014760150   | 0.0060258050252968   | 0.1002628803253170  | 0.0109076504333857  | 23.27087656656900  | 0.1675670706279030 | 72.56201680501300        | 2.834373770616790  | 455.4215223635620              | 0.0                            | 10.4602845509847   | 0.0995066350986747 |
| cora    | GCN   | node         | 0.07          | 0.8487084870848710 | 0.005218500230159    | 5.177481094996140  | 0.115267839639326  | 0.7515375153751540   | 0.0017395000767197   | 0.1343382994333900  | 0.0097608125791822  | 23.69449297587080  | 0.0395788032996186 | 76.70724995930990        | 0.6933564830307060 | 470.2381001231020              | 0.0                            | 10.617900212605800 | 0.1029552059803650 |
| cora    | GCN   | node         | 0.08          | 0.8523985239852400 | 1.11022302462516E-16 | 5.304854075113930  | 0.1551316453749470 | 0.7453874538745390   | 0.0079713907729493   | 0.1236019929250080  | 0.0145800763292509  | 23.652819951375300 | 0.0569787145270929 | 74.54288228352860        | 0.9817269565688510 | 471.9968198604840              | 0.0                            | 10.582382520039900 | 0.0542614412947368 |
| cora    | GCN   | node         | 0.09          | 0.8536285362853630 | 0.0034790001534393   | 5.1945938269297300 | 0.2594850303761670 | 0.7466174661746620   | 0.0034790001534393   | 0.1097153822580970  | 0.0134114774342295  | 24.05794843037920  | 0.0719060106875059 | 80.0522969563802         | 1.2903220642468800 | 474.1998358637670              | 5.6843418860808E-14            | 10.788128852844200 | 0.1510626384472450 |
| cora    | GCN   | node         | 0.1           | 0.8511685116851170 | 0.0017395000767196   | 5.441828330357870  | 0.0806195470253672 | 0.7293972939729400   | 0.0034790001534393   | 0.1412283579508460  | 0.0072483728121166  | 24.26727104187010  | 0.0302877308686056 | 82.59717814127610        | 0.5504999990683110 | 473.3180139515800              | 0.0                            | 10.86750062306720  | 0.055570923478962  |




## Usage of IDEA_attack
The folder `IDEA_attack` is used to attack IDEA. By using the files containing unlearned results from `./attack_materials`, run the following command to attack [node/edge/feauture].

```
cd IDEA_attack
./run_node.sh # attack node
./run_edge.sh # attack edge
./run_feature.sh # attack feature (partial or full)
```
### Note
Unlearned result filenames are supposed to follow the formats below.
```
../IDEA/attack_materials/seed_{args.run_seed_feature}_IDEA_cora_node_0.05_{args.model_name}_{args.remove_feature_ratio}.pth # attack node

../IDEA/attack_materials/seed_{args.run_seed_feature}_IDEA_cora_edge_0.05_{args.model_name}_{args.remove_feature_ratio}.pth # attack edge

../IDEA/attack_materials/seed_{args.run_seed_feature}_IDEA_cora_partial_feature_0.05_{args.model_name}_{args.remove_feature_ratio}.pth # attack feature
```


### Examples
Here we use the unlearned results of IDEA with configurations 

`dataset : Cora | unlearning_mode : node | unlearning_ratio : 0.05 | target_model : GCN`

Run

```
./run_node.sh
```
to attack the model with node unlearned. We present the sample log as follows.

```
shadow training
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 300/300 [00:14<00:00, 20.85it/s]
shadow train done
attack training
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 300/300 [00:22<00:00, 13.27it/s]
attack training done
Test accuracy: 0.492 AUROC: 0.513 precision: 0.492 recall 0.407 F1 score 0.407 ===> Attack Performance!
```
The result is also saved into a CSV file as `./result/node_attack_results.csv`

Dataset | Exp | Unlearn Ratio | AUC Mean | AUC Std | ACC Mean | ACC Std
:--: | :--: | :--: | :--: | :--: | :--: | :--: 
cora | node | 0.05 | 0.513 | 0.0 | 0.492 | 0.0

## Usage of GraphEraser

The folder of `Eraser` is used to reproduce the code of GraphEraser and support implementing both node attack (GraphMIA) and edge attack (StealLink). To choose different running configurations (dataset/target_model/partition_method/etc), change the parameters in run_new_[node/edge/feature].sh. Then run the following command to train & unlearn & attack.

```
cd Eraser
./run_new_node.sh # for node unlearning
./run_new_edge.sh # for edge unlearning
```
### Parameter explanation
```
--exp node # unlearning mode, within node/edge
--is_attack True # edge attack
--is_attack_node True # node attack
--is_ratio True # whether unlearn nodes by ratio/number
```

### Train & unlearn Examples

Here we consider setting as 

`dataset : cora | target_model : GCN | partition_method : random | aggregator : optimal | unlearning_mode : node`

Run
```
./run_new_node.sh
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

### Attack Example

To attack the node unlearning, simply add the `--is_attack_node True` in the .sh script, then rerun.
```
./run_new_node.sh
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


## Usage of CGU

The folder `CGU` is used to reproduce the code of CGU and support implementing both node attack (GraphMIA) and edge attack (StealLink). To choose different running configurations (dataset/unlearning ratio/etc), change the parameters in run_new_[node/edge/feature].sh. Then run the following command to train & unlearn & attack.

```
cd CGU
./run_new_node.sh # for node unlearning
./run_new_edge.sh # for edge unlearning
./run_new_feature.sh # for feature unlearning
```

### Parameter explanation
```
--removal_mode node # unlearning mode, within node/edge
--is_attack (store_true) # edge attack
--is_attack_node (store_true) # node attack
--is_attack_feature (store_true) # feauture attack (require removed feature_dimension data from IDEA)
--is_ratio (store_true) # whether unlearn instances by ratio/number
```


### Train & unlearn Examples

Here we consider setting as 

`dataset : cora | unlearning_mode : edge`

Run
```
./run_new_edge.sh
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

### Attack Example

To attack the node unlearning, simply add the `--is_attack` in the .sh script, then rerun.
```
./run_new_edge.sh
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
