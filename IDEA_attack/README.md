## Usage
This folder is used to attack IDEA. By using the files containing unlearned results from `./attack_materials`, run the following command to attack [node/edge/feauture].

```
bash run_node.sh # attack node
bash run_edge.sh # attack edge
bash run_feature.sh # attack feature
```
### Note
Unlearned result filename should be saved as following formats.
```
../IDEA/attack_materials/seed_{args.run_seed_feature}_IDEA_cora_node_0.05_{args.model_name}_{args.remove_feature_ratio}.pth # attack node

../IDEA/attack_materials/seed_{args.run_seed_feature}_IDEA_cora_edge_0.05_{args.model_name}_{args.remove_feature_ratio}.pth # attack edge

../IDEA/attack_materials/seed_{args.run_seed_feature}_IDEA_cora_partial_feature_0.05_{args.model_name}_{args.remove_feature_ratio}.pth # attack feature
```


## Examples
Here we use the unlearned results of IDEA with configurations 

`dataset : Cora | unlearning_mode : node | unlearning_ratio : 0.05 | target_model : GCN`

Run

```
bash run_node.sh
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