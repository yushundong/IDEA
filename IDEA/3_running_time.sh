#!/bin/bash

for a in cora citeseer pubmed CS Physics
do

    for b in node
    do

        for c in SGC
        do

            for d in 0.001
            do

                # Function to get GPU utilization for a given GPU ID
                get_gpu_load() {
                    gpu_id=$1
                    load=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $gpu_id)
                    echo "$load"
                }

                # Function to choose the GPU with the least load
                choose_gpu_with_least_load() {
                    gpu_count=$(nvidia-smi --list-gpus | wc -l)
                    if [ $gpu_count -eq 0 ]; then
                        echo "No GPUs available."
                        exit 1
                    fi

                    # Initialize variables
                    min_load=100
                    chosen_gpu=""

                    # Loop through available GPUs
                    for ((gpu_id = 0; gpu_id < $gpu_count; gpu_id++)); do
                        load=$(get_gpu_load $gpu_id)
                        if [ -z "$load" ]; then
                            continue
                        fi

                        if ((load < min_load)); then
                            min_load=$load
                            chosen_gpu=$gpu_id
                        fi
                    done

                    echo "$chosen_gpu"
                }

                # Choose GPU with the least load
                chosen_gpu=$(choose_gpu_with_least_load)

                if [ -z "$chosen_gpu" ]; then
                    echo "No available GPUs or unable to determine GPU load."
                    exit 1
                fi

                echo "Selected GPU: $chosen_gpu"

                # Set the CUDA_VISIBLE_DEVICES environment variable to restrict execution to the chosen GPU
                export CUDA_VISIBLE_DEVICES=$chosen_gpu


                info="Dataset = ${a} task unlearn = ${b} model = ${c} unlearn ratio = ${d}"

                echo "Start ${info}"
                output_file="./unlearning_time_logs.txt"

                nohup python main.py \
                    --dataset_name $a \
                    --target_model $c \
                    --is_train_target_model True \
                    --is_use_node_feature True \
                    --num_runs 3 \
                    --unlearn_task $b \
                    --unlearn_ratio $d \
                    --iteration 100 \
                    --file_name unlearning_time \
                    --scale 500  > $output_file 2>&1 &

                pid=$!
                wait $pid

            done
        done
    done
done