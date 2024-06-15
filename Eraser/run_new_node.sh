#!/bin/bash
# node
for a in citeseer 
do

    for b in sage_km sage_km sage_km
    do

        for c in SGC
        do

            for d in optimal
            do
                for e in node 
                do
                    for f in 0.01
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


                        info="Dataset = ${a} partition = ${b} model = ${c} aggregator = ${d} task unlearn = ${e}"

                        echo "Start ${info}"
                        output_file="../Graph-Unlearning/exp/new_setting_eraser.txt"

                        nohup python main.py \
                            --dataset_name $a \
                            --target_model $c \
                            --is_train_target_model True \
                            --exp $e \
                            --partition_method $b \
                            --aggregator $d \
                            --is_use_node_feature True \
                            --is_ratio True \
                            --num_unlearned 0 \
                            --ratio_unlearned $f \
                            --num_runs 3 \
                            --csv_file_name ../Graph-Unlearning/temp_data/new_setting \
                            --file_name ../Graph-Unlearning/temp_data/new_setting.txt > $output_file 2>&1 &

                        pid=$!
                        wait $pid
                    done
                done
            done
        done
    done
done



