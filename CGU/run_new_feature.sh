# !/bin/bash


# node
for a in cora 
do
    for b in 0.05
    do
        for c in feature 
        do
            for e in 2
            do 
                 for g in 0.2 0.5 0.8 # unlearn dimension ratio
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


                    info="Dataset = ${a} removes = ${b} mode = ${c} "

                    echo "Start ${info}"
                    output_file="../sgc_unlearn/logs/new_setting_log.txt"

                    nohup python sgc_feature_node_unlearn.py \
                        --dataset $a \
                        --ratio_removes $b \
                        --removal_mode $c \
                        --trails 1 \
                        --is_ratio \
                        --run_seed_feature $e \
                        --remove_feature_ratio $g \
                        --csv_file_name ../sgc_unlearn/result/new_results > $output_file 2>&1 &
                    pid=$!
                    wait $pid
                done
            done
        done
    done
done

