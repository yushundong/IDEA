# !/bin/bash




# edge
for a in CS 
do
    for b in 0.01 0.05 0.1 0.02 0.03 0.04 0.06 0.07 0.08 0.09
    do
        for c in edge
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
            output_file="../sgc_unlearn/logs/new_setting_log_cs.txt"

            nohup python sgc_edge_unlearn.py \
                --dataset $a \
                --ratio_removes $b \
                --removal_mode $c \
                --trails 3 \
                --is_ratio \
                --csv_file_name ../sgc_unlearn/result/new_results_cs > $output_file 2>&1 &

            pid=$!
            wait $pid
        done
    done

done
