#!/bin/bash
# Run from root folder with: bash scripts/schedule.sh

# Create a logs directory if it doesn't exist
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_DIR="scripts/final_logs/finetuning_scaling_$TIMESTAMP"
mkdir -p "$LOG_DIR"


run_and_log() {
    local experiment=$1
    local log_file="${LOG_DIR}/${2}.log"
    echo "Running: $experiment"
    python $experiment >& "$log_file"
    echo "Output saved to $log_file"
}

# General Configurations
EXPERIMENT_TYPES=("hps_only" "led_only" "finetuning_mixed")
PHYSINET_SETTINGS=(
    "++model.physinet_mode=true ++model.physinet_num_features=3"
    "++model.physinet_mode=false ++model.physinet_num_features=null"
)

# Checkpoints with Dataset Sizes
WITHOUT_EXOPROMPT_CHECKPOINTS=(
    # 1k is missing do it later...
    # "1k:/home/gsoykan/physinet/logs/train/runs/2025-01-08_18-36-22/checkpoints/epoch_000.ckpt"
    "10k:/home/gsoykan/physinet/logs/train/runs/2025-04-03_09-57-03/checkpoints/epoch_048.ckpt"
    "50k:/home/gsoykan/physinet/logs/train/runs/2025-04-03_11-10-15/checkpoints/epoch_018.ckpt"
    "100k:/home/gsoykan/physinet/logs/train/runs/2025-04-03_12-48-47/checkpoints/epoch_048.ckpt"
    "200k:/home/gsoykan/physinet/logs/train/runs/2025-04-03_15-55-08/checkpoints/epoch_049.ckpt"
    "500k:/home/gsoykan/physinet/logs/train/runs/2025-04-03_22-17-01/checkpoints/epoch_047.ckpt"
    "1m:/home/gsoykan/physinet/logs/train/runs/2025-04-03_01-49-33/checkpoints/epoch_049.ckpt"
)

WITH_EXOPROMPT_CHECKPOINTS=(
    # "1k:/home/gsoykan/physinet/logs/train/runs/"
    "10k:/home/gsoykan/physinet/logs/train/runs/2025-04-03_09-30-06/checkpoints/epoch_048.ckpt"
    "50k:/home/gsoykan/physinet/logs/train/runs/2025-04-03_10-23-22/checkpoints/epoch_049.ckpt"
    "100k:/home/gsoykan/physinet/logs/train/runs/2025-04-03_11-37-21/checkpoints/epoch_048.ckpt"
    "200k:/home/gsoykan/physinet/logs/train/runs/2025-04-03_13-54-51/checkpoints/epoch_049.ckpt"
    "500k:/home/gsoykan/physinet/logs/train/runs/2025-04-03_17-46-19/checkpoints/epoch_049.ckpt"
    "1m:/home/gsoykan/physinet/logs/train/runs/2025-04-02_17-19-31/checkpoints/epoch_049.ckpt"
)

# Function to run experiments
run_experiments() {
    local prefix=$1
    local ckpts=("${!2}")
    local base_experiment=$3
    local seed=$4

    # Regular runs with checkpoints
    for ckpt_entry in "${ckpts[@]}"; do
        IFS=':' read -r dataset_size ckpt <<< "$ckpt_entry"
        for experiment_type in "${EXPERIMENT_TYPES[@]}"; do
            for physinet_setting in "${PHYSINET_SETTINGS[@]}"; do
                alias="${prefix}_${dataset_size}_${experiment_type}_$(basename "$ckpt" .ckpt)_$(echo "$physinet_setting" | tr ' ' '_')_seed${seed}"
                experiment="src/train.py $base_experiment ++model.pretrained_ckpt=$ckpt $physinet_setting ++data.experiment_config.type=$experiment_type"
                run_and_log "$experiment" "$alias"
            done
        done
    done

    # Include a special run for ++model.pretrained_ckpt=null
    for experiment_type in "${EXPERIMENT_TYPES[@]}"; do
        for physinet_setting in "${PHYSINET_SETTINGS[@]}"; do
            alias="${prefix}_null_${experiment_type}_$(echo "$physinet_setting" | tr ' ' '_')_seed${seed}"
            experiment="src/train.py $base_experiment ++model.pretrained_ckpt=null $physinet_setting ++data.experiment_config.type=$experiment_type"
            run_and_log "$experiment" "$alias"
        done
    done
}


# Run the function three times with different seeds
for i in {1..3}; do
    # Generate a single random seed for all experiments
    SEED=$((RANDOM % 90000 + 10000))  # Ensures a 5-digit number (10000–99999)
    echo "Using seed: $SEED"

    echo "================== RUN $i =================="
    # Run experiments with Exoprompt
    run_experiments "with_exoprompt" WITH_EXOPROMPT_CHECKPOINTS[@] "experiment=exo_prompt/sys_pre/scaling/two_layer_mlp/pretraining/finetuning_template  ++seed=${SEED}" "$SEED"
    # Run experiments without Exoprompt
    run_experiments "without_exoprompt" WITHOUT_EXOPROMPT_CHECKPOINTS[@] "experiment=exo_prompt/sys_pre/scaling/vanilla/pretraining/finetuning_template  ++seed=${SEED}" "$SEED"
    echo "==========================================="
    echo
done