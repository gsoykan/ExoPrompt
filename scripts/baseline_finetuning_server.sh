#!/bin/bash
# Run from root folder with: bash scripts/schedule.sh

# Create a logs directory if it doesn't exist
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_DIR="scripts/final_logs/finetuning_baseline_$TIMESTAMP"
mkdir -p "$LOG_DIR"

run_and_log() {
  local experiment=$1
  local log_file="${LOG_DIR}/${2}.log"
  echo "Running: $experiment"
  python $experiment >&"$log_file"
  echo "Output saved to $log_file"
}

# General Configurations
EXPERIMENT_TYPES=("hps_only" "led_only" "finetuning_mixed")
PHYSINET_SETTINGS=(
  "++model.physinet_mode=false ++model.physinet_num_features=null"
  "++model.physinet_mode=true ++model.physinet_num_features=3"
)

# Checkpoints for baselines
CHECKPOINTS=(
  # VANILLA
  # "vanilla:/home/gsoykan/physinet/logs/train/runs/2025-01-11_07-13-21/checkpoints/epoch_049.ckpt"

  # BRUTE_CONCAT
  "brute_concat:/home/gsoykan/physinet/logs/train/runs/2025-04-04_17-25-29/checkpoints/epoch_049.ckpt"

  # TWO_LAYER_MLP_RAND_INIT
  "two_layer_mlp_rand_init:/home/gsoykan/physinet/logs/train/runs/2025-04-05_01-07-48/checkpoints/epoch_049.ckpt"

  # TWO_LAYER_MLP
  # "two_layer_mlp:/home/gsoykan/physinet/logs/train/runs/2025-01-10_09-31-49/checkpoints/epoch_048.ckpt"
)

# Baseline-specific configurations
BASELINE_CONFIGS=(
  # currently we  have single exp. run results for vanilla.
  # "vanilla:++model.model_configs_dict.enable_exo_prompt_tuning=false ++model.model_configs_dict.prompt_tuning_type=null ++data.return_random_exo_params=false"
  "brute_concat:++model.model_configs_dict.enable_exo_prompt_tuning=true ++model.model_configs_dict.prompt_tuning_type=brute_concat ++data.return_random_exo_params=false"
  "two_layer_mlp_rand_init:++model.model_configs_dict.enable_exo_prompt_tuning=true ++model.model_configs_dict.prompt_tuning_type=two_layer_mlp ++data.return_random_exo_params=true"
  # "two_layer_mlp:++model.model_configs_dict.enable_exo_prompt_tuning=true ++model.model_configs_dict.prompt_tuning_type=two_layer_mlp ++data.return_random_exo_params=false"
)

# Experiment file template
TEMPLATE_EXPERIMENT_FILE="exo_prompt/sys_pre/baseline/pretraining/finetuning_template.yaml"

# Function to run experiments for each baseline
run_experiments_for_baseline() {
  local baseline=$1
  local ckpt=$2
  local config=$3
  local seed=$4

  for experiment_type in "${EXPERIMENT_TYPES[@]}"; do
    for physinet_setting in "${PHYSINET_SETTINGS[@]}"; do
      log_name="${baseline}_${experiment_type}_$(basename "$ckpt" .ckpt)_$(echo "$physinet_setting" | tr ' ' '_')_seed${seed}"
      experiment="src/train.py experiment=${TEMPLATE_EXPERIMENT_FILE} ++model.pretrained_ckpt=$ckpt $config $physinet_setting ++data.experiment_config.type=$experiment_type"
      run_and_log "$experiment" "$log_name"
      # echo -e "$experiment" "\nLOG NAME: ""$log_name"
    done
  done
}

# Run the function three times with different seeds
for i in {1..3}; do
  # Generate a single random seed for all experiments
  SEED=$((RANDOM % 90000 + 10000)) # Ensures a 5-digit number (10000–99999)
  echo "Using seed: $SEED"

  # Loop through checkpoints and configurations
  for entry in "${CHECKPOINTS[@]}"; do
    # Extract baseline name and checkpoint path
    IFS=':' read -r baseline ckpt <<<"$entry"

    # Match the baseline to its specific configuration
    for config_entry in "${BASELINE_CONFIGS[@]}"; do
      IFS=':' read -r config_baseline config <<<"$config_entry"
      if [[ "$baseline" == "$config_baseline" ]]; then
        echo "------- STARTING EXPERIMENTS FOR $baseline -------"
        echo "Running experiments for baseline: $baseline"
        run_experiments_for_baseline "$baseline" "$ckpt" "$config ++seed=${SEED}" "$SEED"
        echo "------- COMPLETED EXPERIMENTS FOR $baseline -------"
        break
      fi
    done
  done
done
