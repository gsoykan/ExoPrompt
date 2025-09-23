#!/bin/bash

# Check if at least 3 arguments are given
if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <num_runs> <seed> <num_virtual_tokens> [exp_config1 exp_config2 ...]"
  exit 1
fi

NUM_RUNS=$1
SEED=$2
NUM_VIRTUAL_TOKENS=$3
shift 3

# If specific exp_configs are given as arguments, use them; otherwise use defaults
if [ "$#" -gt 0 ]; then
  exp_configs=("$@")
else
  exp_configs=("cleakage_gt_led_finetune_gt/vanilla" "cleakage_gt_led_finetune_gt/two_layer_mlp")
fi

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_DIR="scripts/final_logs/generalization_sim_cleakage_gt_led_finetune_gt_virtual_tokens_${NUM_VIRTUAL_TOKENS}_$TIMESTAMP"
mkdir -p "$LOG_DIR"

run_and_log() {
  local experiment=$1
  local log_file="${LOG_DIR}/${2}.log"
  echo "Running: $experiment"
  python $experiment >&"$log_file"
  echo "Output saved to $log_file"
}

BASELINE_DIR="exo_prompt/sys_pre/generalization_simulation"
SMALL_MODEL_ARGS="++model.model_configs_dict.d_model=256 ++model.model_configs_dict.d_ff=1024"

declare -A ckpts

# === Checkpoints for vanilla ===
ckpts["42460_cleakage_gt_led_finetune_gt/vanilla"]="/gpfs/home2/gsoykan/physinet/logs/train/runs/2025-06-16_12-50-18/checkpoints/epoch_047.ckpt"
ckpts["22248_cleakage_gt_led_finetune_gt/vanilla"]="/gpfs/home2/gsoykan/physinet/logs/train/runs/2025-06-16_12-50-46/checkpoints/epoch_049.ckpt"
ckpts["26199_cleakage_gt_led_finetune_gt/vanilla"]="/gpfs/home2/gsoykan/physinet/logs/train/runs/2025-06-16_12-50-30/checkpoints/epoch_036.ckpt"
ckpts["11339_cleakage_gt_led_finetune_gt/vanilla"]="/gpfs/home2/gsoykan/physinet/logs/train/runs/2025-06-16_12-51-42/checkpoints/epoch_048.ckpt"
ckpts["12481_cleakage_gt_led_finetune_gt/vanilla"]="/gpfs/home2/gsoykan/physinet/logs/train/runs/2025-06-16_12-49-36/checkpoints/epoch_046.ckpt"
ckpts["11649_cleakage_gt_led_finetune_gt/vanilla"]="/gpfs/home2/gsoykan/physinet/logs/train/runs/2025-06-16_12-49-50/checkpoints/epoch_049.ckpt"
ckpts["28820_cleakage_gt_led_finetune_gt/vanilla"]="/gpfs/home2/gsoykan/physinet/logs/train/runs/2025-06-16_12-50-54/checkpoints/epoch_045.ckpt"
ckpts["40263_cleakage_gt_led_finetune_gt/vanilla"]="/gpfs/home2/gsoykan/physinet/logs/train/runs/2025-06-16_12-51-24/checkpoints/epoch_048.ckpt"
ckpts["40173_cleakage_gt_led_finetune_gt/vanilla"]="/gpfs/home2/gsoykan/physinet/logs/train/runs/2025-06-16_12-51-19/checkpoints/epoch_036.ckpt"
ckpts["20237_cleakage_gt_led_finetune_gt/vanilla"]="/gpfs/home2/gsoykan/physinet/logs/train/runs/2025-06-16_12-49-28/checkpoints/epoch_047.ckpt"

# === Checkpoints for two_layer_mlp ===
ckpts["42460_cleakage_gt_led_finetune_gt/two_layer_mlp"]="/gpfs/home2/gsoykan/physinet/logs/train/runs/2025-06-16_12-47-36/checkpoints/epoch_046.ckpt"
ckpts["22248_cleakage_gt_led_finetune_gt/two_layer_mlp"]="/gpfs/home2/gsoykan/physinet/logs/train/runs/2025-06-16_12-48-05/checkpoints/epoch_049.ckpt"
ckpts["26199_cleakage_gt_led_finetune_gt/two_layer_mlp"]="/gpfs/home2/gsoykan/physinet/logs/train/runs/2025-06-16_12-47-51/checkpoints/epoch_048.ckpt"
ckpts["11339_cleakage_gt_led_finetune_gt/two_layer_mlp"]="/gpfs/home2/gsoykan/physinet/logs/train/runs/2025-06-16_12-49-05/checkpoints/epoch_037.ckpt"
ckpts["12481_cleakage_gt_led_finetune_gt/two_layer_mlp"]="/gpfs/home2/gsoykan/physinet/logs/train/runs/2025-06-16_12-47-02/checkpoints/epoch_049.ckpt"
ckpts["11649_cleakage_gt_led_finetune_gt/two_layer_mlp"]="/gpfs/home2/gsoykan/physinet/logs/train/runs/2025-06-16_12-47-19/checkpoints/epoch_049.ckpt"
ckpts["28820_cleakage_gt_led_finetune_gt/two_layer_mlp"]="/gpfs/home2/gsoykan/physinet/logs/train/runs/2025-06-16_12-48-20/checkpoints/epoch_046.ckpt"
ckpts["40263_cleakage_gt_led_finetune_gt/two_layer_mlp"]="/gpfs/home2/gsoykan/physinet/logs/train/runs/2025-06-16_12-48-59/checkpoints/epoch_040.ckpt"
ckpts["40173_cleakage_gt_led_finetune_gt/two_layer_mlp"]="/gpfs/home2/gsoykan/physinet/logs/train/runs/2025-06-16_12-48-37/checkpoints/epoch_045.ckpt"
ckpts["20237_cleakage_gt_led_finetune_gt/two_layer_mlp"]="/gpfs/home2/gsoykan/physinet/logs/train/runs/2025-06-16_12-46-49/checkpoints/epoch_049.ckpt"

run_one_round() {
  echo "Using seed: $SEED"

  for exp_config in "${exp_configs[@]}"; do

    key="${SEED}_${exp_config}"
    ckpt_path="${ckpts[$key]}"

    if [ -z "$ckpt_path" ]; then
      echo "❌ No checkpoint found for $key"
      continue
    fi

    experiment="src/train.py experiment=$BASELINE_DIR/${exp_config} ++model.pretrained_ckpt=$ckpt_path ++model.model_configs_dict.num_virtual_tokens=${NUM_VIRTUAL_TOKENS} ${SMALL_MODEL_ARGS} ++seed=${SEED}"
    # log file name (replaces / with _ to avoid directory confusion)
    log_name="seed${SEED}_generalization_sim_cleakage_gt_led_finetune_gt_virtual_tokens_${NUM_VIRTUAL_TOKENS}_${exp_config//\//_}"
    run_and_log "$experiment" "$log_name"
    echo -e "$experiment" "\nLOG NAME: " "$log_name"
  done
}

for ((i = 1; i <= NUM_RUNS; i++)); do
  echo "========== RUN $i =========="
  run_one_round
  echo "============================"
  echo
done

# bash scripts/16_june_generalization_sim_cleakage_gt_led_finetune_gt.sh 1 42460 0 cleakage_gt_led_finetune_gt/vanilla
# bash scripts/16_june_generalization_sim_cleakage_gt_led_finetune_gt.sh 1 22248 0 cleakage_gt_led_finetune_gt/vanilla
# bash scripts/16_june_generalization_sim_cleakage_gt_led_finetune_gt.sh 1 26199 0 cleakage_gt_led_finetune_gt/vanilla

# bash scripts/16_june_generalization_sim_cleakage_gt_led_finetune_gt.sh 1 42460 1 cleakage_gt_led_finetune_gt/two_layer_mlp
# bash scripts/16_june_generalization_sim_cleakage_gt_led_finetune_gt.sh 1 22248 1 cleakage_gt_led_finetune_gt/two_layer_mlp
# bash scripts/16_june_generalization_sim_cleakage_gt_led_finetune_gt.sh 1 26199 1 cleakage_gt_led_finetune_gt/two_layer_mlp

# bash scripts/16_june_generalization_sim_cleakage_gt_led_finetune_gt.sh 1 42460 2 cleakage_gt_led_finetune_gt/two_layer_mlp
# bash scripts/16_june_generalization_sim_cleakage_gt_led_finetune_gt.sh 1 22248 2 cleakage_gt_led_finetune_gt/two_layer_mlp
# bash scripts/16_june_generalization_sim_cleakage_gt_led_finetune_gt.sh 1 26199 2 cleakage_gt_led_finetune_gt/two_layer_mlp

# bash scripts/16_june_generalization_sim_cleakage_gt_led_finetune_gt.sh 1 42460 3 cleakage_gt_led_finetune_gt/two_layer_mlp
# bash scripts/16_june_generalization_sim_cleakage_gt_led_finetune_gt.sh 1 22248 3 cleakage_gt_led_finetune_gt/two_layer_mlp
# bash scripts/16_june_generalization_sim_cleakage_gt_led_finetune_gt.sh 1 26199 3 cleakage_gt_led_finetune_gt/two_layer_mlp
