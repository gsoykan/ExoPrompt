#!/bin/bash

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_DIR="scripts/final_logs/zero_shot_eval_$TIMESTAMP"
mkdir -p "$LOG_DIR"

run_and_log() {
  local experiment=$1
  local log_file="${LOG_DIR}/${2}.log"
  echo "Running: $experiment"
  python $experiment >&"$log_file"
  echo "Output saved to $log_file"
}

BASELINE_DIR="exo_prompt/sys_pre/baseline/pretraining/zero_shot"

checkpoints_array=(
  # vanilla ckpt
  "/home/gsoykan/physinet/logs/train/runs/2025-04-03_01-49-33/checkpoints/epoch_049.ckpt"
  # two_layer (exo) ckpt
  "/home/gsoykan/physinet/logs/train/runs/2025-04-02_17-19-31/checkpoints/epoch_049.ckpt"
)

exp_configs=("vanilla_hps" "vanilla_led" "vanilla_mixed" "exo_hps" "exo_led" "exo_mixed")

eval_one_round() {
  # Generate a single random seed for all experiments
  SEED=$((RANDOM % 90000 + 10000)) # Ensures a 5-digit number (10000–99999)
  echo "Using seed: $SEED"

  for exp_config in "${exp_configs[@]}"; do
    if [[ $exp_config == *"vanilla"* ]]; then
      current_checkpoint="${checkpoints_array[0]}"
    else
      current_checkpoint="${checkpoints_array[1]}"
    fi

    experiment="src/eval_without_ckpt.py +experiment=$BASELINE_DIR/${exp_config}.yaml ++model.pretrained_ckpt=$current_checkpoint ++seed=${SEED}"
    log_name="seed${SEED}_zero_shot_eval_${exp_config}"
    run_and_log "$experiment" "$log_name"
    # echo -e "$experiment" "\nLOG NAME: " "$log_name"

  done
}

for i in {1..3}; do
  echo "========== RUN $i =========="
  eval_one_round
  echo "============================"
  echo
done
