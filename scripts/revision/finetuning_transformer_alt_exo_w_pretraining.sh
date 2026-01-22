#!/bin/bash
# Script to run alternative exo prompting (brute_concat, direct_concat) finetuning experiments
# Called from jobscript with args: tuning_type experiment_type seed
#
# Usage: bash scripts/revision/finetuning_transformer_alt_exo_w_pretraining.sh <tuning_type> <exp_type> <seed>
# Example: bash scripts/revision/finetuning_transformer_alt_exo_w_pretraining.sh brute_concat hps_only 42460

# Parse arguments
TUNING_TYPE=$1        # e.g., "brute_concat" or "direct_concat"
EXPERIMENT_TYPE=$2   # e.g., "hps_only", "led_only", "finetuning_mixed"
SEED=$3              # e.g., 42460

# Validate arguments
if [ -z "$TUNING_TYPE" ] || [ -z "$EXPERIMENT_TYPE" ] || [ -z "$SEED" ]; then
  echo "Error: Missing required arguments"
  echo "Usage: $0 <tuning_type> <experiment_type> <seed>"
  echo "Example: $0 brute_concat hps_only 42460"
  exit 1
fi

# Create a logs directory if it doesn't exist
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_DIR="scripts/revision_logs/finetuning_alt_exo_w_pretraining"
mkdir -p "$LOG_DIR"

run_and_log() {
  local experiment=$1
  local log_file="${LOG_DIR}/${2}.log"
  echo "Running: $experiment"
  python $experiment >&"$log_file"
  echo "Output saved to $log_file"
}

# Pretrained Checkpoints
PRETRAINED_CHECKPOINTS=(
  "brute_concat:/home/gsoykan/physinet/logs/train/runs/2025-12-18_20-22-52/checkpoints/epoch_049.ckpt"
  "direct_concat:/home/gsoykan/physinet/logs/train/runs/2025-12-18_20-22-36/checkpoints/epoch_047.ckpt"
)

# Find the checkpoint for the current tuning type
CHECKPOINT=""
for entry in "${PRETRAINED_CHECKPOINTS[@]}"; do
  IFS=':' read -r tuning_key ckpt_path <<<"$entry"
  if [[ "$TUNING_TYPE" == "$tuning_key" ]]; then
    CHECKPOINT="$ckpt_path"
    break
  fi
done

# Validate checkpoint was found
if [ -z "$CHECKPOINT" ]; then
  echo "Error: No pretrained checkpoint found for tuning type: $TUNING_TYPE"
  exit 1
fi

echo "============================== EXPERIMENT CONFIGURATION ==============================="
echo "Tuning Type: $TUNING_TYPE"
echo "Experiment Type: $EXPERIMENT_TYPE"
echo "Seed: $SEED"
echo "Pretrained Checkpoint: $CHECKPOINT"
echo "======================================================================================="

# Construct the experiment config path (single template for all tuning types)
EXPERIMENT_CONFIG="revision_baselines/finetuning/alt_exo/transformer_template"

# Create log file name
LOG_NAME="${TUNING_TYPE}_${EXPERIMENT_TYPE}_seed${SEED}_${TIMESTAMP}"

# Run the experiment with pretrained checkpoint
EXPERIMENT_CMD="src/train.py experiment=${EXPERIMENT_CONFIG} ++model.model_configs_dict.prompt_tuning_type=${TUNING_TYPE} ++data.experiment_config.type=${EXPERIMENT_TYPE} ++seed=${SEED} ++model.pretrained_ckpt=${CHECKPOINT}"
run_and_log "$EXPERIMENT_CMD" "$LOG_NAME"

echo "======================================================================================="
echo "✓ Experiment completed"
echo "======================================================================================="