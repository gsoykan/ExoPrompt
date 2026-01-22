#!/bin/bash
# Script to run vanilla baseline finetuning experiments
# Called from jobscript with args: model_name experiment_type seed
#
# Usage: bash scripts/revision/finetuning_vanilla_models_w_pretraining.sh <model> <exp_type> <seed>
# Example: bash scripts/revision/finetuning_vanilla_models_w_pretraining.sh gru hps_only 42460

# Parse arguments
MODEL_NAME=$1        # e.g., "gru", "lstm", "tcn", "dlinear"
EXPERIMENT_TYPE=$2   # e.g., "hps_only", "led_only", "finetuning_mixed"
SEED=$3              # e.g., 42460

# Validate arguments
if [ -z "$MODEL_NAME" ] || [ -z "$EXPERIMENT_TYPE" ] || [ -z "$SEED" ]; then
  echo "Error: Missing required arguments"
  echo "Usage: $0 <model_name> <experiment_type> <seed>"
  echo "Example: $0 gru hps_only 42460"
  exit 1
fi

# Create a logs directory if it doesn't exist
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_DIR="scripts/revision_logs/finetuning_vanilla_w_pretraining"
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
  "dlinear:/home/gsoykan/physinet/logs/train/runs/2025-12-15_20-53-21/checkpoints/epoch_017.ckpt"
  "gru:/home/gsoykan/physinet/logs/train/runs/2025-12-16_10-41-19/checkpoints/epoch_047.ckpt"
  "lstm:/home/gsoykan/physinet/logs/train/runs/2025-12-15_20-54-02/checkpoints/epoch_048.ckpt"
  "tcn:/home/gsoykan/physinet/logs/train/runs/2025-12-15_20-54-40/checkpoints/epoch_049.ckpt"
)

# Find the checkpoint for the current model
CHECKPOINT=""
for entry in "${PRETRAINED_CHECKPOINTS[@]}"; do
  IFS=':' read -r model_key ckpt_path <<<"$entry"
  if [[ "$MODEL_NAME" == "$model_key" ]]; then
    CHECKPOINT="$ckpt_path"
    break
  fi
done

# Validate checkpoint was found
if [ -z "$CHECKPOINT" ]; then
  echo "Error: No pretrained checkpoint found for model: $MODEL_NAME"
  exit 1
fi

echo "============================== EXPERIMENT CONFIGURATION ==============================="
echo "Model: $MODEL_NAME"
echo "Experiment Type: $EXPERIMENT_TYPE"
echo "Seed: $SEED"
echo "Pretrained Checkpoint: $CHECKPOINT"
echo "======================================================================================="

# Construct the experiment config path
EXPERIMENT_CONFIG="revision_baselines/finetuning/vanilla/${MODEL_NAME}_template"

# Create log file name
LOG_NAME="${MODEL_NAME}_${EXPERIMENT_TYPE}_seed${SEED}_${TIMESTAMP}"

# Run the experiment with pretrained checkpoint
EXPERIMENT_CMD="src/train.py experiment=${EXPERIMENT_CONFIG} ++data.experiment_config.type=${EXPERIMENT_TYPE} ++seed=${SEED} ++model.pretrained_ckpt=${CHECKPOINT}"
run_and_log "$EXPERIMENT_CMD" "$LOG_NAME"

echo "======================================================================================="
echo "✓ Experiment completed"
echo "======================================================================================="