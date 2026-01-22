#!/bin/bash
# Script to run vanilla baseline pretraining experiments
# Called from jobscript with args: model_name seed
#
# Usage: bash scripts/revision/pretraining_vanilla_models.sh <model> <seed>
# Example: bash scripts/revision/pretraining_vanilla_models.sh gru 42460

# Parse arguments
MODEL_NAME=$1        # e.g., "gru", "lstm", "tcn", "dlinear"
SEED=$2              # e.g., 42460

# Validate arguments
if [ -z "$MODEL_NAME" ] || [ -z "$SEED" ]; then
  echo "Error: Missing required arguments"
  echo "Usage: $0 <model_name> <seed>"
  echo "Example: $0 gru 42460"
  exit 1
fi

# Create a logs directory if it doesn't exist
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_DIR="scripts/revision_logs/pretraining_vanilla"
mkdir -p "$LOG_DIR"

run_and_log() {
  local experiment=$1
  local log_file="${LOG_DIR}/${2}.log"
  echo "Running: $experiment"
  python $experiment >&"$log_file"
  echo "Output saved to $log_file"
}

echo "============================== EXPERIMENT CONFIGURATION ==============================="
echo "Model: $MODEL_NAME"
echo "Seed: $SEED"
echo "======================================================================================="

# Construct the experiment config path
EXPERIMENT_CONFIG="revision_baselines/pretraining/vanilla/${MODEL_NAME}"

# Create log file name
LOG_NAME="${MODEL_NAME}_seed${SEED}_${TIMESTAMP}"

# Run the experiment
EXPERIMENT_CMD="src/train.py experiment=${EXPERIMENT_CONFIG} ++seed=${SEED}"
run_and_log "$EXPERIMENT_CMD" "$LOG_NAME"

echo "======================================================================================="
echo "✓ Experiment completed"
echo "======================================================================================="