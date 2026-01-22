#!/bin/bash
# Script to run vanilla baseline finetuning experiments
# Called from jobscript with args: model_name experiment_type seed
#
# Usage: bash scripts/revision/finetuning_vanilla_models_wout_pretraining.sh <model> <exp_type> <seed>
# Example: bash scripts/revision/finetuning_vanilla_models_wout_pretraining.sh gru hps_only 42460

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
LOG_DIR="scripts/revision_logs/finetuning_vanilla_wout_pretraining"
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
echo "Experiment Type: $EXPERIMENT_TYPE"
echo "Seed: $SEED"
echo "======================================================================================="

# Construct the experiment config path
EXPERIMENT_CONFIG="revision_baselines/finetuning/vanilla/${MODEL_NAME}_template"

# Create log file name
LOG_NAME="${MODEL_NAME}_${EXPERIMENT_TYPE}_seed${SEED}_${TIMESTAMP}"

# Run the experiment
EXPERIMENT_CMD="src/train.py experiment=${EXPERIMENT_CONFIG} ++data.experiment_config.type=${EXPERIMENT_TYPE} ++seed=${SEED}"
run_and_log "$EXPERIMENT_CMD" "$LOG_NAME"

echo "======================================================================================="
echo "✓ Experiment completed"
echo "======================================================================================="