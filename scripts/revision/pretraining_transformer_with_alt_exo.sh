#!/bin/bash
# Script to run vanilla baseline pretraining experiments
# Called from jobscript with args: model_name seed
#
# Usage: bash scripts/revision/pretraining_vanilla_models.sh <model> <seed>
# Example: bash scripts/revision/pretraining_vanilla_models.sh gru 42460

# Parse arguments
TUNING_TYPE=$1        # e.g., "brute_concat", "direct_concat"
SEED=$2              # e.g., 42460

# Validate arguments
if [ -z "$TUNING_TYPE" ] || [ -z "$SEED" ]; then
  echo "Error: Missing required arguments"
  echo "Usage: $0 <tuning_type> <seed>"
  echo "Example: $0 direct_concat 42460"
  exit 1
fi

# Create a logs directory if it doesn't exist
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_DIR="scripts/revision_logs/pretraining_alt_exo"
mkdir -p "$LOG_DIR"

run_and_log() {
  local experiment=$1
  local log_file="${LOG_DIR}/${2}.log"
  echo "Running: $experiment"
  python $experiment >&"$log_file"
  echo "Output saved to $log_file"
}

echo "============================== EXPERIMENT CONFIGURATION ==============================="
echo "Tuning Type: $TUNING_TYPE"
echo "Seed: $SEED"
echo "======================================================================================="

# Construct the experiment config path
# Will be: revision_baselines/pretraining/alt_exo/transformer_brute_concat
#      or: revision_baselines/pretraining/alt_exo/transformer_direct_concat
EXPERIMENT_CONFIG="revision_baselines/pretraining/alt_exo/transformer_${TUNING_TYPE}"

# Create log file name
LOG_NAME="${TUNING_TYPE}_seed${SEED}_${TIMESTAMP}"

# Run the experiment
EXPERIMENT_CMD="src/train.py experiment=${EXPERIMENT_CONFIG} ++seed=${SEED}"
run_and_log "$EXPERIMENT_CMD" "$LOG_NAME"

echo "======================================================================================="
echo "✓ Experiment completed"
echo "======================================================================================="