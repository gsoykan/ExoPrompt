#!/bin/bash
# Run from root folder with: bash scripts/schedule.sh

# Create a logs directory if it doesn't exist
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_DIR="scripts/final_logs/baselines_$TIMESTAMP"
mkdir -p "$LOG_DIR"

run_and_log() {
    local experiment=$1
    local log_file="${LOG_DIR}/${2}.log"
    echo "Running: $experiment"
    python $experiment >& "$log_file"
    echo "Output saved to $log_file"
}

# Commented baselines are already handled in pretraining scaling
# Uncomment or add baselines as needed
# Available baselines: "brute_concat", "two_layer_mlp", "two_layer_mlp_rand_init", "vanilla"
baselines=("brute_concat" "two_layer_mlp_rand_init")

BASELINE_DIR="exo_prompt/sys_pre/baseline/pretraining"

run_baseline_experiments() {
  # Generate a single random seed for all experiments
  SEED=$((RANDOM % 90000 + 10000))  # Ensures a 5-digit number (10000–99999)
  echo "Using seed: $SEED"

  for baseline in "${baselines[@]}"; do
    experiment="src/train.py experiment=${BASELINE_DIR}/${baseline} ++seed=${SEED}"
    log_name="seed${SEED}_baseline_with_${baseline}"
    run_and_log "$experiment" "$log_name"
    # echo -e "$experiment" "\nLOG NAME: ""$log_name"
  done
}

# Run the function three times with different seeds
for i in {1..3}; do
    echo "================== RUN $i =================="
    run_baseline_experiments
    echo "==========================================="
    echo
done


