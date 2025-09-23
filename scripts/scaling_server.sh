#!/bin/bash
# Run from root folder with: bash scripts/schedule.sh

# Create a logs directory if it doesn't exist
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_DIR="scripts/final_logs/scaling_$TIMESTAMP"
mkdir -p "$LOG_DIR"

run_and_log() {
    local experiment=$1
    local log_file="${LOG_DIR}/${2}.log"
    echo "Running: $experiment"
    python $experiment >& "$log_file"
    echo "Output saved to $log_file"
}

scales=("1m" "10k" "50k" "100k" "200k" "500k") #"2m" "5m" "7m")

BASELINE_DIR="exo_prompt/sys_pre/scaling"

run_scaling_experiments() {
    # Generate a single random seed for all experiments
    SEED=$((RANDOM % 90000 + 10000))  # Ensures a 5-digit number (10000–99999)
    echo "Using seed: $SEED"

    for scale in "${scales[@]}"; do
        experiment="src/train.py experiment=${BASELINE_DIR}/two_layer_mlp/pretraining/${scale}_pretraining_mixed ++seed=${SEED}"
        log_name="seed${SEED}_scaling_with_exo_${scale}"
        run_and_log "$experiment" "$log_name"
        # echo -e "$experiment" "\nLOG NAME: ""$log_name"

        experiment="src/train.py experiment=${BASELINE_DIR}/vanilla/pretraining/${scale}_pretraining_mixed ++seed=${SEED}"
        log_name="seed${SEED}_scaling_without_exo_${scale}"
        run_and_log "$experiment" "$log_name"
        # echo -e "$experiment" "\nLOG NAME: ""$log_name"
    done
}

# Run the function three times with different seeds
for i in {1..1}; do
    echo "================== RUN $i =================="
    run_scaling_experiments
    echo "==========================================="
    echo
done
