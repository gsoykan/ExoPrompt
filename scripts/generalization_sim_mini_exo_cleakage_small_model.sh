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
  exp_configs=("cleakage_mini_exo/vanilla" "cleakage_mini_exo/two_layer_mlp")
fi

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_DIR="scripts/final_logs/generalization_sim_mini_exo_cleakage_small_model_virtual_tokens_${NUM_VIRTUAL_TOKENS}_$TIMESTAMP"
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

run_one_round() {
  echo "Using seed: $SEED"

  for exp_config in "${exp_configs[@]}"; do
    experiment="src/train.py experiment=$BASELINE_DIR/${exp_config} ++model.model_configs_dict.num_virtual_tokens=${NUM_VIRTUAL_TOKENS} ${SMALL_MODEL_ARGS} ++seed=${SEED}"
    # log file name (replaces / with _ to avoid directory confusion)
    log_name="seed${SEED}_generalization_sim_mini_exo_cleakage_small_model_virtual_tokens_${NUM_VIRTUAL_TOKENS}_${exp_config//\//_}"
    # run_and_log "$experiment" "$log_name"
    echo -e "$experiment" "\nLOG NAME: " "$log_name"
  done
}

for ((i = 1; i <= NUM_RUNS; i++)); do
  echo "========== RUN $i =========="
  run_one_round
  echo "============================"
  echo
done

# bash scripts/5_june_generalization_sim_mini_exo_cleakage_small_model.sh 1 42460 0 cleakage_mini_exo/vanilla
# bash scripts/5_june_generalization_sim_mini_exo_cleakage_small_model.sh 1 22248 0 cleakage_mini_exo/vanilla
# bash scripts/5_june_generalization_sim_mini_exo_cleakage_small_model.sh 1 26199 0 cleakage_mini_exo/vanilla

# bash scripts/5_june_generalization_sim_mini_exo_cleakage_small_model.sh 1 42460 1 cleakage_mini_exo/two_layer_mlp
# bash scripts/5_june_generalization_sim_mini_exo_cleakage_small_model.sh 1 22248 1 cleakage_mini_exo/two_layer_mlp
# bash scripts/5_june_generalization_sim_mini_exo_cleakage_small_model.sh 1 26199 1 cleakage_mini_exo/two_layer_mlp

# bash scripts/5_june_generalization_sim_mini_exo_cleakage_small_model.sh 1 42460 2 cleakage_mini_exo/two_layer_mlp
# bash scripts/5_june_generalization_sim_mini_exo_cleakage_small_model.sh 1 22248 2 cleakage_mini_exo/two_layer_mlp
# bash scripts/5_june_generalization_sim_mini_exo_cleakage_small_model.sh 1 26199 2 cleakage_mini_exo/two_layer_mlp

# bash scripts/5_june_generalization_sim_mini_exo_cleakage_small_model.sh 1 42460 3 cleakage_mini_exo/two_layer_mlp
# bash scripts/5_june_generalization_sim_mini_exo_cleakage_small_model.sh 1 22248 3 cleakage_mini_exo/two_layer_mlp
# bash scripts/5_june_generalization_sim_mini_exo_cleakage_small_model.sh 1 26199 3 cleakage_mini_exo/two_layer_mlp
