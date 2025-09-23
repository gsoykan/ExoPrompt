#!/bin/bash

root_dir="$1"
output_file="sim_metrics_summary.csv"

declare -A key_map=(
    ["test/sim_rrmse_tAir"]="test/rrmse_tAir"
    ["test/sim_rrmse_co2Air"]="test/rrmse_co2Air"
    ["test/sim_rrmse_rh"]="test/rrmse_rh"
    ["test/sim_rrmse_vpAir"]="test/rrmse_vpAir"
    ["test/sim_rmse_tAir"]="test/rmse_tAir"
    ["test/sim_rmse_co2Air"]="test/rmse_co2Air"
    ["test/sim_rmse_rh"]="test/rmse_rh"
    ["test/sim_rmse_vpAir"]="test/rmse_vpAir"
    ["test/sim_me_tAir"]="test/me_tAir"
    ["test/sim_me_co2Air"]="test/me_co2Air"
    ["test/sim_me_rh"]="test/me_rh"
    ["test/sim_me_vpAir"]="test/me_vpAir"
)

ordered_keys=(
    "test/rrmse_tAir" "test/rrmse_co2Air" "test/rrmse_rh" "test/rrmse_vpAir"
    "test/rmse_tAir" "test/rmse_co2Air" "test/rmse_rh" "test/rmse_vpAir"
    "test/me_tAir" "test/me_co2Air" "test/me_rh" "test/me_vpAir"
)

# Write CSV header
{
    echo -n "filename"
    for key in "${ordered_keys[@]}"; do echo -n ",$key"; done
    echo
} > "$output_file"

# Process each file
find "$root_dir" -type f -name "*with_exoprompt_1m*" -name "*physinet_num_features=3*" | while read -r file; do
    declare -A extracted
    filename=$(basename "$file")
    content=$(tail -n 100 "$file" | sed 's/\x1B\[[0-9;]*[JKmsu]//g')

    for sim_key in "${!key_map[@]}"; do
        value=$(echo "$content" | grep -F "$sim_key" | awk '{print $NF}')
        extracted["${key_map[$sim_key]}"]="$value"
    done

    {
        echo -n "\"$filename\""
        for key in "${ordered_keys[@]}"; do
            val="${extracted[$key]:-NaN}"
            echo -n ",$val"
        done
        echo
    } >> "$output_file"
done

echo "✅ CSV saved to $output_file"