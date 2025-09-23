import re
import csv
import os
from typing import Dict


def extract_best_ckpt_wandb_run(log_content) -> Dict:
    best_checkpoint = "Unknown"
    view_run_url = "Unknown"

    match_checkpoint = re.search(r"Best ckpt path: ([^\s]+)", log_content)
    if match_checkpoint:
        best_checkpoint = re.sub(r"\x1b\[[0-9;]*m", "", match_checkpoint.group(1))

    match_run_url = re.search(r"🚀 View run .* at: ([^\s]+)", log_content)
    if match_run_url:
        view_run_url = match_run_url.group(1)

    return {
        "best checkpoint": best_checkpoint,
        "view run url": view_run_url,
    }


def infer_metadata_from_filename(filename):
    filename = os.path.split(filename)[-1]

    exo_prompt_availability = "Unknown"
    pretraining_dataset_size = "Unknown"
    finetuning_dataset = "Unknown"
    physinet_availability = "Unknown"

    if "with_exoprompt" in filename:
        exo_prompt_availability = "Yes"
    elif "without_exoprompt" in filename:
        exo_prompt_availability = "No"

    match_pretraining = re.search(r"_(\d+k|\d+m|null)_", filename)
    if match_pretraining:
        pretraining_dataset_size = match_pretraining.group(1).replace("null", "None")

    if "finetuning" in filename:
        finetuning_dataset = "finetuning"
    elif "hps_only" in filename:
        finetuning_dataset = "hps"
    elif "led_only" in filename:
        finetuning_dataset = "led"

    if "physinet_mode=true" in filename:
        physinet_availability = "Yes"
    elif "physinet_mode=false" in filename:
        physinet_availability = "No"

    return {
        "ExoPrompt Availability": exo_prompt_availability,
        "Pretraining Dataset Size": pretraining_dataset_size,
        "Finetuning Dataset": finetuning_dataset,
        "PhysiNet Availability": physinet_availability,
    }


def extract_run_summary(
    log_file_path: str, extract_rmse_and_me: bool = False, only_test_vars: bool = False
) -> Dict:
    """

    Args:
        only_test_vars:
        log_file_path:
        extract_rmse_and_me: flag to extract test RMSE and ME from log file

    Returns:

    """
    run_summary_pattern = r"wandb: Run summary:(.*?)wandb: 🚀 View run"

    metrics_to_extract = [
        "test/rrmse_tAir",
        "test/rrmse_co2Air",
        "test/rrmse_rh",
        "test/rrmse_vpAir",
        "train/rrmse_tAir",
        "train/rrmse_co2Air",
        "train/rrmse_rh",
        "train/rrmse_vpAir",
        "val/rrmse_tAir",
        "val/rrmse_co2Air",
        "val/rrmse_rh",
        "val/rrmse_vpAir",
    ]

    if extract_rmse_and_me:
        metrics_to_extract = (
            metrics_to_extract
            + [
                "test/rmse_tAir",
                "test/rmse_co2Air",
                "test/rmse_rh",
                "test/rmse_vpAir",
            ]
            + [
                "test/me_tAir",
                "test/me_co2Air",
                "test/me_rh",
                "test/me_vpAir",
            ]
        )

    if only_test_vars:
        metrics_to_extract = list(
            filter(lambda m: m.startswith("test/"), metrics_to_extract)
        )

    with open(log_file_path, "r") as file:
        log_content = file.read()

    run_summary_match = re.search(run_summary_pattern, log_content, re.DOTALL)
    if not run_summary_match:
        print(f"Run summary section not found in {log_file_path}")
        return None

    run_summary_content = run_summary_match.group(1)

    metrics = {}
    for metric in metrics_to_extract:
        pattern = rf"wandb:\s+{re.escape(metric)}\s+([\-\d.e]+)"
        match = re.search(pattern, run_summary_content)
        if match:
            metrics[metric] = float(match.group(1))
        else:
            metrics[metric] = None

    file_metadata = infer_metadata_from_filename(log_file_path)
    best_ckpt_and_wandb_run = extract_best_ckpt_wandb_run(log_content)

    return {
        "filename": os.path.split(log_file_path)[-1],
        **file_metadata,
        **metrics,
        **best_ckpt_and_wandb_run,
    }


def process_all_logs_in_folder(
    folder_path,
    output_csv_path,
    *,
    extract_rmse_and_me: bool = False,
    only_test_vars: bool = False,
):
    log_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(".log")
    ]

    if not log_files:
        print(f"No log files found in folder: {folder_path}")
        return

    # sort them by the pretraining dataset size
    try:
        # for finetuning experiments, e.g., physinet21_01_2025_finetuning_with_exo
        log_files = sorted(
            log_files,
            key=lambda x: int(
                os.path.split(x)[-1]
                .split("_")[2]
                .replace("k", "000")
                .replace("m", "000000")
                .replace("null", "0")
            ),
        )
    except ValueError:
        try:
            # for physinet14_01_2025scalingexperimentresults
            log_files = sorted(
                log_files,
                key=lambda x: (
                    os.path.split(x)[-1].replace(".log", "").split("_")[-3],
                    int(
                        os.path.split(x)[-1]
                        .replace(".log", "")
                        .split("_")[-1]
                        .replace("k", "000")
                        .replace("m", "000000")
                        .replace("null", "0")
                    ),
                ),
            )
        except ValueError:
            # for the rest sort by filenames
            log_files = sorted(log_files, key=lambda x: os.path.split(x)[-1])

    all_data = []
    for log_file in log_files:
        print(f"Processing {log_file}")
        data = extract_run_summary(
            log_file,
            extract_rmse_and_me=extract_rmse_and_me,
            only_test_vars=only_test_vars,
        )
        if data:
            all_data.append(data)

    if not all_data:
        print("No valid data extracted from the logs.")
        return

    # Write data to CSV
    with open(output_csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=all_data[0].keys())
        writer.writeheader()
        writer.writerows(all_data)

    print(f"Data successfully saved to {output_csv_path}.")


if __name__ == "__main__":
    input_folder = "/Users/gsoykan/Desktop/yanan-desktop/wur-phd-2024/physinet/scripts/final_logs/snellius_generalization_sim_cleakage_num_datasets_18_per_cleakage"
    output_csv = "generalization_sim_cleakage_num_datasets_18_per_cleakage.csv"

    process_all_logs_in_folder(
        input_folder, output_csv, extract_rmse_and_me=True, only_test_vars=True
    )
