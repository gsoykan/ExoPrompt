import itertools
import json
import os
from typing import Dict, Any

import pandas as pd
import scipy.io
import h5py

from src.data.greenlight_utils.read_climate_model_gt_data import (
    read_climate_model_simulation_csv_data,
)
from src.utils.custom import read_json_file


def read_all_json_files(base_dir):
    """
    Reads all JSON files under the given folder and its subfolders.

    Args:
        base_dir (str): The base directory to search for JSON files.

    Returns:
        dict: A dictionary where the keys are file paths and the values are the JSON content.
    """
    json_data = {}

    # Walk through all directories and subdirectories
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".json"):  # Check for JSON files
                file_path = os.path.join(root, file)
                try:
                    # Open and load JSON file
                    with open(file_path, "r", encoding="utf-8") as f:
                        json_data[file_path] = json.load(f)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    return json_data


def compare_json_keys(first, second):
    """
    Compares the keys of two JSON objects and finds which keys are missing in each.

    :param first: dict, the first JSON object
    :param second: dict, the second JSON object
    :return: dict, keys missing in each JSON object
    """
    missing_in_first = [key for key in second.keys() if key not in first]
    missing_in_second = [key for key in first.keys() if key not in second]

    return {
        "missing_in_first": missing_in_first,
        "missing_in_second": missing_in_second,
    }


def compare_json_values(first, second):
    """
    Compares the values of two JSON objects that have the same keys.

    :param first: dict, the first JSON object
    :param second: dict, the second JSON object
    :return: dict, keys with differing values and their respective values in both JSON objects
    """
    differing_values = {}
    for key in first.keys():
        if key in second:
            if first[key] != second[key]:
                differing_values[key] = {"1": first[key], "2": second[key]}
    return differing_values


def read_mat_file(file_path):
    """
    Reads a .mat file and returns the data.

    :param file_path: str, path to the .mat file
    :return: dict, content of the .mat file
    """
    try:
        data = scipy.io.loadmat(file_path)
        return data
    except FileNotFoundError:
        print(f"The file at {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


def read_mat_file_alternative(file_path):
    """
    Reads a .mat file and returns the data.

    :param file_path: str, path to the .mat file
    :return: dict, content of the .mat file
    """
    try:
        data = scipy.io.loadmat(file_path, struct_as_record=False, squeeze_me=True)
        return data
    except FileNotFoundError:
        print(f"The file at {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


def read_hdf5_mat_file(file_path):
    """
    Reads an HDF5 .mat file and returns the data.

    :param file_path: str, path to the .mat file
    :return: dict, content of the .mat file
    """
    try:
        with h5py.File(file_path, "r") as file:
            data = {}

            def recursively_extract_data(name, object_):
                if isinstance(object_, h5py.Dataset):
                    data[name] = object_[()]
                elif isinstance(object_, h5py.Group):
                    data[name] = {
                        k: recursively_extract_data(k, v) for k, v in object_.items()
                    }
                return data

            recursively_extract_data("/", file["/"])

        return data
    except FileNotFoundError:
        print(f"The file at {file_path} was not found.")
    except OSError as e:
        print(f"An error occurred while reading the HDF5 file: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


def save_all_params_to_csv_for_inspection(
    all_json_files_dict: Dict[str, Dict[str, Any]], output_csv_path: str
) -> pd.DataFrame:
    data_rows = []
    for path, params in all_json_files_dict.items():
        row = {"path": path}  # Add the path as a column
        row.update(params)  # Add the parameters
        data_rows.append(row)

    # Create a DataFrame
    df = pd.DataFrame(data_rows)

    # Save the DataFrame to a CSV file
    df.to_csv(output_csv_path, index=False)

    print(f"CSV file written to: {output_csv_path}")

    return df


def compute_param_stats(df: pd.DataFrame):
    parameters_df = df.drop(columns=["path"])

    # Identify constant (unchanging) and changing parameters
    constant_parameters = []
    changing_parameters = []

    for column in parameters_df.columns:
        if (
            parameters_df[column].nunique() == 1
        ):  # Check if the column has only one unique value
            constant_parameters.append(column)
        else:
            changing_parameters.append(column)

    # Calculate standard deviation for changing parameters
    std_deviation = parameters_df[changing_parameters].std()

    # Create scaling ranges (min and max for each parameter)
    scaling_ranges = {
        column: (parameters_df[column].min(), parameters_df[column].max())
        for column in parameters_df.columns
    }

    # Convert scaling ranges to a DataFrame for easy viewing
    scaling_ranges_df = pd.DataFrame.from_dict(
        scaling_ranges, orient="index", columns=["min", "max"]
    )

    # Save outputs to CSVs
    constant_parameters_df = pd.DataFrame(
        constant_parameters, columns=["Constant Parameters"]
    )
    constant_parameters_df.to_csv("constant_parameters.csv", index=False)

    changing_parameters_df = pd.DataFrame(
        {"Parameter": changing_parameters, "Standard Deviation": std_deviation}
    )
    changing_parameters_df.to_csv("changing_parameters.csv", index=False)

    scaling_ranges_df.to_csv("parameter_scaling_ranges.csv")

    # Print summary
    print("Constant parameters saved to: constant_parameters.csv")
    print("Changing parameters (with std) saved to: changing_parameters.csv")
    print("Parameter scaling ranges saved to: parameter_scaling_ranges.csv")


if __name__ == "__main__":
    # read all new world sim parameters to calculate ranges
    all_new_world_sim_base_dir = "/Users/gsoykan/Desktop/yanan-desktop/wur-phd-2024/GreenLight/Output/new_world_sim_csv"
    all_new_world_json_files = read_all_json_files(all_new_world_sim_base_dir)
    all_new_world_params_df = save_all_params_to_csv_for_inspection(
        all_new_world_json_files, "all_new_world_params.csv"
    )
    print(all_new_world_params_df)
    compute_param_stats(all_new_world_params_df)

    all_old_world_sim_base_dir = (
        "/Users/gsoykan/Desktop/yanan-desktop/wur-phd-2024/GreenLight/Output/csv_output"
    )
    all_old_world_json_files = read_all_json_files(all_old_world_sim_base_dir)
    # all params have 242
    all_params = set(
        itertools.chain(*[list(j.keys()) for j in all_old_world_json_files.values()])
    )

    # checking the parameter differences between world_sim and eval_climate
    world_sim_params_json_path = "/Users/gsoykan/Desktop/yanan-desktop/wur-phd-2024/GreenLight/Output/csv_output/ams_hps_referenceSetting.json"
    current_gl_world_sim_params_json_path = "/Users/gsoykan/Desktop/yanan-desktop/wur-phd-2024/GreenLight/Output/referenceSetting/ams_hps_referenceSetting_params.json"
    evaluate_climate_model_hps_params_json_path = "/Users/gsoykan/Desktop/yanan-desktop/wur-phd-2024/GreenLight/Code/runScenarios/climate_model_hps_params.json"

    sample_world_sim_params = read_json_file(world_sim_params_json_path)
    current_sample_world_sim_params = read_json_file(
        current_gl_world_sim_params_json_path
    )
    evaluate_climate_model_hps_params = read_json_file(
        evaluate_climate_model_hps_params_json_path
    )
    param_key_diffs_world_vs_climate = compare_json_keys(
        sample_world_sim_params, evaluate_climate_model_hps_params
    )
    param_key_diffs_current_world_vs_climate = compare_json_keys(
        current_sample_world_sim_params, evaluate_climate_model_hps_params
    )
    param_key_diffs_old_vs_current_world = compare_json_keys(
        sample_world_sim_params, current_sample_world_sim_params
    )
    print(
        param_key_diffs_world_vs_climate,
        param_key_diffs_current_world_vs_climate,
        param_key_diffs_old_vs_current_world,
    )

    hps_19_params_json_path = "/Users/gsoykan/Desktop/yanan-desktop/wur-phd-2024/GreenLight/Output/referenceSetting/hps_19_params.json"
    hps_23_params_json_path = "/Users/gsoykan/Desktop/yanan-desktop/wur-phd-2024/GreenLight/Output/referenceSetting/hps_23_params.json"
    hps_19_params = read_json_file(hps_19_params_json_path)
    hps_23_params = read_json_file(hps_23_params_json_path)
    print(hps_19_params, hps_23_params)

    # this seems to be working fine!
    sample_simulation_csv_path = "/Users/gsoykan/Desktop/yanan-desktop/wur-phd-2024/GreenLight/Output/csv_output/ams_hps_referenceSetting.csv"
    dfs_by_component = read_climate_model_simulation_csv_data(
        sample_simulation_csv_path
    )
    print(dfs_by_component)

    evaluate_climate_model_hps_params_json_path = "/Users/gsoykan/Desktop/yanan-desktop/wur-phd-2024/GreenLight/Code/runScenarios/climate_model_hps_params.json"
    evaluate_climate_model_led_params_json_path = "/Users/gsoykan/Desktop/yanan-desktop/wur-phd-2024/GreenLight/Code/runScenarios/climate_model_led_params.json"
    data_hps = read_json_file(evaluate_climate_model_hps_params_json_path)
    data_led = read_json_file(evaluate_climate_model_led_params_json_path)
    differing_values = compare_json_values(data_hps, data_led)
    print(differing_values)

    sample_simulation_path = "/Users/gsoykan/Downloads/Data from_ Energy savings in greenhouses by transition from high-pressure sodium to LED lighting_1_all/Data for Katzin Energy savings in greenhouses/referenceSetting/ams_hps_referenceSetting.mat"
    sample_simulation_mat_alternative = read_mat_file_alternative(
        sample_simulation_path
    )
    sample_simulation_mat_h5 = read_hdf5_mat_file(sample_simulation_path)
    sample_simulation_mat = read_mat_file(sample_simulation_path)
    print(sample_simulation_path)
