import json
import os
from collections import defaultdict

import numpy as np

from src.utils.greenlight_scaler import GreenlightScaler

import random
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()


def generate_exp_configs_based_on_world_sim_params(
    num_generations: int = 32, output_file: str = "synth_configs.json"
):
    # Heat Adjustment
    heat_adjustment = {"paramNames": ["heatCorrection"], "paramVals": [1]}

    # More Light Hours
    more_light_hours = {
        "paramNames": ["lampsOffSun", "lampRadSumLimit"],
        "paramVals": [600, 14],
    }

    # Colder
    colder = {"paramNames": ["tSpNight", "tSpDay"], "paramVals": [16.5, 17.5]}

    # Warmer
    warmer = {"paramNames": ["tSpNight", "tSpDay"], "paramVals": [20.5, 21.5]}

    # Low Insulation
    low_insulation = {"paramNames": ["cLeakage", "hRf"], "paramVals": [2e-4, 2e-3]}

    # High Insulation
    high_insulation = {"paramNames": ["cLeakage", "hRf"], "paramVals": [0.5e-4, 8e-3]}

    # PPFD 100
    ppfd_100 = {
        "paramNames": [
            "tauLampPar",
            "tauLampNir",
            "tauLampFir",
            "aLamp",
            "thetaLampMax",
            "capLamp",
            "cHecLampAir",
        ],
        "paramValsHps": [0.99, 0.99, 0.99, 0.01, 100 / 1.8, 50, 0.045],
        "paramValsLed": [0.99, 0.99, 0.99, 0.01, 100 / 3, 5, 1.15],
    }

    # PPFD 400
    ppfd_400 = {
        "paramNames": [
            "tauLampPar",
            "tauLampNir",
            "tauLampFir",
            "aLamp",
            "thetaLampMax",
            "capLamp",
            "cHecLampAir",
        ],
        "paramValsHps": [0.96, 0.96, 0.96, 0.04, 400 / 1.8, 200, 0.18],
        "paramValsLed": [0.96, 0.96, 0.96, 0.04, 400 / 3, 20, 4.6],
    }

    exp_dicts = [
        heat_adjustment,
        more_light_hours,
        colder,
        warmer,
        low_insulation,
        high_insulation,
        ppfd_100,
        ppfd_400,
    ]

    params_values = defaultdict(list)
    for exp in exp_dicts:
        param_names = exp["paramNames"]
        for k, v in exp.items():
            if k != "paramNames":
                values = exp[k]
                for param_name, value in zip(param_names, values):
                    params_values[param_name].append(value)

    print(params_values)

    scaler = GreenlightScaler()
    gl_param_scaling_ranges = scaler.parameter_scaling_ranges
    params = list(params_values.keys())
    params_scaling_ranges = {}
    for param in params:
        params_scaling_ranges[param] = gl_param_scaling_ranges[param]

    print(params_scaling_ranges)

    # generate random initial configs
    num_configs_to_generate = num_generations
    # name - generated config
    generated_configs = {}
    for i in range(num_configs_to_generate):
        generated_config = {}
        for param in params:
            active_range = params_scaling_ranges[param]
            generated_value = random.uniform(active_range[0], active_range[1])
            generated_config[param] = generated_value
        # special treatment for tSpNight, tSpDay
        generated_config["tSpNight"] = random.uniform(
            params_values["tSpNight"][0], params_values["tSpNight"][1]
        )
        generated_config["tSpDay"] = generated_config["tSpNight"] + 1
        generated_configs[f"synth_scenario_{i}"] = generated_config

        # Save to JSON file
    with open(output_file, "w") as f:
        json.dump(generated_configs, f, indent=2)

    print(f"{num_generations} experiment configs saved to {output_file}")


def generate_exp_configs_for_c_leakage(
    num_generations: int = 100,
    output_file: str = "c_leakage_configs.json",
    add_high_low_insulation_values: bool = True,
):
    scaler = GreenlightScaler()
    gl_param_scaling_ranges = scaler.parameter_scaling_ranges
    params = {"cLeakage"}
    params_scaling_ranges = {}
    for param in params:
        params_scaling_ranges[param] = gl_param_scaling_ranges[param]

    print(params_scaling_ranges)

    # generate random initial configs
    num_configs_to_generate = num_generations

    active_range = params_scaling_ranges[param]

    cLeakageValues = np.linspace(
        active_range[0], active_range[1], num=num_configs_to_generate
    )
    cLeakageValues = set(cLeakageValues)

    # for debugging
    # (np.linspace(active_range[0], active_range[1], num=num_configs_to_generate) - active_range[0]) / (active_range[1] - active_range[0] )

    # low - high insulation values
    # 2e-4, 0.5e-4 (5e-5)
    # add these to values we can use them for testing
    if add_high_low_insulation_values:
        cLeakageValues.add(2e-4)
        cLeakageValues.add(0.5e-4)

    cLeakageValues = sorted(list(cLeakageValues))

    # name - generated config
    generated_configs = {}
    for i, generated_value in enumerate(cLeakageValues):
        generated_config = {}
        generated_config[param] = generated_value
        generated_configs[f"cleakage_scenario_{i}"] = generated_config

        # Save to JSON file
    with open(output_file, "w") as f:
        json.dump(generated_configs, f, indent=2)

    print(f"cLeakage =>  {num_generations} experiment configs saved to {output_file}")


def read_all_json_files(root_dir):
    json_data = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".json"):
                file_path = os.path.join(dirpath, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        json_data.append((file_path, data))  # Keep file path and data
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    return json_data


def find_world_sim_changing_exo_params():
    base_dir = os.environ.get("BASE_DIR")
    new_world_sim_dir = os.path.join(
        base_dir, "data/from_david_by_gurkan/new_world_sim"
    )
    all_json = read_all_json_files(new_world_sim_dir)

    all_json_not_ref = list(filter(lambda x: "referenceSetting" not in x[0], all_json))
    all_json_ref = list(filter(lambda x: "referenceSetting" in x[0], all_json))

    def get_changing_exo_params(jsons):
        all_params = defaultdict(list)
        for path, data in tqdm(jsons, desc="reading exo params..."):
            for k, v in data.items():
                all_params[k].append(v)

        all_params_set = {}
        for k, v in all_params.items():
            all_params_set[k] = set(v)

        changing_params = {k: v for k, v in all_params_set.items() if len(v) != 1}
        changing_param_names = list(changing_params.keys())
        print(f"changing params:\n{changing_param_names}")
        return changing_params, changing_param_names

    all_json_changing_params, all_json_changing_param_names = get_changing_exo_params(
        all_json
    )
    all_json_ref_changing_params, all_json_ref_changing_param_names = (
        get_changing_exo_params(all_json_ref)
    )
    all_json_not_ref_changing_params, all_json_not_ref_changing_param_names = (
        get_changing_exo_params(all_json_not_ref)
    )

    print(
        set(all_json_ref_changing_param_names)
        - set(all_json_not_ref_changing_param_names)
    )


if __name__ == "__main__":
    # generating configs that only updates all world sim params...
    # num_generations = 32
    # generate_exp_configs_based_on_world_sim_params(num_generations)

    # find_world_sim_changing_exo_params()
    # generate_exp_configs_for_c_leakage(num_generations=20)
    generate_exp_configs_for_c_leakage(
        num_generations=40,
        output_file="c_leakage_40_configs.json",
        add_high_low_insulation_values=False,
    )
