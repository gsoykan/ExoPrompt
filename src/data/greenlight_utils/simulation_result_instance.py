from dataclasses import dataclass


@dataclass
class SimulationResultInstance:
    simulation_type: str
    parameter_json_path: str
    result_csv_path: str
    # TODO: @gsoykan - we might need to expand this to hold train / test split...
