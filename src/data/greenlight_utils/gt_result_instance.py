from dataclasses import dataclass
from typing import Optional


@dataclass
class GTResultInstance:
    experiment_type: str
    parameter_json_path: str
    gt_result_csv_path: str
    sim_result_csv_path: Optional[str]
    # TODO: @gsoykan - we might need to expand this to hold train / test split...
