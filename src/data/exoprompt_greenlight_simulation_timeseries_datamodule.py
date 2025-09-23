import itertools
import os
import random
from collections import defaultdict
from functools import partial
from typing import Any, Dict, Optional, List

import numpy as np
import torch.utils.data
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset

from src.data.components.exoprompt_greenlight_simulation_timeseries_dataset import (
    ExoPromptGreenlightSimulationTimeSeriesDataset,
)
from src.data.greenlight_utils.simulation_result_instance import (
    SimulationResultInstance,
)
from src.utils.custom import split_dataset_sequentially
from src.utils.pickle_helper import PickleHelper

from multiprocessing import Pool


def process_instance(result_instance, partial_dataset, train_val_test_split):
    """Helper function to process a single result_instance."""
    single_dataset = partial_dataset(simulation_result_instances=[result_instance])
    return split_dataset_sequentially(
        single_dataset, train_val_test_ratio=train_val_test_split
    )


# TODO: @gsoykan - maybe we can create special data structure to handle each param - csv pair?
class ExoPromptGreenLightSimulationTimeSeriesDataModule(LightningDataModule):
    def __init__(
        self,
        root_path,
        # TODO: @gsoykan - make documentation for this... (maybe turn that into a dict or sth)
        experiment_config: Dict,  # has "type"
        exo_params_len: int = 241,  # 242 is for old_world_simulation
        exo_params_to_take: Optional[
            List[str]
        ] = None,  # only to process certain exo params
        return_random_exo_params: bool = False,
        size=(96, 48, 96),
        features="M",
        target="OT",  # redundant in our context
        scale=True,
        timeenc=1,
        freq="t",
        # s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h
        output_feature_idx=(7, 8, 9),
        use_greenlight_scaler: bool = True,
        return_all_output_features: bool = False,
        train_subset_len: Optional[int] = None,
        val_subset_len: Optional[int] = None,
        test_subset_len: Optional[int] = None,
        # loader configs
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """Initialize a DataModule.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split_rate: The train, validation and test split. Defaults to `(0.8, 0.1, 0.1)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = (
                self.hparams.batch_size // self.trainer.world_size
            )

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            random.seed(42)
            experiment_config: Dict[str, Any] = self.hparams.experiment_config
            experiment_type = experiment_config["type"]
            max_train_simulations = experiment_config.get("max_train_simulations", None)
            max_val_simulations = experiment_config.get("max_val_simulations", None)
            max_test_simulations = experiment_config.get("max_test_simulations", None)
            simulation_types = {
                "highInsulation",
                "ppfd100",
                "lowInsulation",
                "heatAdjustment",
                "referenceSetting",
                "moreLightHours",
                "warmer",
                "colder",
                "ppfd400",
            }
            # generated by rand. sampling of
            # 24 world sim parameters
            synth_simulation_types = {
                "synth_scenario_1",
                "synth_scenario_2",
                "synth_scenario_3",
                "synth_scenario_4",
                "synth_scenario_5",
                "synth_scenario_6",
                # "synth_scenario_7",
                "synth_scenario_8",
                "synth_scenario_9",
                "synth_scenario_10",
                "synth_scenario_11",
                "synth_scenario_12",
                "synth_scenario_13",
                "synth_scenario_14",
                "synth_scenario_15",
                "synth_scenario_16",
                "synth_scenario_17",
            }

            c_leakage_simulation_types = set(
                [f"cleakage_scenario_{i}" for i in range(22)]
            )

            # cleakage 40 simulations
            c_leakage_40_simulation_types = set(
                [f"cleakage_scenario_{i}" for i in range(40)]
            )
            c_leakage_40_simulation_types.add("cleakage_scenario_high_insulation")
            c_leakage_40_simulation_types.add("cleakage_scenario_low_insulation")

            # cleakage GT - LED conditions
            c_leakage_gt_led_simulation_types = set(
                [f"cleakage_scenario_{i}" for i in range(22)]
            )
            c_leakage_gt_led_types = {"gt"}.union(c_leakage_gt_led_simulation_types)

            sim_results_by_type = defaultdict(list)
            for root, _, files in os.walk(self.hparams.root_path):
                for file in files:
                    if file.endswith(".json"):  # Check for JSON files
                        simulation_type = root.split("/")[-1]
                        csv_file_path = os.path.join(
                            root, file.replace(".json", ".csv")
                        )
                        json_file_path = os.path.join(root, file)
                        assert os.path.exists(
                            csv_file_path
                        ), f"{csv_file_path} does not exist."
                        assert os.path.exists(
                            json_file_path
                        ), f"{json_file_path} does not exist."
                        sim_result = SimulationResultInstance(
                            simulation_type, json_file_path, csv_file_path
                        )
                        sim_results_by_type[simulation_type].append(sim_result)

            partial_dataset = partial(
                ExoPromptGreenlightSimulationTimeSeriesDataset,
                exo_params_len=self.hparams.exo_params_len,
                exo_params_to_take=self.hparams.exo_params_to_take,
                size=self.hparams.size,
                features=self.hparams.features,
                scale=self.hparams.scale,
                timeenc=1,
                freq=self.hparams.freq,
                output_feature_idx=self.hparams.output_feature_idx,
                use_greenlight_scaler=self.hparams.use_greenlight_scaler,
                return_all_output_features=self.hparams.return_all_output_features,
                return_random_exo_params=self.hparams.return_random_exo_params,
            )

            def create_result_instances(sim_types):
                return list(
                    itertools.chain(*[sim_results_by_type[t] for t in sim_types])
                )

            match experiment_type:
                case "generalization_c_leakage_gt_conditions_finetune_on_gt":
                    # if fine-tune ratio is None then default ratio is 0.1
                    fine_tune_ratio = self.hparams.experiment_config.get(
                        "fine_tune_ratio", 0.1
                    )
                    assert (
                        0 <= fine_tune_ratio <= 0.8
                    ), "fine_tune_percentage must be between 0 and 80."
                    assert (
                        {"cLeakage"} == set(self.hparams.exo_params_to_take)
                    ), "for this experiment we should only have cLeakage exo_params to take"
                    val_ratio = 0.1
                    test_ratio = 1 - (fine_tune_ratio + val_ratio)
                    dataset_type = {"gt"}
                    gt_dataset = partial_dataset(
                        simulation_result_instances=create_result_instances(
                            dataset_type
                        )
                    )
                    self.data_train, self.data_val, self.data_test = (
                        split_dataset_sequentially(
                            gt_dataset,
                            train_val_test_ratio=(
                                fine_tune_ratio,
                                val_ratio,
                                test_ratio,
                            ),
                        )
                    )
                    print("ok")
                case "generalization_c_leakage_gt_conditions_test_on_gt":
                    # c_leakage_gt_led_types, c_leakage_gt_led_simulation_types
                    assert (
                        {"cLeakage"} == set(self.hparams.exo_params_to_take)
                    ), "for this experiment we should only have cLeakage exo_params to take"
                    test_types = {"gt"}
                    # cleakage_values (low - high insulation scen.) 5e-05, 2e-04
                    val_sim_types = {"cleakage_scenario_2", "cleakage_scenario_9"}
                    train_sim_types = c_leakage_gt_led_simulation_types - val_sim_types

                    self.data_train = partial_dataset(
                        simulation_result_instances=create_result_instances(
                            train_sim_types
                        )
                    )
                    self.data_val = partial_dataset(
                        simulation_result_instances=create_result_instances(
                            val_sim_types
                        )
                    )
                    self.data_test = partial_dataset(
                        simulation_result_instances=create_result_instances(test_types)
                    )
                case "generalization_c_leakage_half_train_half_test":
                    assert (
                        {"cLeakage"} == set(self.hparams.exo_params_to_take)
                    ), "for this experiment we should only have cLeakage exo_params to take"
                    test_sim_types = set(
                        [f"cleakage_scenario_{i}" for i in range(20, 40)]
                    )
                    train_val_sim_types = c_leakage_40_simulation_types - test_sim_types
                    val_sim_types = {
                        "cleakage_scenario_low_insulation",
                        "cleakage_scenario_high_insulation",
                    }
                    train_sim_types = train_val_sim_types - val_sim_types

                    self.data_train = partial_dataset(
                        simulation_result_instances=create_result_instances(
                            train_sim_types
                        )
                    )
                    self.data_val = partial_dataset(
                        simulation_result_instances=create_result_instances(
                            val_sim_types
                        )
                    )
                    self.data_test = partial_dataset(
                        simulation_result_instances=create_result_instances(
                            test_sim_types
                        )
                    )
                case "generalization_c_leakage":
                    assert (
                        {"cLeakage"} == set(self.hparams.exo_params_to_take)
                    ), "for this experiment we should only have cLeakage exo_params to take"
                    # cleakage_values (low - high insulation scen.) 5e-05, 2e-04
                    test_sim_types = {"cleakage_scenario_2", "cleakage_scenario_9"}
                    train_val_sim_types = c_leakage_simulation_types - test_sim_types
                    val_sim_types = {"cleakage_scenario_6", "cleakage_scenario_16"}
                    train_sim_types = train_val_sim_types - val_sim_types
                    # To see the effect of number of dataset involved in the process
                    c_leakage_train_dataset_count = self.hparams.experiment_config.get(
                        "c_leakage_train_dataset_count", None
                    )
                    if c_leakage_train_dataset_count is not None:
                        sorted_train_scenarios = sorted(
                            [(int(i.split("_")[-1]), i) for i in train_sim_types],
                            key=lambda x: x[0],
                        )

                        def pick_equally_spaced(data, k):
                            indices = np.linspace(0, len(data) - 1, k, dtype=int)
                            return [data[i] for i in indices]

                        picked_train_sim_types = pick_equally_spaced(
                            sorted_train_scenarios, c_leakage_train_dataset_count
                        )
                        train_sim_types = set(
                            map(lambda x: x[1], picked_train_sim_types)
                        )

                    self.data_train = partial_dataset(
                        simulation_result_instances=create_result_instances(
                            train_sim_types
                        )
                    )
                    self.data_val = partial_dataset(
                        simulation_result_instances=create_result_instances(
                            val_sim_types
                        )
                    )
                    self.data_test = partial_dataset(
                        simulation_result_instances=create_result_instances(
                            test_sim_types
                        )
                    )
                    # TODO: @gsoykan - think about "train_subset_len"
                    print("ok")
                case "generalization_train_24_synth_test_world_sim":
                    local_random = random.Random(42)
                    synth_simulation_types_list = list(synth_simulation_types)
                    val_synth_sim_types = local_random.sample(
                        synth_simulation_types_list, 2
                    )
                    train_synth_sim_types = synth_simulation_types - set(
                        val_synth_sim_types
                    )
                    # handle train and val
                    synth_result_instances_train = list(
                        itertools.chain(
                            *[sim_results_by_type[t] for t in train_synth_sim_types]
                        )
                    )
                    self.data_train = partial_dataset(
                        simulation_result_instances=synth_result_instances_train
                    )
                    synth_result_instances_val = list(
                        itertools.chain(
                            *[sim_results_by_type[t] for t in val_synth_sim_types]
                        )
                    )
                    # validation on a different dataset
                    # to prevent overfitting
                    self.data_val = partial_dataset(
                        simulation_result_instances=synth_result_instances_val
                    )
                    # handle test
                    world_sim_result_instances = list(
                        itertools.chain(
                            *[sim_results_by_type[t] for t in simulation_types]
                        )
                    )
                    self.data_test = partial_dataset(
                        simulation_result_instances=world_sim_result_instances
                    )
                case "generalization_train_world_sim_test_24_synth":
                    # handle train and val
                    world_sim_result_instances = list(
                        itertools.chain(
                            *[sim_results_by_type[t] for t in simulation_types]
                        )
                    )
                    self.data_train = partial_dataset(
                        simulation_result_instances=world_sim_result_instances
                    )
                    self.data_train, self.data_val = random_split(
                        dataset=self.data_train,
                        lengths=(0.9, 0.1),
                        generator=torch.Generator().manual_seed(42),
                    )
                    # handle test
                    synth_result_instances = list(
                        itertools.chain(
                            *[sim_results_by_type[t] for t in synth_simulation_types]
                        )
                    )
                    self.data_test = partial_dataset(
                        simulation_result_instances=synth_result_instances
                    )
                case "generalization_train_with_ref_test_with_ppfd_100_400":
                    training_sims = sim_results_by_type[
                        "referenceSetting"
                    ]  # 30 instance
                    if max_train_simulations is not None:
                        random.shuffle(training_sims)
                        training_sims = training_sims[:max_train_simulations]
                    self.data_train = partial_dataset(
                        simulation_result_instances=training_sims
                    )

                    # ppfd100, ppfd400
                    ppfd_sim_types = {"ppfd100", "ppfd400"}
                    ppfd_result_instances = list(
                        itertools.chain(
                            *[sim_results_by_type[t] for t in ppfd_sim_types]
                        )
                    )

                    self.data_train, self.data_val = random_split(
                        dataset=self.data_train,
                        lengths=(0.9, 0.1),
                        generator=torch.Generator().manual_seed(42),
                    )
                    # the test set contains instances in which their dataset and settings has not been seen before.
                    self.data_test = partial_dataset(
                        simulation_result_instances=ppfd_result_instances
                    )
                case "generalization_train_with_rest_test_with_ref":
                    # handle train and val
                    remaining_sim_types = simulation_types - {
                        "referenceSetting"
                    }  # 72 settings
                    remaining_result_instances = list(
                        itertools.chain(
                            *[sim_results_by_type[t] for t in remaining_sim_types]
                        )
                    )
                    self.data_train = partial_dataset(
                        simulation_result_instances=remaining_result_instances
                    )
                    self.data_train, self.data_val = random_split(
                        dataset=self.data_train,
                        lengths=(0.9, 0.1),
                        generator=torch.Generator().manual_seed(42),
                    )
                    # handle test
                    test_sims = sim_results_by_type["referenceSetting"]  # 30 instance
                    self.data_test = partial_dataset(
                        simulation_result_instances=test_sims
                    )
                case "reference_only" | "generalization_train_with_ref_test_with_rest":
                    # NOTE: this can show pretraining generalization performance
                    #  of the exoprompts (but not entirely)
                    # handle train dataset
                    training_sims = sim_results_by_type[
                        "referenceSetting"
                    ]  # 30 instance
                    if max_train_simulations is not None:
                        random.shuffle(training_sims)
                        training_sims = training_sims[:max_train_simulations]
                    self.data_train = partial_dataset(
                        simulation_result_instances=training_sims
                    )

                    # handle val and test
                    remaining_sim_types = simulation_types - {"referenceSetting"}
                    remaining_result_instances = list(
                        itertools.chain(
                            *[sim_results_by_type[t] for t in remaining_sim_types]
                        )
                    )

                    if experiment_type == "reference_only":
                        if any(
                            [
                                max_val_simulations,
                                max_test_simulations,
                            ]
                        ):
                            temp_num_val = (
                                max_val_simulations
                                if max_val_simulations is not None
                                else 0
                            )
                            temp_num_test = (
                                max_test_simulations
                                if max_test_simulations is not None
                                else 0
                            )
                            total_val_test_simulations = temp_num_val + temp_num_test
                            if total_val_test_simulations > len(
                                remaining_result_instances
                            ):
                                raise ValueError(
                                    f"The total number of simulations ({total_val_test_simulations}) is greater than the number of available simulations ({len(remaining_result_instances)})."
                                )
                            random.shuffle(remaining_result_instances)
                            remaining_result_instances = remaining_result_instances[
                                :total_val_test_simulations
                            ]

                        data_val_test_merged = partial_dataset(
                            simulation_result_instances=remaining_result_instances
                        )
                        self.data_val, self.data_test = random_split(
                            dataset=data_val_test_merged,
                            lengths=(0.34, 0.66),
                            generator=torch.Generator().manual_seed(42),
                        )
                    elif (
                        experiment_type
                        == "generalization_train_with_ref_test_with_rest"
                    ):
                        assert (
                            max_train_simulations is None
                        ), "max_train_simulations should be None for this experiment type."
                        assert (
                            max_val_simulations is None
                        ), "max_val_simulations should be None for this experiment type."
                        assert (
                            max_test_simulations is None
                        ), "max_test_simulations should be None for this experiment type."

                        # this is the main difference from the "reference_only" dataset
                        #  the validation comes from the reference setting as well
                        self.data_train, self.data_val = random_split(
                            dataset=self.data_train,
                            lengths=(0.9, 0.1),
                            generator=torch.Generator().manual_seed(42),
                        )
                        # the test set contains instances in which their dataset and settings has not been seen before.
                        self.data_test = partial_dataset(
                            simulation_result_instances=remaining_result_instances
                        )

                case "pretraining_mixed" | "pretraining_mixed_sequential":
                    # in this setting all simulation results should mix
                    # we can do two things
                    # 1) (pretraining_mixed): create single dataset with all instances and split that
                    # 2) (pretraining_mixed_sequential): create datasets for each experiment
                    #   and split those one by one train-val-test to merge at the end
                    all_result_instances = list(
                        itertools.chain(
                            *[sim_results_by_type[t] for t in simulation_types]
                        )
                    )

                    if any(
                        [
                            max_train_simulations,
                            max_val_simulations,
                            max_test_simulations,
                        ]
                    ):
                        temp_num_train = (
                            max_train_simulations
                            if max_train_simulations is not None
                            else 0
                        )
                        temp_num_val = (
                            max_val_simulations
                            if max_val_simulations is not None
                            else 0
                        )
                        temp_num_test = (
                            max_test_simulations
                            if max_test_simulations is not None
                            else 0
                        )
                        total_num_simulations = (
                            temp_num_train + temp_num_val + temp_num_test
                        )
                        if total_num_simulations > len(all_result_instances):
                            raise ValueError(
                                f"The total number of simulations ({total_num_simulations}) is greater than the number of available simulations ({len(all_result_instances)})."
                            )
                        random.shuffle(all_result_instances)
                        all_result_instances = all_result_instances[
                            :total_num_simulations
                        ]

                    # todo: @gsoykan - make this arg, do not leave magic strings...
                    train_val_test_split = (0.7, 0.1, 0.2)
                    if experiment_type == "pretraining_mixed_sequential":
                        # Parallelized processing
                        with Pool(processes=self.hparams.num_workers or 1) as pool:
                            process_fn = partial(
                                process_instance,
                                partial_dataset=partial_dataset,
                                train_val_test_split=train_val_test_split,
                            )
                            results = pool.map(process_fn, all_result_instances)

                        # Separate train, val, and test sets
                        train_sets, val_sets, test_sets = zip(*results)

                        self.data_train = ConcatDataset(train_sets)
                        self.data_val = ConcatDataset(val_sets)
                        self.data_test = ConcatDataset(test_sets)
                    elif experiment_type == "pretraining_mixed":
                        data_train_val_test_merged = partial_dataset(
                            simulation_result_instances=all_result_instances
                        )
                        self.data_train, self.data_val, self.data_test = random_split(
                            dataset=data_train_val_test_merged,
                            lengths=train_val_test_split,
                            generator=torch.Generator().manual_seed(42),
                        )
                    else:
                        raise ValueError(f"Unknown experiment type: {experiment_type}")
                case _:
                    raise AssertionError(f"Unknown experiment type: {experiment_type}")

            if self.hparams.train_subset_len is not None:
                assert self.hparams.train_subset_len <= len(self.data_train), (
                    f"subset length should be less than the number of training samples. "
                    f"Currently, subset length: {self.hparams.train_subset_len} , dataset length: {len(self.data_train)} "
                )
                indices = torch.randperm(len(self.data_train))[
                    : self.hparams.train_subset_len
                ].tolist()
                self.data_train = torch.utils.data.Subset(self.data_train, indices)

            if self.hparams.val_subset_len is not None:
                assert self.hparams.val_subset_len <= len(self.data_val), (
                    f"subset length should be less than the number of validation samples. "
                    f"Currently, subset length: {self.hparams.val_subset_len} , dataset length: {len(self.data_val)} "
                )
                indices = torch.randperm(len(self.data_val))[
                    : self.hparams.val_subset_len
                ].tolist()
                self.data_val = torch.utils.data.Subset(self.data_val, indices)

            if self.hparams.test_subset_len is not None:
                assert self.hparams.test_subset_len <= len(self.data_test), (
                    f"subset length should be less than the number of test samples. "
                    f"Currently, subset length: {self.hparams.test_subset_len} , dataset length: {len(self.data_test)} "
                )
                indices = torch.randperm(len(self.data_test))[
                    : self.hparams.test_subset_len
                ].tolist()
                self.data_test = torch.utils.data.Subset(self.data_test, indices)

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            drop_last=True,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=True,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=True,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    datamodule = ExoPromptGreenLightSimulationTimeSeriesDataModule(
        root_path="/Users/gsoykan/Desktop/yanan-desktop/wur-phd-2024/physinet/data/from_david_by_gurkan/new_world_sim",
        experiment_config={
            "type": "pretraining_mixed",
            "max_train_simulations": 2,
            "max_val_simulations": 2,
            "max_test_simulations": 2,
        },
        output_feature_idx=None,
        exo_params_len=241,
        batch_size=4,
        train_subset_len=1000,
        val_subset_len=1000,
        test_subset_len=1000,
    )
    datamodule.prepare_data()
    datamodule.setup()
    dataloader = datamodule.train_dataloader()
    dataiter = iter(dataloader)
    batch = next(dataiter)
    PickleHelper.save_object(PickleHelper.exoprompt_old_world_timeseries_batch, batch)
    loaded_batch = PickleHelper.load_object(
        PickleHelper.exopromt_old_world_timeseries_batch
    )
    print(batch)
