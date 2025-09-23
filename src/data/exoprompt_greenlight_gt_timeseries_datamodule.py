import os
import random
from functools import partial
from typing import Any, Dict, Optional, Tuple

import torch.utils.data
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split

from src.data.components.exoprompt_greenlight_gt_timeseries_dataset import (
    ExoPromptGreenlightGTTimeSeriesDataset,
)
from src.data.greenlight_utils.gt_result_instance import GTResultInstance
from src.utils.custom import (
    split_dataset_sequentially,
    get_sequential_slice_from_dataset,
    split_dataset_sequentially_into_two,
)
from src.utils.pickle_helper import PickleHelper


class ExoPromptGreenLightGTTimeSeriesDataModule(LightningDataModule):
    def __init__(
        self,
        root_path,
        # TODO: @gsoykan - make documentation for this... (maybe turn that into a dict or sth)
        experiment_config: Dict,  # has "type"
        exo_params_len: int = 254,  # 241 is for old_world_simulation
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
        discarded_features: Optional[Tuple[str, ...]] = ("sideLee", "sideWind"),
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
            # root_path is going to be sth like => /Users/gsoykan/Desktop/yanan-desktop/wur-phd-2024/physinet/data/from_david_by_gurkan/gt
            # todo: @gsoykan: you need to take these paths out of datamodule - maybe put them in experiment config!
            hps_paths = {
                "parameter_json_path": os.path.join(
                    self.hparams.root_path, "hps/climate_model_hps_params.json"
                ),
                "gt_result_csv_path": os.path.join(
                    self.hparams.root_path, "hps/gt_hps_timeseries.csv"
                ),
                "sim_result_csv_path": os.path.join(
                    self.hparams.root_path, "hps/climateModel_hps_manuscriptParams.csv"
                ),
            }
            led_paths = {
                "parameter_json_path": os.path.join(
                    self.hparams.root_path, "led/climate_model_led_params.json"
                ),
                "gt_result_csv_path": os.path.join(
                    self.hparams.root_path, "led/gt_led_timeseries.csv"
                ),
                "sim_result_csv_path": os.path.join(
                    self.hparams.root_path, "led/climateModel_led_manuscriptParams.csv"
                ),
            }
            hps_result_instance = GTResultInstance(experiment_type="hps", **hps_paths)
            led_result_instance = GTResultInstance(experiment_type="led", **led_paths)

            partial_dataset = partial(
                ExoPromptGreenlightGTTimeSeriesDataset,
                exo_params_len=self.hparams.exo_params_len,
                size=self.hparams.size,
                features=self.hparams.features,
                scale=self.hparams.scale,
                timeenc=1,
                freq=self.hparams.freq,
                output_feature_idx=self.hparams.output_feature_idx,
                use_greenlight_scaler=self.hparams.use_greenlight_scaler,
                return_all_output_features=self.hparams.return_all_output_features,
                return_random_exo_params=self.hparams.return_random_exo_params,
                discarded_features=self.hparams.discarded_features,
            )

            hps_dataset = partial_dataset(gt_result_instances=[hps_result_instance])
            led_dataset = partial_dataset(gt_result_instances=[led_result_instance])

            match experiment_type:
                # led_only and hps_only (train with one - test with other)
                case "led_only":
                    # todo: @gsoykan - make length arg (0.95, 0.05)
                    #   do not leave magic strings...
                    self.data_train, self.data_val = random_split(
                        dataset=led_dataset,
                        lengths=(0.95, 0.05),
                        generator=torch.Generator().manual_seed(42),
                    )
                    self.data_test = hps_dataset
                case "hps_only":
                    self.data_train, self.data_val = random_split(
                        dataset=hps_dataset,
                        lengths=(0.95, 0.05),
                        generator=torch.Generator().manual_seed(42),
                    )
                    self.data_test = led_dataset
                case "finetuning_mixed":
                    train_val_test_split = (0.7, 0.1, 0.2)
                    hps_train, hps_val, hps_test = split_dataset_sequentially(
                        hps_dataset, train_val_test_split
                    )
                    led_train, led_val, led_test = split_dataset_sequentially(
                        led_dataset, train_val_test_split
                    )
                    self.data_train = ConcatDataset([hps_train, led_train])
                    self.data_val = ConcatDataset([hps_val, led_val])
                    self.data_test = ConcatDataset([hps_test, led_test])
                case "finetuning_mixed_preq_sld_bls":
                    # e.g., block number can be 0 and block rate can be 0.2
                    # in that case we get 0 - 0.2 and disregard the rest.
                    block_number = experiment_config.get("block_number", -1)
                    block_rate = experiment_config.get("block_rate", -1)
                    assert block_number != -1, "block_number must be provided."
                    assert block_rate != -1, "block_rate must be provided."
                    start_ratio = block_rate * block_number
                    end_ratio = block_rate * (block_number + 1)
                    print(
                        f"For PREQ_SLD_BLS\n"
                        f"Using block {block_number} ({start_ratio:.2f} - {end_ratio:.2f})"
                    )
                    # get blocks
                    hps_block = get_sequential_slice_from_dataset(
                        hps_dataset, start_ratio=start_ratio, end_ratio=end_ratio
                    )
                    led_block = get_sequential_slice_from_dataset(
                        led_dataset, start_ratio=start_ratio, end_ratio=end_ratio
                    )

                    # create train-val-test splits
                    # the logic is that
                    # - sequential split for train - test
                    # - from train - get random val samples
                    # uses the last .25 of the dataset for test
                    hps_train_val, hps_test = split_dataset_sequentially_into_two(
                        hps_block, split_ratio=0.75
                    )
                    led_train_val, led_test = split_dataset_sequentially_into_two(
                        led_block, split_ratio=0.75
                    )
                    # we are doing random splits for train+val because
                    # We have limited training data (thus, we need variety)
                    hps_train, hps_val = random_split(
                        dataset=hps_train_val,
                        lengths=(0.9, 0.1),
                        # todo: @gsoykan - maybe make seed a config...
                        generator=torch.Generator().manual_seed(42),
                    )
                    led_train, led_val = random_split(
                        dataset=led_train_val,
                        lengths=(0.9, 0.1),
                        # todo: @gsoykan - maybe make seed a config...
                        generator=torch.Generator().manual_seed(42),
                    )

                    # concat datasets
                    self.data_train = ConcatDataset([hps_train, led_train])
                    self.data_val = ConcatDataset([hps_val, led_val])
                    self.data_test = ConcatDataset([hps_test, led_test])
                case _:
                    raise AssertionError(f"Unknown experiment type: {experiment_type}")

            if self.hparams.train_subset_len is not None:
                indices = torch.randperm(len(self.data_train))[
                    : self.hparams.train_subset_len
                ].tolist()
                self.data_train = torch.utils.data.Subset(self.data_train, indices)

            if self.hparams.val_subset_len is not None:
                indices = torch.randperm(len(self.data_val))[
                    : self.hparams.val_subset_len
                ].tolist()
                self.data_val = torch.utils.data.Subset(self.data_val, indices)

            if self.hparams.test_subset_len is not None:
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
    datamodule = ExoPromptGreenLightGTTimeSeriesDataModule(
        root_path="/Users/gsoykan/Desktop/yanan-desktop/wur-phd-2024/physinet/data/from_david_by_gurkan/gt",
        experiment_config={
            "type": "finetuning_mixed",
        },
        output_feature_idx=None,
        exo_params_len=254,
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
    PickleHelper.save_object(PickleHelper.exoprompt_gt_timeseries_batch, batch)
    loaded_batch = PickleHelper.load_object(PickleHelper.exoprompt_gt_timeseries_batch)
    print(batch)
