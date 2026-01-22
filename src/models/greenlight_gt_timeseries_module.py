import os
from functools import partial
from types import SimpleNamespace
from typing import Any, Dict, Tuple, Optional

import numpy as np
import torch
from einops import rearrange
from torch import Tensor
from lightning import LightningModule
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MeanMetric, MeanSquaredError, MinMetric
import torch.nn.functional as F
import matplotlib.pyplot as plt

import importlib

import sys

from src.models.custom_losses.original_scale_mse_loss import OriginalScaleMSELoss
from src.models.custom_losses.weighted_mse import TimeWeightedMSELoss
from src.utils.greenlight_scaler import GreenlightScaler
from src.utils.pickle_helper import PickleHelper

current_file_directory = os.path.dirname(
    os.path.join(os.getenv("BASE_DIR", "iTransformer-official"))
)
sys.path.append(current_file_directory)


# TODO: @gsoykan - log predictions over test set and plot?


class GreenlightGTTimeSeriesLitModule(LightningModule):
    def __init__(
        self,
        model_configs_dict: Dict,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        pretrained_ckpt: Optional[str] = None,
        loss_fn: str = "mse",  # mse, original_scale_mse, time_weighted_mse, linear_decay_mse, adaptive_time_weighted_mse,
        debug: bool = False,
        freeze_exo_prompt_projector: Optional[bool] = None,
    ) -> None:
        """Initialize a `GreenlightGTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # TODO: @gsoykan - when we load the finetuned model this should not be loaded...
        #  do sth for it...
        #  maybe when fine-tuning is over we can flag it
        if pretrained_ckpt is not None:
            pretrained_model = GreenlightGTTimeSeriesLitModule.load_from_checkpoint(
                pretrained_ckpt
            )

        model_configs = SimpleNamespace(**model_configs_dict)
        self.model_configs = model_configs

        if pretrained_ckpt is None:
            match model_configs.model_name:
                case "Transformer":
                    from TimeSeriesLibrary.models.Transformer import (
                        Model as TransformerModel,
                    )

                    self.net = TransformerModel(model_configs)
                case "iTransformer":
                    iTransformer_official = importlib.import_module(
                        "iTransformer-official"
                    )
                    self.net = iTransformer_official.iTransformer.Model(
                        configs=model_configs
                    )
                case "iTransformer-ts-lib":
                    from TimeSeriesLibrary.models.iTransformer import (
                        Model as iTransformerModel,
                    )

                    self.net = iTransformerModel(model_configs)
                case "TimeMixer":
                    from TimeSeriesLibrary.models.TimeMixer import (
                        Model as TimeMixerModel,
                    )

                    self.net = TimeMixerModel(model_configs)
                case "TimesNet":
                    from TimeSeriesLibrary.models.TimesNet import Model as TimesNetModel

                    self.net = TimesNetModel(model_configs)
                case "TimeXer":
                    from TimeSeriesLibrary.models.TimeXer import Model as TimeXerModel

                    self.net = TimeXerModel(model_configs)
                case "DLinear":
                    from TimeSeriesLibrary.models.DLinear import Model as DLinearModel

                    self.net = DLinearModel(model_configs)
                case _:
                    raise ValueError(f"unknown model type: {model_configs.model_name}")
        else:
            assert pretrained_model.model_configs == self.model_configs, (
                f"Model configurations do not match between pretrained model and model to be finetuned!, "
                f"pretrained model configs: "
                f"\n {pretrained_model.model_configs} \n\n model configs: \n {self.model_configs}"
            )
            self.net = pretrained_model.net

        self._maybe_freeze_exo_prompt_projector()

        # TODO: @gsoykan - try -> linear_decay_mse, adaptive_time_weighted_mse
        # loss function
        match loss_fn:
            case "mse":
                self.criterion = torch.nn.MSELoss()
            case "time_weighted_mse":
                self.criterion = TimeWeightedMSELoss(pred_len=model_configs.pred_len)
            case "original_scale_mse":
                # todo: make scaling also args
                #  assumes the order of features in scaler is correct (aligned with model output)
                output_scaling_ranges = GreenlightScaler().output_scaling_ranges
                output_scaling_ranges_idx = {
                    i: values for i, values in enumerate(output_scaling_ranges.values())
                }
                self.criterion = OriginalScaleMSELoss(
                    output_scaling_ranges=output_scaling_ranges_idx,
                    num_features=len(output_scaling_ranges),
                    normalize_ranges_for_max=True,
                )
            case _:
                raise AssertionError(f"unhandled loss function: {loss_fn}")

        # metric objects for calculating and averaging mse across batches
        self.log_idx_and_label = list(
            zip(self.model_configs.output_log_idx, ["tAir", "vpAir", "co2Air"])
        )
        output_dim = len(self.model_configs.output_log_idx)
        self.train_nn_mse = MeanSquaredError(num_outputs=output_dim)
        self.val_nn_mse = MeanSquaredError(num_outputs=output_dim)
        self.test_nn_mse = MeanSquaredError(num_outputs=output_dim)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_module_mse_best = MinMetric()

        # temporary epoch outputs
        self._reset_epoch_outputs(mode="train")
        self._reset_epoch_outputs(mode="val")
        self._reset_epoch_outputs(mode="test")

    # i thought this could be better to write with :)
    # but now i am not so sure :)

    def _maybe_freeze_exo_prompt_projector(self):
        freeze_exo_prompt_projector = self.hparams.freeze_exo_prompt_projector
        if freeze_exo_prompt_projector is not True:
            return

        assert (
            self.hparams.pretrained_ckpt is not None
        ), "Only pretrained model's projector can be frozen! Otherwise, it does not make sense!"

        if hasattr(self.net, "enc_embedding") and hasattr(
            self.net.enc_embedding, "exo_prompt_projector"
        ):
            for param in self.net.enc_embedding.exo_prompt_projector.parameters():
                param.requires_grad = False
            print("[Info] ExoPrompt projector is frozen.")
        else:
            raise AssertionError(
                "ExoPrompt projector can not be accessed to be frozen."
            )

    def _reset_epoch_outputs(self, mode: str) -> None:
        match mode:
            case "train":
                self.training_step_outputs = {
                    "gt": [],
                    "nn": [],
                    "raw_sim": [],
                    "sim": [],
                    "time": [],
                }
            case "val":
                self.val_step_outputs = {
                    "gt": [],
                    "nn": [],
                    "raw_sim": [],
                    "sim": [],
                    "time": [],
                }
            case "test":
                self.test_step_outputs = {
                    "gt": [],
                    "nn": [],
                    "raw_sim": [],
                    "sim": [],
                    "time": [],
                }
            case _:
                raise ValueError(f"Unknown mode: {mode}")

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_nn_mse.reset()

        self.val_module_mse_best.reset()
        self._reset_epoch_outputs(mode="train")
        self._reset_epoch_outputs(mode="val")
        self._reset_epoch_outputs(mode="test")

    @torch.no_grad()
    def evaluate_for_original_scale(self, mode: str) -> None:
        """

        Args:
            mode (str): can be 'train', 'val' or 'test'
        """
        # collect values
        match mode:
            case "train":
                gts = self.training_step_outputs["gt"]
                nns = self.training_step_outputs["nn"]
                raw_sims = self.training_step_outputs["raw_sim"]
                sims = self.training_step_outputs["sim"]
                times = self.training_step_outputs["time"]
            case "val":
                gts = self.val_step_outputs["gt"]
                nns = self.val_step_outputs["nn"]
                raw_sims = self.val_step_outputs["raw_sim"]
                sims = self.val_step_outputs["sim"]
                times = self.val_step_outputs["time"]
            case "test":
                gts = self.test_step_outputs["gt"]
                nns = self.test_step_outputs["nn"]
                raw_sims = self.test_step_outputs["raw_sim"]
                sims = self.test_step_outputs["sim"]
                times = self.test_step_outputs["time"]
            case _:
                raise ValueError(f"Unknown mode: {mode}")

        if len(raw_sims) != 0:
            raw_sims = torch.concat(raw_sims, dim=0)  # no need to scale
        elif len(sims) != 0:
            sims = torch.concat(sims, dim=0)  # need to scale
            sims = rearrange(sims, "b l s -> (b l) s")

        # TODO: @gsoykan - shapes might have issues...
        gts = torch.concat(gts, dim=0)  # [B*S, L, 3] (S: number of steps, L: seq. len.)
        nns = torch.concat(nns, dim=0)  # [B*S, L, 3]
        gts = rearrange(gts, "b l s -> (b l) s")
        nns = rearrange(nns, "b l s -> (b l) s")
        # TODO: @gsoykan - handle times later...
        times = torch.concat(times, dim=0)

        # TODO: @gsoykan - this might be problematic because - we only have outputs here!
        scaler = GreenlightScaler()
        gts = scaler.inverse_transform(gts, is_only_output=True)
        nns = scaler.inverse_transform(nns, is_only_output=True)

        if len(sims) != 0:
            # after scaling it becomes raw_sim
            sims = scaler.inverse_transform(sims, is_only_output=True)
            raw_sims = sims

        def vp_to_rh(vp: Tensor, temp: Tensor) -> Tensor:
            def sat_vp(temp: Tensor) -> Tensor:
                p = [610.78, 238.3, 17.2694, -6140.4, 273, 28.916]
                sat = p[0] * torch.exp(p[2] * temp / (temp + p[1]))
                return sat

            rh = 100 * (vp / sat_vp(temp))

            # bad models can result in higher values than that!
            # float16 (aka Half precision) has a maximum representable finite value of ~65,504,
            # Define finite range for the target dtype
            if torch.isinf(rh).any() or torch.isnan(rh).any():
                orig_dtype = vp.dtype
                dtype_bounds = {
                    torch.float16: (-65504.0, 65504.0),
                    torch.float32: (-3.4e38, 3.4e38),
                    torch.float64: (-1.7e308, 1.7e308),
                }
                min_val, max_val = dtype_bounds.get(
                    orig_dtype, (-float("inf"), float("inf"))
                )
                # Recalculate in float32
                vp_32 = vp.to(torch.float32)
                temp_32 = temp.to(torch.float32)
                rh = 100 * (vp_32 / sat_vp(temp_32))
                # Clamp values
                rh = torch.clamp(rh, min=min_val, max=max_val)
                # Cast back to original dtype
                rh = rh.to(orig_dtype)

            return rh

        # Adding RH dimension
        # key for tAir => 0
        # key for vp => 1
        keys_to_iterate = list(scaler.output_scaling_ranges.keys())
        keys_to_iterate.append("rh")

        rh_gt = vp_to_rh(gts[:, 1], gts[:, 0])
        gts = torch.cat([gts, rh_gt.unsqueeze(1)], dim=1)
        rh_nn = vp_to_rh(nns[:, 1], nns[:, 0])
        nns = torch.cat([nns, rh_nn.unsqueeze(1)], dim=1)
        if len(sims) != 0:
            rh_raw_sims = vp_to_rh(raw_sims[:, 1], raw_sims[:, 0])
            raw_sims = torch.cat([raw_sims, rh_raw_sims.unsqueeze(1)], dim=1)

        # Compute ME, RMSE and RRMSE for each output variable
        # and log them
        for i, key in enumerate(keys_to_iterate):
            gt_col = gts[:, i]
            nn_col = nns[:, i]
            if len(raw_sims) != 0:
                raw_sim_col = raw_sims[:, i]

            # Mean Error (ME)
            me = torch.mean(
                gt_col - nn_col
            )  # this can be misleading because err: +1000 and -1000 cancels each other out...
            if len(raw_sims) != 0:
                sim_me = torch.mean(gt_col - raw_sim_col)
            # RMSE
            rmse = torch.sqrt(F.mse_loss(nn_col, gt_col))
            if len(raw_sims) != 0:
                sim_rmse = torch.sqrt(F.mse_loss(raw_sim_col, gt_col))
            # RRMSE
            rrmse = 100 * (rmse / torch.mean(gt_col))
            if len(raw_sims) != 0:
                sim_rrmse = 100 * (sim_rmse / torch.mean(gt_col))

            self.log(
                f"{mode}/me_{key}",
                me.item(),
                prog_bar=True,
            )
            self.log(
                f"{mode}/rmse_{key}",
                rmse.item(),
                prog_bar=True,
            )
            self.log(
                f"{mode}/rrmse_{key}",
                rrmse.item(),
                prog_bar=True,
            )
            if len(raw_sims) != 0:
                self.log(
                    f"{mode}/sim_me_{key}",
                    sim_me.item(),
                    prog_bar=True,
                )
                self.log(
                    f"{mode}/sim_rmse_{key}",
                    sim_rmse.item(),
                    prog_bar=True,
                )
                self.log(
                    f"{mode}/sim_rrmse_{key}",
                    sim_rrmse.item(),
                    prog_bar=True,
                )

        # TODO: @gsoykan - sort by time and save to csv especially for test...

        if mode in ["train", "test", "val"]:
            if mode == "train" and gts.size(0) > 1000:
                # for train inspect small subset of it
                subset_ind = torch.randint(0, gts.size(0), (1000,))
                gts = gts[subset_ind]
                nns = nns[subset_ind]
                if len(raw_sims) != 0:
                    raw_sims = raw_sims[subset_ind]

            self._plot_simulation_results(
                None,  # times
                gts,
                nns,
                raw_sims if len(raw_sims) != 0 else None,
                mode=mode,
                save_path="./",
            )

    def _plot_simulation_results(
        self, times: Optional, gts, nns, raw_sims: Optional, mode="test", save_path=None
    ):
        """
        Plot the measured, NN predictions, and simulated climate trajectories for T_Air, RH_Air, and CO2_Air.

        Args:
            times (torch.Tensor): The time steps to plot on the x-axis.
            gts (torch.Tensor): Ground truth values [B*S, 3].
            nns (torch.Tensor): Neural network predictions [B*S, 3].
            raw_sims (torch.Tensor): Raw simulation values [B*S, 3].
            mode (str): 'train', 'val', or 'test'.
        """
        # Convert times to list or NumPy array for plotting
        if times is not None:
            time_steps = times.cpu().numpy()
        else:
            time_steps = np.arange(
                0,
                len(gts),
            )
        gts = gts.cpu().numpy()
        nns = nns.cpu().numpy()
        if raw_sims is not None:
            raw_sims = raw_sims.cpu().numpy()

        # Subplot layout: 3 rows, 1 column
        fig, axes = plt.subplots(4, 1, figsize=(12, 12))

        # Set the overall title based on the mode
        fig.suptitle(
            f"GT x NN x Simulation Results - {mode.capitalize()} Mode", fontsize=16
        )

        # T_Air plot (temperature)
        axes[0].plot(time_steps, gts[:, 0], label="Measured", color="blue")
        axes[0].plot(time_steps, nns[:, 0], label="NN Prediction", color="red")
        if raw_sims is not None:
            axes[0].plot(time_steps, raw_sims[:, 0], label="Simulated", color="brown")
        axes[0].set_title("T_Air (Temperature)")
        axes[0].set_ylabel("T_Air (°C)")
        axes[0].legend()

        # Vapor Pressure plot
        axes[1].plot(time_steps, gts[:, 1], label="Measured", color="blue")
        axes[1].plot(time_steps, nns[:, 1], label="NN Prediction", color="red")
        if raw_sims is not None:
            axes[1].plot(time_steps, raw_sims[:, 1], label="Simulated", color="brown")
        axes[1].set_title("vp_Air (Vapor Pressure)")
        axes[1].set_ylabel("vp_Air (Pascal)")

        # CO2_Air plot (CO2 levels)
        axes[2].plot(time_steps, gts[:, 2], label="Measured", color="blue")
        axes[2].plot(time_steps, nns[:, 2], label="NN Prediction", color="red")
        if raw_sims is not None:
            axes[2].plot(time_steps, raw_sims[:, 2], label="Simulated", color="brown")
        axes[2].set_title("CO2_Air (CO2 Levels)")
        axes[2].set_ylabel("CO2_Air (ppm)")

        # RH_Air plot
        axes[3].plot(time_steps, gts[:, 3], label="Measured", color="blue")
        axes[3].plot(time_steps, nns[:, 3], label="NN Prediction", color="red")
        if raw_sims is not None:
            axes[3].plot(time_steps, raw_sims[:, 3], label="Simulated", color="brown")
        axes[3].set_title("RH_Air (Relative Humidity)")
        axes[3].set_ylabel("RH_Air (%)")

        # Set common x-label
        axes[2].set_xlabel("Time")

        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        if save_path is not None:
            save_file = os.path.join(save_path, f"in_training_results_{mode}.png")
            plt.savefig(save_file)

            # Only show the plot in interactive mode
            if hasattr(sys, "ps1") or os.environ.get("PYCHARM_HOSTED"):
                plt.show()
            else:
                if os.environ.get("MODE") != "server":
                    print("Skipping plt.show() in non-interactive mode")

        if self.hparams.debug:
            if os.environ.get("MODE") != "server":
                print(f"Plot is shown for {mode} mode.")

        plt.close()

    def forward(
        self,
        x: Tensor | Any,
        y_physical: Optional[Tensor] = None,
        custom_merge_ops: Optional = None,
        exo_prompt: Optional[Tensor] = None,
    ) -> Tensor | Tuple[Tensor, Tensor, Tensor]:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of inputs.
        :param y_physical: A tensor of outputs coming from physics based simulator
        :return: A tensor of logits.
        """
        if exo_prompt is not None:
            if isinstance(x, tuple):
                y_nn = self.net(*x, exo_prompt=exo_prompt)
            else:
                y_nn = self.net(x, exo_prompt=exo_prompt)
        else:
            if isinstance(x, tuple):
                y_nn = self.net(*x)
            else:
                y_nn = self.net(x)
        return y_nn

    def custom_physinet_merge_ops(
        self, w_physical, w_nn, y_nn, y_physical
    ) -> Tuple[Tuple[Tensor, Tensor], Tensor, Optional[Tensor]]:
        y_nn = y_nn[:, -self.model_configs.pred_len :]
        y_nn_raw = None
        if self.model_configs.output_feature_idx is not None:
            # since 7, 8, 9 are the indices for t, vp, co2 indoor
            y_nn_output = y_nn[:, :, self.model_configs.output_feature_idx]
            y_nn_raw = y_nn
        else:
            y_nn_output = y_nn

        y_physical = y_physical[:, -self.model_configs.pred_len :]

        y_combined = (w_physical * y_physical) + (w_nn * y_nn_output)
        return (y_nn_output, y_combined), y_physical, y_nn_raw

    def model_step(
        self, batch: Dict[str, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor | Tuple[Tensor, ...]]]:
        batch_x, batch_y, batch_x_mark, batch_y_mark = (
            batch["seq_x"],
            batch["seq_y"],
            batch["seq_x_mark"],
            batch["seq_y_mark"],
        )
        batch_y_sim = None
        if "seq_y_sim" in batch:
            batch_y_sim = batch["seq_y_sim"]

        # decoder input
        if (
            hasattr(self.model_configs, "use_all_features_for_decoder")
            and self.model_configs.use_all_features_for_decoder
        ):
            assert "seq_y_all" in batch, "All output features should be in the batch"
            batch_y_all = batch["seq_y_all"][
                :, -self.model_configs.pred_len :
            ]  # [B, pred_len, 20 (num_all_features)]
            dec_inp = torch.zeros_like(
                batch_y_all[:, -self.model_configs.pred_len :, :]
            )
            dec_inp = (
                torch.cat(
                    [batch_y_all[:, : self.model_configs.label_len, :], dec_inp], dim=1
                )
                # .to(self.device)
            )
        else:
            dec_inp = torch.zeros_like(batch_y[:, -self.model_configs.pred_len :, :])
            dec_inp = (
                torch.cat(
                    [batch_y[:, : self.model_configs.label_len, :], dec_inp], dim=1
                )
                # .to(self.device)
            )


        outputs = self.forward(
            (batch_x, batch_x_mark, dec_inp, batch_y_mark),
            exo_prompt=batch.get("exo_params", None),
        )
        outputs = outputs[:, -self.model_configs.pred_len :]
        if self.model_configs.output_feature_idx is not None:
            # since 7, 8, 9 are the indices for t, vp, co2 indoor
            outputs = outputs[:, :, self.model_configs.output_feature_idx]

        batch_y = batch_y[:, -self.model_configs.pred_len :]  # .to(self.device)


        loss = self.criterion(outputs, batch_y)

        if self.hparams.debug:
            for i in [0, 1, 2]:
                pred_std = outputs[:, :, i].std()
                gt_std = batch_y[:, :, i].std()
                print(f"for {i}th feature, pred_std: {pred_std}, gt_std: {gt_std}")

        return loss, (batch_y, outputs)

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        seq_y_mark = batch["seq_y_mark"]
        loss, (batch_y, outputs) = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log(
            "train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True
        )

        # Log each MSE component separately
        self.train_nn_mse(
            rearrange(batch_y, "b s f -> (b s) f"),
            rearrange(outputs, "b s f -> (b s) f"),
        )
        for i, name in self.log_idx_and_label:
            self.log(
                f"train/nn_mse_{name}",
                self.train_nn_mse.compute()[i],  # log each output separately
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        self.training_step_outputs["gt"].append(batch_y)
        self.training_step_outputs["nn"].append(
             outputs
        )
        if "output_raw_sim" in batch:
            # TODO: @gsoykan - this might be changed...
            self.training_step_outputs["raw_sim"].append(batch["output_raw_sim"])
        self.training_step_outputs["time"].append(seq_y_mark)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        self.evaluate_for_original_scale(mode="train")
        self._reset_epoch_outputs(mode="train")

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> None:
        seq_y_mark = batch["seq_y_mark"]
        loss, (batch_y, outputs) = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        # Log each MSE component separately
        self.val_nn_mse(
            rearrange(batch_y, "b s f -> (b s) f"),
            rearrange(outputs, "b s f -> (b s) f"),
        )

        for i, name in self.log_idx_and_label:
            self.log(
                f"val/nn_mse_{name}",
                self.val_nn_mse.compute()[i],  # log each output separately
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        self.val_step_outputs["gt"].append(batch_y)
        self.val_step_outputs["nn"].append(
             outputs
        )
        if "output_raw_sim" in batch:
            # TODO: @gsoykan - this might be changed...
            self.val_step_outputs["raw_sim"].append(batch["output_raw_sim"])
        self.val_step_outputs["time"].append(seq_y_mark)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        mse = self.val_nn_mse.compute()  # get current val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch

        mean_mse = torch.mean(mse)  # Compute mean MSE to track the best performance
        # we are doing this becase _best is a MinMetric
        self.val_module_mse_best(mean_mse)  # Update best so far val MSE
        self.log(
            "val/module_mse_mean_best",
            self.val_module_mse_best.compute(),
            sync_dist=True,
            prog_bar=True,
        )
        self.evaluate_for_original_scale(mode="val")
        self._reset_epoch_outputs(mode="val")

    def test_step(self, batch: Dict[str, Tensor], batch_idx: int) -> None:
        seq_y_mark = batch["seq_y_mark"]
        loss, (batch_y, outputs) = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.log(
            "test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True
        )

        # Log each MSE component separately
        self.test_nn_mse(
            rearrange(batch_y, "b s f -> (b s) f"),
            rearrange(outputs, "b s f -> (b s) f"),
        )
        for i, name in self.log_idx_and_label:
            self.log(
                f"test/nn_mse_{name}",
                self.val_nn_mse.compute()[i],  # log each output separately
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        self.test_step_outputs["gt"].append(batch_y)
        self.test_step_outputs["nn"].append(
           outputs
        )
        if "output_raw_sim" in batch:
            # TODO: @gsoykan - this might be changed...
            self.test_step_outputs["raw_sim"].append(batch["output_raw_sim"])
        self.test_step_outputs["time"].append(seq_y_mark)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        self.evaluate_for_original_scale(mode="test")
        self._reset_epoch_outputs(mode="test")

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    # exo_prompt forward pass test - BEGIN
    # Define the model configurations
    model_configs_dict = {
        "model_name": "Transformer",
        "task_name": "long_term_forecast",
        "seq_len": 96,
        "pred_len": 96,
        "label_len": 48,
        "e_layers": 2,
        "d_layers": 1,
        "factor": 3,
        "embed": "timeF",
        "freq": "t",
        "enc_in": 18,
        "dec_in": 18,
        "c_out": 18,
        "d_model": 512,
        "d_ff": 2048,
        "channel_independence": 1,
        "dropout": 0.1,
        "use_norm": 1,
        "top_k": 5,
        "num_kernels": 6,
        "activation": "gelu",
        "n_heads": 8,
        "expand": 2,
        "d_conv": 4,
        "inverse": False,
        "use_all_features_for_decoder": True,
        "input_num_features": 18,
        "output_feature_idx": [7, 8, 9],
        "output_log_idx": [0, 1, 2],
        # exo_prompt_tuning configs
        "enable_exo_prompt_tuning": True,
        "prompt_tuning_type": "two_layer_mlp",
        "num_virtual_tokens": 10,
        "exo_prompt_dim": 241,
        "exo_prompt_projector_hidden_size": 512,
    }

    optimizer = partial(Adam, lr=0.0001, weight_decay=0.0)
    scheduler = partial(
        ReduceLROnPlateau, optimizer=optimizer, mode="min", factor=0.1, patience=10
    )

    # Define whether to compile the model (PyTorch 2.0 feature)
    compile = False

    loss_fn = "mse"

    # Instantiate the module
    module = GreenlightGTTimeSeriesLitModule(
        model_configs_dict=model_configs_dict,
        optimizer=optimizer,
        scheduler=scheduler,
        compile=compile,
        loss_fn=loss_fn,
    )

    exo_prompt_batch = PickleHelper.load_object(
        PickleHelper.exoprompt_old_world_timeseries_batch
    )
    exo_prompt_batch["seq_y_all"] = exo_prompt_batch["seq_y"]
    exo_prompt_batch["seq_y"] = exo_prompt_batch["seq_y"][
        :, :, model_configs_dict["output_feature_idx"]
    ]
    result = module.model_step(exo_prompt_batch)
    print(result)

    # exo_prompt forward pass test - END

    model_configs = {
        "seq_len": 96,  # Length of the input sequence
        "pred_len": 96,  # Length of the prediction horizon
        "output_attention": False,  # Whether to output attention weights
        "use_norm": True,  # Whether to apply normalization
        "d_model": 512,  # Model dimension
        "embed": "timeF",  # Embedding type, could be time-based
        "freq": "t",  # Data frequency (e.g., hourly)
        "dropout": 0.1,  # Dropout rate
        "class_strategy": "projection",  # Strategy for handling classes
        "factor": 1,  # Attention factor
        "n_heads": 8,  # Number of attention heads
        "d_ff": 512,  # Feed-forward network dimension
        "activation": "gelu",  # Activation function
        "e_layers": 3,  # Number of encoder layers
        "output_feature_idx": (7, 8, 9),
        "output_log_idx": (0, 1, 2),
        "model": "iTransformer",
    }
    model_configs = SimpleNamespace(**model_configs)
    # iTransformer_official = importlib.import_module("iTransformer-official")
    # net = iTransformer_official.iTransformer.Model(configs=model_configs)

    time_mixer_model_configs = {
        "seq_len": 96,  # Length of the input sequence
        "pred_len": 96,  # Length of the prediction horizon
        "label_len": 0,  # Label length (for decoder input)
        "e_layers": 3,  # Number of encoder layers
        "d_layers": 1,  # Number of decoder layers
        "factor": 3,  # Attention factor (if applicable)
        "d_model": 16,  # Model dimension
        "d_ff": 32,  # Dimension of feed-forward layers
        "down_sampling_layers": 3,  # Number of down-sampling layers
        "down_sampling_window": 2,  # Down-sampling window size
        "down_sampling_method": "avg",  # Method for down-sampling
        "model_name": "TimeMixer",  # Model name
        "task_name": "long_term_forecast",  # Task name
        "embed": "timeF",  # Embedding type, could be time-based
        "freq": "t",  # Data frequency (e.g., hourly)
        "channel_independence": 1,  # '0: channel dependence 1: channel independence for FreTS model' (What is this?)
        "dropout": 0.1,  # Dropout rate
        #  parser.add_argument('--decomp_method', type=str, default='moving_avg',
        #                         help='method of series decompsition, only support moving_avg or dft_decomp')
        "decomp_method": "moving_avg",
        #     parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
        "moving_avg": 25,
        # --enc_in 20 --dec_in 20 --c_out 3
        "enc_in": 20,  # depending on our dataset!
        # parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
        "use_norm": 1,
        # "batch_size": 128,  # Batch size for training
        # "learning_rate": 0.01,  # Learning rate for the optimizer
        # "train_epochs": 20,  # Number of training epochs
        # "patience": 10,  # Patience for early stopping
        # "is_training": 1,  # Flag for training
        # "root_path": "./dataset/weather/",  # Path to the dataset root
        # "data_path": "weather.csv",  # Path to the dataset file
        # "model_id": "weather_96_96",  # ID for the model instance
        # "data": "custom",  # Type of dataset
        # "features": "M",  # Features mode (M for multi-variable)
        # "des": "Exp",  # Description of the experiment
        # "itr": 1,  # Number of iterations
    }
    time_mixer_model_configs = SimpleNamespace(**time_mixer_model_configs)
    from TimeSeriesLibrary.models.TimeMixer import Model as TimeMixerModel

    time_mixer_net = TimeMixerModel(time_mixer_model_configs)
    print(time_mixer_net)

    # TimesNet
    times_net_model_configs = {
        "seq_len": 96,  # Length of the input sequence
        "pred_len": 96,  # Length of the prediction horizon
        "label_len": 48,  # Label length (for decoder input)
        "e_layers": 2,  # Number of encoder layers
        "d_layers": 1,  # Number of decoder layers
        "factor": 3,  # Attention factor (if applicable)
        "d_model": 32,  # Model dimension
        "d_ff": 32,  # Dimension of feed-forward layers
        "model_name": "TimesNet",  # Model name
        "task_name": "long_term_forecast",  # Task name
        "embed": "timeF",  # Embedding type, could be time-based
        "freq": "t",  # Data frequency (e.g., hourly)
        "channel_independence": 1,  # '0: channel dependence 1: channel independence for FreTS model' (What is this?)
        "dropout": 0.1,  # Dropout rate
        "enc_in": 20,  # depending on our dataset!
        "c_out": 20,
        "use_norm": 1,
        "top_k": 5,
        # parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
        "num_kernels": 6,
    }

    times_net_model_configs = SimpleNamespace(**times_net_model_configs)
    from TimeSeriesLibrary.models.TimesNet import Model as TimesNetModel

    times_net = TimesNetModel(times_net_model_configs)
    print(times_net)

    _ = GreenlightGTTimeSeriesLitModule(None, None, None, None)
