import os
from typing import Any, Dict, Optional

import numpy as np
import torch
from einops import rearrange
from torch import Tensor
from lightning import LightningModule
from torchmetrics import MeanMetric, MeanSquaredError, MinMetric
import torch.nn.functional as F
import matplotlib.pyplot as plt


import sys
from pathlib import Path

from src.models.custom_losses.original_scale_mse_loss import OriginalScaleMSELoss
from src.models.custom_losses.weighted_mse import TimeWeightedMSELoss
from src.utils.greenlight_scaler import GreenlightScaler
from abc import ABC, abstractmethod

# Dynamically resolve project root and add iTransformer-official to path
# This ensures the code works on any machine regardless of absolute paths
current_file = Path(__file__).resolve()
project_root = current_file.parents[1]  # Go up from src/models/ to project root
itransformer_path = project_root / "iTransformer-official"
if itransformer_path.exists() and str(itransformer_path) not in sys.path:
    sys.path.append(str(itransformer_path))


class AbstractGreenlightTimeSeriesModule(ABC, LightningModule):
    """
    Abstract base class for greenhouse time series forecasting models.

    This class provides common functionality for training, validation, and testing
    of time series models for greenhouse climate prediction. It handles:
    - Model instantiation and configuration
    - Loss function setup (MSE, time-weighted MSE, original-scale MSE)
    - Metrics tracking (MSE, RMSE, RRMSE for tAir, vpAir, co2Air, RH)
    - Exogenous prompt tuning support
    - Evaluation in original scale with automatic inverse transformation
    - Visualization of predictions

    Subclasses must implement:
    - instantiate_net(): Create the specific model architecture
    - maybe_freeze_exo_prompt_projector(): Handle freezing of prompt projector if applicable
    """

    @abstractmethod
    def instantiate_net(self, *args, **kwargs):
        """
        Instantiate the neural network model.

        This method should:
        1. Create the base model (Transformer, GRU, TCN, etc.)
        2. Load pretrained weights if pretrained_ckpt is provided
        3. Store the instantiated model in self.net
        4. Store model configuration in self.model_configs

        Args:
            model_configs_dict (Dict): Dictionary of model configuration parameters.
                Expected keys depend on the specific model but typically include:
                - model_name: Name of the model architecture
                - seq_len: Input sequence length
                - pred_len: Prediction horizon length
                - label_len: Label length for decoder (transformer models)
                - enc_in: Number of input features
                - dec_in: Number of decoder input features
                - output_feature_idx: Indices of output features to predict
                - output_log_idx: Indices for logging individual metrics
                - enable_exo_prompt_tuning: Whether to use exogenous prompts
                - exo_prompt_dim: Dimension of exogenous parameters
                And model-specific parameters (d_model, n_heads, etc.)
            pretrained_ckpt (Optional[str]): Path to pretrained checkpoint to load

        Returns:
            None (stores model in self.net)

        Example:
            ```python
            def instantiate_net(self, model_configs_dict, pretrained_ckpt):
                model_configs = SimpleNamespace(**model_configs_dict)
                self.model_configs = model_configs

                if pretrained_ckpt is None:
                    # Create new model
                    from TimeSeriesLibrary.models.DLinear import Model as DLinearModel
                    self.net = DLinearModel(model_configs)
                else:
                    # Load pretrained model
                    pretrained = self.load_from_checkpoint(pretrained_ckpt)
                    self.net = pretrained.net
            ```
        """
        pass

    @abstractmethod
    def maybe_freeze_exo_prompt_projector(self):
        """
        Freeze exogenous prompt projector parameters if applicable.

        This method should:
        1. Check if self.hparams.freeze_exo_prompt_projector is True
        2. Verify that a pretrained checkpoint was loaded (only makes sense for fine-tuning)
        3. Access the exo_prompt_projector in the model's embedding layer
        4. Set requires_grad=False for all its parameters to freeze them

        Notes:
        - Only applicable for models with exogenous prompt support (Transformer, iTransformer)
        - For models without exo_prompt support (GRU, LSTM, MLP, DLinear), this should be a no-op
        - Freezing is typically used during fine-tuning when you want to keep the
          prompt projector trained during pretraining and only adapt the main model

        Raises:
            AssertionError: If freeze is requested but no pretrained checkpoint was provided
            AssertionError: If the exo_prompt_projector cannot be accessed in the model

        Example:
            ```python
            def maybe_freeze_exo_prompt_projector(self):
                freeze = self.hparams.freeze_exo_prompt_projector
                if freeze is not True:
                    return  # Nothing to do

                assert self.hparams.pretrained_ckpt is not None, \
                    "Only pretrained model's projector can be frozen!"

                # For transformer-based models with exo_prompt support
                if hasattr(self.net, "enc_embedding") and \
                   hasattr(self.net.enc_embedding, "exo_prompt_projector"):
                    for param in self.net.enc_embedding.exo_prompt_projector.parameters():
                        param.requires_grad = False
                    print("[Info] ExoPrompt projector is frozen.")
                # For models without exo_prompt support, silently skip
            ```
        """
        pass

    def setup_loss_fn(
        self,
        model_configs,
        loss_fn: str = "mse",  # mse, original_scale_mse, time_weighted_mse, linear_decay_mse, adaptive_time_weighted_mse,
    ):
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

    def setup_metrics(self):
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
