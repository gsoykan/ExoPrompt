from functools import partial
from types import SimpleNamespace
from typing import Any, Dict, Tuple, Optional

import torch
from einops import rearrange
from torch import Tensor
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, LRScheduler

from src.models.abstract_greenlight_timeseries_module import AbstractGreenlightTimeSeriesModule
from src.models.components.tcn import VanillaTCNModel
from src.models.components.vanilla_rnn import VanillaGRUModel, VanillaLSTMModel


class GreenlightCustomBaselinesTimeSeriesLitModule(AbstractGreenlightTimeSeriesModule):
    """
    Lightning module for custom baseline models (GRU, LSTM, TCN) for greenhouse time series forecasting.

    This module provides a unified interface for training and evaluating various baseline models
    with optional ExoPrompt support for exogenous parameter conditioning.

    Supported Models:
        - GRU: Vanilla GRU with configurable embedding and decoder types
        - LSTM: Vanilla LSTM with configurable embedding and decoder types
        - TCN: Temporal Convolutional Network with configurable embedding

    All models support:
        - Token/Linear/None embedding types
        - ExoPrompt tuning with two_layer_mlp or brute_concat modes
        - Various loss functions (MSE, time-weighted MSE, original-scale MSE)
    """

    def __init__(self,
                 model_configs_dict: Dict,
                 optimizer: Optimizer | partial[Optimizer],
                 scheduler: LRScheduler | partial[LRScheduler],
                 compile: bool,
                 pretrained_ckpt: Optional[str] = None,
                 loss_fn: str = "mse",
                 # mse, original_scale_mse, time_weighted_mse, linear_decay_mse, adaptive_time_weighted_mse,
                 freeze_exo_prompt_projector: Optional[bool] = None,
                 debug: bool = False,
                 ):
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = None
        self.model_configs = None
        self.instantiate_net(model_configs_dict, pretrained_ckpt)
        self.maybe_freeze_exo_prompt_projector()
        self.setup_loss_fn(model_configs=self.model_configs, loss_fn=loss_fn)

        # attributes that are to be set
        self.log_idx_and_label = None
        self.train_nn_mse = None
        self.val_nn_mse = None
        self.test_nn_mse = None
        self.train_loss = None
        self.val_loss = None
        self.test_loss = None
        self.val_module_mse_best = None
        self.training_step_outputs = None
        self.val_step_outputs = None
        self.test_step_outputs = None
        self.setup_metrics()

    def instantiate_net(self,
                        model_configs_dict,
                        pretrained_ckpt, ):
        if pretrained_ckpt is not None:
            pretrained_model = GreenlightCustomBaselinesTimeSeriesLitModule.load_from_checkpoint(
                pretrained_ckpt
            )

        model_configs = SimpleNamespace(**model_configs_dict)
        self.model_configs = model_configs

        if pretrained_ckpt is None:
            match model_configs.model_name:
                case "GRU":
                    self.net = VanillaGRUModel(model_configs)
                case "LSTM":
                    self.net = VanillaLSTMModel(model_configs)
                case "TCN":
                    self.net = VanillaTCNModel(model_configs)
                case _:
                    raise ValueError(f"unknown model type: {model_configs.model_name}")
        else:
            assert pretrained_model.model_configs == self.model_configs, (
                f"Model configurations do not match between pretrained model and model to be finetuned!, "
                f"pretrained model configs: "
                f"\n {pretrained_model.model_configs} \n\n model configs: \n {self.model_configs}"
            )
            self.net = pretrained_model.net

    def maybe_freeze_exo_prompt_projector(self):
        freeze_exo_prompt_projector = self.hparams.freeze_exo_prompt_projector
        if freeze_exo_prompt_projector is not True:
            return

        assert (
                self.hparams.pretrained_ckpt is not None
        ), "Only pretrained model's projector can be frozen! Otherwise, it does not make sense!"

        # Check if model is one of the supported baseline models
        if isinstance(self.net, (VanillaGRUModel, VanillaLSTMModel, VanillaTCNModel)):
            match self.net.embedding_type:
                case "token":
                    for param in self.net.embedding.exo_prompt_projector.parameters():
                        param.requires_grad = False
                    print("[Info] ExoPrompt projector is frozen.")
                case "linear":
                    for param in self.net.exo_prompt_projector.parameters():
                        param.requires_grad = False
                    print("[Info] ExoPrompt projector is frozen.")
                case _:
                    raise ValueError(
                        f"unknown or invalid embedding type: {self.net.embedding_type} "
                        f"for model {self.model_configs.model_name}"
                    )
        else:
            raise ValueError(f"unknown model type: {self.model_configs.model_name}")

    def forward(
            self,
            x: Tensor | Tuple | Any,
            exo_prompt: Optional[Tensor] = None,
            target: Optional[Tensor] = None,
    ) -> Tensor | Tuple[Tensor, Tensor, Tensor]:
        '''
        target is for teacher forcing and is redundant in most cases...
        '''
        if exo_prompt is not None:
            if isinstance(x, tuple):
                y_nn = self.net(*x, exo_prompt=exo_prompt, target=target)
            else:
                y_nn = self.net(x, exo_prompt=exo_prompt, target=target)
        else:
            if isinstance(x, tuple):
                y_nn = self.net(*x, target=target)
            else:
                y_nn = self.net(x, target=target)
        return y_nn

    def model_step(
            self, batch: Dict[str, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor | Tuple[Tensor, ...]]]:
        batch_x, batch_y, batch_x_mark, batch_y_mark = (
            batch["seq_x"],
            batch["seq_y"],
            batch["seq_x_mark"],
            batch["seq_y_mark"],
        )

        # decoder input
        if (
                hasattr(self.model_configs, "use_all_features_for_decoder")
                and self.model_configs.use_all_features_for_decoder
        ):
            assert "seq_y_all" in batch, "All output features should be in the batch"
            # batch_y_all is the full sequence, take label_len + pred_len
            batch_y_all = batch["seq_y_all"]
            # Create decoder input: [label_len of real data, pred_len of zeros]
            dec_inp = torch.zeros_like(batch_y_all[:, -self.model_configs.pred_len:, :])
            dec_inp = torch.cat(
                [batch_y_all[:, : self.model_configs.label_len, :], dec_inp], dim=1
            )
        else:
            dec_inp = torch.zeros_like(batch_y[:, -self.model_configs.pred_len:, :])
            dec_inp = (
                torch.cat(
                    [batch_y[:, : self.model_configs.label_len, :], dec_inp], dim=1
                )
            )

        batch_y = batch_y[:, -self.model_configs.pred_len:]
        outputs = self.forward(
            (batch_x, batch_x_mark, dec_inp, batch_y_mark),
            exo_prompt=batch.get("exo_params", None),
            target=batch_y.detach()
        )

        outputs = outputs[:, -self.model_configs.pred_len:]
        if self.model_configs.output_feature_idx is not None:
            # since 7, 8, 9 are the indices for t, vp, co2 indoor
            outputs = outputs[:, :, self.model_configs.output_feature_idx]

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
        self.training_step_outputs["nn"].append(outputs)
        if "output_raw_sim" in batch:
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
        self.val_step_outputs["nn"].append(outputs)
        if "output_raw_sim" in batch:
            self.val_step_outputs["raw_sim"].append(batch["output_raw_sim"])
        self.val_step_outputs["time"].append(seq_y_mark)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        mse = self.val_nn_mse.compute()  # get current val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch

        mean_mse = torch.mean(mse)  # Compute mean MSE to track the best performance
        # we are doing this because _best is a MinMetric
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
                self.test_nn_mse.compute()[i],  # log each output separately
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        self.test_step_outputs["gt"].append(batch_y)
        self.test_step_outputs["nn"].append(
            outputs
        )
        if "output_raw_sim" in batch:
            self.test_step_outputs["raw_sim"].append(batch["output_raw_sim"])
        self.test_step_outputs["time"].append(seq_y_mark)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        self.evaluate_for_original_scale(mode="test")
        self._reset_epoch_outputs(mode="test")
