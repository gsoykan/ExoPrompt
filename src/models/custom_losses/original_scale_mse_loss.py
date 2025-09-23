from typing import Tuple, Dict

import torch
import torch.nn as nn
from torch import Tensor


class OriginalScaleMSELoss(nn.Module):
    def __init__(
        self,
        output_scaling_ranges: Dict[int, Tuple[int | float, int | float]],
        num_features: int,
        normalize_ranges_for_max: bool = False,
    ):
        """
        Initialize the loss function.

        Args:
            output_scaling_ranges: A dictionary mapping feature indices to (min, max) scaling ranges.
            num_features: Total number of output features.
            normalize_ranges_for_max: If True, normalizes feature contributions to the loss based on their range.
                Features with smaller ranges are weighted higher to balance their impact on the loss. If False,
                all features contribute equally regardless of their range.
        """
        super().__init__()

        assert len(set(output_scaling_ranges.keys())) == len(
            output_scaling_ranges
        ), "output_scaling_ranges must be a dict with unique keys"
        assert (
            sorted(list(output_scaling_ranges.keys())) == list(range(num_features))
        ), "output_scaling_ranges must have keys corresponding to the indices of the output features"

        self.output_scaling_ranges = output_scaling_ranges
        self.normalize_ranges_for_max = normalize_ranges_for_max

        # register as PyTorch buffers, they will be moved automatically
        #  to the appropriate device (e.g., GPU/CPU).
        self.register_buffer(
            "output_scaling_indices",
            torch.tensor(list(output_scaling_ranges.keys()), dtype=torch.long),
        )
        self.register_buffer(
            "output_scaling_lower_bound",
            torch.tensor(
                [min(a, b) for (a, b) in output_scaling_ranges.values()],
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "output_scaling_upper_bound",
            torch.tensor(
                [max(a, b) for (a, b) in output_scaling_ranges.values()],
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "max_min_diff",
            self.output_scaling_upper_bound - self.output_scaling_lower_bound,
        )

        if self.normalize_ranges_for_max:
            max_range = self.max_min_diff.max()
            feature_weights = max_range / self.max_min_diff
            self.register_buffer("feature_weights", feature_weights)

    def forward(self, outputs: Tensor, targets: Tensor) -> Tensor:
        """
        Compute the MSE loss in the original scale.

        Args:
            outputs: Tensor of predicted values, shape [..., n_features].
            targets: Tensor of ground truth values, shape [..., n_features].

        Returns:
            Tensor: The MSE loss computed on the original scale.
        """
        assert outputs.shape[-1] >= len(self.output_scaling_indices), (
            f"Outputs must have at least {len(self.output_scaling_indices)} features, "
            f"but got {outputs.shape[-1]}."
        )
        assert targets.shape[-1] >= len(self.output_scaling_indices), (
            f"Targets must have at least {len(self.output_scaling_indices)} features, "
            f"but got {targets.shape[-1]}."
        )

        scaled_outputs = (
            outputs[..., self.output_scaling_indices] * self.max_min_diff
            + self.output_scaling_lower_bound
        )
        scaled_targets = (
            targets[..., self.output_scaling_indices] * self.max_min_diff
            + self.output_scaling_lower_bound
        )

        if self.normalize_ranges_for_max:
            # normalize wrt features weights
            normalized_loss = (
                (scaled_outputs * self.feature_weights)
                - (scaled_targets * self.feature_weights)
            ) ** 2
            loss = normalized_loss.mean()
        else:
            featurewise_loss = (scaled_outputs - scaled_targets) ** 2
            loss = featurewise_loss.mean()

        return loss
