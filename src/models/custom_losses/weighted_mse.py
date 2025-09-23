import torch
import torch.nn as nn


class TimeWeightedMSELoss(nn.Module):
    def __init__(self, pred_len: int, alpha: float = 0.1):
        """
        Initialize time-weighted MSE loss
        Args:
            pred_len: Number of prediction timesteps
            alpha: Decay factor (higher means faster decay)
        """
        super().__init__()
        # Create weights that decay exponentially
        weights = torch.exp(-alpha * torch.arange(pred_len))

        # TODO: @gsoykan - maybe make this optional? based on args
        # Normalize weights to sum to pred_len (to maintain scale with regular MSE)
        self.weights = (weights * pred_len) / weights.sum()

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            outputs: Predictions of shape [batch_size, pred_len, n_features]
            targets: Ground truth of shape [batch_size, pred_len, n_features]
        """
        # Compute squared errors
        squared_errors = (outputs - targets) ** 2

        # Apply time-based weights
        weights = self.weights.to(outputs.device)
        weighted_errors = squared_errors * weights.unsqueeze(0).unsqueeze(-1)

        return weighted_errors.mean()


# TODO: @gsoykan - try this
class LinearDecayMSELoss(nn.Module):
    def __init__(self, pred_len: int):
        """
        Initialize linearly decaying MSE loss
        Args:
            pred_len: Number of prediction timesteps
        """
        super().__init__()
        # Create linearly decreasing weights
        weights = torch.linspace(2, 0, pred_len)
        # Normalize weights to sum to pred_len
        self.weights = (weights * pred_len) / weights.sum()

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        squared_errors = (outputs - targets) ** 2
        weights = self.weights.to(outputs.device)
        weighted_errors = squared_errors * weights.unsqueeze(0).unsqueeze(-1)
        return weighted_errors.mean()


# TODO: @gsoykan - try this
class AdaptiveTimeWeightedLoss(nn.Module):
    def __init__(self, pred_len: int, init_alpha: float = 0.1):
        """
        Initialize adaptive time-weighted loss where decay rate is learned
        Args:
            pred_len: Number of prediction timesteps
            init_alpha: Initial decay factor
        """
        super().__init__()
        self.pred_len = pred_len
        # Make alpha a learnable parameter
        self.alpha = nn.Parameter(torch.tensor(init_alpha))

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Compute dynamic weights based on learned alpha
        weights = torch.exp(
            -self.alpha * torch.arange(self.pred_len).to(outputs.device)
        )
        weights = (weights * self.pred_len) / weights.sum()

        squared_errors = (outputs - targets) ** 2
        weighted_errors = squared_errors * weights.unsqueeze(0).unsqueeze(-1)
        return weighted_errors.mean()
