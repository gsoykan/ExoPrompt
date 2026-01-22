"""
Temporal Convolutional Network (TCN) for time series forecasting.

Based on "An Empirical Evaluation of Generic Convolutional and Recurrent Networks
for Sequence Modeling" (Bai et al., 2018).

Key features:
- Dilated causal convolutions (only looks at past, not future)
- Exponentially growing receptive field (dilation = 2^layer)
- Residual connections for gradient flow
- Weight normalization for stable training

Supports configurable embedding types (same as RNN models):
- "token": TokenEmbedding (1D conv) + TemporalEmbedding - same as transformers
- "linear": Simple linear projection - typical baseline
- "none": Raw input directly to TCN - simplest possible

ExoPrompt support with "two_layer_mlp" and "brute_concat" modes:
- "token" embedding: Full support (same as transformers)
- "linear" embedding: Full support
- "none" embedding: No ExoPrompt support (simplest baseline)
"""

import torch
import torch.nn as nn
from typing import Optional, Literal
from torch import Tensor
from einops import rearrange, repeat
from torch.nn.utils.parametrizations import weight_norm

# Import shared RNN embedding layer and config (reusable for TCN too)
from src.models.components.rnn_embed import (
    DataEmbeddingRNNWithExoPromptTuning,
    ExoPromptConfig,
)

EmbeddingType = Literal["token", "linear", "none"]


class CausalConv1d(nn.Module):
    """
    Causal 1D convolution: only uses past information.

    Applies left-padding so output length matches input length,
    and output at time t only depends on inputs at times <= t.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            dilation: int = 1,
            use_weight_norm: bool = True,
    ):
        super(CausalConv1d, self).__init__()
        # Padding on left side only (causal)
        self.padding = (kernel_size - 1) * dilation

        conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
        )

        if use_weight_norm:
            self.conv = weight_norm(conv)
        else:
            self.conv = conv

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, C, L] input tensor
        Returns:
            [B, C_out, L] output tensor (same length as input, causal)
        """
        out = self.conv(x)
        # Remove right padding to maintain causality
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TemporalBlock(nn.Module):
    """
    A single temporal block in the TCN.

    Structure:
        Input → Conv1 → ReLU → Dropout → Conv2 → ReLU → Dropout → + Residual → Output

    Uses dilated causal convolutions and residual connections.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            dilation: int,
            dropout: float = 0.1,
            use_weight_norm: bool = True,
    ):
        super(TemporalBlock, self).__init__()

        # First conv layer
        self.conv1 = CausalConv1d(
            in_channels, out_channels, kernel_size, dilation, use_weight_norm
        )

        # Second conv layer
        self.conv2 = CausalConv1d(
            out_channels, out_channels, kernel_size, dilation, use_weight_norm
        )

        # Activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Residual connection (1x1 conv if channels change)
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, C_in, L] input tensor
        Returns:
            [B, C_out, L] output tensor
        """
        # Two-layer conv with ReLU and dropout
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.dropout(out)

        # Residual connection
        residual = x if self.downsample is None else self.downsample(x)

        return self.relu(out + residual)


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network backbone.

    Stacks multiple TemporalBlocks with exponentially increasing dilation.
    Receptive field grows as: rf = 2 * kernel_size * (2^num_layers - 1)
    """

    def __init__(
            self,
            num_inputs: int,
            num_channels: list,
            kernel_size: int = 3,
            dropout: float = 0.1,
            use_weight_norm: bool = True,
    ):
        """
        Args:
            num_inputs: Number of input channels
            num_channels: List of output channels for each layer (determines depth)
            kernel_size: Kernel size for all convolutions
            dropout: Dropout rate
            use_weight_norm: Whether to use weight normalization
        """
        super(TemporalConvNet, self).__init__()

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation = 2 ** i  # Exponentially growing dilation
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    dilation,
                    dropout,
                    use_weight_norm,
                )
            )

        self.network = nn.Sequential(*layers)

        # Calculate receptive field
        # Each TemporalBlock has 2 convs with the same dilation
        # RF = 1 + 2 * (kernel_size - 1) * sum(dilations)
        # where dilations = 1, 2, 4, ..., 2^(num_levels-1)
        # sum = 2^num_levels - 1
        self.receptive_field = 1 + 2 * (kernel_size - 1) * (2 ** num_levels - 1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, C_in, L] input tensor (channel-first)
        Returns:
            [B, C_out, L] output tensor
        """
        return self.network(x)


class VanillaTCNModel(nn.Module):
    """
    Vanilla TCN baseline for time series forecasting with configurable embedding.

    Same interface as VanillaGRU/LSTM for fair comparison.

    Embedding Types:
        - "token": TokenEmbedding (1D conv) + TemporalEmbedding - same as transformers
          Supports ExoPrompt. Use for fair architecture comparison.

        - "linear": Simple linear projection to d_model - typical baseline
          Supports ExoPrompt. Use for standard baseline with/without exo conditioning.

        - "none": Raw input directly to TCN - simplest possible
          No ExoPrompt support. Use for absolute simplest baseline.

    Architecture:
        Input → Embedding → TCN (dilated causal conv) → Projection → Output

    For long-term forecasting, uses the full sequence output from TCN
    (captures temporal patterns) rather than just the last timestep.
    """

    def __init__(self, configs):
        """
        Initialize the TCN model with configurable embedding.

        Args:
            configs: Configuration object with the following attributes:
                - seq_len: Input sequence length
                - pred_len: Prediction horizon length
                - enc_in: Number of input features
                - d_model: Hidden dimension (used as TCN channel width)
                - e_layers: Number of TCN layers (default: 4)
                - kernel_size: Convolution kernel size (default: 3)
                - dropout: Dropout rate (default: 0.1)
                - embedding_type: One of "token", "linear", "none" (default: "token")
                - use_weight_norm: Whether to use weight normalization (default: True)
                - enable_exo_prompt_tuning: Whether to use ExoPrompt
                - prompt_tuning_type: "two_layer_mlp" or "brute_concat"
                - exo_prompt_dim: Dimension of exogenous parameters
        """
        super(VanillaTCNModel, self).__init__()

        # Basic configuration
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.task_name = getattr(configs, "task_name", "long_term_forecast")
        self.d_model = configs.d_model
        self.num_layers = getattr(configs, "e_layers", 4)  # Default 4 for TCN
        self.kernel_size = getattr(configs, "kernel_size", 3)
        self.dropout_rate = getattr(configs, "dropout", 0.1)
        self.use_weight_norm = getattr(configs, "use_weight_norm", True)

        # Embedding type configuration
        self.embedding_type: EmbeddingType = getattr(configs, "embedding_type", "token")
        if self.embedding_type not in ["token", "linear", "none"]:
            raise ValueError(
                f"embedding_type must be 'token', 'linear', or 'none', "
                f"got '{self.embedding_type}'"
            )

        # Temporal embedding configuration (for "token" type)
        self.embed_type = getattr(configs, "embed", "timeF")
        self.freq = getattr(configs, "freq", "t")

        # Output configuration
        if (
                hasattr(configs, "output_feature_idx")
                and configs.output_feature_idx is not None
        ):
            self.output_dim = len(configs.output_feature_idx)
        else:
            self.output_dim = getattr(configs, "dec_in", configs.enc_in)

        # Build embedding based on embedding_type
        self._build_embedding(configs)

        # TCN backbone - input size depends on embedding type
        tcn_input_size = self.d_model if self.embedding_type != "none" else self.enc_in

        # Create channel list: all layers have d_model channels
        num_channels = [self.d_model] * self.num_layers

        self.tcn = TemporalConvNet(
            num_inputs=tcn_input_size,
            num_channels=num_channels,
            kernel_size=self.kernel_size,
            dropout=self.dropout_rate,
            use_weight_norm=self.use_weight_norm,
        )

        # Output projection
        # Option 1: Use last timestep (like RNN last_hidden)
        # Option 2: Use adaptive pooling over sequence
        # Option 3: Use learnable projection from full sequence
        # We use option 2 (pool) + linear for flexibility
        self.output_projection = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.d_model, self.pred_len * self.output_dim),
        )

    def _build_embedding(self, configs):
        """Build embedding layer based on embedding_type."""
        self.use_exo_prompt = False
        self.prompt_tuning_type = None
        self.exo_prompt_dim = None
        self.num_virtual_tokens = None

        # Check if ExoPrompt is enabled
        exo_enabled = (
                hasattr(configs, "enable_exo_prompt_tuning")
                and configs.enable_exo_prompt_tuning
        )

        if self.embedding_type == "token":
            # Full transformer-style embedding with optional ExoPrompt
            exo_prompt_config = None
            if exo_enabled:
                self.use_exo_prompt = True
                self.prompt_tuning_type = getattr(configs, "prompt_tuning_type", "two_layer_mlp")
                self.exo_prompt_dim = configs.exo_prompt_dim
                self.num_virtual_tokens = getattr(configs, "num_virtual_tokens", 10)
                exo_prompt_config = ExoPromptConfig(
                    prompt_tuning_type=self.prompt_tuning_type,
                    exo_prompt_dim=self.exo_prompt_dim,
                    num_virtual_tokens=self.num_virtual_tokens,
                    exo_prompt_projector_hidden_size=getattr(
                        configs, "exo_prompt_projector_hidden_size", 512
                    ),
                )

            self.embedding = DataEmbeddingRNNWithExoPromptTuning(
                c_in=self.enc_in,
                d_model=self.d_model,
                embed_type=self.embed_type,
                freq=self.freq,
                dropout=self.dropout_rate,
                exo_prompt_config=exo_prompt_config,
            )

        elif self.embedding_type == "linear":
            # Simple linear projection with optional ExoPrompt
            if exo_enabled:
                self.use_exo_prompt = True
                self.prompt_tuning_type = getattr(configs, "prompt_tuning_type", "two_layer_mlp")
                self.exo_prompt_dim = configs.exo_prompt_dim
                self.num_virtual_tokens = getattr(configs, "num_virtual_tokens", 10)

                if self.prompt_tuning_type == "brute_concat":
                    # Concatenate exo at input level before linear projection
                    self.linear_embed = nn.Sequential(
                        nn.Linear(self.enc_in + self.exo_prompt_dim, self.d_model),
                        nn.Dropout(self.dropout_rate),
                    )
                else:  # two_layer_mlp
                    # Linear projection + ExoPrompt projector for virtual tokens
                    self.linear_embed = nn.Sequential(
                        nn.Linear(self.enc_in, self.d_model),
                        nn.Dropout(self.dropout_rate),
                    )
                    # ExoPrompt projector (same as transformer implementation)
                    self.exo_prompt_projector = nn.Sequential(
                        nn.Linear(
                            self.exo_prompt_dim,
                            getattr(configs, "exo_prompt_projector_hidden_size", 512),
                        ),
                        nn.Tanh(),
                        nn.Linear(
                            getattr(configs, "exo_prompt_projector_hidden_size", 512),
                            self.d_model * self.num_virtual_tokens,
                        ),
                    )
            else:
                # No ExoPrompt - simple linear projection
                self.linear_embed = nn.Sequential(
                    nn.Linear(self.enc_in, self.d_model),
                    nn.Dropout(self.dropout_rate),
                )

        else:  # "none"
            # No embedding - raw input directly to TCN (no ExoPrompt support)
            self.embedding = nn.Dropout(self.dropout_rate)

    def forward(
            self,
            x_enc: Tensor,
            x_mark_enc: Optional[Tensor] = None,
            x_dec: Optional[Tensor] = None,
            x_mark_dec: Optional[Tensor] = None,
            exo_prompt: Optional[Tensor] = None,
            mask: Optional[Tensor] = None,
            target: Optional[Tensor] = None,  # Unused, kept for interface compatibility
    ) -> Tensor:
        """
        Forward pass with configurable embedding.

        Args:
            x_enc: Input sequence [B, seq_len, enc_in]
            x_mark_enc: Time features [B, seq_len, time_features] (only used with "token" embedding)
            x_dec: Decoder input (unused)
            x_mark_dec: Decoder time features (unused)
            exo_prompt: Exogenous parameters [B, exo_dim] (used with "token" or "linear" + ExoPrompt)
            mask: Attention mask (unused)
            target: Ground truth (unused, kept for interface compatibility with RNN)

        Returns:
            Predictions [B, pred_len, output_dim]
        """
        # Embed based on embedding type
        if self.embedding_type == "token":
            x_embedded = self.embedding(x_enc, x_mark_enc, exo_prompt if self.use_exo_prompt else None)

        elif self.embedding_type == "linear":
            if self.use_exo_prompt and exo_prompt is not None:
                if self.prompt_tuning_type == "brute_concat":
                    # Expand exo_prompt across time and concatenate
                    exo_expanded = repeat(exo_prompt, "b d -> b l d", l=x_enc.size(1))
                    x_with_exo = torch.cat([x_enc, exo_expanded], dim=2)
                    x_embedded = self.linear_embed(x_with_exo)
                else:  # two_layer_mlp
                    # Linear embed input
                    x_embedded = self.linear_embed(x_enc)
                    # Project exo_prompt to virtual tokens
                    exo_tokens = self.exo_prompt_projector(exo_prompt)
                    exo_tokens = rearrange(
                        exo_tokens, "b (l d) -> b l d", l=self.num_virtual_tokens
                    )
                    # Prepend virtual tokens
                    x_embedded = torch.cat([exo_tokens, x_embedded], dim=1)
            else:
                x_embedded = self.linear_embed(x_enc)

        else:  # "none"
            x_embedded = self.embedding(x_enc)

        # TCN expects [B, C, L] but we have [B, L, C]
        x_tcn = x_embedded.transpose(1, 2)  # [B, d_model, seq_len]

        # Apply TCN
        tcn_output = self.tcn(x_tcn)  # [B, d_model, seq_len]

        # Use last timestep for prediction (similar to RNN last_hidden)
        last_output = tcn_output[:, :, -1]  # [B, d_model]

        # Project to predictions
        predictions = self.output_projection(last_output)  # [B, pred_len * output_dim]
        predictions = predictions.view(-1, self.pred_len, self.output_dim)

        return predictions

    def get_receptive_field(self) -> int:
        """Return the receptive field of the TCN."""
        return self.tcn.receptive_field
