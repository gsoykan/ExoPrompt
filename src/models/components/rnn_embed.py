"""
Embedding layers for RNN models with ExoPrompt support.

Similar to TimeSeriesLibrary/layers/Embed.py but tailored for RNN architectures:
- No PositionalEmbedding (RNNs have inherent sequential ordering)
- Uses TokenEmbedding for feature transformation
- Uses TemporalEmbedding for time features
- Supports ExoPrompt with "two_layer_mlp" and "brute_concat" modes
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch import Tensor

# Import embedding layers from TimeSeriesLibrary
from TimeSeriesLibrary.layers.Embed import (
    TokenEmbedding,
    TemporalEmbedding,
    TimeFeatureEmbedding,
)


@dataclass
class ExoPromptConfig:
    """
    Configuration for ExoPrompt tuning.

    Attributes:
        prompt_tuning_type: Type of prompt tuning:
            - "two_layer_mlp": MLP projector → N prefix tokens (ExoPrompt)
            - "brute_concat": Feature-level concat (repeat at every timestep)
            - "direct_concat": Linear → 1 prefix token (simplest baseline)
        exo_prompt_dim: Dimension of exogenous parameters
        num_virtual_tokens: Number of virtual tokens (only for "two_layer_mlp" mode)
        exo_prompt_projector_hidden_size: Hidden size for MLP projector (only for "two_layer_mlp" mode)
    """

    prompt_tuning_type: str
    exo_prompt_dim: int
    num_virtual_tokens: int = 10
    exo_prompt_projector_hidden_size: int = 512

    def __post_init__(self):
        """Validate configuration."""
        valid_types = ["two_layer_mlp", "brute_concat", "direct_concat"]
        if self.prompt_tuning_type not in valid_types:
            raise ValueError(
                f"prompt_tuning_type must be one of {valid_types}, "
                f"got '{self.prompt_tuning_type}'"
            )


class DataEmbeddingRNNWithExoPromptTuning(nn.Module):
    """
    Data embedding for RNN models with optional ExoPrompt support.

    Similar to DataEmbeddingWithExoPromptTuning but without PositionalEmbedding.
    RNNs inherently process sequences in order, so positional encoding is not needed.

    Architecture:
        Input → TokenEmbedding → [+ExoPrompt (optional)] → [+TemporalEmbedding] → Dropout → Output

    If exo_prompt_config is provided, supports three ExoPrompt modes:
    - "two_layer_mlp": Projects exo params through MLP to N virtual tokens (prefix tuning style)
    - "brute_concat": Concatenates exo params directly at feature level (repeat at every timestep)
    - "direct_concat": Projects exo params through Linear to 1 prefix token (simplest baseline)
    """

    def __init__(
            self,
            c_in: int,
            d_model: int,
            embed_type: str = "fixed",
            freq: str = "h",
            dropout: float = 0.1,
            exo_prompt_config: Optional[ExoPromptConfig] = None,
    ):
        """
        Initialize RNN embedding layer with optional ExoPrompt support.

        Args:
            c_in: Number of input features
            d_model: Model dimension
            embed_type: Type of temporal embedding ("fixed" or "timeF")
            freq: Frequency for time features (h/t/s/m/a/w/d/b)
            dropout: Dropout rate
            exo_prompt_config: Optional ExoPrompt configuration. If None, no ExoPrompt is used.
        """
        super(DataEmbeddingRNNWithExoPromptTuning, self).__init__()

        self.exo_prompt_config = exo_prompt_config

        # ExoPrompt projector (only if config is provided)
        if exo_prompt_config is not None:
            if exo_prompt_config.prompt_tuning_type == "two_layer_mlp":
                # Inspired from prefix-tuning
                self.exo_prompt_projector = nn.Sequential(
                    nn.Linear(
                        exo_prompt_config.exo_prompt_dim,
                        exo_prompt_config.exo_prompt_projector_hidden_size,
                    ),
                    nn.Tanh(),
                    nn.Linear(
                        exo_prompt_config.exo_prompt_projector_hidden_size,
                        d_model * exo_prompt_config.num_virtual_tokens,
                    ),
                )
            elif exo_prompt_config.prompt_tuning_type == "direct_concat":
                self.exo_prompt_projector = nn.Linear(
                    exo_prompt_config.exo_prompt_dim,
                    d_model
                )
            elif exo_prompt_config.prompt_tuning_type == "brute_concat":
                self.exo_prompt_projector = None
        else:
            self.exo_prompt_projector = None

        # TokenEmbedding for input features
        # For brute_concat, concatenate exo at feature level before embedding
        token_embedding_c_in = c_in
        if (
                exo_prompt_config is not None
                and exo_prompt_config.prompt_tuning_type == "brute_concat"
        ):
            token_embedding_c_in = c_in + exo_prompt_config.exo_prompt_dim

        self.value_embedding = TokenEmbedding(
            c_in=token_embedding_c_in, d_model=d_model
        )

        # TemporalEmbedding for time features
        self.temporal_embedding = (
            TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
            if embed_type != "timeF"
            else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        )

        # Note: No PositionalEmbedding - RNNs have inherent sequential ordering
        self.dropout = nn.Dropout(p=dropout)

    def forward(
            self,
            x: Tensor,
            x_mark: Optional[Tensor] = None,
            exo_prompt: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass with optional ExoPrompt conditioning.

        Args:
            x: Input features [B, seq_len, c_in]
            x_mark: Time features [B, seq_len, time_features] (optional)
            exo_prompt: Exogenous parameters [B, exo_dim] (optional, ignored if exo_prompt_config is None)

        Returns:
            Embedded input [B, seq_len (+virtual_tokens), d_model]
        """
        # Process exo_prompt if config is provided and exo_prompt is not None
        if self.exo_prompt_config is not None and exo_prompt is not None:
            if self.exo_prompt_config.prompt_tuning_type == "brute_concat":
                # Expand across time dimension
                exo_prompt = repeat(exo_prompt, "b v -> b l v", l=x.size(1))
            elif self.exo_prompt_config.prompt_tuning_type == "two_layer_mlp":
                # Project and reshape to virtual tokens
                exo_prompt = self.exo_prompt_projector(exo_prompt)
                exo_prompt = rearrange(
                    exo_prompt,
                    "b (l d) -> b l d",
                    l=self.exo_prompt_config.num_virtual_tokens,
                )
            elif self.exo_prompt_config.prompt_tuning_type == "direct_concat":
                exo_prompt = self.exo_prompt_projector(exo_prompt)  # [B, d_model]
                exo_prompt = rearrange(exo_prompt, "b d -> b 1 d")

        # Apply embeddings
        if x_mark is None:
            # No time features
            if self.exo_prompt_config is not None and exo_prompt is not None:
                if self.exo_prompt_config.prompt_tuning_type == "brute_concat":
                    x = torch.cat([exo_prompt, x], dim=2)
                    x = self.value_embedding(x)
                else:  # two_layer_mlp or direct_concat
                    x = self.value_embedding(x)
                    x = torch.cat([exo_prompt, x], dim=1)
            else:
                x = self.value_embedding(x)
        else:
            # With time features
            if self.exo_prompt_config is not None and exo_prompt is not None:
                if self.exo_prompt_config.prompt_tuning_type == "brute_concat":
                    x = torch.cat([exo_prompt, x], dim=2)
                    x = self.value_embedding(x)
                else:  # two_layer_mlp or direct_concat
                    x = self.value_embedding(x)
                    x = torch.cat([exo_prompt, x], dim=1)

                # Add temporal embedding
                temporal_embedding = self.temporal_embedding(x_mark)
                if self.exo_prompt_config.prompt_tuning_type != "brute_concat":
                    # Pad with zeros for virtual tokens
                    temporal_filler = torch.zeros_like(exo_prompt)
                    temporal_embedding = torch.cat(
                        [temporal_filler, temporal_embedding], dim=1
                    )
                x = x + temporal_embedding
            else:
                x = self.value_embedding(x) + self.temporal_embedding(x_mark)

        return self.dropout(x)
