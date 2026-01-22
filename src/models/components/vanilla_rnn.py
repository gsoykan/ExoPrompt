"""
Vanilla RNN models (GRU, LSTM) for time series forecasting.

Supports configurable embedding types:
- "token": TokenEmbedding (1D conv) + TemporalEmbedding - same as transformers for fair comparison
- "linear": Simple linear projection - typical RNN baseline
- "none": Raw input directly to RNN - simplest possible baseline

ExoPrompt support with "two_layer_mlp" and "brute_concat" modes:
- "token" embedding: Full support (same as transformers)
- "linear" embedding: Full support (shows ExoPrompt works without fancy embedding)
- "none" embedding: No ExoPrompt support (simplest baseline)

Supports two decoder types for long-term forecasting:
- "last_hidden": Simple MLP from last hidden state (simplest baseline)
- "seq2seq": Encoder-decoder with separate decoder RNN (recommended for long horizons,
             generates predictions autoregressively with proper temporal correlation)

For seq2seq decoder, teacher forcing is supported during training:
- teacher_forcing_ratio=1.0: Always use ground truth (fast, parallelized)
- teacher_forcing_ratio=0.0: Always use own predictions (autoregressive)
- 0 < ratio < 1: Randomly choose per batch (scheduled sampling)
"""

import torch
import torch.nn as nn
import random
from typing import Optional, Literal
from torch import Tensor
from einops import rearrange, repeat

# Import shared RNN embedding layer and config
from src.models.components.rnn_embed import (
    DataEmbeddingRNNWithExoPromptTuning,
    ExoPromptConfig,
)

DecoderType = Literal["last_hidden", "seq2seq"]
EmbeddingType = Literal["token", "linear", "none"]


class VanillaGRUModel(nn.Module):
    """
    Vanilla GRU baseline for time series forecasting with configurable embedding.

    Embedding Types:
        - "token": TokenEmbedding (1D conv) + TemporalEmbedding - same as transformers
          Supports ExoPrompt. Use for fair architecture comparison.

        - "linear": Simple linear projection to d_model - typical RNN baseline
          Supports ExoPrompt. Use for standard RNN baseline with/without exo conditioning.

        - "none": Raw input directly to RNN - simplest possible
          No ExoPrompt support. Use for absolute simplest baseline.

    Decoder Types:
        - "last_hidden": Last hidden state → MLP → [pred_len * output_dim]
          Simplest baseline. Compresses all future predictions into single vector.
          May struggle with long horizons due to information bottleneck.

        - "seq2seq": Encoder → Decoder GRU → [pred_len, output_dim]
          Recommended for long-term forecasting. Generates predictions step-by-step
          with proper temporal correlation. Each prediction builds on the previous.
    """

    def __init__(self, configs):
        """
        Initialize the GRU model with configurable embedding.

        Args:
            configs: Configuration object with the following attributes:
                - seq_len: Input sequence length
                - pred_len: Prediction horizon length
                - enc_in: Number of input features
                - d_model: Hidden dimension
                - e_layers: Number of RNN layers (default: 2)
                - dropout: Dropout rate (default: 0.1)
                - embedding_type: One of "token", "linear", "none" (default: "token")
                - decoder_type: One of "last_hidden", "seq2seq" (default: "seq2seq")
                - teacher_forcing_ratio: Ratio for teacher forcing during training (default: 1.0)
                    1.0 = always use ground truth (fast, parallelized)
                    0.0 = always use own predictions (autoregressive)
                - enable_exo_prompt_tuning: Whether to use ExoPrompt (only with "token" embedding)
                - prompt_tuning_type: "two_layer_mlp" or "brute_concat"
                - exo_prompt_dim: Dimension of exogenous parameters
        """
        super(VanillaGRUModel, self).__init__()

        # Basic configuration
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.task_name = getattr(configs, "task_name", "long_term_forecast")
        self.d_model = configs.d_model
        self.num_layers = getattr(configs, "e_layers", 2)
        self.dropout_rate = getattr(configs, "dropout", 0.1)

        # Embedding type configuration
        self.embedding_type: EmbeddingType = getattr(configs, "embedding_type", "token")
        if self.embedding_type not in ["token", "linear", "none"]:
            raise ValueError(
                f"embedding_type must be 'token', 'linear', or 'none', "
                f"got '{self.embedding_type}'"
            )

        # Decoder type configuration
        self.decoder_type: DecoderType = getattr(configs, "decoder_type", "seq2seq")
        self.teacher_forcing_ratio = getattr(configs, "teacher_forcing_ratio", 1.0)
        if self.decoder_type not in ["last_hidden", "seq2seq"]:
            raise ValueError(
                f"decoder_type must be 'last_hidden' or 'seq2seq', "
                f"got '{self.decoder_type}'"
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

        # GRU encoder - input size depends on embedding type
        encoder_input_size = self.d_model if self.embedding_type != "none" else self.enc_in
        self.encoder = nn.GRU(
            input_size=encoder_input_size,
            hidden_size=self.d_model,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout_rate if self.num_layers > 1 else 0.0,
        )

        # Decoder based on decoder_type
        self._build_decoder()

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
            # No embedding - raw input directly to RNN (no ExoPrompt support)
            self.embedding = nn.Dropout(self.dropout_rate)

    def _build_decoder(self):
        """Build decoder layers based on decoder_type."""
        if self.decoder_type == "last_hidden":
            # MLP decoder: last hidden → all predictions at once
            self.decoder = nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.d_model, self.pred_len * self.output_dim),
            )

        elif self.decoder_type == "seq2seq":
            # Decoder GRU + projection
            self.decoder_gru = nn.GRU(
                input_size=self.output_dim,  # Feed back previous prediction
                hidden_size=self.d_model,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=self.dropout_rate if self.num_layers > 1 else 0.0,
            )
            self.projection = nn.Linear(self.d_model, self.output_dim)
            # Learnable start token for decoder
            self.start_token = nn.Parameter(torch.zeros(1, 1, self.output_dim))

    def forward(
            self,
            x_enc: Tensor,
            x_mark_enc: Optional[Tensor] = None,
            x_dec: Optional[Tensor] = None,
            x_mark_dec: Optional[Tensor] = None,
            exo_prompt: Optional[Tensor] = None,
            mask: Optional[Tensor] = None,
            target: Optional[Tensor] = None,
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
            target: Ground truth for teacher forcing [B, pred_len, output_dim] (optional)
                    Required for teacher forcing during training with seq2seq decoder.

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

        # Encode
        encoder_output, encoder_hidden = self.encoder(x_embedded)

        # Decode based on decoder_type
        if self.decoder_type == "last_hidden":
            return self._decode_last_hidden(encoder_hidden)
        else:  # seq2seq
            return self._decode_seq2seq(encoder_hidden, target)

    def _decode_last_hidden(self, encoder_hidden: Tensor) -> Tensor:
        """Decode using only the last hidden state."""
        # encoder_hidden: [num_layers, B, d_model]
        last_hidden = encoder_hidden[-1]  # [B, d_model]
        predictions = self.decoder(last_hidden)  # [B, pred_len * output_dim]
        return predictions.view(-1, self.pred_len, self.output_dim)

    def _decode_seq2seq(
            self, encoder_hidden: Tensor, target: Optional[Tensor] = None
    ) -> Tensor:
        """
        Decode using a separate decoder GRU with teacher forcing support.

        Args:
            encoder_hidden: Final hidden state from encoder [num_layers, B, d_model]
            target: Ground truth for teacher forcing [B, pred_len, output_dim] (optional)

        Returns:
            Predictions [B, pred_len, output_dim]
        """
        batch_size = encoder_hidden.size(1)

        # Determine whether to use teacher forcing for this batch
        use_teacher_forcing = (
                self.training
                and target is not None
                and random.random() < self.teacher_forcing_ratio
        )

        if use_teacher_forcing:
            # Teacher forcing: use ground truth as input (parallelized, fast)
            # Prepend start token to target (shifted right)
            start_tokens = self.start_token.expand(
                batch_size, 1, -1
            )  # [B, 1, output_dim]
            decoder_input = torch.cat(
                [start_tokens, target[:, :-1, :]], dim=1
            )  # [B, pred_len, output_dim]

            # Single forward pass through decoder (parallelized)
            decoder_output, _ = self.decoder_gru(decoder_input, encoder_hidden)
            predictions = self.projection(decoder_output)  # [B, pred_len, output_dim]
        else:
            # Autoregressive: use own predictions (for inference or scheduled sampling)
            decoder_input = self.start_token.expand(
                batch_size, 1, -1
            )  # [B, 1, output_dim]
            decoder_hidden = encoder_hidden

            predictions = []
            for t in range(self.pred_len):
                decoder_output, decoder_hidden = self.decoder_gru(
                    decoder_input, decoder_hidden
                )
                pred_t = self.projection(decoder_output)  # [B, 1, output_dim]
                predictions.append(pred_t)
                decoder_input = pred_t

            predictions = torch.cat(predictions, dim=1)  # [B, pred_len, output_dim]

        return predictions


class VanillaLSTMModel(nn.Module):
    """
    Vanilla LSTM baseline - identical to GRU but uses LSTM cells.

    LSTM has additional cell state for potentially better long-term dependency modeling.
    Supports the same embedding types and decoder types as VanillaGRUModel.

    Embedding Types:
        - "token": TokenEmbedding (1D conv) + TemporalEmbedding - same as transformers
        - "linear": Simple linear projection to d_model - typical RNN baseline
        - "none": Raw input directly to RNN - simplest possible
    """

    def __init__(self, configs):
        """Initialize LSTM model (same config as GRU, including embedding_type)."""
        super(VanillaLSTMModel, self).__init__()

        # Basic configuration (same as GRU)
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.task_name = getattr(configs, "task_name", "long_term_forecast")
        self.d_model = configs.d_model
        self.num_layers = getattr(configs, "e_layers", 2)
        self.dropout_rate = getattr(configs, "dropout", 0.1)

        # Embedding type configuration
        self.embedding_type: EmbeddingType = getattr(configs, "embedding_type", "token")
        if self.embedding_type not in ["token", "linear", "none"]:
            raise ValueError(
                f"embedding_type must be 'token', 'linear', or 'none', "
                f"got '{self.embedding_type}'"
            )

        # Decoder type configuration
        self.decoder_type: DecoderType = getattr(configs, "decoder_type", "seq2seq")
        self.teacher_forcing_ratio = getattr(configs, "teacher_forcing_ratio", 1.0)
        if self.decoder_type not in ["last_hidden", "seq2seq"]:
            raise ValueError(
                f"decoder_type must be 'last_hidden' or 'seq2seq', "
                f"got '{self.decoder_type}'"
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

        # LSTM encoder - input size depends on embedding type
        encoder_input_size = self.d_model if self.embedding_type != "none" else self.enc_in
        self.encoder = nn.LSTM(
            input_size=encoder_input_size,
            hidden_size=self.d_model,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout_rate if self.num_layers > 1 else 0.0,
        )

        # Decoder based on decoder_type
        self._build_decoder()

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
            # No embedding - raw input directly to RNN (no ExoPrompt support)
            self.embedding = nn.Dropout(self.dropout_rate)

    def _build_decoder(self):
        """Build decoder layers based on decoder_type."""
        if self.decoder_type == "last_hidden":
            self.decoder = nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.d_model, self.pred_len * self.output_dim),
            )

        elif self.decoder_type == "seq2seq":
            self.decoder_lstm = nn.LSTM(
                input_size=self.output_dim,
                hidden_size=self.d_model,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=self.dropout_rate if self.num_layers > 1 else 0.0,
            )
            self.projection = nn.Linear(self.d_model, self.output_dim)
            self.start_token = nn.Parameter(torch.zeros(1, 1, self.output_dim))

    def forward(
            self,
            x_enc: Tensor,
            x_mark_enc: Optional[Tensor] = None,
            x_dec: Optional[Tensor] = None,
            x_mark_dec: Optional[Tensor] = None,
            exo_prompt: Optional[Tensor] = None,
            mask: Optional[Tensor] = None,
            target: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass (same interface as GRU, with configurable embedding)."""
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

        # Encode
        encoder_output, (encoder_hidden, encoder_cell) = self.encoder(x_embedded)

        # Decode based on decoder_type
        if self.decoder_type == "last_hidden":
            return self._decode_last_hidden(encoder_hidden)
        else:  # seq2seq
            return self._decode_seq2seq(encoder_hidden, encoder_cell, target)

    def _decode_last_hidden(self, encoder_hidden: Tensor) -> Tensor:
        """Decode using only the last hidden state."""
        last_hidden = encoder_hidden[-1]
        predictions = self.decoder(last_hidden)
        return predictions.view(-1, self.pred_len, self.output_dim)

    def _decode_seq2seq(
            self,
            encoder_hidden: Tensor,
            encoder_cell: Tensor,
            target: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Decode using a separate decoder LSTM with teacher forcing support.

        Args:
            encoder_hidden: Final hidden state from encoder [num_layers, B, d_model]
            encoder_cell: Final cell state from encoder [num_layers, B, d_model]
            target: Ground truth for teacher forcing [B, pred_len, output_dim] (optional)

        Returns:
            Predictions [B, pred_len, output_dim]
        """
        batch_size = encoder_hidden.size(1)

        # Determine whether to use teacher forcing for this batch
        use_teacher_forcing = (
                self.training
                and target is not None
                and random.random() < self.teacher_forcing_ratio
        )

        if use_teacher_forcing:
            # Teacher forcing: use ground truth as input (parallelized, fast)
            start_tokens = self.start_token.expand(
                batch_size, 1, -1
            )  # [B, 1, output_dim]
            decoder_input = torch.cat(
                [start_tokens, target[:, :-1, :]], dim=1
            )  # [B, pred_len, output_dim]

            # Single forward pass through decoder (parallelized)
            decoder_output, _ = self.decoder_lstm(
                decoder_input, (encoder_hidden, encoder_cell)
            )
            predictions = self.projection(decoder_output)  # [B, pred_len, output_dim]
        else:
            # Autoregressive: use own predictions (for inference or scheduled sampling)
            decoder_input = self.start_token.expand(batch_size, 1, -1)
            decoder_hidden = encoder_hidden
            decoder_cell = encoder_cell

            predictions = []
            for t in range(self.pred_len):
                decoder_output, (decoder_hidden, decoder_cell) = self.decoder_lstm(
                    decoder_input, (decoder_hidden, decoder_cell)
                )
                pred_t = self.projection(decoder_output)
                predictions.append(pred_t)
                decoder_input = pred_t

            predictions = torch.cat(predictions, dim=1)  # [B, pred_len, output_dim]

        return predictions
