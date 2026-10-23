import torch
import torch.nn as nn

class ImprovedRULLSTM(nn.Module):
    """
    Two-layer stacked LSTM network for turbofan engine Remaining Useful Life estimation.
    Ingests sliding sequence windows of shape (batch_size, sequence_length, input_size).
    """
    def __init__(self, input_size: int = 19, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        # Pool the output of the final cycle in the sequence
        return self.fc(output[:, -1, :])


class RULTransformer(nn.Module):
    """
    Transformer Encoder architecture for turbofan engine Remaining Useful Life estimation.
    Applies learnable positional encodings and self-attention across the 30-cycle temporal window.
    """
    def __init__(
        self,
        input_size: int = 19,
        sequence_length: int = 30,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = nn.Parameter(
            torch.randn(1, sequence_length, d_model) * 0.02
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        self.regressor = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = x + self.positional_encoding
        x = self.transformer(x)
        # Use final cycle representation
        x = x[:, -1, :]
        return self.regressor(x)
