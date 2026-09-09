"""ChronoGuard LSTM Forecaster with Self-Attention Pooling.

Implements the 2-layer sequence-forecasting neural network defined in
AI-INSTRUCTIONS/05-ml-algorithms.md Section 4:
- Linear input projection
- 2-Layer LSTM with recurrent dropout
- Self-attention pooling layer (serves as the primary explainability mechanism)
- Dual output heads:
  1. Infiltration probability regressor (K forecast steps ahead)
  2. MITRE ATT&CK stage classifier (6-class distribution)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional


class SelfAttentionPooling(nn.Module):
    """Learns an attention weight per timestep and aggregates into a context vector.
    
    Acts as the primary explainability mechanism: the extracted attention
    weights show which historical time windows drove the model's forecast.
    """
    def __init__(self, hidden_dim: int, attention_dim: int = 32):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1, bias=False),
        )

    def forward(self, lstm_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_outputs: Tensor of shape (batch_size, seq_len, hidden_dim)
        Returns:
            context_vector: Tensor of shape (batch_size, hidden_dim)
            attention_weights: Tensor of shape (batch_size, seq_len)
        """
        # scores: (batch_size, seq_len, 1)
        scores = self.projection(lstm_outputs)
        # attention_weights: (batch_size, seq_len)
        attention_weights = F.softmax(scores.squeeze(-1), dim=1)
        # context: (batch_size, hidden_dim)
        context_vector = torch.sum(lstm_outputs * attention_weights.unsqueeze(-1), dim=1)
        return context_vector, attention_weights


class ChronoGuardLSTM(nn.Module):
    """Sequence forecaster for cyber-attack build-up detection."""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        forecast_horizon_k: int = 3,
        num_stages: int = 6,
        use_max_pool: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.forecast_horizon_k = forecast_horizon_k
        self.num_stages = num_stages
        self.use_max_pool = use_max_pool

        # 1. Feature Projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 2. 2-Layer Recurrent Core
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # 3. Self-Attention Pooling (Explainability Hook)
        self.attention_pool = SelfAttentionPooling(hidden_dim=hidden_dim, attention_dim=32)

        # Dimension entering the classification heads
        head_in_dim = hidden_dim * 2 if use_max_pool else hidden_dim

        # 4. Multi-Task Output Heads
        # Infiltration probability across K future horizons
        self.prob_head = nn.Sequential(
            nn.Linear(head_in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, forecast_horizon_k),
            nn.Sigmoid(),
        )

        # MITRE stage classification head (raw logits for CrossEntropyLoss)
        self.stage_head = nn.Sequential(
            nn.Linear(head_in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_stages),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            prob_forecast: (batch_size, forecast_horizon_k) in [0, 1]
            stage_logits: (batch_size, num_stages)
            attention_weights: (batch_size, seq_len) in [0, 1], sum=1.0
        """
        batch_size, seq_len, _ = x.shape
        
        # Projection
        proj = self.input_proj(x)
        
        # Recurrent temporal encoding
        lstm_out, _ = self.lstm(proj)
        
        # Explainable Attention Pooling
        context, attention_weights = self.attention_pool(lstm_out)
        
        # Optional Max-Pooling branch for sharp volumetric spikes
        if self.use_max_pool:
            max_pooled, _ = torch.max(lstm_out, dim=1)
            pooled = torch.cat([context, max_pooled], dim=-1)
        else:
            pooled = context
        
        # Heads
        prob_forecast = self.prob_head(pooled)
        stage_logits = self.stage_head(pooled)
        
        return prob_forecast, stage_logits, attention_weights

    @torch.no_grad()
    def predict_step(
        self, x: torch.Tensor
    ) -> Dict[str, Any]:
        """Single-sequence inference method for real-time web deployment."""
        self.eval()
        if x.dim() == 2:
            x = x.unsqueeze(0)
            
        prob_forecast, stage_logits, attention_weights = self.forward(x)
        stage_probs = F.softmax(stage_logits, dim=-1)
        pred_stage_id = torch.argmax(stage_probs, dim=-1)
        
        return {
            "infiltration_probabilities": prob_forecast.cpu().numpy()[0].tolist(),
            "stage_probabilities": stage_probs.cpu().numpy()[0].tolist(),
            "predicted_stage_id": int(pred_stage_id.cpu().numpy()[0]),
            "attention_weights": attention_weights.cpu().numpy()[0].tolist(),
        }
