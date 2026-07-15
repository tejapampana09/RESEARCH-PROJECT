"""
Fault Detection and Noise Prediction Analytics
==============================================

This module implements predictive diagnostics and noise analytics for the silicon
quantum dot digital twin.

1. Fault Detection:
   Diagnoses anomalies in the quantum dot array based on electrostatic and capacitive
   measurements. It classifies device health into:
   - HEALTHY: Normal operation.
   - DEAD_DOT: Plunger gate has very weak lever arm (alpha_ii < 0.04), indicating the dot
     is physically blocked, not formed, or the electrode is disconnected.
   - SHORTED_GATES: High cross-correlation (>0.95) between different gate coupling profiles,
     suggesting a lithographic short-circuit.
   - HIGH_NOISE: Charge noise standard deviation exceeds a critical threshold (>2e-4 V).

2. Noise Prediction:
   Uses a PyTorch LSTM model to forecast low-frequency 1/f charge noise drifts, enabling
   proactive feedback stabilization.

Mathematical Model (LSTM Noise Predictor)
-----------------------------------------
Input: Sequence of past noise observations: x = [n_{t-W}, ..., n_{t-1}, n_t] (Window size W)
Output: Projected noise value at next step: y_pred = n_{t+1}
Network: Input -> LSTM(hidden_dim) -> Dense(1) -> Output

References
----------
1. S. S. Kalantre et al., "Machine learning for autonomous tuning of silicon quantum dots,"
   npj Quantum Information 5, 6 (2019).
2. T. McJunkin et al., "Active tuning of a double quantum dot using machine learning,"
   Applied Physics Reviews 8, 021402 (2021).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List, Tuple

class FaultDetector:
    """
    Analyzes physical parameters of the quantum dot array to detect hardware anomalies.
    """
    def __init__(self, critical_noise_level: float = 2.0e-4):
        self.critical_noise_level = critical_noise_level

    def diagnose(self, 
                 lever_arm_matrix: np.ndarray, 
                 nominal_noise_amp: float, 
                 num_dots: int) -> Dict[str, Any]:
        """
        Runs device diagnostic checks.
        
        Args:
            lever_arm_matrix: Matrix alpha of shape (num_dots, num_gates)
            nominal_noise_amp: Current standard deviation of gate noise
            num_dots: Number of dots in the device
            
        Returns:
            Dict containing health status and diagnostics details.
        """
        status = "HEALTHY"
        issues = []
        
        # Check 1: Dead Dots (Diagonal elements of alpha are too small)
        for i in range(min(num_dots, lever_arm_matrix.shape[0])):
            if i < lever_arm_matrix.shape[1]:
                diagonal_coupling = lever_arm_matrix[i, i]
                if diagonal_coupling < 0.04:
                    status = "FAULTY"
                    issues.append(f"Dead Dot Detected: Dot {i+1} has extremely weak plunger coupling ({diagonal_coupling:.4f} eV/V). Electrode may be disconnected.")
                    
        # Check 2: Shorted Gates (High correlation between column couplings)
        # If two gates have identical or highly correlated coupling vectors, they are shorted.
        num_gates = lever_arm_matrix.shape[1]
        for g1 in range(num_gates):
            for g2 in range(g1 + 1, num_gates):
                vec1 = lever_arm_matrix[:, g1]
                vec2 = lever_arm_matrix[:, g2]
                
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                
                if norm1 > 0 and norm2 > 0:
                    cos_similarity = np.dot(vec1, vec2) / (norm1 * norm2)
                    if cos_similarity > 0.96:
                        status = "FAULTY"
                        issues.append(f"Gate Short Detected: Plunger/Barrier gates {g1+1} and {g2+1} have highly correlated coupling profiles (similarity {cos_similarity:.3f}). Potential lithographic short.")
                        
        # Check 3: High Noise Level
        if nominal_noise_amp > self.critical_noise_level:
            if status != "FAULTY":
                status = "WARNING"
            issues.append(f"Excessive Charge Noise: Current noise level ({nominal_noise_amp:.3e} V) exceeds threshold ({self.critical_noise_level:.3e} V). Spin coherence time T2* may be significantly reduced.")
            
        return {
            "status": status,
            "issues": issues,
            "timestamp": np.round(np.datetime64('now').astype(float) / 1000)
        }

class LSTMModel(nn.Module):
    """
    LSTM architecture for noise sequence forecasting.
    """
    def __init__(self, input_dim: int = 1, hidden_dim: int = 32, num_layers: int = 1):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        # Take the output of the last time step
        out = self.fc(out[:, -1, :])
        return out

class NoisePredictor:
    """
    Manages the PyTorch LSTM model to predict low frequency drift.
    """
    def __init__(self, window_size: int = 15, hidden_dim: int = 24):
        self.window_size = window_size
        self.model = LSTMModel(input_dim=1, hidden_dim=hidden_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.loss_fn = nn.MSELoss()

    def train_on_history(self, history_data: List[float], epochs: int = 5) -> float:
        """
        Dynamically trains the LSTM model on a history stream.
        
        Args:
            history_data: List of historical noise values.
        """
        if len(history_data) < self.window_size + 5:
            return 0.0
            
        # Create sequences
        X, Y = [], []
        for i in range(len(history_data) - self.window_size):
            X.append(history_data[i:i+self.window_size])
            Y.append(history_data[i+self.window_size])
            
        X_t = torch.FloatTensor(np.array(X)).unsqueeze(-1)  # (N, seq_len, 1)
        Y_t = torch.FloatTensor(np.array(Y)).unsqueeze(-1)  # (N, 1)
        
        self.model.train()
        epoch_loss = 0.0
        for _ in range(epochs):
            pred = self.model(X_t)
            loss = self.loss_fn(pred, Y_t)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss = loss.item()
            
        return epoch_loss

    def predict_next(self, history_data: List[float]) -> float:
        """
        Predicts the next noise value given past history.
        """
        if len(history_data) < self.window_size:
            return 0.0
            
        # Take the last window size elements
        seq = history_data[-self.window_size:]
        seq_t = torch.FloatTensor(np.array(seq)).view(1, self.window_size, 1)
        
        self.model.eval()
        with torch.no_grad():
            pred = self.model(seq_t)
            
        return float(pred.item())
