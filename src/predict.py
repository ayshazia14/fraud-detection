import torch
import torch.nn as nn
import joblib
import os
import numpy as np

# Define your model structure
class FraudMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

class FraudPredictor:
    def __init__(self, model_path, scaler_path, input_dim=30):
        self.scaler = joblib.load(scaler_path)
        self.model = FraudMLP(input_dim)
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()

    def predict_proba(self, features):
        scaled_features = self.scaler.transform(features)
        tensor_features = torch.tensor(scaled_features, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(tensor_features)
            probs = torch.sigmoid(logits).squeeze().numpy()
        return probs