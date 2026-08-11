import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, classification_report

from imblearn.over_sampling import SMOTE
import xgboost as xgb
from xgboost import XGBClassifier

# Configuration & Setup
MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

df = pd.read_csv("data/creditcard.csv")

# Data Splitting & Scaling
print("\nPreprocessing and splitting data...")
X = df.drop(columns=["Class"])
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Calculate class weights for imbalanced handling
neg_count, pos_count = np.bincount(y_train)
scale_pos_weight = neg_count / pos_count
print(f"--> Scale Pos Weight (Imbalance Ratio): {scale_pos_weight:.2f}")

# Resampling (SMOTE)
print("\nApplying SMOTE to training set")
smote = SMOTE(random_state=42)
X_train_smote_scaled, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
print(f"--> SMOTE Train Shape: {X_train_smote_scaled.shape}")

# Save scaler and processed test data for notebook analysis
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
joblib.dump((X_test_scaled, y_test), os.path.join(MODEL_DIR, "test_data.joblib"))

# PyTorch Datasets & Loaders
print("\n Creating PyTorch DataLoaders")

# DataLoaders for Weighted Model (Original Scaled Data)
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)

train_dataset_weighted = TensorDataset(X_train_tensor, y_train_tensor)
train_loader_weighted = DataLoader(train_dataset_weighted, batch_size=256, shuffle=True)

# DataLoaders for SMOTE Model
X_train_smote_tensor = torch.tensor(X_train_smote_scaled, dtype=torch.float32)
y_train_smote_tensor = torch.tensor(y_train_smote.values, dtype=torch.float32).unsqueeze(1)

train_dataset_smote = TensorDataset(X_train_smote_tensor, y_train_smote_tensor)
train_loader_smote = DataLoader(train_dataset_smote, batch_size=256, shuffle=True)

# DataLoaders for Testing
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# PyTorch Model Architecture & Helpers
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
            # Output raw logits for compatibility with BCEWithLogitsLoss
        )
    def forward(self, x):
        return self.net(x)

def train_model(model, loader, pos_weight=None, epochs=10, lr=0.001):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")
    return model

def evaluate_torch_model(model, loader, name):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(y_batch.numpy().flatten())

    ap_score = average_precision_score(all_labels, all_probs)
    print(f"PyTorch ({name})")
    print(f"PR-AUC (Average Precision): {ap_score:.4f}")
    y_pred = (np.array(all_probs) >= 0.5).astype(int)
    print(classification_report(all_labels, y_pred, target_names=["Legit", "Fraud"]))

# Model Training Execution
print("[5/6] Training Weighted PyTorch Model...")
pos_weight_tensor = torch.tensor([scale_pos_weight], dtype=torch.float32).to(device)
model_weighted = FraudMLP(input_dim=X_train_scaled.shape[1])
model_weighted = train_model(
    model_weighted, train_loader_weighted, pos_weight=pos_weight_tensor, epochs=10
)

print("[5/6] Training SMOTE PyTorch Model...")
model_smote = FraudMLP(input_dim=X_train_smote_scaled.shape[1])
model_smote = train_model(
    model_smote, train_loader_smote, pos_weight=None, epochs=10
)

# Model Evaluation & Artifact Saving
print("\n[6/6] Evaluating Models on Test Set...")
evaluate_torch_model(model_weighted, test_loader, "Class Weights")
evaluate_torch_model(model_smote, test_loader, "SMOTE")

# Save models
torch.save(
    model_weighted.state_dict(),
    os.path.join(MODEL_DIR, "pytorch_weighted_model.pth"),
)
torch.save(
    model_smote.state_dict(),
    os.path.join(MODEL_DIR, "pytorch_smote_model.pth"),
)
print("\nFinished! Artifacts successfully saved in directory:", MODEL_DIR)