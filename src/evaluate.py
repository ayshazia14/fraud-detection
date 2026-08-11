import os
import torch
import torch.nn as nn
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix, ConfusionMatrixDisplay

MODEL_DIR = "saved_models"

# 1. Plotting Function
def generate_and_save_artifacts(y_true, probs_weighted, probs_smote):
    sns.set_theme(style="whitegrid")
    
    # Calculate Precision & Recall points
    prec_w, rec_w, _ = precision_recall_curve(y_true, probs_weighted)
    ap_w = average_precision_score(y_true, probs_weighted)
    
    prec_s, rec_s, _ = precision_recall_curve(y_true, probs_smote)
    ap_s = average_precision_score(y_true, probs_smote)

    # Artifact 1: Comparative Precision-Recall Curve Plot
    plt.figure(figsize=(8, 6))
    plt.plot(rec_w, prec_w, label=f"Class-Weighted MLP (PR-AUC = {ap_w:.4f})", color="#d62728", lw=2)
    plt.plot(rec_s, prec_s, label=f"SMOTE MLP (PR-AUC = {ap_s:.4f})", color="#1f77b4", lw=2)
    
    plt.xlabel("Recall (Sensitivity)", fontsize=11)
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=11)
    plt.title("Precision-Recall Curve Comparison (Test Set)", fontsize=13, fontweight='bold')
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    
    pr_curve_path = os.path.join(MODEL_DIR, "pr_curve_comparison.png")
    plt.savefig(pr_curve_path, dpi=300)
    plt.close()
    print(f" Saved: {pr_curve_path}")

    # Artifact 2: Confusion Matrices
    preds_s = (probs_smote >= 0.5).astype(int)
    preds_w = (probs_weighted >= 0.5).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    cm_w = confusion_matrix(y_true, preds_w)
    ConfusionMatrixDisplay(cm_w, display_labels=["Legit", "Fraud"]).plot(
        ax=axes[0], cmap="Reds", values_format="d", colorbar=False
    )
    axes[0].set_title(f"Class-Weighted Matrix\n(Fraud Precision: {cm_w[1,1]/(cm_w[0,1]+cm_w[1,1]):.2%})")

    cm_s = confusion_matrix(y_true, preds_s)
    ConfusionMatrixDisplay(cm_s, display_labels=["Legit", "Fraud"]).plot(
        ax=axes[1], cmap="Blues", values_format="d", colorbar=False
    )
    axes[1].set_title(f"SMOTE Matrix\n(Fraud Precision: {cm_s[1,1]/(cm_s[0,1]+cm_s[1,1]):.2%})")

    plt.tight_layout()
    cm_path = os.path.join(MODEL_DIR, "confusion_matrices.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f" Saved: {cm_path}")

    # Artifact 3: Threshold vs Precision/Recall Curve
    prec_s_t, rec_s_t, thresholds = precision_recall_curve(y_true, probs_smote)
    
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, prec_s_t[:-1], label="Precision", color="#1f77b4", lw=2)
    plt.plot(thresholds, rec_s_t[:-1], label="Recall", color="#ff7f0e", lw=2)
    plt.axvline(x=0.5, color="gray", linestyle="--", alpha=0.7, label="Default Threshold (0.5)")
    
    plt.xlabel("Decision Threshold", fontsize=11)
    plt.ylabel("Score", fontsize=11)
    plt.title("SMOTE Model: Precision vs. Recall Across Thresholds", fontsize=13, fontweight='bold')
    plt.legend(loc="best")
    plt.tight_layout()
    
    thresh_path = os.path.join(MODEL_DIR, "threshold_tradeoff_smote.png")
    plt.savefig(thresh_path, dpi=300)
    plt.close()
    print(f" Saved: {thresh_path}")


# 2. Main Execution Entry Point
if __name__ == "__main__":
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

    print("Loading test data and trained models...")
    X_test_scaled, y_test = joblib.load(os.path.join(MODEL_DIR, "test_data.joblib"))

    input_dim = X_test_scaled.shape[1]
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

    # Load Weighted Model
    model_w = FraudMLP(input_dim)
    model_w.load_state_dict(torch.load(os.path.join(MODEL_DIR, "pytorch_weighted_model.pth")))
    model_w.eval()
    with torch.no_grad():
        probs_weighted = torch.sigmoid(model_w(X_test_tensor)).squeeze().numpy()

    # Load SMOTE Model
    model_s = FraudMLP(input_dim)
    model_s.load_state_dict(torch.load(os.path.join(MODEL_DIR, "pytorch_smote_model.pth")))
    model_s.eval()
    with torch.no_grad():
        probs_smote = torch.sigmoid(model_s(X_test_tensor)).squeeze().numpy()

    print("Generating artifact plots...")
    generate_and_save_artifacts(y_test, probs_weighted, probs_smote)