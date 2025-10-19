# src/metrics.py
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix

# Ensure the output directory for figures exists
FIGURES_DIR = "reports/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

def calculate_segmentation_metrics(y_true, y_pred):
    """
    Calculates a suite of segmentation metrics from flattened masks.
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()

    # Get the components of the confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true_f, y_pred_f).ravel()

    # Calculate metrics, handling division by zero
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0 # Also called Sensitivity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Dice and IoU are very common in segmentation
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f)
    
    dice_score = (2. * intersection) / (union) if union > 0 else 1.0
    iou_score = intersection / (union - intersection) if (union - intersection) > 0 else 1.0 # Also Jaccard Index

    return {
        "dice_score": dice_score,
        "f1_score": f1_score,
        "iou_score": iou_score,
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": specificity,
        "raw_confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    }

def plot_confusion_matrix(cm_data, model_name):
    """
    Creates and saves a heatmap visualization of the confusion matrix.
    """
    tn = cm_data['tn']
    fp = cm_data['fp']
    fn = cm_data['fn']
    tp = cm_data['tp']
    
    cm_array = np.array([[tn, fp], [fn, tp]])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Healthy', 'Tumor'], yticklabels=['Healthy', 'Tumor'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Pixel-Level Confusion Matrix ({model_name} Model)')
    
    output_path = os.path.join(FIGURES_DIR, f"{model_name.lower()}_confusion_matrix.png")
    plt.savefig(output_path)
    plt.close()
    print(f"[✔] Confusion matrix plot saved to {output_path}")