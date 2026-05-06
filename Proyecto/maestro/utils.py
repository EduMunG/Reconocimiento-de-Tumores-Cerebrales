import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix


def calculate_metrics(y_true, y_pred, y_probs):
    """
    Calculates accuracy, f1_macro and auc.
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_probs: Predicted probabilities for each class.
    Returns:
        dict: dictionary with 'accuracy', 'f1_macro' and 'auc'.
    """
    # Convert to numpy if they are torch tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if isinstance(y_probs, torch.Tensor):
        y_probs = y_probs.cpu().numpy()

    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    auc = roc_auc_score(y_true, y_probs, multi_class='ovr')
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'auc': auc
    }


def save_confusion_matrix(y_true, y_pred, title, path, labels=None):
    """
    Generates and saves a confusion matrix image.
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        title: Title of the plot.
        path: Path to save the image.
        labels (list, optional): List of class names for labels.
    """
    # Convert to numpy if they are torch tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    # Create the directory if it does not exist
    os.makedirs(os.path.dirname(path), exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels if labels else 'auto', 
                yticklabels=labels if labels else 'auto')
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(path)
    plt.close()
