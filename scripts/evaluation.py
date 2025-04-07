### Evaluating Stress Models ###
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve
)
from sklearn.preprocessing import label_binarize
"""
    Plot feature importances or coefficients from a fitted model.

    Parameters:
    - model: Trained model 
    - X_train: Training features
    - y_train: Training labels
    - X_test: Test features
    - y_test: Test labels
    - classes: List of classes
    - model_name: Name of model
    - save_path: Directory where to save the plot
    """

def full_evaluation(model, X_train, y_train, X_test, y_test, classes=None, model_name="Model", save_dir="/directiory"):
    os.makedirs(save_dir, exist_ok=True) # Create directory if it doesn't exist
    results = {}

    # Predict on train and test
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Try predict_proba for AUC-ROC
    try:
        y_train_proba = model.predict_proba(X_train)
        y_test_proba = model.predict_proba(X_test)
    except:
        y_train_proba = y_test_proba = None

    # Evaluation for both sets
    for split_name, y_true, y_pred, y_proba in [
        ("Train", y_train, y_train_pred, y_train_proba),
        ("Test", y_test, y_test_pred, y_test_proba)
    ]:
        split_results = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'F1 Score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        split_results['Confusion Matrix'] = cm
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"{model_name} - {split_name} - Confusion Matrix")
        fig_path_cm = os.path.join(save_dir, f"{model_name}_{split_name}_ConfusionMatrix.png")
        plt.savefig(fig_path_cm)
        plt.show()

        # AUC-ROC
        if y_proba is not None and classes is not None:
            try:
                y_true_bin = label_binarize(y_true, classes=classes)
                auc = roc_auc_score(y_true_bin, y_proba, multi_class='ovr')
                split_results['AUC-ROC'] = auc

                for i, label in enumerate(classes):
                    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
                    plt.plot(fpr, tpr, label=f'Class {label} (AUC={roc_auc_score(y_true_bin[:, i], y_proba[:, i]):.2f})')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.title(f"{model_name} - {split_name} - ROC Curve")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.legend()
                plt.grid(True)
                fig_path_roc = os.path.join(save_dir, f"{model_name}_{split_name}_ROCCurve.png")
                plt.savefig(fig_path_roc)
                plt.show()
            except Exception as e:
                split_results['AUC-ROC'] = None
                print(f"Error calculating AUC-ROC on {split_name}: {e}")
        else:
            split_results['AUC-ROC'] = None

        results[split_name] = split_results

    return results

