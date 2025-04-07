### Feature Importances of Stress Models ###
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_feature_importance(model, feature_names, model_name="Model", coef_type="auto", top_n=15, save_path=None):
    """
    Plot feature importances or coefficients from a fitted model.

    Parameters:
    - model: Trained model object
    - feature_names: List of feature names
    - model_name: Name to display on the plot
    - coef_type: "auto", "coef", or "importance"
    - top_n: Number of top features to show
    - save_path: Full file path to save the plot
    """

    # Determine how to access importance
    if coef_type == "auto":
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        else:
            raise ValueError("Model does not have feature_importances_ or coef_ attribute.")
    elif coef_type == "coef":
        importances = np.abs(model.coef_[0])
    elif coef_type == "importance":
        importances = model.feature_importances_
    else:
        raise ValueError("Invalid coef_type. Use 'auto', 'coef', or 'importance'.")

    # Ensure feature_names and importances have the same length
    feature_names = feature_names[:len(importances)]

    # Create dataframe
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(top_n)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=importance_df)
    plt.title(f"{model_name} - Top {top_n} Feature Importances")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()
