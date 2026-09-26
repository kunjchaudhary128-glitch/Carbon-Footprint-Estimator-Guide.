# %% [markdown]
# # 🤖 Model Training & Evaluation Notebook
# 
# Train multiple regression models, compare performance, and interpret
# with SHAP values.

# %%
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.metrics import mean_absolute_error, r2_score

from src.utils import load_config, PROJECT_ROOT
from src.data_loader import load_lifestyle_transport, load_geo_profiles, merge_with_geo
from src.geo_features import add_geo_features
from src.preprocessing import preprocess, split_data

plt.style.use("seaborn-v0_8-whitegrid")
cfg = load_config()

# %% [markdown]
# ## 1. Load & Preprocess

# %%
df = load_lifestyle_transport()
geo = load_geo_profiles()
df = merge_with_geo(df, geo)
df = add_geo_features(df)
X, y, feature_names, artifacts = preprocess(df, fit=True)
X_train, X_test, y_train, y_test = split_data(X, y)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# %% [markdown]
# ## 2. Model Comparison (already trained via src/model.py)

# %%
results_path = PROJECT_ROOT / "models" / "comparison_results.json"
if results_path.exists():
    with open(results_path) as f:
        results = json.load(f)
    
    results_df = pd.DataFrame(results["results"])
    print(f"\n🏆 Best Model: {results['best_model']}\n")
    print(results_df.to_string(index=False))
    
    # Bar chart comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    colors = ["#e74c3c" if m == results["best_model"] else "#3498db" 
              for m in results_df["model"]]
    
    for ax, metric in zip(axes, ["MAE", "RMSE", "R2"]):
        bars = ax.bar(results_df["model"], results_df[metric], color=colors)
        ax.set_title(metric, fontsize=14)
        ax.tick_params(axis="x", rotation=30)
        for bar, val in zip(bars, results_df[metric]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.4f}", ha="center", fontsize=10)
    
    plt.suptitle("Model Comparison", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(str(PROJECT_ROOT / "data/processed/model_comparison.png"), dpi=150)
    plt.show()
else:
    print("⚠️ Run `python src/model.py` first to generate results!")

# %% [markdown]
# ## 3. Prediction vs Actual Plot

# %%
import joblib

model = joblib.load(PROJECT_ROOT / "models" / "best_model.pkl")
y_pred = model.predict(X_test)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Scatter: predicted vs actual
axes[0].scatter(y_test, y_pred, alpha=0.3, s=15, c="#2ecc71")
lims = [min(y_test.min(), y_pred.min()) - 0.5, max(y_test.max(), y_pred.max()) + 0.5]
axes[0].plot(lims, lims, "r--", alpha=0.8, label="Perfect prediction")
axes[0].set_xlabel("Actual (tonnes CO₂)")
axes[0].set_ylabel("Predicted (tonnes CO₂)")
axes[0].set_title("Predicted vs Actual Carbon Emissions")
axes[0].legend()

# Residuals
residuals = y_test - y_pred
axes[1].hist(residuals, bins=50, edgecolor="white", alpha=0.8, color="#9b59b6")
axes[1].axvline(0, color="red", ls="--")
axes[1].set_xlabel("Residual (Actual − Predicted)")
axes[1].set_ylabel("Frequency")
axes[1].set_title(f"Residual Distribution (MAE={mean_absolute_error(y_test, y_pred):.3f})")

plt.tight_layout()
plt.savefig(str(PROJECT_ROOT / "data/processed/prediction_analysis.png"), dpi=150)
plt.show()

# %% [markdown]
# ## 4. Feature Importance

# %%
if hasattr(model, "feature_importances_"):
    importance = pd.Series(model.feature_importances_, index=feature_names)
    importance = importance.sort_values(ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    importance.plot(kind="barh", ax=ax, color="#1abc9c")
    ax.set_xlabel("Feature Importance")
    ax.set_title("Feature Importance (Best Model)")
    plt.tight_layout()
    plt.savefig(str(PROJECT_ROOT / "data/processed/feature_importance.png"), dpi=150)
    plt.show()

# %% [markdown]
# ## 5. SHAP Analysis

# %%
try:
    import shap
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test[:500])
    
    # Summary plot
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test[:500], feature_names=feature_names, show=False)
    plt.title("SHAP Feature Impact on Carbon Emissions")
    plt.tight_layout()
    plt.savefig(str(PROJECT_ROOT / "data/processed/shap_summary.png"), dpi=150, bbox_inches="tight")
    plt.show()
    
    # Bar plot
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test[:500], feature_names=feature_names, 
                      plot_type="bar", show=False)
    plt.title("SHAP Mean Absolute Impact")
    plt.tight_layout()
    plt.savefig(str(PROJECT_ROOT / "data/processed/shap_bar.png"), dpi=150, bbox_inches="tight")
    plt.show()
    
except ImportError:
    print("⚠️ Install shap: pip install shap")
except Exception as e:
    print(f"⚠️ SHAP error: {e}")

# %% [markdown]
# ## 6. Cross-Validation Scores

# %%
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1)
print(f"\n5-Fold Cross-Validation MAE: {-scores.mean():.3f} ± {scores.std():.3f} tonnes CO₂")

r2_scores = cross_val_score(model, X, y, cv=5, scoring="r2", n_jobs=-1)
print(f"5-Fold Cross-Validation R²:  {r2_scores.mean():.4f} ± {r2_scores.std():.4f}")
