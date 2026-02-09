import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


# Step 0: Global settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300


# Step 1: Data loading
print("Loading XGBoost training data...")
filename = 'datasets/DWTS_XGBoost_Training_Data.csv'
df = pd.read_csv(filename)

exclude_cols = ['season', 'week', 'celebrity_name', 'Target_Judge_Scores_Sum', 'Target_Fan_Votes']
feature_cols = [c for c in df.columns if c not in exclude_cols]

print(f"Input Feature List ({len(feature_cols)} items):")
print(f"    {feature_cols}")


# Step 2: Spearman Correlation Analysis
print("\nStart spearman correlation analysis...")

# Calculate correlation matrix
corr_matrix = df[feature_cols].corr(method='spearman')

# Draw heatmap
plt.figure(figsize=(16, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', center=0, square=True, linewidths=1)
plt.title('Spearman Correlation Heatmap (Constructed Features)', fontsize=16)
plt.tight_layout()
plt.savefig('visualization_outputs/Feature_Correlation_Heatmap.png')

print("    Spearman Correlation Heatmap saved.")

threshold = 0.85
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [c for c in upper.columns if any(upper[c].abs() > threshold)]

if to_drop:
    print(f"    High correlation detected, considering removal: {to_drop}")
    # feature_cols = [f for f in feature_cols if f not in to_drop]

else:
    print("    No extremely collinear features detected.")


# Step 3: Train XGBoost models
# Prepare final training data
X = df[feature_cols]
y_judge = df['Target_Judge_Scores_Sum']
y_fan_log = np.log1p(df['Target_Fan_Votes'])

print("\nTraining XGBoost models...")

xgb_params = {
    'n_estimators': 2000,
    'learning_rate': 0.01,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_jobs': -1,
    'random_state': 2026
}

# Model A: Judge Scores
model_judge = xgb.XGBRegressor(**xgb_params)
model_judge.fit(X, y_judge)
r2_judge = model_judge.score(X, y_judge)
print(f"    Model A (Judge Scores) R2 Score: {r2_judge:.4f}")

# Model B: Fan Votes
model_fan = xgb.XGBRegressor(**xgb_params)
model_fan.fit(X, y_fan_log)
r2_fan = model_fan.score(X, y_fan_log)
print(f"    Model B (Fan Votes) R2 Score: {r2_fan:.4f}")


# Step 4: SHAP analysis
print("\nCalculating SHAP values...")

explainer_judge = shap.TreeExplainer(model_judge)
shap_values_judge = explainer_judge.shap_values(X)

explainer_fan = shap.TreeExplainer(model_fan)
shap_values_fan = explainer_fan.shap_values(X)

print("\nGenerating different analysis plots...")

# Plot 1: Judge Impact Factors Beeswarm Plot
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values_judge, X, show=False, cmap='viridis')
plt.title("Factor Impact on Judge Scores (What Judges Care About)", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('visualization_outputs/SHAP_Judge_Impact_Factors.png')

# Plot 2: Fan Impact Factors Beeswarm Plot
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values_fan, X, show=False, cmap='magma')
plt.title("Factor Impact on Fan Votes (What Fans Care About)", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('visualization_outputs/SHAP_Fan_Impact_Factors.png')

# Calculate global importance (mean absolute SHAP)
importance_j = np.abs(shap_values_judge).mean(axis=0)
importance_f = np.abs(shap_values_fan).mean(axis=0)
scaler = MinMaxScaler()
imp_normalized = scaler.fit_transform(np.column_stack((importance_j, importance_f)))

# Build dataframe for plotting
df_imp = pd.DataFrame({
    'Feature': X.columns,
    'Judge_Importance': imp_normalized[:, 0],
    'Fan_Importance': imp_normalized[:, 1]
})

df_imp = df_imp.sort_values('Judge_Importance', ascending=True)

# Start plotting
fig, ax = plt.subplots(figsize=(12, 10))

y_pos = np.arange(len(df_imp))
bar_height = 0.35

# Plot blue bar chart (Judge Scores)
rects_j = ax.barh(y_pos + bar_height/2, df_imp['Judge_Importance'], bar_height, label='Importance to Judges', color='#3498DB', alpha=0.9, edgecolor='white')

# Plot red bar chart (Fan Votes)
rects_f = ax.barh(y_pos - bar_height/2, df_imp['Fan_Importance'], bar_height, label='Importance to Fans', color='#E74C3C', alpha=0.9, edgecolor='white')

# Chart Beautification
ax.set_yticks(y_pos)
ax.set_yticklabels(df_imp['Feature'], fontsize=12)
ax.set_xlabel('Normalized Feature Importance (0-1)', fontsize=12)
ax.set_title("Feature Importance Comparison (Judge Scores and Fan Votes)", fontsize=16, fontweight='bold')
ax.legend(loc='lower right', fontsize=12, frameon=True)

plt.tight_layout()
plt.savefig('visualization_outputs/Feature_Importance_Comparison_Visualization.png')

print("\nSuccessfully generated:")
print("1. Feature_Correlation_Heatmap.png")
print("2. SHAP_Judge_Impact_Factors.png")
print("3. SHAP_Fan_Impact_Factors.png")
print("4. Feature_Importance_Comparison_Visualization.png")
