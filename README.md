# DWTS Predictive Modeling & Analysis Project

## Overview

This project implements a **Dual-Target Machine Learning Framework** to analyze and predict success factors in the TV show *Dancing with the Stars* (DWTS). By leveraging historical data, the model differentiates between technical performance (**Judge Scores**) and public popularity (**Fan Votes**), dealing with the discrepancy between professional evaluation and audience preference using **XGBoost** and **SHAP** interpretability mechanisms.

## Project Structure

```text
MCM-2026-Problem-C/
├── README.md                        # Project documentation
├── data_processor.py                # Feature engineering & dataset construction
├── xgboost_model.py                 # XGBoost training & SHAP analysis
├── visualization_1.py               # 3D/2D Viewership trend analysis
├── visualization_2.py               # Competitor industry distribution analysis
├── datasets/
│   ├── 2026_MCM_Problem_C_Data.csv  # Raw competition data
│   ├── Fan_Votes.csv                # Fan engagement metrics
│   ├── Total_Votes.csv              # Aggregated voting data
│   └── DWTS_XGBoost_Training_Data.csv # Processed model input
└── visualization_outputs/           # Generated charts & figures
```

## Core Methodology

### 1. Feature Engineering & Preprocessing
Data is transformed to quantify qualitative attributes into computable features:
- **Industry Clustering**: Mapping diverse occupations (e.g., 'Astronaut', 'Influencer') into 5 refined categories (`Athletes`, `Performing_Arts`, etc.).
- **Pro-Partner Quantization**: 
  - `Pro_Experience`: Number of past seasons competed.
  - `Pro_Pedigree`: Historical average placement of the professional partner.
  - `Pro_Lift`: Impact factor relative to global averages.

### 2. Dual-Target XGBoost Modeling
Two independent **XGBoost Regressors** are trained to capture distinct evaluation criteria:
- **Model A (Technical)**: Predicts `Target_Judge_Scores_Sum`.
- **Model B (Popularity)**: Predicts `Log(Target_Fan_Votes)` (Log-transformed to handle skewness).
- **Hyperparameters**: `n_estimators=2000`, `max_depth=5`, `learning_rate=0.01`.

### 3. SHAP Interpretability Analysis
Uses **TreeExplainer** to calculate SHAP values ($ \phi_i $), identifying feature contributions:
- **Contrast Analysis**: Comparing feature importance rankings between Judge Model and Fan Model to reveal divergence in evaluation standards.

## Quick Start

### Installation
Ensure Python 3.8+ and install dependencies:
```bash
pip install pandas numpy matplotlib seaborn xgboost shap scikit-learn openpyxl
```

### Usage
Execute the pipeline in the following order:

1. **Process Data**:
   ```bash
   python data_processor.py
   ```
   *Generates `datasets/DWTS_XGBoost_Training_Data.csv`*

2. **Train Models & Analyze**:
   ```bash
   python xgboost_model.py
   ```
   *Outputs SHAP summary plots and Correlation heatmaps.*

3. **Visualize Trends**:
   ```bash
   python visualization_1.py
   python visualization_2.py
   ```

## Outputs & Visualization

The project generates high-resolution figures for report integration:

- **Model Analysis**:
  - `SHAP_Judge_Impact_Factors.png`: Drivers of technical scores.
  - `SHAP_Fan_Impact_Factors.png`: Drivers of audience voting.
  - `Feature_Importance_Comparison.png`: Side-by-side importance bar chart.
- **Trend Analysis**:
  - `DWTS_Viewership_Combined_Visualization.png`: 3D surface plot of viewership over seasons/weeks.
  - `Industry_Comparison_Visualization.png`: Demographics distribution pie charts.

## Key Features

- **Differentiated Prediction**: Explicitly models the "Judge-Fan Gap" rather than treating success as a single metric.
- **Robust Feature Construction**: Introduces specific "Pro Stats" to isolate the partner's impact from the celebrity's ability.
- **Interpretable AI**: Moving beyond black-box predictions to explain *why* certain contestants succeed.

---

**Data Sources**: DWTS Historical Data, Fan Voting Estimates.  
**Models**: Gradient Boosting Trees (XGBoost), SHAP value estimation.  
**Time Range**: Comprehensive historical seasons.
