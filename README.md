# MSDS590_Capstone
Forecasting stress, anxiety, and affect using time-series analysis of HRV and EDA from the WESAD dataset.

## 📌 Overview

This repository will contain all code, data processing steps, models, and results for my Biomedical Data Science Capstone project at Meharry Medical College. 

The project explores whether physiological signals from wearable devices—specifically Electrodermal Activity (EDA) and Heart Rate Variability (HRV)—can be used to forecast minute-by-minute self-reported psychological states. These states include:

- Stress levels (3-class classification)
- Anxiety levels (3-class classification)
- Negative affect (regression)
- Positive affect (regression) (excluded from final evaluation due to weak label validation)

## 🎯 Objectives
- Forecast the next minute of self-reported pyschological states (stress, anxiety, affect) using HRV and EDA data.
- Compare performance between static and time-series modeling approaches.
- Identify the most predictive physiological features.
- Simulate real-time use while ensuring models generalize across unseen participants.

## 🔬 Methodology Summary
- Dataset: WESAD (Wearable Stress and Affect Detection Dataset)
- Signal Processing: Raw EDA and ECG signals segmented into 1-minute windows using NeuroKit2.
- Model Types:
- > Classification (stress, anxiety)
- > Regression (positive and negative affect)
- Algorithms: Random Forest, LightGBM, XGBoost, AdaBoost, Logistic Regression, Linear Regression, SVM
- Evaluation: GroupKFold cross-validation (k=5) to prevent data leakage across individuals.
- Metrics: Accuracy, F1-score, AUC (for classification); R², MAE, MSE (for regression)

🔎 Note: PANAS-based positive affect scores did not validate well against SAM valence, so PA results were excluded from final conclusions.
## 📈 Key Findings
- Time-series models consistently outperformed static models for all tasks.
- Best performing models achieved:
- > Over 85% test accuracy for stress and anxiety classification.
  > R² = 0.68 for negative affect regression (SVM model).
- Most important predictors:
- > SCL_Mean (EDA) and HRV_HF (High-Frequency HRV)
  > Temporal/lags in physiological features were essential for predictive accuracy.

## ⚠️ Limitations
- Small sample size (15 participants, mostly male).
- Data collected in controlled lab environments—results may not generalize to daily life.
- Fairness assessments across demographic subgroups not feasible.
- PANAS PA scores poorly captured momentary positive affect in this setting.


## 🗃️ Directory Structure
```
MSDS590_Capstone/
├── data/                                            # Raw and Preprocessed files
│   ├── processed data
│
│   ├── raw data link
│
├── notebooks/                                       # Jupyter notebooks for analysis & modeling
│
│   ├── 00_Data_Extraction_and_Preprocessing.ipynb
│   ├── 01_static_stress_modeling.ipynb
│   ├── 02_time-series_stress_modeling.ipynb
│   ├── 03_static_anxiety_modeling.ipynb
│   ├── 04_time-series_anxiety_modeling.ipynb
│   ├── 05_static_PA_modeling.ipynb
│   ├── 06_time-series_PA_modeling.ipynb
│   ├── 07_static_NA_modeling.ipynb
│   ├── 08_time-series_NA_modeling.ipynb
│   ├── 09_EDA_and_Visualizations.ipynb
│
├── figures/                     # Model performance plots
├── results/                     # Evaluation outputs and tables
├── scripts/                      # Custom scripts
└── README.md                    # Project overview
```


