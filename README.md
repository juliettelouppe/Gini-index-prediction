Juliette LOUPPE

# Economic Inequality Prediction: Gini Index Modeling
Predicting economic wages inequality (Gini coefficient) using socio-economic data and machine learning models.

# Research Question
Which machine learning model best predicts a country's Gini Index (economic inequality)  
based on macroeconomic and social indicators?

The project compares multiple models (Random Forest, XGBoost, Linear Regression, etc.)  
to evaluate which one achieves the best predictive performance.

# Setup
# 1. Create the environment
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
.venv\Scripts\activate     # Windows
pip install -r requirements.txt

## Usage 
python main.py

Expected output:
A cvs file results/model_comparison.csv summarizing model performance.

## Project Structure
Gini-index-prediction/
│
├── main.py                         # Main entry point to run the full pipeline
│
├── notebooks/                      # Jupyter notebooks (full analysis)
│   ├── 01-data-loading.ipynb       # Downloads raw World Bank indicators
│   ├── 02-eda.ipynb                # Cleans and preprocesses inequality dataset
│   ├── 03-modeling.ipynb           # Train–test split 80/20 + model training
│   └── 04-results.ipynb            # Final evaluation + visualizations
│
├── src/                            # Source code
│   ├── data_loader.py              # Downloads raw World Bank indicators
│   ├── preprocessing.py            # Cleans and preprocesses inequality dataset
│   ├── split_data.py               # Train–test split 80/20
│   ├── models.py                   # Model training and comparison
│   └── plot_feature_importance.py  # Feature importance plots
│
├── data/                           # Raw and processed datasets
│
├── results/                        # Stored model outputs and figures
│   ├── boxplot_gini_us_fr_br.png
│   ├── heatmap_correlations.png
│   ├── model_comparison.csv
│   ├── rf_feature_importances.csv
│   ├── top10_inequality.png
│   ├── X_test.csv
│   ├── X_train.csv
│   ├── xgb_feature_importances_plot.png
│   ├── xgb_feature_importances.csv
│   ├── xgb_true_vs_pred.png
│   ├── y_test.csv
│   └── y_train.csv
│
└── requirements.txt                # Reproducible Python environment


## Results
model, r2, rmse, mae
Linear Regression, 0.6066, 5.4935, 4.3785
Random Forest , 0.9287, 2.3381, 1.6117
XGBoost, 0.9383, 2.1749, 1.6435

Best Model: XGBoost (typically shows best performance on nonlinear relationships)

## Requirements
python 3.11, pandas, numpy, statsmodels, jupyter scikit-learn, xgboost, matplotlib, seaborn, openxyl (for Excel reading)
