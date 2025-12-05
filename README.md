Juliette LOUPPE

# Economic Inequality Prediction: Gini Index Modeling
Predicting economic wages inequality (Gini coefficient) using socio-economic data and machine learning models.

# Research Question
Which machine learning model best predicts a country's Gini Index (economic inequality)  
based on macroeconomic and social indicators?

The project compares multiple models (Random Forest, XGBoost, Linear Regression, etc.)  
to evaluate which one achieves the best predictive performance.

# Setup
# 1. Create the conda environment
```bash
conda env create -f environment.yml
conda activate gini-project

## Usage 
python main.py

Expected output:
A cvs file results/model_comparison.csv summarizing model performance.

## Project Structure
Gini-index-prediction/
│
├── main.py                 # Main entry point to run the full pipeline
│
├── src/                    # Source code
│   ├── data_loader.py      # Downloads raw World Bank indicators
│   ├── preprocessing.py    # Cleans and preprocesses inequality dataset
│   ├── split_data.py       # Train-test split 80/20
│   ├── models.py           # Model training and comparison
│   └── plot_feature_importance.py   # Optional plots
│
├── data/                   # Raw and processed datasets
│
├── results/                # Stored model outputs and figures
│   └── model_comparison.csv
│
└── environment.yml         # Reproducible environment specification

## Results
model, r2, rmse, mae
Linear Regression, 0.6066, 5.4935, 4.3785
Random Forest , 0.9287, 2.3381, 1.6117
XGBoost, 0.9383, 2.1749, 1.6435

Best Model: XGBoost (typically shows best performance on nonlinear relationships)

##Requirements
python 3.11, pandas, numpy, statsmodels, jupyter scikit-learn, xgboost, matplotlib, seaborn, openxyl (for Excel reading)
