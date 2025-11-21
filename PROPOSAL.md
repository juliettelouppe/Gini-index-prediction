I will use data from the World Bank and WID.world to predict country-level GINI coefficients and income distribution indicators. The analysis will employ linear regression and random forest models, using features such as female labor force participation, education levels, and GDP. Model performance will be evaluated using an 80/20 train–test split.


#Answer TA with guidance :
Train/test split: Since you have country-year panel data, use temporal validation - train on earlier years, test on recent years. Don't randomly split countries or you'll have data leakage.

Sample size: How many country-year observations do you have? World Bank + WID covers maybe 100-150 countries over 20-30 years, so you should have enough data.

Consider adding gradient boosting (XGBoost) as a third model to compare alongside linear and random forest. Gives you linear vs two nonlinear methods.

Metrics: Report R², RMSE, MAE. GINI is a percentage, so errors are interpretable.

Feature engineering: Consider interaction terms like "female participation × education level" since these likely interact.