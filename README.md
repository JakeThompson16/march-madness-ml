# march-madness-ml
Machine Learning tool to assign win probabilities to march madness games and construct brackets with differing levels of variance.

## The Model
Using an ensemble of Neural Network and Random Forest models (using the same features).
Initially tested using Logistic Regression and XGBoost models which struggled in cross validation
especially in heavy upset seasons. Predictions implement temperature scaling to reduce overconfidence in models.
Neural Network specifically was very overconfident resulting in assigned temperature of 2.22 resulting in significant
flattening. 

## Metrics
*targets indicate a strong model*
- log loss: measures accuracy of predictions with regard to confidence (target < .60)
- brier score: mean squared difference between actual outcome and probability predicted (target < .20)
- auc roc: accuracy of model ranking teams against each other (target > .70)
- accuracy: binary indicator of rate that higher prob team wins (target > .70)


## Model Performance

*The following evaluations are based on 6-fold cross validation from 2019-2025 excluding 2020 (no tournament)*

========== Random Forest Averages ==========
- log_loss: 0.5769
- brier: 0.1960
- auc_roc: 0.6372
- accuracy: 0.7103

========== Neural Network Averages ==========
- log_loss: 0.6857
- brier: 0.2202
- auc_roc: 0.6394
- accuracy: 0.6919

========== Ensemble Averages ==========
- log_loss: 0.5888
- brier: 0.2014
- auc_roc: 0.6464
- accuracy: 0.7041

*Overall, model performance is very strong, especially considering all data sources are free. By using paid KenPom subscription for additional/enhanced features I could likely improve all metrics.*



## Data Sources
 - College Basketball Data (CBBD)
 - sportsdataverse (sdv)
 - Kaggle

## Features
 - Seed (Kaggle)
 - Adjusted Efficiency Margin (CBBD)
 - Adjusted Offensive Efficiency (CBBD)
 - Adjusted Defensive Efficiency (CBBD)
 - Effective FG% (CBBD)
 - Turnover rate (CBBD)
 - Offensive Rebound % (CBBD)
 - Free throw rate (CBBD)
 - 3 pt rate (CBBD)
 - Tempo (CBBD)
 - Road win percentage (sdv) 
 - Last 10 Game Win pct (sdv)
 - Strength of Schedule (calculated from sdv and CBBD data)