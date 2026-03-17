# march-madness-ml
Machine Learning tool to assign win probabilities to march madness games and construct brackets with differing levels of variance.

## Model Performance

*The following evaluations are based on training from 2019,2021,2022,2023 and testing on 2024 season*

Log Loss:  0.5832

Brier:     0.2063

AUC-ROC:   0.7226

Accuracy:  0.6716



## Data Sources
 - College Basketball Data (CBBD)
 - sportsdataverse (sdv)
 - Kaggle

## Features
 - Seed (Kaggle)
 - AdjEM (CBBD)
 - AdjO (CBBD)
 - AdjD (CBBD)
 - eFG% (CBBD)
 - Turnover rate (CBBD)
 - oRB% (CBBD)
 - Free throw rate (CBBD)
 - 3 pt rate (CBBD)
 - Tempo (CBBD)
 - Road win percentage (sdv)