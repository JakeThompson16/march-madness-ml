# march-madness-ml
Machine Learning tool to assign win probabilities to march madness games and construct brackets with differing levels of variance.

## The Model
Using an ensemble of XGBoost and Random Forest models (using the same features).
Initially tested using Logistic Regression models which struggled in cross validation
especially in heavy upset seasons. Also tried to implement Neural Network, but did not have 
enough meaningful training data from tournament games, and could not establish high signal
features to train using regular season games.

## Metrics
*targets indicate a strong model*
- log loss: measures accuracy of predictions with regard to confidence (target < .60)
- brier score: mean squared difference between actual outcome and probability predicted (target < .20)
- auc roc: accuracy of model ranking teams against each other (target > .70)
- accuracy: binary indicator of rate that higher prob team wins (target > .70)


## Model Performance

*The following evaluations are based on 8-fold cross validation from 2015-2025 excluding 2020 (no tournament)*

========== Random Forest Averages ==========
- log_loss: 0.5549
- brier: 0.1865
- auc_roc: 0.6902
- accuracy: 0.7297

========== XGBoost Averages ==========
- log_loss: 0.5629
- brier: 0.1897
- auc_roc: 0.6671
- accuracy: 0.7302

========== Ensemble Averages ==========
- log_loss: 0.5550
- brier: 0.1867
- auc_roc: 0.6878
- accuracy: 0.7260

## Simulations
*the following are win probabilities generated from 10,000 monte carlo simulations*

========== Championship Probabilities ==========
* Duke (1):     20.9%
* UConn (2):    12.5%
* Arizona (1):  8.3%
* Michigan (1): 8.2%
* Kansas (4):   5.2%
* Michigan State (3):   4.9%
* Gonzaga (3):  4.7%
* Iowa State (2):       4.7%
* Purdue (2):   4.2%
* Virginia (3): 4.1%
* Florida (1):  2.9%
* Houston (2):  2.4%
* Louisville (6):       1.7%
* St. John's (5):       1.6%
* Alabama (4):  1.5%
* Nebraska (4): 1.3%
* Illinois (3): 1.2%
* Arkansas (4): 1.2%
* UCLA (7):     1.0%
* Texas Tech (5):       0.8%
* Ohio State (8):       0.8%
* Wisconsin (5):        0.8%
* UCF (10):     0.5%
* Villanova (8):        0.5%
* BYU (6):      0.4%
* Tennessee (6):        0.4%
* Georgia (8):  0.4%
* Miami (7):    0.3%
* Vanderbilt (5):       0.3%
* Santa Clara (10):     0.3%
* Kentucky (7): 0.2%
* South Florida (11):   0.2%
* TCU (9):      0.2%
* Saint Louis (9):      0.2%
* Saint Mary's (7):     0.2%
* Northern Iowa (12):   0.2%
* North Carolina (6):   0.1%
* Texas (11):   0.1%
* Utah State (9):       0.1%
* Akron (12):   0.1%
* Iowa (9):     0.1%
* Miami (OH) (11):      0.1%
* Missouri (10):        0.1%
* Clemson (8):  0.1%
* Siena (16):   0.0%
* Texas A&M (10):       0.0%
* Hofstra (13): 0.0%
* Wright State (14):    0.0%
* Furman (15):  0.0%
* High Point (12):      0.0%
* VCU (11):     0.0%
* Queens University (15):       0.0%
* North Dakota State (14):      0.0%

========== Final Four Probabilities ==========
* Duke (1):     40.4%
* Michigan (1): 37.3%
* Arizona (1):  35.7%
* Florida (1):  33.1%
* UConn (2):    23.4%
* Iowa State (2):       23.2%
* Houston (2):  22.8%
* Purdue (2):   22.8%
* Gonzaga (3):  19.0%
* Illinois (3): 16.9%
* Virginia (3): 16.9%
* Nebraska (4): 14.1%
* Kansas (4):   11.0%
* Michigan State (3):   10.6%
* Alabama (4):  8.2%
* Arkansas (4): 7.8%
* Texas Tech (5):       4.9%
* Wisconsin (5):        4.9%
* Louisville (6):       4.3%
* St. John's (5):       3.7%
* Vanderbilt (5):       3.5%
* Villanova (8):        3.0%
* North Carolina (6):   2.6%
* BYU (6):      2.6%
* Tennessee (6):        2.6%
* Saint Mary's (7):     2.4%
* UCLA (7):     2.2%
* Miami (7):    2.1%
* Georgia (8):  1.8%
* Ohio State (8):       1.7%
* Clemson (8):  1.7%
* Kentucky (7): 1.6%
* Santa Clara (10):     1.1%
* Saint Louis (9):      1.1%
* UCF (10):     1.1%
* Iowa (9):     1.0%
* Texas A&M (10):       0.8%
* Missouri (10):        0.7%
* Utah State (9):       0.7%
* Akron (12):   0.6%
* VCU (11):     0.5%
* South Florida (11):   0.5%
* Texas (11):   0.5%
* TCU (9):      0.5%
* Miami (OH) (11):      0.4%
* Northern Iowa (12):   0.4%
* Idaho (15):   0.2%
* High Point (12):      0.2%
* Hofstra (13): 0.2%
* McNeese (12): 0.1%
* Prairie View A&M (16):        0.1%
* Hawai'i (13): 0.1%
* Wright State (14):    0.1%
* Queens University (15):       0.1%
* Furman (15):  0.1%
* Pennsylvania (14):    0.1%
* Kennesaw State (14):  0.1%
* Troy (13):    0.0%
* Siena (16):   0.0%
* North Dakota State (14):      0.0%
* California Baptist (13):      0.0%
* Tennessee State (15): 0.0%
* Howard (16):  0.0%

## Data Sources
 - College Basketball Data (CBBD)
 - sportsdataverse (sdv)
 - Kaggle

## Features
*Each feature is included for both teams independently*
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
