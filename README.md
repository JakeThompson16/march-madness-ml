# march-madness-ml
Machine Learning tool to assign win probabilities to march madness games and construct brackets with differing levels of variance.

## Model Win Probabilities
*based on 100,000 simulations*

- Houston (2):                                 29.428%
- Michigan (1):                                10.342%
- Duke (1):                                    8.576%
- Gonzaga (3):                                 8.484%
- Illinois (3):                                6.845%
- Iowa State (2):                              6.821%
- Arizona (1):                                 4.636%
- Purdue (2):                                  4.169%
- Nebraska (4):                                2.796%
- Florida (1):                                 2.545%
- UConn (2):                                   2.055%
- Arkansas (4):                                1.796%
- Texas Tech (5):                              1.669%
- Louisville (6):                              1.607%
- Wisconsin (5):                               1.025%
- Michigan State (3):                          0.892%
- Virginia (3):                                0.66%
- Vanderbilt (5):                              0.644%
- BYU (6):                                     0.617%
- UCLA (7):                                    0.581%
- Alabama (4):                                 0.347%
- Iowa (9):                                    0.346%
- Kansas (4):                                  0.331%
- Villanova (8):                               0.301%
- Kentucky (7):                                0.279%
- Georgia (8):                                 0.27%
- Tennessee (6):                               0.265%
- Ohio State (8):                              0.264%
- North Carolina (6):                          0.229%
- Saint Mary's (7):                            0.193%
- St. John's (5):                              0.188%
- SMU (11):                                     0.184%
- Utah State (9):                              0.168%
- Santa Clara (10):                             0.138%
- Clemson (8):                                 0.057%
- Miami (7):                                   0.051%
- Northern Iowa (12):                           0.05%
- UCF (10):                                     0.043%
- South Florida (11):                           0.038%
- Saint Louis (9):                             0.038%
- Akron (12):                                   0.007%
- Texas A&M (10):                               0.006%
- Texas (11):                                   0.006%
- TCU (9):                                     0.004%
- McNeese (12):                                 0.003%
- VCU (11):                                     0.003%
- High Point (12):                              0.002%
- Hofstra (13):                                 0.001%


## Model Performance

*The following evaluations are based on training from 2019,2021,2022,2023 and testing on 2024 season*

- Log Loss:  0.5607
- Brier:     0.1916
- AUC-ROC:   0.7127
- Accuracy:  0.7015

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