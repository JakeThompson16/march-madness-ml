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
*the following are win probabilities generated from 10,000 monte carlo simulations, teams that don't show 
up did not win in any trials*

========== Championship Probabilities ==========
1. Duke (1):    20.62%
2. UConn (2):   12.53%
3. Michigan (1):        8.90%
4. Arizona (1): 7.55%
5. Gonzaga (3): 5.30%
6. Michigan State (3):  5.26%
7. Kansas (4):  4.73%
8. Purdue (2):  4.60%
9. Iowa State (2):      4.49%
10. Virginia (3):       3.88%
11. Florida (1):        3.04%
12. Houston (2):        2.34%
13. Louisville (6):     1.93%
14. St. John's (5):     1.43%
15. Alabama (4):        1.42%
16. Nebraska (4):       1.25%
17. Illinois (3):       1.18%
18. Arkansas (4):       1.10%
19. Wisconsin (5):      1.01%
20. UCLA (7):   0.89%
21. Ohio State (8):     0.81%
22. Texas Tech (5):     0.73%
23. Villanova (8):      0.62%
24. BYU (6):    0.49%
25. UCF (10):   0.34%
26. Santa Clara (10):   0.32%
27. Tennessee (6):      0.31%
28. Georgia (8):        0.31%
29. Miami (7):  0.29%
30. Saint Louis (9):    0.24%
31. Kentucky (7):       0.22%
32. Vanderbilt (5):     0.18%
33. South Florida (11): 0.17%
34. TCU (9):    0.17%
35. North Carolina (6): 0.17%
36. Clemson (8):        0.17%
37. Northern Iowa (12): 0.16%
38. Saint Mary's (7):   0.13%
39. Texas (11): 0.12%
40. Utah State (9):     0.11%
41. Missouri (10):      0.10%
42. Akron (12): 0.09%
43. Miami (OH) (11):    0.08%
44. Texas A&M (10):     0.05%
45. Iowa (9):   0.04%
46. North Dakota State (14):    0.03%
47. Furman (15):        0.03%
48. Long Island University (16):        0.02%
49. High Point (12):    0.02%
50. VCU (11):   0.01%
51. Hofstra (13):       0.01%
52. Tennessee State (15):       0.01%

========== Final Four Probabilities ==========
1. Duke (1):    39.53%
2. Michigan (1):        37.62%
3. Florida (1): 33.92%
4. Arizona (1): 33.84%
5. UConn (2):   23.68%
6. Houston (2): 23.19%
7. Purdue (2):  23.01%
8. Iowa State (2):      22.59%
9. Gonzaga (3): 19.89%
10. Virginia (3):       17.11%
11. Illinois (3):       16.60%
12. Nebraska (4):       13.28%
13. Michigan State (3): 11.46%
14. Kansas (4): 10.70%
15. Alabama (4):        8.26%
16. Arkansas (4):       7.77%
17. Wisconsin (5):      4.85%
18. Texas Tech (5):     4.64%
19. Louisville (6):     4.54%
20. St. John's (5):     3.70%
21. Vanderbilt (5):     3.64%
22. Villanova (8):      3.05%
23. BYU (6):    2.82%
24. North Carolina (6): 2.70%
25. Tennessee (6):      2.36%
26. UCLA (7):   2.34%
27. Miami (7):  2.31%
28. Saint Mary's (7):   2.21%
29. Georgia (8):        1.92%
30. Clemson (8):        1.68%
31. Ohio State (8):     1.67%
32. Kentucky (7):       1.55%
33. Santa Clara (10):   1.45%
34. Saint Louis (9):    1.28%
35. UCF (10):   0.95%
36. Iowa (9):   0.95%
37. Texas A&M (10):     0.87%
38. Utah State (9):     0.72%
39. Texas (11): 0.71%
40. Missouri (10):      0.63%
41. Miami (OH) (11):    0.55%
42. Akron (12): 0.53%
43. South Florida (11): 0.48%
44. VCU (11):   0.47%
45. TCU (9):    0.41%
46. Northern Iowa (12): 0.32%
47. High Point (12):    0.22%
48. Idaho (15): 0.17%
49. McNeese (12):       0.12%
50. Hofstra (13):       0.11%
51. North Dakota State (14):    0.10%
52. Troy (13):  0.09%
53. Furman (15):        0.08%
54. Pennsylvania (14):  0.07%
55. Queens University (15):     0.06%
56. Kennesaw State (14):        0.05%
57. Prairie View A&M (16):      0.04%
58. Long Island University (16):        0.04%
59. Hawai'i (13):       0.03%
60. Tennessee State (15):       0.02%
61. Siena (16): 0.02%
62. California Baptist (13):    0.02%
63. Wright State (14):  0.01%


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
