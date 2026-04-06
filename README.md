# march-madness-ml
Machine Learning tool to assign win probabilities to march madness games and run full tournament simulations.

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

*The following evaluations are based on walk forward cross validation from 2015-2025 excluding 2020 (no tournament)*

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

*AUC ROC narrowly missed target, but other metrics exceeded targets considerably*

## 2026 Tournament Results and Validation
An assessment of the model's predictions compared to true outcomes of the 2026 March 
Madness Tournament, see simulation results in section below.

*Note: The 2026 tournament produced only 6 upsets by the official NCAA definition, 
making it a relatively chalk-friendly field that generally favors efficiency-based models.*

### Hits
* UConn entered as the model's **#2 championship probability (11.10%)** and reached 
the national championship game as a finalist
* All four Final Four teams (Michigan, Arizona, UConn, Illinois) were identified in 
the model's **top 11 Final Four probabilities** pre-tournament, with Michigan at 37.39%, 
Arizona at 35.63%, UConn at 23.67%, and Illinois at 15.68%
* Championship game implied probabilities (**Michigan: 74.5%, UConn: 25.5%**) closely 
matched opening sports betting lines of -280 to -300 for Michigan, translating to implied 
win probabilities of **73.88% to 75.00%** — within 1% of the model's output
* Illinois was identified as a legitimate Final Four contender at **15.68% Final Four 
probability** despite being a 3-seed, correctly anticipating their run over Houston and Iowa

### Misses
* Duke was overvalued as the outright championship favorite at **17.93%**, falling in 
the Elite Eight to UConn — worth noting that UConn eliminating Duke was within the model's 
probability space given UConn's **#2 championship ranking**
* Gonzaga (**4.91%**) and Kansas (**4.56%**) were both overvalued relative to their 
performance, with neither advancing past the Round of 32
* Florida was assigned a **34.51% Final Four probability** — the model's largest miss — 
as they were eliminated by 9-seed Iowa in the Round of 32
* No meaningful signal toward Iowa's Cinderella run, likely due to their **3-7 record 
in their final 10 regular season games** suppressing their last 10 game win % feature, 
despite underlying efficiency metrics suggesting a stronger team — a candidate feature 
to revisit in future iterations

## Full Bracket Simulations
*the following are win probabilities generated from 10,000 monte carlo simulations, teams that don't show 
up did not win in any trials, all data is pre-tournament*

========== Championship Probabilities ==========
1. Duke (1):    17.93%
2. UConn (2):   11.10%
3. Michigan (1):        9.99%
4. Arizona (1): 8.68%
5. Purdue (2):  4.97%
6. Gonzaga (3): 4.91%
7. Florida (1): 4.87%
8. Iowa State (2):      4.86%
9. Kansas (4):  4.56%
10. Michigan State (3): 4.17%
11. Virginia (3):       3.64%
12. Houston (2):        3.00%
13. Louisville (6):     1.92%
14. Illinois (3):       1.50%
15. St. John's (5):     1.38%
16. Nebraska (4):       1.36%
17. Alabama (4):        1.25%
18. Arkansas (4):       1.18%
19. Texas Tech (5):     1.08%
20. Wisconsin (5):      1.02%
21. UCLA (7):   0.97%
22. BYU (6):    0.70%
23. Ohio State (8):     0.66%
24. Villanova (8):      0.49%
25. Tennessee (6):      0.38%
26. UCF (10):   0.38%
27. Saint Louis (9):    0.28%
28. Vanderbilt (5):     0.26%
29. Miami (7):  0.26%
30. South Florida (11): 0.24%
31. North Carolina (6): 0.23%
32. Georgia (8):        0.22%
33. Santa Clara (10):   0.21%
34. Northern Iowa (12): 0.21%
35. TCU (9):    0.20%
36. Kentucky (7):       0.14%
37. Texas (11): 0.13%
38. Saint Mary's (7):   0.10%
39. Clemson (8):        0.07%
40. Missouri (10):      0.07%
41. Utah State (9):     0.07%
42. Miami (OH) (11):    0.07%
43. VCU (11):   0.06%
44. Akron (12): 0.04%
45. North Dakota State (14):    0.04%
46. Iowa (9):   0.04%
47. Texas A&M (10):     0.03%
48. High Point (12):    0.02%
49. Tennessee State (15):       0.02%
50. Kennesaw State (14):        0.01%
51. Hofstra (13):       0.01%
52. Furman (15):        0.01%
53. Siena (16): 0.01%

========== Final Four Probabilities ==========
1. Duke (1):    40.86%
2. Michigan (1):        37.39%
3. Arizona (1): 35.63%
4. Florida (1): 34.51%
5. UConn (2):   23.67%
6. Purdue (2):  22.74%
7. Iowa State (2):      22.68%
8. Houston (2): 22.65%
9. Gonzaga (3): 18.87%
10. Virginia (3):       17.54%
11. Illinois (3):       15.68%
12. Nebraska (4):       13.92%
13. Michigan State (3): 11.04%
14. Kansas (4): 10.09%
15. Alabama (4):        7.87%
16. Arkansas (4):       7.15%
17. Texas Tech (5):     5.11%
18. Wisconsin (5):      5.01%
19. Louisville (6):     4.62%
20. Vanderbilt (5):     3.77%
21. St. John's (5):     3.71%
22. BYU (6):    3.30%
23. North Carolina (6): 2.97%
24. Villanova (8):      2.81%
25. Tennessee (6):      2.60%
26. Miami (7):  2.32%
27. Saint Mary's (7):   2.24%
28. UCLA (7):   2.14%
29. Georgia (8):        1.74%
30. Kentucky (7):       1.61%
31. Clemson (8):        1.56%
32. Saint Louis (9):    1.33%
33. Ohio State (8):     1.32%
34. Santa Clara (10):   1.17%
35. Iowa (9):   1.07%
36. UCF (10):   0.97%
37. Utah State (9):     0.70%
38. Texas A&M (10):     0.66%
39. Missouri (10):      0.57%
40. VCU (11):   0.52%
41. TCU (9):    0.50%
42. South Florida (11): 0.50%
43. Texas (11): 0.50%
44. Akron (12): 0.44%
45. Northern Iowa (12): 0.44%
46. Miami (OH) (11):    0.35%
47. Idaho (15): 0.22%
48. Queens University (15):     0.17%
49. High Point (12):    0.17%
50. McNeese (12):       0.11%
51. Tennessee State (15):       0.08%
52. North Dakota State (14):    0.08%
53. Hofstra (13):       0.05%
54. Troy (13):  0.05%
55. Hawai'i (13):       0.04%
56. Prairie View A&M (16):      0.04%
57. Wright State (14):  0.03%
58. California Baptist (13):    0.03%
59. Pennsylvania (14):  0.03%
60. Kennesaw State (14):        0.02%
61. Furman (15):        0.02%
62. Howard (16):        0.01%
63. Siena (16): 0.01%

## Sweet 16 Simulations
*Simulations run from the actual 2026 Sweet 16; it is worth noting the model only uses 
pre tournament data, so these probabilities more closely
reflect each teams chance to win if we knew they would be in the sweet 16 pre-tournament*

========== Championship Probabilities ==========
1. Duke (1):    23.39%
2. Michigan (1):        16.73%
3. Arizona (1): 15.02%
4. UConn (2):   11.74%
5. Houston (2): 6.93%
6. Purdue (2):  5.92%
7. Iowa State (2):      5.38%
8. Michigan State (3):  4.08%
9. St. John's (5):      2.27%
10. Nebraska (4):       2.25%
11. Alabama (4):        1.70%
12. Arkansas (4):       1.40%
13. Illinois (3):       1.32%
14. Texas (11): 0.82%
15. Tennessee (6):      0.72%
16. Iowa (9):   0.33%

========== Final Four Probabilities ==========
1. Duke (1):    57.89%
2. Arizona (1): 56.99%
3. Michigan (1):        54.53%
4. Houston (2): 52.20%
5. Iowa State (2):      29.82%
6. Purdue (2):  29.09%
7. UConn (2):   27.58%
8. Nebraska (4):        23.08%
9. Illinois (3):        18.76%
10. Alabama (4):        9.94%
11. Michigan State (3): 9.07%
12. Arkansas (4):       8.62%
13. Iowa (9):   5.96%
14. Tennessee (6):      5.71%
15. St. John's (5):     5.46%
16. Texas (11): 5.30%

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
