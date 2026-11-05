# Benchmark (validation set)

Train 27,978 URLs, validation 6,120 URLs, 29 features.

| Model | Accuracy | Precision | Recall | F1 | FPR |
|---|---|---|---|---|---|
| Logistic Regression | 88.6 % | 92.1 % | 85.7 % | 88.8 % | 8.2 % |
| Decision Tree | 86.2 % | 90.8 % | 82.0 % | 86.2 % | 9.2 % |
| Random Forest | 89.4 % | 90.6 % | 89.0 % | 89.8 % | 10.2 % |
| LightGBM | 90.0 % | 92.0 % | 88.6 % | 90.3 % | 8.5 % |

Training time: Logistic Regression 0.0 s, Decision Tree 0.1 s, Random Forest 0.6 s, LightGBM 1.3 s

## Learning curve (LightGBM, validation F1)

| Training URLs | F1 |
|---|---|
| 6,994 (25%) | 89.5 % |
| 13,989 (50%) | 90.2 % |
| 20,983 (75%) | 90.0 % |
| 27,978 (100%) | 90.3 % |
