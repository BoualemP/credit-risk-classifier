# Credit Risk Classifier

Classification model that predicts, at origination time, whether a borrower will repay a loan or end up in collection. Built on a small, imbalanced dataset of 400 loans, with a focus on explainable decisions.

## Results

Hold-out test set: 54 loans, 14 in collection.

| Model | Balanced accuracy | Macro F1 | Defaults caught | Good borrowers flagged |
|---|---|---|---|---|
| **Decision Tree** (depth 2) | **0.76** | **0.67** | **13 / 14** | 16 / 40 |
| SVM | 0.74 | 0.67 | 12 / 14 | 15 / 40 |
| Weekend rule (benchmark) | 0.70 | 0.59 | 13 / 14 | 21 / 40 |
| Logistic Regression | 0.66 | 0.58 | 11 / 14 | 19 / 40 |
| Always "paid off" (baseline) | 0.50 | 0.43 | 0 / 14 | 0 / 40 |

## Key findings

- Loans originated from Friday onwards end in collection about 40% of the time, against about 2% earlier in the week.
- The selected decision tree catches as many defaults as a simple business rule while flagging fewer good borrowers, and every decision reads as two lending rules.
- Class weighting matters more than the choice of algorithm on this imbalanced dataset.
- Logistic regression remains the preferred option when a risk score is needed rather than a yes/no decision.

## Methodology

- Exploratory analysis of borrower profiles against loan outcomes
- Feature engineering on origination date, gender and education
- Stratified cross-validation with leakage-free pipelines
- Evaluation focused on the minority class, against a business rule and a naive baseline

## Tech stack

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · Jupyter

## Project structure

```
credit-risk-classifier/
├── credit_risk_classification.ipynb   # full analysis
├── requirements.txt
└── README.md
```

## Author

**Boualem Pereira**
