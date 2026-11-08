# Credit Risk Classifier

Predicting whether a borrower will **repay** a loan or end up **in collection**. Candidate models are chosen for the problem (a small, imbalanced dataset where decisions must be explainable), then tuned and compared against a business-rule benchmark.

## Highlights

- **Feature engineering:** a weekend-origination flag, gender encoding and one-hot education levels, produced by one reusable function so training and test data are processed identically.
- **Justified model shortlist:** Logistic Regression (primary scoring model), Decision Tree (explainable rules) and SVM (non-linear challenger), benchmarked against a one-line business rule and a majority-class baseline.
- **Leakage-free tuning:** `Pipeline(StandardScaler → classifier)` with `GridSearchCV` on stratified 5-fold cross-validation, selected on macro F1.
- **Imbalance-aware evaluation:** balanced accuracy and recall of the minority `COLLECTION` class on an independent hold-out set.
- **Interpretability:** decision tree visualisation, feature importance and logistic regression coefficients.

## Results (hold-out test set: 54 loans, 14 in collection)

| Model | Balanced accuracy | Macro F1 | Defaults caught | Good borrowers flagged |
|---|---|---|---|---|
| **Decision Tree** (depth 2, balanced) | **0.76** | **0.67** | **13 / 14** | 16 / 40 |
| SVM (RBF, balanced) | 0.74 | 0.67 | 12 / 14 | 15 / 40 |
| Weekend rule (benchmark) | 0.70 | 0.59 | 13 / 14 | 21 / 40 |
| Logistic Regression (balanced) | 0.66 | 0.58 | 11 / 14 | 19 / 40 |
| Always "paid off" (baseline) | 0.50 | 0.43 | 0 / 14 | 0 / 40 |

**Key insight:** loans originated from Friday onwards end in collection about 40% of the time, compared with about 2% for loans issued earlier in the week.

**Selected model:** the depth-2 decision tree. It catches as many defaults as the business rule while flagging fewer good borrowers, and it reads as two simple lending rules. Logistic regression remains the preferred option when a calibrated risk score is needed.

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
