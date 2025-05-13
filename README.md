# Predicting Mortgage Defaults Using Neural Networks

Meili Gupta, Salahuddin Khan, Rheia Edgar-Nemec

# Abstract

The mortgage industry, a 20.2 trillion dollar sector in the
U.S., relies heavily on third-party vendors for borrower risk
assessment. This often comes at the cost of transparency
and flexibility. This project investigates whether machine
learning can offer an internal, cost-effective alternative by
predicting the likelihood of mortgage loan default. We im-
plemented four classification models; Logistic Regression,
Support Vector Machine, Naive Bayes, and a fully con-
nected Artificial Neural Network (ANN). Each were trained
from scratch and evaluated using accuracy, precision, re-
call, and F1 score. The ANN outperformed all other mod-
els, achieving the highest F1 score and demonstrating strong
performance in identifying potential defaulters. Results are
deployed via an interactive Streamlit app that allows finan-
cial analysts to upload mortgage datasets and receive pre-
dictive insights. This work not only validates the use of
custom-built ML tools in the mortgage space but also con-
tributes a replicable and accessible framework for financial
institutions seeking greater control over risk modeling.

# Repository Content

```
.
├── .github/workflows/           # GitHub Actions config
├── pages/                       # Streamlit app pages (A, B, C)
├── .gitignore                   # Files/folders to ignore in version control
├── Loan_default.csv             # Dataset
├── README.md                    # Project overview and instructions
├── experiments.ipynb            # Data pre-processing, imports models, runs experiments
├── helper_functions.py          # Utility functions
├── mortgage.py                  # Streamlit app entry point
├── setup.cfg                    # Python package configuration
```

# Run Streamlit

```
streamlit run mortgage.py
```


