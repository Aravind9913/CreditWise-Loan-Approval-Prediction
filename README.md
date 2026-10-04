# CreditWise — Loan Approval Prediction

CreditWise is a small machine-learning project that explores applicant and loan information to predict whether a loan is approved. The project is implemented as a Jupyter notebook and compares several classification models after exploratory data analysis and feature preparation.

> **Suggested project name:** **CreditWise — Loan Approval Prediction**  
> “CreditWise” is short, easy to remember, and describes the project’s credit-risk focus. Check name availability before using it as a public product or brand.

## Project overview

The notebook uses applicant, financial, employment, and loan attributes to predict the `Loan_Approved` outcome (`Yes` or `No`). It includes:

- Initial data inspection and missing-value handling
- Exploratory plots for approval balance, income, education, credit score, and other financial attributes
- Categorical feature encoding and numerical feature scaling
- Feature engineering using squared credit-score and debt-to-income values
- A reproducible 80/20 train/test split
- Comparison of Logistic Regression, K-Nearest Neighbors (KNN), and Gaussian Naive Bayes
- Evaluation with accuracy, precision, recall, F1 score, and a confusion matrix

This is an educational analysis project, not a deployed application or a financial decision system.

## Project files

```text
.
├── credit_wise.ipynb       # Data exploration, preprocessing, model training, and evaluation
├── loan_approval_data.csv  # Applicant and loan dataset (1,000 rows)
└── README.md
```

## Dataset

The CSV contains **1,000 records and 20 columns**, including an applicant identifier and the prediction target. The notebook removes `Applicant_ID` before training.

| Column | Description |
|---|---|
| `Applicant_ID` | Applicant identifier; excluded from model features |
| `Applicant_Income` | Applicant's income |
| `Coapplicant_Income` | Co-applicant's income |
| `Employment_Status` | Employment status |
| `Age` | Applicant's age |
| `Marital_Status` | Marital status |
| `Dependents` | Number of dependents |
| `Credit_Score` | Credit score |
| `Existing_Loans` | Number of existing loans |
| `DTI_Ratio` | Debt-to-income ratio |
| `Savings` | Savings amount |
| `Collateral_Value` | Value of collateral |
| `Loan_Amount` | Requested loan amount |
| `Loan_Term` | Loan term |
| `Loan_Purpose` | Purpose of the loan |
| `Property_Area` | Property area category |
| `Education_Level` | Education category |
| `Gender` | Gender category |
| `Employer_Category` | Employer category |
| `Loan_Approved` | Target label: `Yes` or `No` |

The dataset's original source, collection methodology, units, and license are not documented in the project. Confirm these details before redistributing or using the data outside this project.

## Requirements

- Python 3.10 or newer
- Jupyter Notebook
- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn 1.2 or newer (`OneHotEncoder` uses the `sparse_output` argument)

## Setup and run

Run these commands from the project folder in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install jupyter pandas numpy matplotlib seaborn "scikit-learn>=1.2"
jupyter notebook credit_wise.ipynb
```

If PowerShell blocks virtual-environment activation, either allow activation according to your machine's policy or run the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install jupyter pandas numpy matplotlib seaborn "scikit-learn>=1.2"
.\.venv\Scripts\python.exe -m jupyter notebook credit_wise.ipynb
```

Keep `loan_approval_data.csv` in the same project folder as the notebook. In Jupyter, choose **Kernel → Restart Kernel and Run All Cells** to execute the analysis from top to bottom.

## Notebook workflow

1. Load the CSV with pandas and inspect its columns and data types.
2. Fill missing numerical values with the column mean and missing categorical values with the most frequent category.
3. Explore the class balance and visualize selected applicant and financial attributes.
4. Remove the applicant identifier.
5. Encode categorical fields using label and one-hot encoding.
6. Create squared `Credit_Score` and `DTI_Ratio` features, then remove their original untransformed columns for the feature-engineered comparison.
7. Split the data into 80% training and 20% testing data with `random_state=42`.
8. Scale features with `StandardScaler` and fit the three classifiers.
9. Compare predictions using precision, recall, F1 score, accuracy, and confusion matrices.

## Results

The following values are from the notebook's saved output for the **feature-engineered model comparison**. The positive class is the encoded `Yes` approval label.

| Model | Accuracy | Precision | Recall | F1 score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.880 | 0.785 | 0.836 | 0.810 |
| K-Nearest Neighbors (5 neighbors) | 0.785 | 0.673 | 0.574 | 0.619 |
| Gaussian Naive Bayes | 0.860 | 0.811 | 0.705 | 0.754 |

Logistic Regression has the highest accuracy and F1 score in this saved comparison. Gaussian Naive Bayes has slightly higher precision. Which model is preferable depends on the cost of false approvals versus false rejections; accuracy alone should not determine a lending decision.

These are results from one holdout split, not a guarantee of future performance. Re-running the notebook or changing the data and software versions may produce different results.

## Important limitations and next steps

- **Preprocessing leakage:** The notebook imputes missing values and fits categorical encoders before splitting into training and test sets. This lets information from the test set influence preprocessing and may make the reported metrics optimistic. Fit preprocessing only on training data, ideally in a scikit-learn `Pipeline` or `ColumnTransformer`.
- **Single split:** Add stratified cross-validation and tune model parameters before drawing stronger conclusions.
- **Data quality and provenance:** Document the dataset source, units, collection process, and license; check for duplicates, unusual values, and class imbalance.
- **Responsible use:** Loan decisions can materially affect people. Evaluate subgroup performance and fairness, protect sensitive information, and do not use this demo as the sole basis for real credit decisions.
- **No inference interface:** The notebook trains and evaluates models but does not save a reusable model or provide an interface for predictions on new applicants.

Possible extensions include leakage-safe preprocessing pipelines, cross-validation, threshold analysis, ROC/precision-recall curves, explainability, and a small demo interface.

## License

No license is currently specified. Add a license before distributing or reusing this project, and make sure the dataset's own license permits your intended use.
