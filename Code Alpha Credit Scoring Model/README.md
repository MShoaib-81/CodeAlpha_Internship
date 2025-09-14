📊 Credit Scoring Model – Give Me Some Credit Dataset

This project builds a Credit Scoring Model to predict whether an individual is likely to default on a loan, using the Give Me Some Credit dataset from Kaggle.
The goal is to simulate real-world credit risk assessment by applying data preprocessing, feature engineering, and machine learning models.

📂 Dataset

Source: Give Me Some Credit (Kaggle)

Records: ~150,000

Target variable:

SeriousDlqin2yrs → 1 = default / bad credit, 0 = non-default / good credit

Features include:

RevolvingUtilizationOfUnsecuredLines (credit utilization ratio)

age (borrower’s age)

NumberOfTime30-59DaysPastDueNotWorse

DebtRatio

MonthlyIncome

NumberOfOpenCreditLinesAndLoans

NumberOfTimes90DaysLate

NumberRealEstateLoansOrLines

NumberOfTime60-89DaysPastDueNotWorse

NumberOfDependents

🛠️ Project Workflow

Data Loading & Exploration

Load dataset with Pandas.

Check missing values, distribution, target imbalance.

Data Cleaning

Handle missing values (MonthlyIncome, NumberOfDependents).

Remove unrealistic ages (< 18).

Drop ID column.

Feature Engineering

Create ratios:

IncomePerDependent = MonthlyIncome / (Dependents + 1)

DebtToIncomeRatio = DebtRatio * MonthlyIncome

Cap extreme outliers.

Train/Test Split & Scaling

Standardize features using StandardScaler.

Stratified split to handle imbalance.

Modeling

Logistic Regression (baseline).

Random Forest Classifier.

Class imbalance handled via class_weight="balanced".

Evaluation

Metrics: Precision, Recall, F1-score, ROC-AUC.

ROC Curve comparison.

Model Saving

Save trained models with joblib.

Save scaler to apply on new data.

⚡ Results

Logistic Regression: Solid baseline.

Random Forest: Higher ROC-AUC, better at capturing non-linear patterns.

ROC-AUC chosen as main metric due to class imbalance (~7% defaults).

💾 Saving the Model
import joblib

# Save model and scaler
joblib.dump(rf, "random_forest_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Load later
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

🚀 How to Run

Clone repo / open notebook in Colab/Kaggle.

Install dependencies (if not already available):

pip install pandas numpy scikit-learn seaborn matplotlib joblib kagglehub


Download dataset via:

import kagglehub
path = kagglehub.dataset_download("saleha07/give-me-some-credit")


Run notebook cells step by step.

Models + scaler will be saved as .pkl files for reuse.

📌 Future Improvements

Apply SMOTE or other resampling methods to further handle class imbalance.

Try XGBoost / LightGBM for better performance.

Deploy as a Flask/FastAPI/Streamlit app for real-world use.
