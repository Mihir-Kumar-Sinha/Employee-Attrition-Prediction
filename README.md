# Employee Attrition Prediction — IBM HR Analytics

This project implements a complete machine learning pipeline to predict employee attrition and identify key drivers behind why employees leave. Using the IBM HR Analytics dataset, it covers exploratory data analysis (EDA), preprocessing, addressing class imbalance, model training, evaluation, and actionable HR recommendations.

---

## Project Structure

*   **`Analysis.ipynb`**: The main Jupyter Notebook containing the end-to-end pipeline (EDA, preprocessing, modeling, evaluation, and visualizations).
*   **`WA_Fn-UseC_-HR-Employee-Attrition.csv`**: The raw IBM HR Analytics dataset.
*   **`cleaned_hr_data.csv`**: The preprocessed dataset ready for modeling.
*   **`README.md`**: Project documentation.

---

## Machine Learning Pipeline

### 1. Data Loading & Exploration (EDA)
*   Analyzed employee demographics, job roles, satisfaction levels, and compensation.
*   Identified class imbalance: only **~16%** of the employees in the dataset have left the company.

### 2. Data Cleaning & Preprocessing
*   Removed redundant/zero-variance features (e.g., `EmployeeCount`, `Over18`, `StandardHours`).
*   One-Hot Encoded categorical variables using `drop_first=True` to avoid the dummy variable trap (critical for Logistic Regression).
*   **Data Leakage Prevention**: Split the dataset into train/test (80/20 stratified split) *before* fitting the `StandardScaler` to ensure no information from the test set leaked into the training process.

### 3. Model Building & Comparison
Three models were trained and compared:
1.  **Logistic Regression** (with `class_weight='balanced'`)
2.  **Random Forest Classifier** (with `class_weight='balanced'`)
3.  **Gradient Boosting Classifier** (using sample weights computed via `compute_sample_weight` to address class imbalance since it lacks a native `class_weight` parameter)

### 4. Model Evaluation & Selection
Because of the imbalanced target class, models were compared using **ROC-AUC** as the primary metric, with **F1-Score** as a secondary metric:

| Model | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.345** | **0.617** | **0.443** | **0.798** |
| Random Forest | 0.571 | 0.085 | 0.148 | 0.783 |
| Gradient Boosting | 0.439 | 0.383 | 0.409 | 0.766 |

*   **Best Model**: **Logistic Regression** achieved the highest ROC-AUC of **0.798** and a balanced Recall of **0.617**, making it the most suitable model for identifying employees at risk of leaving while remaining highly interpretable for HR decision-makers.

---

## Key Drivers of Attrition

Based on the feature coefficients of the Logistic Regression model, the top 5 factors driving employee attrition are:
1.  **Job Role (Laboratory Technicians & Sales Representatives)**: Employees in these roles show significantly higher attrition rates.
2.  **Overtime (Yes)**: Working overtime is one of the strongest positive predictors of attrition, indicating potential burnout.
3.  **Business Travel (Frequently)**: Frequent travel strongly correlates with higher attrition.
4.  **Total Working Years & Job Level**: Employees with fewer working years or lower job levels are more likely to leave.
5.  **Years Since Last Promotion**: A lack of recent promotion increases the likelihood of attrition.

---

## Getting Started

### Prerequisites
Make sure you have the following libraries installed:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Running the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/Mihir-Kumar-Sinha/Employee-Attrition-Prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Employee-Attrition-Prediction
   ```
3. Start the Jupyter Notebook:
   ```bash
   jupyter notebook Analysis.ipynb
   ```
