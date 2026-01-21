# HR Analytics – Employee Attrition Analysis 📊

## 📌 Project Overview
This project focuses on **HR Analytics** to analyze employee data and identify factors responsible for **employee attrition (employees leaving the company)**.  
Using Python and machine learning techniques, we perform data exploration, visualization, and build a **Logistic Regression model** to predict attrition.

---

## 🎯 Objectives
- Understand employee behavior and attrition trends
- Perform Exploratory Data Analysis (EDA)
- Visualize attrition across departments and salary levels
- Prepare data for machine learning
- Build a Logistic Regression model
- Evaluate model performance using Confusion Matrix

---

## 📂 Dataset Information
The dataset contains **14,999 employee records** with the following features:

| Column Name | Description |
|------------|------------|
| satisfaction_level | Employee satisfaction score |
| last_evaluation | Last performance evaluation |
| number_project | Number of projects handled |
| average_montly_hours | Average monthly working hours |
| time_spend_company | Years spent in the company |
| Work_accident | Whether employee had work accident |
| promotion_last_5years | Promotion in last 5 years |
| Department | Employee department |
| salary | Salary level (low, medium, high) |
| left | Target variable (1 = Left, 0 = Stayed) |

---

## 🛠️ Technologies Used
- Python 🐍
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

---

## 🔍 Exploratory Data Analysis (EDA)
The following analyses were performed:
- Dataset shape, info, and missing value check
- Statistical summary using `describe()`
- Attrition comparison using:
  - Department vs Attrition
  - Salary vs Attrition
- Group-wise mean comparison using `groupby()`

### 📊 Visualizations
- Bar charts using `pd.crosstab()`
- Attrition distribution across departments
- Attrition distribution across salary levels

---

## ⚙️ Data Preprocessing
- Checked for missing values (No missing data found ✅)
- Converted categorical variables using **One-Hot Encoding**
- Created dummy variables for salary column

---

## 🤖 Machine Learning Model
- **Algorithm Used:** Logistic Regression
- **Train-Test Split:** Applied
- **Evaluation Metric:**
  - Confusion Matrix
  - Accuracy Score (to be added)

---

## 📈 Model Evaluation
- Confusion Matrix to analyze:
  - True Positives
  - True Negatives
  - False Positives
  - False Negatives

---

## 🚀 Future Improvements
- Add accuracy, precision, recall, F1-score
- Try other models (Random Forest, XGBoost)
- Handle class imbalance
- Deploy model using Flask / Streamlit

---

## 📌 Conclusion
This project demonstrates how **HR data analytics** can help organizations:
- Predict employee attrition
- Improve employee retention strategies
- Make data-driven HR decisions

---

## 👨‍💻 Author
**Shweta**

---

⭐ If you like this project, don't forget to star the repository!
