import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Page Configuration
st.set_page_config(page_title="HR Attrition Predictor", layout="wide", page_icon="📊")

# Custom CSS for better aesthetics
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_excel('data/hr_analytics.xlsx')
    # Preprocessing for the model
    df_model = pd.get_dummies(df, columns=['Department', 'salary'], drop_first=True)
    return df, df_model

def train_model(df_model):
    X = df_model.drop('left', axis=1)
    y = df_model['left']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X.columns

# Sidebar info
st.sidebar.title("HR Analytics Dashboard")
st.sidebar.markdown("---")
st.sidebar.write("Analyze and Predict Employee Attrition")

try:
    df, df_model = load_data()
    model, features = train_model(df_model)
    st.sidebar.success("Model Trained Successfully!")
    
    # Summary Metrics in Sidebar
    attrition_rate = (df['left'].mean() * 100)
    st.sidebar.metric("Company Attrition Rate", f"{attrition_rate:.1f}%")
    st.sidebar.metric("Total Employees", len(df))
    
except Exception as e:
    st.error(f"Error loading data: {e}. Please ensure 'data/hr_analytics.xlsx' exists.")
    st.stop()

# Main UI
st.title("👨‍💼 Employee Attrition Prediction")
st.write("Enter employee details below to predict the probability of them leaving the company.")

col1, col2 = st.columns(2)

with col1:
    satisfaction = st.slider("Satisfaction Level", 0.0, 1.0, 0.5)
    last_eval = st.slider("Last Evaluation Score", 0.0, 1.0, 0.5)
    num_projects = st.number_input("Number of Projects", min_value=1, max_value=10, value=3)
    avg_hours = st.number_input("Average Monthly Hours", min_value=50, max_value=350, value=200)

with col2:
    tenure = st.number_input("Years at Company", min_value=0, max_value=20, value=3)
    accident = st.selectbox("Work Accident?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    promotion = st.selectbox("Promotion in last 5 years?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    dept = st.selectbox("Department", df['Department'].unique())
    salary = st.selectbox("Salary Level", ['low', 'medium', 'high'])

# Preparation of Input for Prediction
if st.button("Predict Attrition Risk"):
    # Create a template for the input data matching the dummy variable structure
    input_data = pd.DataFrame(columns=features)
    input_data.loc[0] = 0
    
    input_data.at[0, 'satisfaction_level'] = satisfaction
    input_data.at[0, 'last_evaluation'] = last_eval
    input_data.at[0, 'number_project'] = num_projects
    input_data.at[0, 'average_montly_hours'] = avg_hours
    input_data.at[0, 'time_spend_company'] = tenure
    input_data.at[0, 'Work_accident'] = accident
    input_data.at[0, 'promotion_last_5years'] = promotion
    
    # Handle Department Dummy
    dept_col = f"Department_{dept}"
    if dept_col in input_data.columns:
        input_data.at[0, dept_col] = 1
        
    # Handle Salary Dummy
    salary_col = f"salary_{salary}"
    if salary_col in input_data.columns:
        input_data.at[0, salary_col] = 1

    # Prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.markdown(f"""
        <div class='prediction-box' style='background-color: #ffcccc; color: #cc0000; border: 2px solid #cc0000;'>
            <h2>⚠️ High Risk of Attrition</h2>
            <p>Predicting employee is likely to leave.</p>
            <p>Confidence: {probability*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='prediction-box' style='background-color: #ccffcc; color: #006600; border: 2px solid #006600;'>
            <h2>✅ Low Risk of Attrition</h2>
            <p>Predicting employee is likely to stay.</p>
            <p>Retention Probability: {(1-probability)*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.write("Developed by Shweta | HR Analytics Project Refactored 🚀")
