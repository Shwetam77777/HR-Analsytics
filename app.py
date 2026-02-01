import streamlit as st
import os
import sys

# 1. PAGE CONFIG
try:
    st.set_page_config(page_title="HR Insight Dashboard", layout="wide", page_icon="🏢")
except Exception as e:
    st.error(f"Page Config Error: {e}")

# 2. IMPORTS WRAPPER
try:
    import pandas as pd
    import numpy as np
    import joblib
    import plotly.express as px
    import plotly.graph_objects as go
    from sklearn.ensemble import RandomForestClassifier
except Exception as e:
    st.error(f"Import Error: Missing library? Details: {e}")
    st.stop()

# 3. GLOBAL STYLE
st.markdown("""
<style>
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
</style>
""", unsafe_allow_html=True)

# 4. ROBUST LOADER
@st.cache_data
def get_data():
    path = 'data/hr_analytics.xlsx'
    if not os.path.exists(path):
        return None, f"File not found at {path}. Current Dir: {os.getcwd()}"
    try:
        return pd.read_excel(path), None
    except Exception as e:
        return None, str(e)

@st.cache_resource
def get_model(df):
    try:
        # Simplest possible fallback: Always train fresh on cloud to avoid OS pickle issues
        df_model = pd.get_dummies(df, columns=['Department', 'salary'], drop_first=True)
        X = df_model.drop('left', axis=1)
        y = df_model['left']
        model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42) # Lighter model
        model.fit(X, y)
        return model, list(X.columns), None
    except Exception as e:
        return None, None, str(e)

# 5. INITIALIZATION
with st.spinner("Initializing Dashboard..."):
    df, err = get_data()
    if err:
        st.error(f"CRITICAL: Data Load Failed. {err}")
        st.stop()

    model, features, err = get_model(df)
    if err:
        st.error(f"CRITICAL: Model Training Failed. {err}")
        st.stop()

# 6. SIDEBAR
st.sidebar.title("🏢 HR Insights Pro")
st.sidebar.info("v2.3 Safe Mode | Auto-Healing Enabled")

# 7. MAIN UI
st.title("HR Analytics & Retention Dashboard")
m1, m2, m3, m4 = st.columns(4)
attrition_rate = (df['left'].mean() * 100)
with m1: st.metric("Attrition Rate", f"{attrition_rate:.1f}%")
with m2: st.metric("Total Workforce", f"{len(df):,}")
with m3: st.metric("Avg Satisfaction", f"{df['satisfaction_level'].mean():.2f}")
with m4: st.metric("Avg Monthly Hours", f"{df['average_montly_hours'].mean():.0f}h")

st.markdown("---")

tab1, tab2 = st.tabs(["🔮 Predict Attrition", "📊 Workforce Analytics"])

with tab1:
    st.header("Predict Individual Employee Risk")
    with st.expander("📝 Employee Input Form", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            satisfaction = st.slider("Satisfaction Level", 0.0, 1.0, 0.5)
            last_eval = st.slider("Last Evaluation", 0.0, 1.0, 0.7)
            num_projects = st.number_input("Number of Projects", 1, 10, 3)
        with col2:
            avg_hours = st.number_input("Average Monthly Hours", 50, 350, 200)
            tenure = st.number_input("Tenure (Years)", 1, 15, 3)
            accident = st.selectbox("Work Accident?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        with col3:
            promotion = st.selectbox("Recent Promotion?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
            dept = st.selectbox("Department", df['Department'].unique())
            salary = st.selectbox("Salary Level", ['low', 'medium', 'high'])

    if st.button("🚀 Analyze Risk Instance"):
        input_df = pd.DataFrame(columns=features)
        input_df.loc[0] = 0
        input_df.at[0, 'satisfaction_level'] = satisfaction
        input_df.at[0, 'last_evaluation'] = last_eval
        input_df.at[0, 'number_project'] = num_projects
        input_df.at[0, 'average_montly_hours'] = avg_hours
        input_df.at[0, 'time_spend_company'] = tenure
        input_df.at[0, 'Work_accident'] = accident
        input_df.at[0, 'promotion_last_5years'] = promotion
        
        dept_col = f"Department_{dept}"
        if dept_col in input_df.columns: input_df.at[0, dept_col] = 1
        salary_col = f"salary_{salary}"
        if salary_col in input_df.columns: input_df.at[0, salary_col] = 1

        prob = model.predict_proba(input_df)[0][1]
        
        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            st.metric("Attrition Probability", f"{prob*100:.1f}%")
        with res_col2:
            if prob > 0.5: st.error("High Risk")
            else: st.success("Low Risk")

with tab2:
    st.header("Workforce Insights")
    st.bar_chart(df['Department'].value_counts()) # Fallback to native charts if Plotly fails (unlikely)
    
