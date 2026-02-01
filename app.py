import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="HR Insight Dashboard", layout="wide", page_icon="🏢")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# --- CACHING & DATA LOADING ---
@st.cache_data
def load_data():
    try:
        # Check if file exists first to avoid vague errors
        if not os.path.exists('data/hr_analytics.xlsx'):
            st.error("CRITICAL ERROR: 'data/hr_analytics.xlsx' not found in repository.")
            return None
        return pd.read_excel('data/hr_analytics.xlsx')
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        return None

@st.cache_resource
def load_model(df):
    # 1. Try Loading Pre-trained
    try:
        if os.path.exists('models/hr_model.joblib') and os.path.exists('models/features.joblib'):
            model = joblib.load('models/hr_model.joblib')
            features = joblib.load('models/features.joblib')
            return model, features
    except Exception:
        pass # Silently fail to fallback

    # 2. Fallback: Train Fresh (Cloud Safe Mode)
    # This guarantees the model works even if pickles match different OS versions
    try:
        if df is None: return None, None
        
        df_model = pd.get_dummies(df, columns=['Department', 'salary'], drop_first=True)
        X = df_model.drop('left', axis=1)
        y = df_model['left']
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model, list(X.columns)
    except Exception as e:
        st.error(f"Model Training Failed: {e}")
        return None, None

# --- INITIALIZATION ---
with st.spinner("Initializing Dashboard..."):
    df = load_data()
    if df is None:
        st.stop()
        
    model, features = load_model(df)
    if model is None:
        st.error("Application Failed to Start: Could not load or train model.")
        st.stop()

# --- SIDEBAR ---
st.sidebar.title("🏢 HR Insights Pro")
st.sidebar.info("v2.4 Stable | Production Ready")
st.sidebar.markdown("---")

# --- MAIN DASHBOARD ---
st.title("HR Analytics & Retention Dashboard")

# Metrics
m1, m2, m3, m4 = st.columns(4)
attrition_rate = (df['left'].mean() * 100)
with m1: st.metric("Attrition Rate", f"{attrition_rate:.1f}%")
with m2: st.metric("Total Workforce", f"{len(df):,}")
with m3: st.metric("Avg Satisfaction", f"{df['satisfaction_level'].mean():.2f}")
with m4: st.metric("Avg Monthly Hours", f"{df['average_montly_hours'].mean():.0f}h")

st.markdown("---")

# Tabs
tab1, tab2 = st.tabs(["🔮 Prediction Tool", "📊 Analytics Dashboard"])

# TAB 1: PREDICTION
with tab1:
    st.header("Employee Risk Assessment")
    
    with st.expander("📝 Employee Details", expanded=True):
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

    if st.button("🚀 Calculate Attrition Risk"):
        input_data = pd.DataFrame(columns=features)
        input_data.loc[0] = 0
        input_data.at[0, 'satisfaction_level'] = satisfaction
        input_data.at[0, 'last_evaluation'] = last_eval
        input_data.at[0, 'number_project'] = num_projects
        input_data.at[0, 'average_montly_hours'] = avg_hours
        input_data.at[0, 'time_spend_company'] = tenure
        input_data.at[0, 'Work_accident'] = accident
        input_data.at[0, 'promotion_last_5years'] = promotion
        
        dept_col = f"Department_{dept}"
        if dept_col in input_data.columns: input_data.at[0, dept_col] = 1
        salary_col = f"salary_{salary}"
        if salary_col in input_data.columns: input_data.at[0, salary_col] = 1

        prob = model.predict_proba(input_data)[0][1]
        
        c_res1, c_res2 = st.columns([1, 2])
        with c_res1:
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Risk Probability"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#3366cc"},
                    'steps' : [{'range': [0, 100], 'color': "lightgray"}]
                }))
            st.plotly_chart(fig, use_container_width=True)
            
        with c_res2:
            st.subheader("Analysis Result")
            if prob > 0.5:
                st.error(f"High Risk ({prob:.1%}): Employee is likely to leave.")
            else:
                st.success(f"Low Risk ({prob:.1%}): Employee is likely to stay.")
                
            st.caption("Key Drivers (Top 5 Factors):")
            importances = model.feature_importances_
            feat_imp = pd.Series(importances, index=features).sort_values(ascending=False).head(5)
            st.bar_chart(feat_imp)

# TAB 2: ANALYTICS
with tab2:
    st.header("Workforce Trends")
    graph_col1, graph_col2 = st.columns(2)
    
    with graph_col1:
        st.subheader("Attrition by Department")
        dept_df = df.groupby('Department')['left'].mean().reset_index()
        fig_dept = px.bar(dept_df, x='Department', y='left', color='left', title="Avg Attrition Rate")
        st.plotly_chart(fig_dept, use_container_width=True)
        
    with graph_col2:
        st.subheader("Satisfaction Impact")
        fig_sat = px.box(df, x='left', y='satisfaction_level', color='left', title="Satisfaction Distribution (0=Stay, 1=Left)")
        st.plotly_chart(fig_sat, use_container_width=True)
