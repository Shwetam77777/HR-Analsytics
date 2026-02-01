import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
import os

# Page Configuration
st.set_page_config(page_title="HR Insight Dashboard", layout="wide", page_icon="🏢")

# Custom Metric Styling
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

@st.cache_data
def load_data():
    try:
        return pd.read_excel('data/hr_analytics.xlsx')
    except Exception as e:
        st.error(f"Error loading Excel data: {e}")
        return None

@st.cache_resource
def load_model():
    # Attempt to load pre-trained assets
    if os.path.exists('models/hr_model.joblib') and os.path.exists('models/features.joblib'):
        try:
            model = joblib.load('models/hr_model.joblib')
            features = joblib.load('models/features.joblib')
            return model, features
        except Exception as e:
            st.warning(f"Note: Saved model loading failed ({e}). Re-training...")
    
    # Fallback: Train model on the fly
    data = load_data()
    if data is not None:
        try:
            df_model = pd.get_dummies(data, columns=['Department', 'salary'], drop_first=True)
            X = df_model.drop('left', axis=1)
            y = df_model['left']
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)
            return model, list(X.columns)
        except Exception as e:
            st.error(f"Error training fallback model: {e}")
            return None, None
    return None, None

# Initialization
df = load_data()
model, features = load_model()

if df is None or model is None:
    st.error("The application could not initialize. Please check your data folder.")
    st.stop()

# Sidebar
st.sidebar.title("🏢 HR Insights Pro")
st.sidebar.markdown("---")
st.sidebar.info("Advanced platform for employee retention analysis and predictive modeling.")

# Top Metrics
st.title("HR Analytics & Retention Dashboard")
m1, m2, m3, m4 = st.columns(4)
attrition_rate = (df['left'].mean() * 100)
with m1: st.metric("Attrition Rate", f"{attrition_rate:.1f}%")
with m2: st.metric("Total Workforce", f"{len(df):,}")
with m3: st.metric("Avg Satisfaction", f"{df['satisfaction_level'].mean():.2f}")
with m4: st.metric("Avg Monthly Hours", f"{df['average_montly_hours'].mean():.0f}h")

st.markdown("---")

# Main Tabs
tab1, tab2 = st.tabs(["🔮 Predict Attrition", "📊 Workforce Analytics"])

# TAB 1: PREDICTION
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
        # Prepare Data
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

        # Predict
        prob = model.predict_proba(input_df)[0][1]
        
        # Results Display
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            st.subheader("Risk Score")
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#007bff"},
                    'steps' : [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "orange"},
                        {'range': [70, 100], 'color': "red"}],
                }))
            st.plotly_chart(fig, use_container_width=True)

        with res_col2:
            st.subheader("Decision Breakdown")
            if prob > 0.7:
                st.error("⚠️ CRITICAL: High Risk of Attrition.")
            elif prob > 0.3:
                st.warning("⚡ WARNING: Moderate Risk.")
            else:
                st.success("✅ STABLE: Low Risk.")
            
            st.write("**Top Drivers for this Prediction:**")
            importances = model.feature_importances_
            feat_imp = pd.Series(importances, index=features).sort_values(ascending=False).head(5)
            fig_imp = px.bar(feat_imp, orientation='h', labels={'value':'Importance', 'index':'Feature'})
            fig_imp.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig_imp, use_container_width=True)

# TAB 2: ANALYTICS
with tab2:
    st.header("Workforce Insights Dashboard")
    
    c1, c2 = st.columns(2)
    with c1:
        dept_attr = df.groupby('Department')['left'].mean().reset_index().sort_values(by='left', ascending=False)
        fig_dept = px.bar(dept_attr, x='Department', y='left', title="Attrition Rate by Department")
        st.plotly_chart(fig_dept, use_container_width=True)
    with c2:
        fig_sat = px.histogram(df, x='satisfaction_level', color='left', barmode='overlay', title="Satisfaction vs Retention")
        st.plotly_chart(fig_sat, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        tenure_attr = df.groupby('time_spend_company')['left'].mean().reset_index()
        fig_ten = px.line(tenure_attr, x='time_spend_company', y='left', title="Risk by Tenure", markers=True)
        st.plotly_chart(fig_ten, use_container_width=True)
    with c4:
        sal_attr = df.groupby('salary')['left'].mean().reset_index()
        fig_sal = px.pie(sal_attr, values='left', names='salary', title="Salary vs Attrition", hole=.4)
        st.plotly_chart(fig_sal, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("v2.2 Stable Release | Developed by Shweta")
