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
    .stAlert {
        padding: 10px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- CACHING & DATA LOADING ---
@st.cache_data
def load_data():
    try:
        if not os.path.exists('data/hr_analytics.xlsx'):
            st.error("CRITICAL ERROR: 'data/hr_analytics.xlsx' not found.")
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
        pass 

    # 2. Fallback: Train Fresh
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

def predict_risk(model, features, input_data):
    # Helper to format input correctly for model
    row = pd.DataFrame(columns=features)
    row.loc[0] = 0 # Initialize with 0s
    
    # Map basic numericals
    for col in ['satisfaction_level', 'last_evaluation', 'number_project', 
                'average_montly_hours', 'time_spend_company', 'Work_accident', 'promotion_last_5years']:
        if col in input_data:
            row.at[0, col] = input_data[col]
            
    # Map Categoricals
    dept_col = f"Department_{input_data['dept']}"
    if dept_col in features: row.at[0, dept_col] = 1
    
    salary_col = f"salary_{input_data['salary']}"
    if salary_col in features: row.at[0, salary_col] = 1
    
    return model.predict_proba(row)[0][1]

# --- INITIALIZATION ---
with st.spinner("Initializing Dashboard..."):
    df = load_data()
    if df is None: st.stop()
    model, features = load_model(df)
    if model is None: st.stop()

# --- SIDEBAR & FILTERS ---
st.sidebar.title("🏢 HR Insights Pro")
st.sidebar.caption("v3.0 Advanced | What-If Analysis Enabled")
st.sidebar.markdown("---")
st.sidebar.header("Global Filters (Analytics Tab)")
selected_dept = st.sidebar.selectbox("Filter Department", ["All Departments"] + list(df['Department'].unique()))

# --- MAIN PAGE ---
st.title("HR Analytics & Retention Command Center")

# Filtered Data for Analytics
if selected_dept != "All Departments":
    analytics_df = df[df['Department'] == selected_dept]
else:
    analytics_df = df

m1, m2, m3, m4 = st.columns(4)
attrition = (analytics_df['left'].mean() * 100)
with m1: st.metric("Attrition Rate", f"{attrition:.1f}%", help="Percentage of employees who left")
with m2: st.metric("Active Employees", f"{len(analytics_df):,}")
with m3: st.metric("Avg Satisfaction", f"{analytics_df['satisfaction_level'].mean():.2f}")
with m4: st.metric("Avg Monthly Hours", f"{analytics_df['average_montly_hours'].mean():.0f}h")

st.markdown("---")

tab1, tab2 = st.tabs(["🔮 Prediction & What-If Analysis", "📊 Deep-Dive Analytics"])

# --- TAB 1: PREDICTION & SIMULATION ---
with tab1:
    col_input, col_result = st.columns([1, 1.2])
    
    with col_input:
        st.subheader("👤 Employee Profile")
        with st.container():
            # Collecting Inputs
            sat = st.slider("Satisfaction (0-1)", 0.0, 1.0, 0.5)
            eval_scr = st.slider("Last Evaluation (0-1)", 0.0, 1.0, 0.7)
            projects = st.number_input("Projects", 1, 10, 3)
            hours = st.number_input("Monthly Hours", 50, 350, 200)
            tenure = st.number_input("Tenure (Years)", 1, 15, 3)
            accident = st.selectbox("Work Accident?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
            promo = st.selectbox("Promotion (5 yrs)?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
            dept = st.selectbox("Department", df['Department'].unique())
            salary = st.selectbox("Salary", ['low', 'medium', 'high'])

            current_input = {
                'satisfaction_level': sat, 'last_evaluation': eval_scr, 'number_project': projects,
                'average_montly_hours': hours, 'time_spend_company': tenure, 'Work_accident': accident,
                'promotion_last_5years': promo, 'dept': dept, 'salary': salary
            }
        
        btn_predict = st.button("🚀 Analyze Risk", type="primary")

    with col_result:
        st.subheader("🎯 Risk Analysis Results")
        
        # We run prediction if button clicked OR if we already have a session state (to keep result visible)
        if btn_predict:
            risk_score = predict_risk(model, features, current_input)
            st.session_state['risk_score'] = risk_score
            st.session_state['run_sim'] = True
        
        if 'risk_score' in st.session_state and st.session_state.get('run_sim'):
            current_risk = st.session_state['risk_score']
            
            # GAUGE CHART
            fig = go.Figure(go.Indicator(
                mode = "gauge+number", value = current_risk * 100,
                title = {'text': "Current Attrition Probability"},
                gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': "#ff4b4b" if current_risk > 0.5 else "#00cc96"}}
            ))
            fig.update_layout(height=250, margin=dict(l=20,r=20,t=30,b=20))
            st.plotly_chart(fig, use_container_width=True)
            
            if current_risk > 0.5:
                st.error(f"⚠️ HIGH RISK: This employee is {current_risk:.1%} likely to leave.")
            else:
                st.success(f"✅ STABLE: This employee is likely to stay ({current_risk:.1%} risk).")

            st.markdown("---")
            
            # --- WHAT-IF SIMULATOR ---
            with st.expander("🧩 Retention Strategy Simulator (What-If)", expanded=True):
                st.write("**Simulate Interventions**: Adjust these likely scenarios to see if risk drops.")
                
                sim_col1, sim_col2 = st.columns(2)
                with sim_col1:
                    new_sat = st.slider("Target Satisfaction", 0.0, 1.0, float(sat), key="sim_sat")
                    new_salary = st.selectbox("Target Salary", ['low', 'medium', 'high'], index=['low', 'medium', 'high'].index(salary), key="sim_sal")
                with sim_col2:
                    new_hours = st.slider("Target Hours", 50, 350, int(hours), key="sim_hours")
                    new_promo = st.selectbox("Offer Promotion?", [0, 1], index=promo, format_func=lambda x: "Yes" if x==1 else "No", key="sim_promo")
                
                # Dynamic Re-prediction
                sim_input = current_input.copy()
                sim_input.update({
                    'satisfaction_level': new_sat, 'average_montly_hours': new_hours,
                    'salary': new_salary, 'promotion_last_5years': new_promo
                })
                
                sim_risk = predict_risk(model, features, sim_input)
                delta = current_risk - sim_risk
                
                st.markdown(f"#### Predicted New Risk: `{sim_risk:.1%}`")
                
                if delta > 0.01:
                    st.success(f"📉 **Impact:** Risk reduced by **{delta*100:.1f}%** points!")
                    st.progress(max(0.0, min(1.0, delta * 2))) # Visual progress of "saving"
                elif delta < -0.01:
                    st.warning(f"📈 **Impact:** Risk INCREASED by **{abs(delta)*100:.1f}%** points.")
                else:
                    st.info("No significant change in risk.")

# --- TAB 2: ANALYTICS ---
with tab2:
    st.header(f"Workforce Insights: {selected_dept}")
    
    g1, g2 = st.columns(2)
    with g1:
        # 1. Satisfaction vs Evaluation Cluster
        fig_scatter = px.scatter(analytics_df.sample(min(1000, len(analytics_df))), x='satisfaction_level', y='last_evaluation', color='left',
                                title="Satisfaction vs Evaluation (Cluster View)", opacity=0.6,
                                color_continuous_scale='Bluered')
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with g2:
        # 2. Salary Distribution
        fig_pie = px.pie(analytics_df, names='salary', title="Salary Distribution", 
                        color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig_pie, use_container_width=True)
        
    g3, g4 = st.columns(2)
    with g3:
        # 3. Hours Distribution
        fig_hist = px.histogram(analytics_df, x='average_montly_hours', color='left', barmode='overlay',
                               title="Monthly Hours Workload")
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with g4:
        # 4. Tenure Impact
        tenure_risk = analytics_df.groupby('time_spend_company')['left'].mean().reset_index()
        fig_line = px.line(tenure_risk, x='time_spend_company', y='left', markers=True,
                          title="Attrition Risk by Tenure (Years)")
        st.plotly_chart(fig_line, use_container_width=True)
