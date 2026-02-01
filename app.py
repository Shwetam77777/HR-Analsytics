import streamlit as st
import os
import sys

# 1. Immediate Page Config
st.set_page_config(page_title="Diagnostic Mode", layout="wide")

st.title("🛠️ System Diagnostic Mode")
st.write("If you can see this, Streamlit is running!")

# 2. Check Python Environment
st.subheader("1. Environment Details")
st.write(f"Python Version: {sys.version}")
st.write(f"Current Directory: {os.getcwd()}")

# 3. Check File System
st.subheader("2. File System Check")
if os.path.exists("data"):
    st.success("found 'data' folder")
    files = os.listdir("data")
    st.write(f"Files in data/: {files}")
else:
    st.error("'data' folder NOT found!")
    st.write(f"Root files: {os.listdir('.')}")

# 4. Check Dependencies
st.subheader("3. Dependency Check")
try:
    import pandas as pd
    st.success(f"Pandas imported: {pd.__version__}")
except ImportError as e:
    st.error(f"Pandas missing: {e}")

try:
    import sklearn
    st.success(f"Scikit-learn imported: {sklearn.__version__}")
except ImportError as e:
    st.error(f"Scikit-learn missing: {e}")

try:
    import plotly
    st.success(f"Plotly imported: {plotly.__version__}")
except ImportError as e:
    st.error(f"Plotly missing: {e}")

# 5. Attempt Data Load
st.subheader("4. Data Load Test")
try:
    df = pd.read_excel("data/hr_analytics.xlsx")
    st.success(f"Data loaded successfully! Shape: {df.shape}")
    st.dataframe(df.head())
except Exception as e:
    st.error(f"Failed to load data: {e}")

# 6. Attempt Model Train
st.subheader("5. Model Training Test")
try:
    from sklearn.ensemble import RandomForestClassifier
    st.write("Training dummy model...")
    # Train on a tiny fraction just to test imports/logic
    df_small = df.head(50)
    df_model = pd.get_dummies(df_small, columns=['Department', 'salary'], drop_first=True)
    X = df_model.drop('left', axis=1)
    y = df_model['left']
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    st.success("Model trained successfully!")
except Exception as e:
    st.error(f"Model training failed: {e}")
