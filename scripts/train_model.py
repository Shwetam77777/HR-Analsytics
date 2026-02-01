import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_and_save():
    print("Loading data...")
    df = pd.read_excel('data/hr_analytics.xlsx')
    
    print("Preprocessing...")
    df_model = pd.get_dummies(df, columns=['Department', 'salary'], drop_first=True)
    
    X = df_model.drop('left', axis=1)
    y = df_model['left']
    
    # We use all data for the production model deployed in the app
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    print("Saving model and feature list...")
    joblib.dump(model, 'models/hr_model.joblib')
    joblib.dump(list(X.columns), 'models/features.joblib')
    print("Done! Saved to models/ directory.")

if __name__ == "__main__":
    train_and_save()
