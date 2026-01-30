import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

# Config
DATA_DIR = "."  # Since we generated them in root, or I can move them. Assuming root based on previous run.
MODEL_DIR = "nexus_analytics/models"

def train_rul_model():
    print("Training RUL Prediction Model...")
    df = pd.read_csv(os.path.join(DATA_DIR, "bearings_data.csv"))
    
    # Feature Engineering / Selection
    features = ['Operating_Hours', 'RPM', 'Temperature_C', 'Vibration_mm_s', 'Lubrication_Level_Pct', 'Load_Factor']
    target = 'RUL_Days'
    
    X = df[features]
    y = df[target]
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model: Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Evaluate
    score = rf.score(X_test, y_test)
    print(f"RUL Model R^2 Score: {score:.4f}")
    
    # Save Model
    joblib.dump(rf, os.path.join(MODEL_DIR, "rul_model.pkl"))
    print("Saved RUL model.")

def train_dealer_clustering():
    print("\nTraining Dealer Segmentation Model...")
    df = pd.read_csv(os.path.join(DATA_DIR, "dealer_network.csv"))
    
    # Features for clustering
    features = ['Inventory_Level', 'Service_Responsiveness_Score', 'Turnaround_Time_Hrs', 'Customer_Satisfaction_Index']
    X = df[features]
    
    # Scaling is important for clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # KMeans Clustering
    # We want maybe 3 segments: Gold, Silver, Bronze performance
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Analyze clusters to label them sensibly (heuristic)
    # High CSI + Low Turnaround = Best
    cluster_means = df.groupby('Cluster')['Customer_Satisfaction_Index'].mean()
    sorted_clusters = cluster_means.sort_values(ascending=False).index.tolist()
    
    # Map cluster ID to meaningful labels
    label_map = {
        sorted_clusters[0]: 'Premium Partner',
        sorted_clusters[1]: 'Standard Partner',
        sorted_clusters[2]: 'At-Risk Partner'
    }
    
    # Save the model and the label logic/scaler to be used in inference
    model_bundle = {
        'model': kmeans,
        'scaler': scaler,
        'label_map': label_map
    }
    
    joblib.dump(model_bundle, os.path.join(MODEL_DIR, "dealer_segmentation.pkl"))
    print("Saved Dealer Segmentation model.")

if __name__ == "__main__":
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    train_rul_model()
    train_dealer_clustering()
