# app.py - Streamlit deployment script for Hepatitis C Prediction Project

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from PIL import Image

# Load the saved model
@st.cache_resource
def load_model():
    return joblib.load('Hepatitis Model.joblib')

model = load_model()

# Function to load and preprocess data (for EDA and scaler fitting)
@st.cache_data
def load_data():
    df = pd.read_csv('HepatitisCdata.csv')
    # Handle missing values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mean())
    # Encode Sex
    df['Sex'] = df['Sex'].map({'m': 0, 'f': 1})
    # Encode Category (for reference)
    category_map = {
        '0=Blood Donor': 0,
        '0s=suspect Blood Donor': 0,
        '1=Hepatitis': 1,
        '2=Fibrosis': 1,
        '3=Cirrhosis': 1
    }
    df['Category'] = df['Category'].map(category_map)
    # Drop unnecessary column
    df = df.drop(columns=['Unnamed: 0'])
    return df

df = load_data()

# Fit scaler on the entire dataset (features only)
features = ['Age', 'Sex', 'ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']
X = df[features]
scaler = StandardScaler()
scaler.fit(X)

# Reverse category map for output
reverse_category_map = {
    0: 'Blood Donor',
    0: 'Suspect Blood Donor',
    1: 'Hepatitis',
    1: 'Fibrosis',
    1: 'Cirrhosis'
}

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Project Description", "Prediction"])

if page == "Project Description":
    st.title("Hepatitis C Prediction Project")
    st.markdown("""
    ### Project Overview
    This project focuses on predicting Hepatitis C infection status using machine learning techniques. 
    Hepatitis C is a viral infection that can lead to severe liver damage if left untreated. 
    Early detection through predictive modeling can aid healthcare professionals in identifying at-risk individuals based on clinical biomarkers.
    
    This final year project demonstrates the application of supervised learning algorithms to classify patients into categories such as 
    "Blood Donor," "Suspect Blood Donor," "Hepatitis," "Fibrosis," or "Cirrhosis" using a dataset of laboratory values.
    
    **Key highlights:**
    - **Dataset**: UCI Machine Learning Repository's Hepatitis C Virus (HCV) dataset (615 records, 14 features).
    - **Models**: Support Vector Machine (SVM), Random Forest Classifier, and a Stacking Ensemble.
    - **Evaluation Metrics**: Accuracy, Confusion Matrix, Classification Report, and ROC-AUC Curves.
    - **Best Model**: Random Forest Classifier (deployed here).
    
    ### Dataset Details
    - **Features**: Age, Sex, ALB (Albumin), ALP (Alkaline Phosphatase), ALT (Alanine Aminotransferase), AST (Aspartate Aminotransferase), 
      BIL (Bilirubin), CHE (Cholinesterase), CHOL (Cholesterol), CREA (Creatinine), GGT (Gamma-Glutamyl Transferase), PROT (Proteins).
    - **Target**: Category (multiclass: 0-4).
    - Preprocessing: Missing values imputed with means, categorical encoding, standardization.
    
    ### Methodology
    1. **Data Preprocessing**: Load data, handle missing values, encode categoricals, scale features.
    2. **Exploratory Data Analysis (EDA)**: Statistical summaries, class distribution, correlations.
    3. **Model Training**: Trained SVM, Random Forest, and Stacking Classifier. Random Forest performed best (~95% accuracy).
    4. **Evaluation**: Used accuracy, confusion matrices, classification reports, and ROC-AUC curves.
    
    ### Results
    - Random Forest achieved the highest accuracy.
    - ROC-AUC curves indicate excellent model discrimination.
    """)
    
    # Display ROC image if available
    try:
        roc_image = Image.open('roc_auc_curves.png')
        st.image(roc_image, caption='ROC-AUC Curves for Models', use_column_width=True)
    except FileNotFoundError:
        st.warning("ROC AUC image not found. Please ensure 'roc_auc_curves.png' is in the directory.")
    
    # Example EDA in app
    st.subheader("Sample EDA: Class Distribution")
    fig, ax = plt.subplots()
    df['Category'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
    st.pyplot(fig)
    
    st.subheader("Correlation Heatmap")
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    
    st.markdown("""
    ### Conclusion
    This project demonstrates effective use of ML for medical prediction. For defense, note the handling of class imbalance and biomarker importance (e.g., AST/ALT).
    
    **Future Work**: SMOTE for imbalance, deep learning integration, external validation.
    """)

elif page == "Prediction":
    st.title("Hepatitis C Prediction")
    st.markdown("Enter patient details below to predict Hepatitis C status.")
    
    # Input fields
    age = st.number_input("Age", min_value=19, max_value=77, value=32)
    sex = st.selectbox("Sex", options=["Male", "Female"])
    sex_encoded = 0 if sex == "Male" else 1
    alb = st.number_input("ALB (Albumin)", min_value=14.9, max_value=82.2, value=41.6)
    alp = st.number_input("ALP (Alkaline Phosphatase)", min_value=11.3, max_value=416.6, value=68.3)
    alt = st.number_input("ALT (Alanine Aminotransferase)", min_value=0.9, max_value=325.3, value=28.5)
    ast = st.number_input("AST (Aspartate Aminotransferase)", min_value=10.6, max_value=324.0, value=34.8)
    bil = st.number_input("BIL (Bilirubin)", min_value=0.8, max_value=254.0, value=11.4)
    che = st.number_input("CHE (Cholinesterase)", min_value=1.42, max_value=16.41, value=8.2)
    chol = st.number_input("CHOL (Cholesterol)", min_value=1.43, max_value=9.67, value=5.37)
    crea = st.number_input("CREA (Creatinine)", min_value=8.0, max_value=1079.1, value=81.3)
    ggt = st.number_input("GGT (Gamma-Glutamyl Transferase)", min_value=4.5, max_value=650.9, value=39.5)
    prot = st.number_input("PROT (Proteins)", min_value=44.8, max_value=90.0, value=72.0)
    
    if st.button("Predict"):
        # Prepare input
        input_data = np.array([[age, sex_encoded, alb, alp, alt, ast, bil, che, chol, crea, ggt, prot]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        #category = reverse_category_map.get(prediction, "Unknown")
        #st.success(f"Predicted Category: {category} ({prediction})")
        if prediction ==0:
            st.success(f"The model predicts: [{prediction}]. The patient is NOT infected.")
        elif prediction ==1:
            st.warning(f"The model predicts: [{prediction}]. The patient is infected.")
        else:
            st.warning(f"Prediction result is unexpected: {prediction}")

