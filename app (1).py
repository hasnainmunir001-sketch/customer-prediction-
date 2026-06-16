
import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide"
)

st.title("📊 Customer Churn Prediction App")
st.write("Machine Learning Based Customer Churn Prediction System")

# =========================
# AUTO DOWNLOAD DATASET
# =========================

csv_file = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

if not os.path.exists(csv_file):

    st.info("Downloading dataset automatically...")

    os.system(
        "wget https://raw.githubusercontent.com/blastchar/telco-customer-churn/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

# =========================
# LOAD DATA
# =========================

@st.cache_data
def load_data():

    df = pd.read_csv(csv_file)

    # CLEANING
    df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])

    df['TotalCharges'].fillna(
        df['TotalCharges'].median(),
        inplace=True
    )

    # DROP ID
    df.drop(columns=['customerID'], inplace=True)

    # TARGET ENCODING
    df['Churn'] = df['Churn'].map({
        'Yes': 1,
        'No': 0
    })

    # LABEL ENCODING
    le = LabelEncoder()

    categorical_cols = df.select_dtypes(
        include=['object']
    ).columns

    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    return df

df = load_data()

# =========================
# MODEL TRAINING
# =========================

X = df.drop(columns=['Churn'])
y = df['Churn']

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# MODEL ACCURACY
# =========================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

st.success(f"✅ Model Accuracy: {accuracy:.2f}")

# =========================
# SIDEBAR INPUTS
# =========================

st.sidebar.header("Customer Information")

tenure = st.sidebar.slider(
    "Tenure (Months)",
    1,
    72,
    12
)

monthly_charges = st.sidebar.slider(
    "Monthly Charges",
    10,
    150,
    70
)

total_charges = st.sidebar.slider(
    "Total Charges",
    10,
    9000,
    2000
)

# =========================
# SAMPLE INPUT
# =========================

sample = X.iloc[0].copy()

sample['tenure'] = tenure
sample['MonthlyCharges'] = monthly_charges
sample['TotalCharges'] = total_charges

# =========================
# PREDICTION
# =========================

if st.sidebar.button("Predict Churn"):

    input_data = pd.DataFrame([sample])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error("⚠ Customer will likely churn")

    else:

        st.success("✅ Customer will stay")

    st.write(
        f"Churn Probability: {probability:.2f}"
    )

# =========================
# DATASET PREVIEW
# =========================

st.subheader("📁 Dataset Preview")

st.dataframe(df.head())

# =========================
# CHURN DISTRIBUTION
# =========================

st.subheader("📈 Churn Distribution")

st.bar_chart(df['Churn'].value_counts())

# =========================
# FEATURE OVERVIEW
# =========================

st.subheader("📊 Feature Statistics")

st.write(df.describe())
