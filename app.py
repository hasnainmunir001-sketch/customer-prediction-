import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="AI Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

# ================= MODERN UI (GRADIENT) =================
st.markdown("""
<style>
    .main-title {
        font-size: 45px;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(90deg,#00C9FF,#92FE9D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #aaa;
    }

    .card {
        background: linear-gradient(135deg,#1f1c2c,#928dab);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown("<div class='main-title'>AI Customer Churn Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Multi Model ML System with Modern UI</div>", unsafe_allow_html=True)

# ================= SIDEBAR =================
st.sidebar.title("⚙ Navigation")
uploaded_file = st.sidebar.file_uploader("Upload Dataset", type=["csv"])

menu = st.sidebar.radio("Go To", ["🏠 Home", "📊 Analytics", "🔮 Prediction"])

# ================= LOAD DATA =================
if uploaded_file:

    df = pd.read_csv(uploaded_file)

    # CLEANING
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # ENCODING
    cat_cols = df.select_dtypes(include=['object']).columns
    le = LabelEncoder()

    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # MODELS
    models = {
        "Random Forest": RandomForestClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Naive Bayes": GaussianNB(),
        "KNN": KNeighborsClassifier()
    }

    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

    # ================= HOME =================
    if menu == "🏠 Home":

        st.markdown("### 📌 Dashboard Overview")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"<div class='card'>Rows<br><h2>{df.shape[0]}</h2></div>", unsafe_allow_html=True)

        with col2:
            st.markdown(f"<div class='card'>Features<br><h2>{df.shape[1]-1}</h2></div>", unsafe_allow_html=True)

        with col3:
            st.markdown("<div class='card'>Models<br><h2>4</h2></div>", unsafe_allow_html=True)

    # ================= ANALYTICS =================
    elif menu == "📊 Analytics":

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.subheader("Statistics")
        st.write(df.describe())

        st.subheader("Correlation")
        st.write(df.corr())

    # ================= PREDICTION =================
    elif menu == "🔮 Prediction":

        st.subheader("Single Customer Prediction")

        input_data = []

        cols = st.columns(3)

        for i, col in enumerate(X.columns):
            with cols[i % 3]:
                val = st.number_input(col, value=float(df[col].mean()))
                input_data.append(val)

        model_name = st.selectbox("Select Model", list(models.keys()))

        if st.button("Predict Now 🚀"):

            input_array = np.array(input_data).reshape(1, -1)
            input_scaled = scaler.transform(input_array)

            model = trained_models[model_name]
            prediction = model.predict(input_scaled)[0]

            st.markdown("---")

            if prediction == 1:
                st.error(f"⚠ Customer WILL CHURN ({model_name})")
            else:
                st.success(f"✅ Customer WILL STAY ({model_name})")

else:
    st.warning("⬅ Please upload dataset from sidebar")
