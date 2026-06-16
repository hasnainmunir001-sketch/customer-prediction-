import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# =========================
# PAGE CONFIG (MODERN UI)
# =========================
st.set_page_config(
    page_title="AI Churn Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CUSTOM CSS (MODERN LOOK)
# =========================
st.markdown("""
<style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        color: #4CAF50;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: gray;
        margin-bottom: 20px;
    }
    .card {
        padding: 20px;
        border-radius: 15px;
        background-color: #111827;
        color: white;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.3);
        text-align: center;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #00FFAA;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("<div class='main-title'>📊 AI Customer Churn Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload data → Train model → Predict churn instantly</div>", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("⚙ Navigation Panel")
uploaded_file = st.sidebar.file_uploader("📁 Upload Dataset", type=["csv"])

page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Analytics", "🔮 Prediction"])

# =========================
# LOAD DATA
# =========================
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # CLEANING
    df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    df.drop(columns=['customerID'], inplace=True)
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

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    # =========================
    # HOME PAGE
    # =========================
    if page == "🏠 Home":

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div class="card">
                📊 Dataset Rows<br>
                <div class="metric-value">{df.shape[0]}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="card">
                🧠 Model Accuracy<br>
                <div class="metric-value">{accuracy:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="card">
                📁 Features<br>
                <div class="metric-value">{df.shape[1]-1}</div>
            </div>
            """, unsafe_allow_html=True)

        st.info("👈 Use sidebar to explore Analytics or Prediction")

    # =========================
    # ANALYTICS PAGE
    # =========================
    elif page == "📊 Analytics":

        st.subheader("📈 Dataset Overview")

        tab1, tab2, tab3 = st.tabs(["Data Preview", "Statistics", "Correlation"])

        with tab1:
            st.dataframe(df.head(20))

        with tab2:
            st.write(df.describe())

        with tab3:
            st.write(df.corr())

    # =========================
    # PREDICTION PAGE
    # =========================
    elif page == "🔮 Prediction":

        st.subheader("🔮 Predict Customer Churn")

        st.write("Fill customer details below:")

        input_data = []

        cols = st.columns(3)

        for i, col in enumerate(X.columns):
            with cols[i % 3]:
                val = st.number_input(col, value=float(df[col].mean()))
                input_data.append(val)

        if st.button("🚀 Predict Now"):

            input_array = np.array(input_data).reshape(1, -1)
            input_scaled = scaler.transform(input_array)

            prediction = model.predict(input_scaled)[0]

            st.markdown("---")

            if prediction == 1:
                st.error("⚠ HIGH RISK: Customer will CHURN")
            else:
                st.success("✅ LOW RISK: Customer will STAY")

else:
    st.warning("⬅ Please upload CSV file from sidebar to start")
