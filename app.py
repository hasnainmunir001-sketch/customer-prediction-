import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────
# PAGE CONFIG
# ─────────────────────────────
st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

# ─────────────────────────────
# MODERN DARK UI + SIDEBAR FIX
# ─────────────────────────────
st.markdown("""
<style>

/* MAIN BACKGROUND */
.main {
    background-color: #0e1117;
}

/* HEADER */
.header {
    background: linear-gradient(135deg,#111827,#1f2937);
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    margin-bottom: 20px;
}
.header h1 { color: white; margin: 0; }
.header p { color: #9ca3af; }

/* CARDS */
.card {
    background: #1f2937;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #374151;
    color: white;
    text-align: center;
}

/* BUTTON */
.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
}

/* ───── SIDEBAR FIX ───── */

/* Sidebar background */
section[data-testid="stSidebar"] {
    background-color: #000000 !important;
}

/* Sidebar ALL TEXT WHITE */
section[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

/* Radio buttons */
div[data-testid="stRadio"] label {
    color: #ffffff !important;
}

/* File uploader */
div[data-testid="stFileUploader"] {
    color: #ffffff !important;
}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────
# HEADER
# ─────────────────────────────
st.markdown("""
<div class="header">
    <h1>📊 Customer Churn Analysis</h1>
    <p>Machine Learning Dashboard</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────
# SIDEBAR
# ─────────────────────────────
with st.sidebar:
    st.title("📌 MENU")

    page = st.radio("Navigate", [
        "🏠 Overview",
        "📊 Data Analysis",
        "⚙️ ML Pipeline",
        "🤖 Model Results",
        "👥 Clustering",
        "🔮 Prediction"
    ])

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

# ─────────────────────────────
# LOAD DATA + TRAIN MODELS
# ─────────────────────────────
@st.cache_data
def process_data(df):
    df = df.copy()

    df = df.drop(columns=[c for c in ['RowNumber','CustomerId','Surname'] if c in df.columns])

    encoders = {}
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    X = df.drop("Exited", axis=1)
    y = df["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = {
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB()
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results[name] = accuracy_score(y_test, pred) * 100

    df["Cluster"] = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(X)

    return df, X, scaler, encoders, models, results

# ─────────────────────────────
# DATA LOAD
# ─────────────────────────────
if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)
else:
    try:
        df_raw = pd.read_csv("Customer-Churn-Records.csv")
    except:
        df_raw = None

# ─────────────────────────────
# MAIN APP
# ─────────────────────────────
if df_raw is not None:

    df, X, scaler, encoders, models, results = process_data(df_raw)

    # ───── OVERVIEW ─────
    if page == "🏠 Overview":
        st.subheader("Overview")

        col1, col2, col3 = st.columns(3)

        col1.metric("Rows", len(df))
        col2.metric("Churn Rate", f"{df['Exited'].mean()*100:.2f}%")
        col3.metric("Features", df.shape[1])

        st.bar_chart(df["Exited"].value_counts())

    # ───── DATA ANALYSIS ─────
    elif page == "📊 Data Analysis":
        st.subheader("Data Analysis")

        fig, ax = plt.subplots()
        sns.heatmap(df.corr(), ax=ax, cmap="coolwarm")
        st.pyplot(fig)

    # ───── ML PIPELINE ─────
    elif page == "⚙️ ML Pipeline":
        st.subheader("ML Steps")
        st.write("✔ Cleaning")
        st.write("✔ Encoding")
        st.write("✔ Scaling")
        st.write("✔ Training")

    # ───── MODEL RESULTS ─────
    elif page == "🤖 Model Results":
        st.subheader("Accuracy Results")

        for name, acc in results.items():
            st.write(f"{name}: {acc:.2f}%")

    # ───── CLUSTERING ─────
    elif page == "👥 Clustering":
        st.subheader("Customer Clusters")

        fig, ax = plt.subplots()
        ax.scatter(df["Age"], df["Balance"], c=df["Cluster"])
        st.pyplot(fig)

    # ───── PREDICTION ─────
    elif page == "🔮 Prediction":
        st.subheader("Live Prediction")

        credit = st.number_input("Credit Score", 300, 900, 650)
        age = st.number_input("Age", 18, 100, 30)
        balance = st.number_input("Balance", 0, 200000, 50000)

        model_name = st.selectbox("Model", list(models.keys()))

        if st.button("Predict"):
            model = models[model_name]

            sample = np.array([[credit, age, balance] + [0]*(X.shape[1]-3)])
            sample = scaler.transform(sample)

            pred = model.predict(sample)[0]

            if pred == 1:
                st.error("❌ Customer will churn")
            else:
                st.success("✅ Customer will stay")

else:
    st.warning("Upload dataset from sidebar")
