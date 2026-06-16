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
from sklearn.metrics import accuracy_score, confusion_matrix

# ================= PAGE CONFIG =================
st.set_page_config(page_title="AI Churn Dashboard", layout="wide")

# ================= MODERN UI (NEW DESIGN) =================
st.markdown("""
<style>

/* BACKGROUND */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0a0f1c, #111827, #0a0f1c);
}

/* TITLE */
.title {
    font-size: 48px;
    font-weight: 900;
    text-align: center;
    background: linear-gradient(90deg,#00f5ff,#8b5cf6,#ff00d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* SUBTITLE */
.subtitle {
    text-align: center;
    color: #94a3b8;
    font-size: 16px;
    margin-bottom: 30px;
}

/* CARD */
.card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(15px);
    border-radius: 18px;
    padding: 20px;
    border: 1px solid rgba(255,255,255,0.1);
    text-align: center;
    transition: 0.3s;
}

.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 25px rgba(0,0,0,0.5);
}

/* BUTTON */
.stButton>button {
    background: linear-gradient(90deg,#7c3aed,#06b6d4);
    color: white;
    border-radius: 12px;
    padding: 10px 20px;
    font-weight: bold;
    border: none;
}

.stButton>button:hover {
    transform: scale(1.05);
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0f172a,#1e293b);
}

/* TABS */
div[data-testid="stTabs"] button {
    color: #94a3b8 !important;
}

div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #22d3ee !important;
    border-bottom: 2px solid #22d3ee !important;
}

</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown('<div class="title">AI Customer Churn Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Modern ML System with Multi-Model Prediction & Analytics</div>', unsafe_allow_html=True)

# ================= UPLOAD =================
uploaded_file = st.file_uploader("📁 Upload CSV Dataset", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.success("Dataset Loaded Successfully!")

    st.dataframe(df.head())

    # ================= CLEANING =================
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # ================= ENCODING =================
    cat_cols = df.select_dtypes(include=['object']).columns
    le = LabelEncoder()

    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    # ================= SPLIT =================
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # ================= MODELS =================
    models = {
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB()
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        results[name] = model

    # ================= TABS =================
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📈 Analytics", "🔮 Prediction"])

    # ================= TAB 1 =================
    with tab1:

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"<div class='card'><h3>Rows</h3><h2>{df.shape[0]}</h2></div>", unsafe_allow_html=True)

        with col2:
            st.markdown(f"<div class='card'><h3>Features</h3><h2>{df.shape[1]-1}</h2></div>", unsafe_allow_html=True)

        with col3:
            st.markdown(f"<div class='card'><h3>Models</h3><h2>4</h2></div>", unsafe_allow_html=True)

    # ================= TAB 2 =================
    with tab2:

        st.subheader("Model Accuracy")

        for name, model in models.items():
            acc = model.score(X_test, y_test)
            st.write(f"✔ {name} → {acc:.2f}")

        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(), ax=ax, cmap="coolwarm")
        st.pyplot(fig)

    # ================= TAB 3 =================
    with tab3:

        st.subheader("Single Customer Prediction")

        input_data = []

        cols = st.columns(3)

        for i, col in enumerate(X.columns):
            with cols[i % 3]:
                val = st.number_input(col, value=float(df[col].mean()))
                input_data.append(val)

        model_choice = st.selectbox("Select Model", list(models.keys()))

        if st.button("Predict Now 🚀"):

            model = results[model_choice]

            input_array = np.array(input_data).reshape(1, -1)
            input_scaled = scaler.transform(input_array)

            prediction = model.predict(input_scaled)[0]

            st.markdown("---")

            if prediction == 1:
                st.error(f"⚠ Customer WILL CHURN ({model_choice})")
            else:
                st.success(f"✅ Customer WILL STAY ({model_choice})")

else:
    st.warning("⬅ Upload dataset to start analysis")
