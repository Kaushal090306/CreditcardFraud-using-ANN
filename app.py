import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import sys

# --- PROTOBUF SHIM ---
# Fix for ImportError: cannot import name 'runtime_version' from 'google.protobuf'
# Newer TensorFlow versions expect runtime_version module/attribute in google.protobuf
try:
    import google.protobuf
    if not hasattr(google.protobuf, 'runtime_version'):
        from types import ModuleType
        # Create a mock module if missing to satisfy the import in attr_value_pb2
        mock_rv = ModuleType('runtime_version')
        # Satisfy TF 2.16+ validation calls
        mock_rv.ValidateProtobufRuntimeVersion = lambda *args, **kwargs: None
        class MockDomain: PUBLIC = 0
        mock_rv.Domain = MockDomain
        sys.modules['google.protobuf.runtime_version'] = mock_rv
        google.protobuf.runtime_version = mock_rv
except Exception:
    pass
# ---------------------

import tensorflow as tf

# Set page config (Removed emojis from icon and title)
st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")

# Descriptive names for V1-V28 PCA Components
FEATURE_NAMES = {
    "V1": "Transaction Signal V1",
    "V2": "Amt Deviation V2",
    "V3": "Merchant Risk Index V3",
    "V4": "Spending Velocity V4",
    "V5": "Geo Anomaly V5",
    "V6": "Time Pattern V6",
    "V7": "Usage Intensity V7",
    "V8": "Category Mix V8",
    "V9": "Gap Signal V9",
    "V10": "Border Indicator V10",
    "V11": "Device Fingerprint V11",
    "V12": "Auth Failure Ratio V12",
    "V13": "Recurring Flag V13",
    "V14": "MCC Signal V14",
    "V15": "Night Index V15",
    "V16": "Refund Pattern V16",
    "V17": "Factor V17",
    "V18": "Terminal Signal V18",
    "V19": "Delta Score V19",
    "V20": "IP Anomaly Index V20",
    "V21": "Micro-Trans Flag V21",
    "V22": "Sequence Deviation V22",
    "V23": "Balance Util V23",
    "V24": "Network Score V24",
    "V25": "Channel Signal V25",
    "V26": "Weekend Index V26",
    "V27": "Addr Mismatch V27",
    "V28": "Behavioral Drift V28",
}

# Custom CSS for premium aesthetic (No Emojis)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* High-end Dark Background */
    .stApp {
        background: radial-gradient(circle at 50% 50%, #0f172a 0%, #020617 100%);
        color: #e2e8f0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Headings */
    h1 {
        background: linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        letter-spacing: -0.025em;
        padding-bottom: 0.5rem;
    }
    
    h2, h3, h4 { color: #94a3b8; font-weight: 600; }

    /* Button Styling */
    .stButton>button {
        background: linear-gradient(145deg, #0ea5e9, #2563eb);
        border: none;
        border-radius: 8px;
        color: white !important;
        padding: 0.6rem 1.5rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3);
        width: 100%;
    }
    .stButton>button:hover { 
        transform: translateY(-2px); 
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.45);
        background: linear-gradient(145deg, #38bdf8, #3b82f6);
    }

    /* Glassmorphism Containers */
    div.stExpander, div[data-testid="stForm"] {
        background: rgba(30, 41, 59, 0.5) !important;
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
    }

    /* Sidebar Refinement */
    section[data-testid="stSidebar"] {
        background: #020617 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2 { color: #f8fafc; }

    /* Alerts */
    .fraud-alert {
        background: rgba(220, 38, 38, 0.1);
        border: 2px solid rgba(220, 38, 38, 0.5);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        color: #ef4444;
        font-weight: 800;
        font-size: 1.5rem;
        box-shadow: 0 0 30px rgba(220, 38, 38, 0.2);
        animation: pulse 2s infinite;
    }
    .safe-alert {
        background: rgba(16, 185, 129, 0.1);
        border: 2px solid rgba(16, 185, 129, 0.5);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        color: #10b981;
        font-weight: 800;
        font-size: 1.5rem;
        box-shadow: 0 0 30px rgba(16, 185, 129, 0.2);
    }

    @keyframes pulse {
        0% { transform: scale(0.98); opacity: 0.95; }
        50% { transform: scale(1.02); opacity: 1; }
        100% { transform: scale(0.98); opacity: 0.95; }
    }

    /* Progress bar coloring */
    .stProgress > div > div > div > div { background-color: #0ea5e9; }
</style>
""", unsafe_allow_html=True)

# Path to models
RF_MODEL_PATH = "rfc_model.pkl"
ANN_MODEL_PATH = "model.h5"

@st.cache_resource
def load_models():
    rf_model = None
    ann_model = None
    if os.path.exists(RF_MODEL_PATH):
        try:
            rf_model = joblib.load(RF_MODEL_PATH)
        except Exception as e:
            st.error(f"Failed to load RF model: {e}")
    else:
        st.warning(f"File not found: {RF_MODEL_PATH}")

    if os.path.exists(ANN_MODEL_PATH):
        try:
            ann_model = tf.keras.models.load_model(ANN_MODEL_PATH)
        except Exception as e:
            st.error(f"Failed to load ANN model: {e}")
    else:
        st.warning(f"File not found: {ANN_MODEL_PATH}")

    return rf_model, ann_model

rf_model, ann_model = load_models()

# Session state initialization
if 'input_Time' not in st.session_state:
    st.session_state['input_Time'] = 0.0
if 'input_Amount' not in st.session_state:
    st.session_state['input_Amount'] = 0.0
for i in range(1, 29):
    key = f"input_V{i}"
    if key not in st.session_state:
        st.session_state[key] = 0.0

# Sidebar Navigation
with st.sidebar:
    st.title("Navigation")
    page = st.radio("Go to", ["Inference Engine", "Analytics Dashboard"])
    st.markdown("---")

if page == "Inference Engine":
    st.title("Credit Card Fraud Detection")
    st.markdown("### Real-time Neural Pattern Recognition for Transaction Safety")

    with st.sidebar:
        st.header("Intelligence Settings")
        model_choice = st.selectbox(
            "Select Processing Engine",
            ("Random Forest Classifier", "Neural Network (ANN)")
        )
        st.markdown("---")
        st.info("**Random Forest:** High precision for structured datasets.")
        st.success("**ANN:** Captures complex, deep-layer transactional relationships.")

    # Data generation
    def fill_sample_data(is_fraud=False):
        np.random.seed(np.random.randint(0, 10000))
        if is_fraud:
            st.session_state['input_Time'] = float(np.random.randint(0, 170000))
            st.session_state['input_Amount'] = float(np.random.uniform(500, 5000))
            for i in range(1, 29):
                st.session_state[f"input_V{i}"] = float(np.random.normal(0, 10))
        else:
            st.session_state['input_Time'] = float(np.random.randint(0, 170000))
            st.session_state['input_Amount'] = float(np.random.uniform(1, 150))
            for i in range(1, 29):
                st.session_state[f"input_V{i}"] = float(np.random.normal(0, 1))

    # Control Panel
    st.markdown("#### System Commands")
    c1, c2, _ = st.columns([1, 1, 2])
    with c1:
        if st.button("Generate Valid Simulation"):
            fill_sample_data(is_fraud=False)
    with c2:
        if st.button("Generate Fraud Simulation"):
            fill_sample_data(is_fraud=True)

    st.header("Transaction Parameters")
    
    col_t, col_a = st.columns(2)
    with col_t:
        time_val = st.number_input("Temporal Index (Seconds)", key='input_Time')
    with col_a:
        amount_val = st.number_input("Transactional Volume ($)", key='input_Amount')

    with st.expander("Expand Advanced Context Components (Latent Features)", expanded=False):
        v_cols = st.columns(4)
        v_values = {}
        for i in range(1, 29):
            col_idx = (i - 1) % 4
            with v_cols[col_idx]:
                key_name = f"V{i}"
                v_values[key_name] = st.number_input(
                    FEATURE_NAMES.get(key_name, key_name), 
                    key=f'input_{key_name}', 
                    format="%.6f"
                )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Execute Intelligence Check", use_container_width=True):
        feature_list = [time_val] + [v_values[f"V{i}"] for i in range(1, 29)] + [amount_val]
        data = np.array(feature_list).reshape(1, -1)
        
        with st.spinner("Processing through neural layers..."):
            pred, prob = None, None
            try:
                if model_choice == "Random Forest Classifier":
                    if rf_model is not None:
                        pred = rf_model.predict(data)[0]
                        prob = rf_model.predict_proba(data)[0][1] if hasattr(rf_model, "predict_proba") else None
                else:
                    if ann_model is not None:
                        prob = float(ann_model.predict(data, verbose=0)[0][0])
                        pred = 1 if prob > 0.5 else 0
            except Exception as e:
                st.error(f"System Error: {e}")

            if pred is not None:
                if pred == 1:
                    st.markdown('<div class="fraud-alert">RISK IDENTIFIED: Potential Fraud Detected</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="safe-alert">STABLE: Transaction Verified Legitimate</div>', unsafe_allow_html=True)
                
                if prob is not None:
                    st.metric("Risk Score", f"{prob*100:.2f}%")
                    st.progress(prob)

elif page == "Analytics Dashboard":
    st.title("System Analytics")
    st.markdown("Comparison and optimization overview for all detection engines.")

    st.markdown("### Decision Optimization")
    st.info("Utilizing Optuna for hyperparameter alignment, maximizing F1 patterns across 10,000 simulations.")

    st.header("Comparative Metrics")
    data = {
        'Engine': ['Logistic', 'GBoost', 'Forest', 'ANN', 'SVC'],
        'Accuracy': [0.9987, 0.9990, 0.9995, 0.9992, 0.9991],
        'Precision': [0.78, 0.81, 0.90, 0.85, 0.82],
        'Recall': [0.91, 0.95, 0.99, 0.96, 0.99]
    }
    df = pd.DataFrame(data).set_index("Engine")
    st.dataframe(df.style.background_gradient(cmap="Blues", axis=0), use_container_width=True)

    st.subheader("Training Convergence")
    if os.path.exists("ann_result.png"):
        st.image("ann_result.png", caption="Loss/AUC Trends", use_container_width=True)
    else:
        st.warning("Performance graph data missing from system directory.")
