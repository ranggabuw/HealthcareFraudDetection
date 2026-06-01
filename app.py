import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, date

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Healthcare Fraud Detection",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Exo+2:ital,wght@0,700;0,800;1,700&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* Background */
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 50%, #0a1628 100%);
        color: #e0e8f0;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2a 0%, #111d2e 100%);
        border-right: 1px solid rgba(0, 200, 255, 0.15);
    }

    section[data-testid="stSidebar"] * {
        color: #c8d8e8 !important;
    }

    /* Header */
    .hero-header {
        background: linear-gradient(90deg, rgba(0,200,255,0.08) 0%, rgba(0,120,200,0.05) 100%);
        border: 1px solid rgba(0,200,255,0.2);
        border-radius: 16px;
        padding: 32px 40px;
        margin-bottom: 32px;
        position: relative;
        overflow: hidden;
    }
    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(0,200,255,0.06) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-title {
        font-family: 'Exo 2', sans-serif;
        font-size: 2.4rem;
        font-weight: 1200;
        color: #ffffff;
        letter-spacing: -0.5px;
        line-height: 1.2;
    }
    .hero-title span {
        color: #00c8ff;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #7a9ab8;
        margin-top: 8px;
        font-weight: 300;
    }

    /* Cards */
    .card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(0,200,255,0.12);
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 20px;
        transition: border-color 0.2s;
    }
    .card:hover {
        border-color: rgba(0,200,255,0.3);
    }
    .card-title {
        font-family: 'Syne', sans-serif;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #00c8ff;
        margin-bottom: 16px;
    }

    .card-background {
        background: rgba(255,255,255,0);
        padding: 20px;
        margin-bottom: 18px;            
    }
    
    /* Result badges */
    .result-fraud {
        background: linear-gradient(135deg, rgba(255,59,59,0.15), rgba(200,20,20,0.1));
        border: 1.5px solid rgba(255,80,80,0.5);
        border-radius: 16px;
        padding: 28px 32px;
        text-align: center;
    }
    .result-safe {
        background: linear-gradient(135deg, rgba(0,220,130,0.12), rgba(0,180,100,0.08));
        border: 1.5px solid rgba(0,220,130,0.4);
        border-radius: 16px;
        padding: 28px 32px;
        text-align: center;
    }
    .result-label {
        font-family: 'Syne', sans-serif;
        font-size: 1.6rem;
        font-weight: 800;
        margin: 0;
    }
    .result-desc {
        font-size: 0.85rem;
        color: #8aabb8;
        margin-top: 6px;
    }

    /* Metric boxes */
    .metric-box {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .metric-val {
        font-family: 'Syne', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #00c8ff;
    }
    .metric-lbl {
        font-size: 0.72rem;
        color: #6a8a9e;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Input styling override */
    .stSelectbox label, .stNumberInput label, .stSlider label,
    .stDateInput label, .stRadio label, .stCheckbox label {
        color: #8aabb8 !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.8px !important;
    }

    div[data-baseweb="select"] > div {
        background: rgba(255,255,255,0.05) !important;
        border-color: rgba(0,200,255,0.2) !important;
        color: #e0e8f0 !important;
    }

    .stNumberInput input {
        background: rgba(255,255,255,0.05) !important;
        border-color: rgba(0,200,255,0.2) !important;
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        opacity: 1 !important;
    }

    .stNumberInput {
    border: none !important;
    box-shadow: none !important;
    }
            
    /* Button */
    .stButton > button {
        background: linear-gradient(90deg, #0088cc, #00aaf0) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.5px !important;
        padding: 14px 0 !important;
        width: 110% !important;
        transition: opacity 0.2s !important;
        box-shadow: 0 4px 20px rgba(0,150,255,0.3) !important;
    }
    .stButton > button:hover {
        opacity: 0.88 !important;
        box-shadow: 0 6px 28px rgba(0,150,255,0.5) !important;
    }

    /* Cursor pada selectbox */
    div[data-baseweb="select"] > div {
        cursor: pointer !important;
    }

    div[data-baseweb="select"] {
        cursor: pointer !important;
    }

    /* Saat hover */
    div[data-baseweb="select"] > div:hover {
        cursor: pointer !important;
    }

    /* Dropdown option list */
    ul[data-baseweb="menu"] li {
        cursor: pointer !important;
    }       
    
    /* Divider */
    hr {
        border-color: rgba(0,200,255,0.1) !important;
    }

    /* Risk bar */
    .risk-bar-wrap {
        background: rgba(255,255,255,0.06);
        border-radius: 999px;
        height: 8px;
        margin: 8px 0 4px;
        overflow: hidden;
    }
    .risk-bar-fill {
        height: 100%;
        border-radius: 999px;
        transition: width 0.5s ease;
    }

    /* Tab / section label */
    .section-label {
        font-family: 'Syne', sans-serif;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 2.5px;
        text-transform: uppercase;
        color: #4a7a9b;
        margin-bottom: 12px;
    }

    /* Hide default streamlit menu */
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the trained pipeline from disk."""
    model_path = "fraud_pipeline.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

pipeline = load_model()

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:8px 0 24px'>
        <p style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:800;color:#fff;margin:0'>
            🏥 FraudGuard
        </p>
        <p style='font-size:0.75rem;color:#4a7a9b;margin:4px 0 0'>Healthcare ML System</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <p class='section-label'>About</p>
    <p style='font-size:0.82rem;color:#6a8a9e;line-height:1.6'>
    Sistem deteksi fraud klaim asuransi kesehatan berbasis <b style='color:#00c8ff'>XGBoost</b>,
    dilatih dengan SMOTENC untuk menangani ketidakseimbangan data.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("<p class='section-label'>Model Performance</p>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div class='metric-box'>
            <div class='metric-val'>89%</div>
            <div class='metric-lbl'>Accuracy</div>
        </div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class='metric-box'>
            <div class='metric-val'>74%</div>
            <div class='metric-lbl'>Recall</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("""
        <div class='metric-box'>
            <div class='metric-val'>73%</div>
            <div class='metric-lbl'>Precision</div>
        </div>""", unsafe_allow_html=True)
    with col_d:
        st.markdown("""
        <div class='metric-box'>
            <div class='metric-val'>73%</div>
            <div class='metric-lbl'>F1-Score</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    if pipeline is None:
        st.warning("⚠️ Model belum dimuat. Pastikan file `fraud_pipeline.pkl` ada.")
    else:
        st.success("✅ Model berhasil dimuat")

    st.markdown("""
    <p style='font-size:0.72rem;color:#4a7a9b;margin-top:32px'>
    Top Feature:<br>
    <b style='color:#00c8ff'>Days Between Service & Claim</b><br>
    (importance: 56.4%)
    </p>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Main Content
# ─────────────────────────────────────────────
st.markdown("""
<div class='hero-header'>
    <p class='hero-title'>Healthcare <span>Fraud</span> Detection</p>
    <p class='hero-subtitle'>Masukkan data klaim untuk memprediksi potensi kecurangan asuransi kesehatan</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Input Form
# ─────────────────────────────────────────────
col_left, col_right = st.columns([1.1, 0.9], gap="large")

with col_left:
    # ── Patient Info ──────────────────────────
    # st.markdown("<div class='card-background'>", unsafe_allow_html=True)
    st.markdown("<p class='card-title'>👤 Patient Information</p>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        patient_age = st.number_input(
            "Patient Age",
            min_value=0, max_value=120, value=45,
            help="Usia pasien dalam tahun"
        )
    with c2:
        patient_gender = st.selectbox(
            "Patient Gender",
            options=["Male", "Female"],
            help="Jenis kelamin pasien"
        )

    chronic_condition = st.selectbox(
        "Chronic Condition",
        options=["Yes", "No"],
        help="Apakah pasien memiliki kondisi kronis?"
    )
    chronic_condition_flag = 1 if "Yes" in chronic_condition else 0

    st.markdown("<div class='card-background'>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Provider & Insurance ─────────────────
    st.markdown("<p class='card-title'>🏨 Provider & Insurance</p>", unsafe_allow_html=True)

    insurance_type = st.selectbox(
        "Insurance Type",
        options=["Medicaid", "Self-Pay", "Medicare", "Private"],
        help="Jenis asuransi yang digunakan"
    )

    provider_specialty = st.selectbox(
        "Provider Specialty",
        options=[
            "Internal Medicine",
            "General Practice",
            "Orthopedics",
            "Neurology",
            "Cardiology",
            "Pulmonology"
        ],
        help="Spesialisasi penyedia layanan kesehatan"
    )

    c3, c4 = st.columns(2)
    with c3:
        visit_type = st.selectbox(
            "Visit Type",
            options=["Outpatient", "Emergency", "Inpatient"],
            help="Jenis kunjungan pasien"
        )

    with c4:
        length_of_stay = st.number_input(
            "Length of Stay (days)",
            min_value=0, max_value=365, value=0,
            help="Lama rawat inap (0 = rawat jalan)"
        )

    num_claims_monthly = st.number_input(
        "Number of Claims Per Provider (Monthly)",
        min_value=0, max_value=1000, value=50,
        help="Jumlah klaim bulanan dari provider ini"
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    # ── Provider & Insurance ─────────────────
    st.markdown("<p class='card-title'>💵 Claim Submission</p>", unsafe_allow_html=True)
    # st.markdown("<div class='card-background'>", unsafe_allow_html=True)
    claim_amount = st.number_input(
        "Claim Amount (IDR)",
        min_value=0.0, max_value=1000000000000000.0, value=150000.0, step=1000.0,
        help="Total nilai klaim yang diajukan"
    )

    days_between = st.number_input(
        "Days Between Service & Claim",
        min_value=0, max_value=365, value=10,
        help="Selisih hari antara layanan dan pengajuan klaim"
    )

    submission_date = st.date_input(
        "Claim Submission Date",
        value=date.today(),
        help="Tanggal pengajuan klaim"
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Predict Button ───────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔍  Analyze Claim for Fraud")

    # ── Result Display ───────────────────────
    if predict_btn:
        # Build input DataFrame
        input_data = pd.DataFrame({
            "Patient_Age": [patient_age],
            "Patient_Gender": [patient_gender],
            "Claim_Amount": [claim_amount],
            "Insurance_Type": [insurance_type],
            "Days_Between_Service_and_Claim": [days_between],
            "Number_of_Claims_Per_Provider_Monthly": [num_claims_monthly],
            "Provider_Specialty": [provider_specialty],
            "Length_of_Stay": [length_of_stay],
            "Visit_Type": [visit_type],
            "Chronic_Condition_Flag": [chronic_condition_flag],
            "Claim_Submission_mth": [submission_date.month],
            "Claim_Submission_yr": [submission_date.year],
        })

        if pipeline is not None:
            try:
                prediction = pipeline.predict(input_data)[0]
                proba = pipeline.predict_proba(input_data)[0]
                fraud_prob = proba[1]
                safe_prob = proba[0]

                if prediction == 1:
                    risk_color = "#ff4040"
                    st.markdown(f"""
                    <div class='result-fraud'>
                        <p class='result-label' style='color:#ff6060'>⚠️ FRAUD DETECTED</p>
                        <p class='result-desc'>Klaim ini terindikasi <b style='color:#ff8080'>penipuan</b></p>
                        <div style='margin-top:16px'>
                            <p style='font-size:2rem;font-family:Syne,sans-serif;font-weight:800;color:#ff4040;margin:0'>
                                {fraud_prob:.1%}
                            </p>
                            <p style='font-size:0.75rem;color:#8a6a6a;margin:2px 0 0'>Fraud Probability</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    risk_color = "#00dc82"
                    st.markdown(f"""
                    <div class='result-safe'>
                        <p class='result-label' style='color:#00dc82'>✅ CLAIM LEGITIMATE</p>
                        <p class='result-desc'>Klaim ini tampak <b style='color:#00ff99'>sah</b> dan valid</p>
                        <div style='margin-top:16px'>
                            <p style='font-size:2rem;font-family:Syne,sans-serif;font-weight:800;color:#00dc82;margin:0'>
                                {safe_prob:.1%}
                            </p>
                            <p style='font-size:0.75rem;color:#4a8a6a;margin:2px 0 0'>Legitimacy Confidence</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Risk bar
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<p class='section-label'>Risk Score Breakdown</p>", unsafe_allow_html=True)

                for label, val, color in [
                    ("Fraud Risk", fraud_prob, "#ff5050"),
                    ("Safe Score", safe_prob, "#00dc82"),
                ]:
                    pct = int(val * 100)
                    st.markdown(f"""
                    <div style='margin-bottom:10px'>
                        <div style='display:flex;justify-content:space-between;font-size:0.78rem;color:#8aabb8'>
                            <span>{label}</span><span style='color:{color};font-weight:600'>{pct}%</span>
                        </div>
                        <div class='risk-bar-wrap'>
                            <div class='risk-bar-fill' style='width:{pct}%;background:{color}'></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.info("Pastikan kolom input sesuai dengan data training.")
        else:
            st.warning("⚠️ Model belum dimuat. Jalankan notebook terlebih dahulu dan simpan pipeline ke `fraud_pipeline.pkl`.")

            # Demo mode
            st.markdown("**[Demo Mode — tanpa model]**")
            demo_risk = min(1.0, (days_between / 30) * 0.4 + (claim_amount / 10000) * 0.3 + (0.3 if insurance_type == "Self-Pay" else 0.1))
            if demo_risk > 0.5:
                st.markdown(f"""
                <div class='result-fraud'>
                    <p class='result-label' style='color:#ff6060'>⚠️ HIGH RISK (Demo)</p>
                    <p class='result-desc'>Estimasi risiko fraud: <b>{demo_risk:.1%}</b></p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='result-safe'>
                    <p class='result-label' style='color:#00dc82'>✅ LOW RISK (Demo)</p>
                    <p class='result-desc'>Estimasi risiko fraud: <b>{demo_risk:.1%}</b></p>
                </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Feature Importance Reference (bottom)
# ─────────────────────────────────────────────
# st.markdown("---")
# st.markdown("<p class='section-label'>📊 Top 10 Feature Importance (Model Reference)</p>", unsafe_allow_html=True)

# features = [
#     ("Days Between Service and Claim", 56.4),
#     ("Chronic Condition Flag",          8.9),
#     ("Claim Submission Year",           5.9),
#     ("Claim Amount",                    4.3),
#     ("Length of Stay",                  4.1),
#     ("Provider Specialty – Internal Medicine", 2.0),
#     ("Claim Submission Month",          2.0),
#     ("Insurance Type – Self-Pay",       1.8),
#     ("Provider Specialty – Pulmonology",1.6),
#     ("Insurance Type – Medicare",       1.5),
# ]

# cols = st.columns(2)
# for i, (feat, imp) in enumerate(features):
#     with cols[i % 2]:
#         bar_w = int(imp / features[0][1] * 100)
#         st.markdown(f"""
#         <div style='margin-bottom:10px'>
#             <div style='display:flex;justify-content:space-between;font-size:0.78rem;color:#8aabb8'>
#                 <span>{feat}</span>
#                 <span style='color:#00c8ff;font-weight:600'>{imp}%</span>
#             </div>
#             <div class='risk-bar-wrap'>
#                 <div class='risk-bar-fill' style='width:{bar_w}%;background:linear-gradient(90deg,#0088cc,#00c8ff)'></div>
#             </div>
#         </div>
#         """, unsafe_allow_html=True)

st.markdown("<br><p style='font-size:0.72rem;color:#2a4a5e;text-align:center'>Healthcare Fraud Detection System · XGBoost + SMOTENC Pipeline</p>", unsafe_allow_html=True)
