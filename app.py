"""
Latihan Deploy (2) - Web App Prediksi Diabetes dengan Naive Bayes
Nama : Dafagiandra Dwi Maulana
NIM  : 23071003
"""

import streamlit as st
import pickle
import numpy as np

# ============================================
# 1. Konfigurasi halaman
# ============================================
st.set_page_config(
    page_title="Prediksi Risiko Diabetes",
    page_icon="🩺",
    layout="centered"
)

# ============================================
# 2. Custom CSS
# ============================================
st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}

    .main-banner {
        background: linear-gradient(135deg, #0ea5a4 0%, #0369a1 100%);
        padding: 2.6rem 1.8rem;
        border-radius: 18px;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 24px rgba(3, 105, 161, 0.25);
    }
    .main-banner h1 {
        color: white;
        font-size: 2.1rem;
        margin-bottom: 0.5rem;
    }
    .main-banner p {
        color: rgba(255,255,255,0.92);
        font-size: 0.95rem;
        margin: 0;
    }
    .badge-row {
        display: flex;
        justify-content: center;
        gap: 0.8rem;
        margin-bottom: 1.6rem;
        flex-wrap: wrap;
    }
    .badge {
        background: white;
        border-radius: 24px;
        padding: 0.5rem 1.2rem;
        font-size: 0.9rem;
        color: #0f172a;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }
    .badge b { color: #0369a1; }

    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin: 0.5rem 0 1rem 0;
        color: #0f172a;
    }

    div[data-testid="stNumberInput"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 0.8rem 1rem 0.4rem 1rem;
        margin-bottom: 0.8rem;
    }
    div[data-testid="stNumberInput"] label p {
        font-weight: 600;
        color: #0369a1;
        font-size: 0.9rem;
    }

    .stButton button {
        border-radius: 12px;
        height: 3rem;
        font-weight: 700;
        font-size: 1rem;
        background: linear-gradient(135deg, #0ea5a4 0%, #0369a1 100%);
        border: none;
    }

    .result-card {
        padding: 1.8rem;
        border-radius: 16px;
        text-align: center;
        margin-top: 1.2rem;
    }
    .result-positive { background: #fef2f2; border: 1px solid #fecaca; }
    .result-negative { background: #f0fdf4; border: 1px solid #bbf7d0; }
    .result-title { font-size: 1.4rem; font-weight: 800; margin-bottom: 0.4rem; }
    .result-sub { font-size: 0.9rem; color: #475569; margin-bottom: 0.8rem; }
</style>
""", unsafe_allow_html=True)

# ============================================
# 3. Load model
# ============================================
with open("Naive_DafagiandraDwiMaulana.pkl", "rb") as file:
    model = pickle.load(file)

# ============================================
# 4. Banner Judul
# ============================================
st.markdown("""
<div class="main-banner">
    <h1>🩺 Prediksi Risiko Diabetes</h1>
    <p>Didukung oleh model Gaussian Naive Bayes yang telah dilatih pada dataset Pima Indians Diabetes</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="badge-row">
    <div class="badge">👤 Nama: <b>Dafagiandra Dwi Maulana</b></div>
    <div class="badge">🎓 NIM: <b>23071003</b></div>
</div>
""", unsafe_allow_html=True)

# ============================================
# 5. Form Input
# ============================================
st.markdown('<div class="section-title">📋 Masukkan Data Pasien</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    Pregnancies = st.number_input("🤰 Pregnancies (jumlah kehamilan)", min_value=0, max_value=20, value=1)
    BloodPressure = st.number_input("❤️ Blood Pressure (tekanan darah)", min_value=0, max_value=200, value=70)
    Insulin = st.number_input("💉 Insulin", min_value=0, max_value=900, value=79)
    DiabetesPedigreeFunction = st.number_input("🧬 Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)

with col2:
    Glucose = st.number_input("🍬 Glucose (kadar glukosa)", min_value=0, max_value=300, value=100)
    SkinThickness = st.number_input("📏 Skin Thickness", min_value=0, max_value=100, value=20)
    BMI = st.number_input("⚖️ BMI (indeks massa tubuh)", min_value=0.0, max_value=70.0, value=25.0)
    Age = st.number_input("🎂 Age (usia)", min_value=1, max_value=120, value=30)

st.write("")

# ============================================
# 6. Tombol Prediksi + Hasil
# ============================================
if st.button("🔍  Prediksi Sekarang", use_container_width=True):
    data_baru = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                            Insulin, BMI, DiabetesPedigreeFunction, Age]])

    hasil = model.predict(data_baru)
    proba = model.predict_proba(data_baru)[0]

    if hasil[0] == 1:
        keyakinan = proba[1]
        st.markdown(f"""
        <div class="result-card result-positive">
            <div class="result-title" style="color:#dc2626;">⚠️ Berpotensi Diabetes</div>
            <div class="result-sub">Tingkat keyakinan model terhadap prediksi ini</div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(keyakinan, text=f"{keyakinan*100:.1f}% keyakinan")
    else:
        keyakinan = proba[0]
        st.markdown(f"""
        <div class="result-card result-negative">
            <div class="result-title" style="color:#16a34a;">✅ Tidak Berpotensi Diabetes</div>
            <div class="result-sub">Tingkat keyakinan model terhadap prediksi ini</div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(keyakinan, text=f"{keyakinan*100:.1f}% keyakinan")

    st.write("")
    with st.expander("📄 Lihat data yang dimasukkan"):
        st.write({
            "Pregnancies": Pregnancies,
            "Glucose": Glucose,
            "Blood Pressure": BloodPressure,
            "Skin Thickness": SkinThickness,
            "Insulin": Insulin,
            "BMI": BMI,
            "Diabetes Pedigree Function": DiabetesPedigreeFunction,
            "Age": Age
        })

st.write("")
st.caption("Tugas Data Mining & Data Warehouse — Latihan Deploy (2)")
