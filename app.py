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
    page_title="Prediksi Diabetes - Naive Bayes",
    page_icon="🩺",
    layout="centered"
)

# ============================================
# 2. Load model yang sudah di-training
# ============================================
with open("Naive_DafagiandraDwiMaulana.pkl", "rb") as file:
    model = pickle.load(file)

# ============================================
# 3. Sidebar - Info identitas
# ============================================
with st.sidebar:
    st.header("ℹ️ Tentang Aplikasi")
    st.write("Aplikasi ini memprediksi potensi diabetes seseorang menggunakan algoritma **Naive Bayes**.")
    st.markdown("---")
    st.write("**Nama :** Dafagiandra Dwi Maulana")
    st.write("**NIM  :** 23071003")
    st.markdown("---")
    st.caption("Tugas Data Mining & Data Warehouse - Latihan Deploy (2)")

# ============================================
# 4. Header utama
# ============================================
st.title("🩺 Prediksi Diabetes")
st.markdown("Menggunakan model **Naive Bayes** yang telah dilatih pada dataset diabetes.")
st.markdown("---")

st.subheader("📋 Masukkan Data Pasien")

# ============================================
# 5. Form Input dibuat 2 kolom biar rapi
# ============================================
col1, col2 = st.columns(2)

with col1:
    Pregnancies = st.number_input("Pregnancies (jumlah kehamilan)", min_value=0, max_value=20, value=1)
    BloodPressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
    Insulin = st.number_input("Insulin", min_value=0, max_value=900, value=79)
    DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)

with col2:
    Glucose = st.number_input("Glucose", min_value=0, max_value=300, value=100)
    SkinThickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    BMI = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
    Age = st.number_input("Age", min_value=1, max_value=120, value=30)

st.markdown("---")

# ============================================
# 6. Tombol Prediksi + Hasil
# ============================================
if st.button("🔍 Prediksi Sekarang", use_container_width=True):
    data_baru = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                            Insulin, BMI, DiabetesPedigreeFunction, Age]])

    hasil = model.predict(data_baru)

    st.markdown("### Hasil Prediksi")
    if hasil[0] == 1:
        st.error("⚠️ Pasien **Berpotensi DIABETES**")
    else:
        st.success("✅ Pasien **Tidak Berpotensi Diabetes**")

    with st.expander("Lihat data yang dimasukkan"):
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
