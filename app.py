"""
Latihan Deploy (2) - Web App Prediksi Diabetes dengan Naive Bayes
Nama : Dafagiandra Dwi Maulana
NIM  : 23071003
"""

import streamlit as st
import pickle
import numpy as np

# ============================================
# 1. Load model yang sudah di-training
# ============================================
with open("Naive_DafagiandraDwiMaulana.pkl", "rb") as file:
    model = pickle.load(file)

# ============================================
# 2. Tampilan Halaman Web
# ============================================
st.title("Prediksi Diabetes - Naive Bayes")
st.write("Nama  : Dafagiandra Dwi Maulana")
st.write("NIM   : 23071003")
st.write("Masukkan data pasien di bawah ini untuk memprediksi apakah pasien berpotensi diabetes atau tidak.")

# ============================================
# 3. Form Input (sesuai kolom dataset diabetes.csv)
# ============================================
Pregnancies = st.number_input("Pregnancies (jumlah kehamilan)", min_value=0, max_value=20, value=1)
Glucose = st.number_input("Glucose", min_value=0, max_value=300, value=100)
BloodPressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
SkinThickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
Insulin = st.number_input("Insulin", min_value=0, max_value=900, value=79)
BMI = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
Age = st.number_input("Age", min_value=1, max_value=120, value=30)

# ============================================
# 4. Tombol Prediksi
# ============================================
if st.button("Prediksi"):
    data_baru = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                            Insulin, BMI, DiabetesPedigreeFunction, Age]])

    hasil = model.predict(data_baru)

    if hasil[0] == 1:
        st.error("Hasil Prediksi: Berpotensi DIABETES")
    else:
        st.success("Hasil Prediksi: TIDAK Berpotensi Diabetes")
