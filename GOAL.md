# Goal & Requirements: Supervised Learning Mini Project

File ini mencatat seluruh tujuan, ruang lingkup, dan deliverables yang disepakati pada percakapan ini agar seluruh alur *machine learning* dan integrasi aplikasi terjaga hingga selesai.

## Konteks Tugas
- **Mata Ajar:** Mini Project Supervised Learning (Bootcamp Data Science — diskusi kelas bersama mentor **Kak Vanya Mayazura**).
- **Direktori Target:** `ujian 1 mat/mini project 1/`
- **Dataset:** Kaggle [Credit Risk Dataset (`laotse/credit-risk-dataset`)](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) → `credit_risk_dataset.csv` (`32,581` baris × `12` kolom).

## Tahapan Wajib yang Diminta
1. **Data Understanding:** Audit skema, tipe data, missing values, duplikat, distribusi target, dan anomali domain (`person_age > 100`, `person_emp_length > 60`).
2. **Data Cleaning:** Hapus duplikat persis dan anomali domain yang mustahil tanpa membocorkan informasi statistik imputasi (`32,409` baris bersih).
3. **Feature Engineering:** Rancang fitur turunan domain keuangan yang *stateless* (`cred_hist_to_age_ratio`, `income_to_loan_ratio`, `emp_to_age_ratio`, `high_burden_flag`) serta pipeline preprocessing bebas *data leakage* (`SimpleImputer` + missing indicator, `StandardScaler`, `OrdinalEncoder`, `OneHotEncoder`).
4. **Exploratory Data Analysis (EDA):** Visualisasi distribusi numerik, tingkat gagal bayar & rata-rata suku bunga per kategori, matriks korelasi, dan peta interaksi risiko.
5. **Data Modeling and Evaluations (Dual Supervised Learning Tracks):**
   - **Track A — Classification (`loan_status`, Prediksi Gagal Bayar):**
     - Bandingkan 8 model: `DummyClassifier`, `LogisticRegression` (Default & Balanced), `KNeighborsClassifier`, `DecisionTreeClassifier` (Unregularized & Regularized), `RandomForestClassifier`, `XGBClassifier`.
     - Validasi 5-Fold Stratified CV, Hyperparameter Tuning (`GridSearchCV`, `RandomizedSearchCV`, `Optuna`), Learning Curve, Threshold Tuning (`0.50` vs `0.60` F1-Optimal vs `0.38` High-Recall), Evaluasi Test Set (**ROC-AUC `0.9564`**, **PR-AUC `0.9168`**, **F1 `0.8434`**) + Bootstrap 95% CI, Slice-Based Error Analysis, dan Interpretabilitas.
   - **Track B — Multiple Regression (`loan_int_rate`, Prediksi Suku Bunga Wajar Berbasis Risiko):**
     - Bandingkan 9 model: `DummyRegressor`, `LinearRegression` (*Multiple Linear Regression*), `Ridge`, `Lasso`, `KNeighborsRegressor`, `DecisionTreeRegressor` (Unregularized & Regularized), `RandomForestRegressor`, `XGBRegressor`.
     - Bebas *target leakage* (`loan_status` dikeluarkan, `29,315` baris dengan `loan_int_rate` valid).
     - Validasi 5-Fold CV (`R²`, `RMSE`, `MAE`), Tuning (`RandomizedSearchCV`), Evaluasi Test Set (**Test R² `0.9010`**, **RMSE `1.0194`**, **MAE `0.7949`**) + Bootstrap 95% CI, Plot Actual vs Predicted & Residual Diagnostics, serta Feature Importance.
6. **Streamlit Interactive App (`app.py`):**
   - Antarmuka web interaktif 4-tab melayani kedua model (**Prediksi Risiko Default / Classification** + **Estimasi Suku Bunga / Multiple Regression**), preset profil pemohon (`Low Risk`, `Medium Risk`, `High Risk`), *policy threshold slider*, prediksi batch CSV + unduh hasil, serta dashboard EDA & perbandingan evaluasi seluruh model.

## Checklist Deliverables
- [x] `credit_risk_dataset.csv` — Dataset mentah dari Kaggle (`32,581` baris × `12` kolom)
- [x] `GOAL.md` — Daftar target & checklist tugas percakapan
- [x] `features.py` — Custom transformer & pipeline helpers untuk notebook dan Streamlit
- [x] `mini-project-credit-risk.ipynb` — Notebook lengkap tereksekusi (`68` sel, `24` kode tereksekusi, `12` grafik inline, analisis lengkap di setiap sel)
- [x] `model_credit_risk_xgb.joblib` — Artifak model Classification (`Tuned XGBoost`) + metadata evaluasi
- [x] `model_interest_rate_reg.joblib` — Artifak model Multiple Regression (`Tuned Random Forest Regressor`) + metadata evaluasi
- [x] `app.py` — Aplikasi web interaktif Streamlit 5-tab (termasuk halaman panduan pemula, preset profil Budi/Sinta/Andi, terverifikasi via headless `AppTest`)
- [x] `requirements.txt`, `.python-version`, `.gitignore`, `.streamlit/config.toml` — Konfigurasi siap deploy ke GitHub & Streamlit Community Cloud (`https://github.com/Rifaditya/mini-project-1`)
- [x] `README.md` — Dokumentasi teknis project & panduan deploy
- [x] `PANDUAN_DISKUSI_KELAS.md` — Ringkasan presentasi & tanya-jawab untuk diskusi kelas
