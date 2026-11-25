# Panduan Diskusi Kelas — Mini Project Supervised Learning

Catatan ringkas ini disusun untuk presentasi 3–5 menit dan sesi tanya-jawab bersama mentor (**Kak Vanya Mayazura**).

---

## 1. Alur Presentasi 3–5 Menit

1. **Problem Statement & Konteks Bisnis (30 detik)**
   - Dataset dari Kaggle: *Credit Risk Dataset* (`32,581` pemohon pinjaman, target default `21.82%`).
   - Saya membangun **dua model Supervised Learning** sekaligus:
     - **Track A (Classification):** Memprediksi gagal bayar (`loan_status`) karena salah meloloskan nasabah macet (False Negative) jauh lebih mahal dibanding kehilangan bunga (False Positive).
     - **Track B (Multiple Regression):** Mengestimasi suku bunga wajar berbasis risiko (`loan_int_rate`) **tanpa memakai kolom `loan_status`** agar bebas *target leakage*.
2. **Data Understanding, Cleaning & Feature Engineering (1 menit)**
   - Audit menemukan `165` duplikat persis, `7` anomali domain mustahil (usia `123–144` tahun, lama kerja `123` tahun pada usia `21–22`), serta nilai kosong pada `person_emp_length` (`2.75%`) dan `loan_int_rate` (`9.56%`).
   - Duplikat dan anomali dihapus (`32,409` baris bersih), namun baris dengan *missing values* tetap disimpan agar diimputasi di dalam `Pipeline` setelah split (`SimpleImputer(strategy="median", add_indicator=True)`).
   - Menambahkan 4 fitur domain (`cred_hist_to_age_ratio`, `income_to_loan_ratio`, `emp_to_age_ratio`, `high_burden_flag`). Terbukti `high_burden_flag = 1` (rasio pinjaman ≥ 30% pendapatan) punya default rate **`65.56%` vs `15.25%`**.
3. **EDA Highlights (45 detik)**
   - `loan_grade` naik monotonik baik pada *default rate* (`9.96%` di Grade A → `98.44%` di Grade G) maupun rata-rata suku bunga (`7.33%` → `20.25%`).
   - Pemohon penyewa rumah (`RENT`) berisiko `31.61%` dibanding pemilik rumah (`OWN`: `7.49%`), dan riwayat default (`Y`) berisiko `37.86%` vs `18.44%`.
4. **Data Modeling & Evaluasi: Perbandingan Lengkap Algoritma (1 menit)**
   - **Track A (8 Model Klasifikasi):** `Decision Tree` tanpa regularisasi menghafal data (`Train Acc = 1.0000` vs `Val Acc = 0.8866`). Regularisasi pohon menaikkan `Val ROC-AUC` ke `0.9075`, mengungguli `KNN` (`0.8790`) dan `Logistic Regression` (`0.8719`). Juara: **`XGBoost` hasil tuning `Optuna`** dengan **Test ROC-AUC `0.9564`**, **PR-AUC `0.9168`**, dan **F1 `0.8434`** (`threshold = 0.60`).
   - **Track B (9 Model Multiple Regression):** `Multiple Linear Regression`, `Ridge`, dan `Lasso` stabil di `Val R² = 0.8761` (`Val RMSE = 1.1640`). `KNN Regressor` hanya `0.8489`, dan `Decision Tree` tanpa regularisasi *overfit* (`Train R² = 1.0000` → `Val R² = 0.8189`). Model *Tree Ensemble* **`Random Forest Regressor (Tuned)`** menurunkan error ke **Test R² `0.9010`** dan **Test MAE `0.7949% poin`**.
5. **Demo Live Aplikasi Streamlit (`app.py`) (1 menit)**
   - Tunjukkan **Tab 1** dengan klik cepat 3 preset (`Low Risk`, `Medium Risk`, `High Risk`) + geser *Policy Threshold* di sidebar (misal `0.38` untuk kebijakan bank konservatif vs `0.60` F1-optimal).
   - Tunjukkan **Tab 2** untuk memprediksi suku bunga wajar pemohon secara *real-time*.
   - Tunjukkan **Tab 3** (*Batch CSV* dengan tombol `20 Sampel Acak`) dan **Tab 4** (Dashboard EDA & perbandingan semua 17 model).

---

## 2. Antisipasi Pertanyaan Mentor (Q&A)

- **Kenapa tidak pakai Accuracy saja pada Classification?**
  - Karena data tidak seimbang (`78.18%` lunas vs `21.82%` default). Contoh nyata di tabel evaluasi: `DummyClassifier` dapat `Accuracy = 78.13%` tapi `F1 = 0.0000` (gagal mendeteksi satu pun nasabah macet). Karena itu kita pakai `ROC-AUC`, `PR-AUC`, `F1-Score`, dan `Recall`.
- **Bagaimana mencegah Data Leakage pada preprocessing dan Regression?**
  - Pertama, data dipecah menjadi **Train (64%) / Val (16%) / Test (20%)** sebelum *scaling*, *encoding*, atau *imputasi* (`SimpleImputer` di-*fit* hanya pada Train di dalam `Pipeline`). Kedua, pada Track B Regression kita **membuang `loan_status`** karena di dunia nyata status gagal bayar belum diketahui saat suku bunga ditentukan.
- **Kenapa `loan_grade` pakai OrdinalEncoder sementara `loan_intent` pakai OneHotEncoder?**
  - `loan_grade` memiliki urutan risiko eksplisit (`A < B < C < D < E < F < G`, terverifikasi *default rate* `9.96%` → `98.44%` dan bunga `7.33%` → `20.25%`). Sedangkan `loan_intent` (`MEDICAL`, `VENTURE`, dll.) dan `person_home_ownership` tidak punya peringkat numerik alami, sehingga memakai `OneHotEncoder`.
- **Mengapa Multiple Linear Regression mendapat `R² = 0.8761`, kalah ~3.5 poin dari Random Forest (`0.9110`)?**
  - Karena hubungan antara `loan_grade` dan suku bunga bersifat non-linear/*stepwise* dan bergantung pada interaksi dengan riwayat gagal bayar (`cb_person_default_on_file`), yang lebih mudah ditangkap oleh pemisahan pohon (`Random Forest` / `XGBoost`).
- **Apa kelemahan model yang ditemukan dari Slice Error Analysis?**
  - Pada segmen **`MORTGAGE`** (pemilik KPR), `Recall` turun ke `0.5015` (`ROC-AUC = 0.8997`) dibanding `OWN` (`0.9998`) dan `RENT` (`0.9684`). Ke depan perlu fitur tambahan terkait cicilan KPR atau *debt-service ratio* agar kasus macet pada nasabah KPR lebih terdeteksi.
