# Mini Project Supervised Learning: Credit Risk & Interest Rate Workbench

Proyek *end-to-end* **Supervised Learning** menggunakan dataset publik Kaggle [Credit Risk Dataset (`laotse/credit-risk-dataset`)](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) yang mencakup **6 tahapan wajib**:
1. **Data Understanding**
2. **Data Cleaning**
3. **Feature Engineering**
4. **Exploratory Data Analysis (EDA)**
5. **Data Modeling & Evaluations** (Classification + Multiple Regression)
6. **Interactive Streamlit Web App (`app.py`)**

---

## Struktur Folder

```text
ujian 1 mat/mini project 1/
├── .streamlit/
│   └── config.toml                # Tema UI Streamlit (Teal & Slate)
├── .gitignore                       # Mengecualikan __pycache__, .venv, dll. (tetap menyertakan .joblib & .csv)
├── .python-version                  # Pin Python 3.13 untuk Streamlit Community Cloud
├── requirements.txt                 # Dependensi deploy Streamlit Community Cloud
├── GOAL.md                          # Ringkasan tujuan & checklist tugas percakapan
├── credit_risk_dataset.csv          # Dataset mentah Kaggle (32,581 baris x 12 kolom)
├── features.py                      # Custom Feature Engineer, Preprocessor & Inference Helpers
├── mini-project-credit-risk.ipynb   # Notebook lengkap (68 sel, tereksekusi penuh)
├── model_credit_risk_xgb.joblib     # Bundle model Classification (Tuned XGBoost) + metadata
├── model_interest_rate_reg.joblib   # Bundle model Multiple Regression (Tuned RF) + metadata
├── app.py                           # Aplikasi interaktif Streamlit (5 Tab)
├── README.md                        # Dokumentasi proyek & panduan deploy
└── PANDUAN_DISKUSI_KELAS.md         # Panduan presentasi & tanya-jawab diskusi kelas
```

---

## Ringkasan Metodologi

- **Data Understanding & Cleaning:**
  - Dataset mentah terdiri dari **`32,581` baris × `12` kolom** dengan tingkat default **`21.82%`** (`7,108` default vs `25,473` lunas).
  - Dihapus **`165` baris duplikat persis** dan **`7` baris anomali mustahil** (`person_age > 100` dan `person_emp_length > 60`), menyisakan **`32,409` baris bersih**.
  - Nilai kosong pada `person_emp_length` (`887`) dan `loan_int_rate` (`3,094`) dipertahankan untuk diimputasi secara *leak-free* di dalam `Pipeline` pasca-split (dengan `SimpleImputer(strategy="median", add_indicator=True)`).
- **Feature Engineering (`features.py`):**
  - Ditambahkan 4 fitur domain keuangan: `cred_hist_to_age_ratio`, `income_to_loan_ratio`, `emp_to_age_ratio`, dan `high_burden_flag` (`loan_percent_income >= 0.30`, memisahkan *default rate* `65.56%` vs `15.25%`).
- **Anti-Leakage Split:**
  - Data dibagi secara stratifikasi **64% Train / 16% Validation / 20% Test** sebelum fitting imputer, scaler, maupun encoder.
  - Pada Track B (Multiple Regression `loan_int_rate`), kolom `loan_status` dikeluarkan dan baris tanpa label `loan_int_rate` disaring (`29,315` baris valid).

---

## Hasil Evaluasi Model

### Track A — Classification (`loan_status`, 8 Model dibandingkan)

| Model | CV ROC-AUC | Val ROC-AUC | Val PR-AUC | Val F1 | Train Acc | Val Acc |
|---|---:|---:|---:|---:|---:|---:|
| **XGBoost (Optuna Tuned)** | **0.9457** | **0.9495** | **0.9047** | **0.8238** | — | — |
| XGBoost (Default) | 0.9432 | 0.9480 | 0.9012 | 0.8192 | 0.9454 | 0.9317 |
| Random Forest (Default) | 0.9272 | 0.9320 | 0.8795 | 0.8082 | 1.0000 | 0.9279 |
| Decision Tree (Regularized) | 0.9082 | 0.9075 | 0.8439 | 0.7988 | 0.9302 | 0.9250 |
| KNN (`k=11`) | 0.8740 | 0.8790 | 0.7594 | 0.7035 | 0.9031 | 0.8914 |
| Logistic Regression (Balanced) | 0.8735 | 0.8728 | 0.7226 | 0.6452 | 0.8063 | 0.8149 |
| Logistic Regression (Default) | 0.8724 | 0.8719 | 0.7308 | 0.6435 | 0.8694 | 0.8677 |
| Decision Tree (Unregularized) | 0.8442 | 0.8404 | 0.6084 | 0.7452 | 1.0000 | 0.8866 |
| Dummy (Most Frequent) | 0.5000 | 0.5000 | 0.2187 | 0.0000 | 0.7813 | 0.7813 |

**Evaluasi Final Test Set Track A (`6,482` baris, 1,000 Bootstrap Resamples):**
- **Test ROC-AUC:** **`0.9564`** (95% CI: `0.9496 – 0.9625`)
- **Test PR-AUC:** **`0.9168`** (95% CI: `0.9056 – 0.9273`)
- **Test F1 (`@ threshold 0.60`):** **`0.8434`** (95% CI: `0.8287 – 0.8574`) | Precision `0.9537` | Recall `0.7560` | Accuracy `0.9386`
- **Kebijakan 3-Tier (Streamlit):** `< 0.38` = *Low Risk (Approve)*, `0.38 – 0.59` = *Medium Risk (Manual Review, Recall >= 80%)*, `>= 0.60` = *High Risk (Reject / Deep Audit)*.

---

### Track B — Multiple Regression (`loan_int_rate`, 9 Model dibandingkan)

| Model | CV R² | CV RMSE | Val R² | Val RMSE | Val MAE | Train R² |
|---|---:|---:|---:|---:|---:|---:|
| **Random Forest Regressor (Tuned, depth=8)** | — | — | **0.9110** | **0.9866** | **0.7797** | — |
| Random Forest Regressor (Default) | 0.9052 | 0.9927 | 0.9113 | 0.9847 | 0.7781 | 0.9456 |
| XGBoost Regressor (Default) | 0.9052 | 0.9929 | 0.9109 | 0.9869 | 0.7804 | 0.9191 |
| Decision Tree Reg (`Depth=7`) | 0.9031 | 1.0040 | 0.9089 | 0.9981 | 0.7890 | 0.9090 |
| Multiple Linear Regression | 0.8737 | 1.1461 | 0.8761 | 1.1640 | 0.9126 | 0.8742 |
| Ridge (`alpha=1.0`) | 0.8737 | 1.1461 | 0.8761 | 1.1640 | 0.9127 | 0.8742 |
| Lasso (`alpha=0.01`) | 0.8737 | 1.1460 | 0.8760 | 1.1647 | 0.9133 | 0.8740 |
| KNN Regressor (`k=11`) | 0.8450 | 1.2698 | 0.8489 | 1.2854 | 1.0132 | 0.8764 |
| Decision Tree Reg (Unregularized) | 0.8121 | 1.3980 | 0.8189 | 1.4073 | 1.0693 | 1.0000 |
| Dummy (Mean) | -0.0003 | 3.2258 | -0.0000 | 3.3071 | 2.7573 | 0.0000 |

**Evaluasi Final Test Set Track B (`5,863` baris, 1,000 Bootstrap Resamples):**
- **Test R²:** **`0.9010`** (95% CI: `0.8941 – 0.9075`)
- **Test RMSE:** **`1.0194` % poin** (95% CI: `0.9878 – 1.0546`)
- **Test MAE:** **`0.7949` % poin** (95% CI: `0.7786 – 0.8120`)

---

## Cara Menjalankan (Lokal)

Jalankan perintah berikut dari root workspace (`c:\Users\fmrif.RIFA_LOQ_ESSE15\OneDrive\Documents\data science`):

1. **Menjalankan Aplikasi Interaktif Streamlit:**
   ```powershell
   .\.venv\Scripts\streamlit.exe run "ujian 1 mat\mini project 1\app.py"
   ```
2. **Membuka / Menjalankan Ulang Notebook:**
   Buka `ujian 1 mat\mini project 1\mini-project-credit-risk.ipynb` dan pilih kernel Python `.venv`.

---

## Deploy ke Streamlit Community Cloud

1. Pastikan repositori GitHub (`https://github.com/Rifaditya/mini-project-1`) sudah disinkronkan (`git push -u origin main`).
2. Buka [share.streamlit.io](https://share.streamlit.io/) dan klik **Create app**.
3. Pilih:
   - **Repository:** `Rifaditya/mini-project-1`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Di bagian **Advanced settings**, pastikan **Python version** diset ke **`3.13`** (sesuai `.python-version` dan `requirements.txt`), lalu klik **Deploy!**.
