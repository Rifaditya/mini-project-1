"""Interactive Fintech Workbench for Credit Risk (Classification) & Interest Rate (Regression)."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from features import (  # noqa: E402
    CLF_RAW_COLS,
    REG_RAW_COLS,
    CreditFeatureEngineer,
    prediksi_risiko_kredit,
    prediksi_suku_bunga,
)

st.set_page_config(
    page_title="Smart Credit Analyst AI",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------------------
# Custom Modern Fintech CSS
# ------------------------------------------------------------------------------
st.markdown(
    """
<style>
.block-container {
    padding-top: 1.4rem;
    padding-bottom: 2.5rem;
}
.hero-banner {
    background: linear-gradient(135deg, #0F172A 0%, #134E4A 55%, #0F766E 100%);
    color: #FFFFFF;
    padding: 1.6rem 2rem;
    border-radius: 16px;
    margin-bottom: 1.2rem;
    box-shadow: 0 10px 25px rgba(15, 23, 42, 0.14);
}
.hero-title {
    font-size: 1.9rem;
    font-weight: 800;
    margin: 0 0 0.35rem 0;
    letter-spacing: -0.02em;
}
.hero-subtitle {
    font-size: 1.02rem;
    color: #E2E8F0;
    margin: 0 0 0.9rem 0;
    line-height: 1.5;
}
.pill-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
}
.hero-pill {
    background: rgba(255, 255, 255, 0.14);
    border: 1px solid rgba(255, 255, 255, 0.25);
    color: #F8FAFC;
    padding: 0.28rem 0.75rem;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 600;
}
.info-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 1.15rem 1.25rem;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    height: 100%;
}
.card-accent-teal {
    border-top: 4px solid #0F766E;
}
.card-accent-blue {
    border-top: 4px solid #0284C7;
}
.step-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-left: 4px solid #0F766E;
    border-radius: 12px;
    padding: 0.95rem 1.1rem;
    margin-bottom: 0.75rem;
    box-shadow: 0 2px 6px rgba(15, 23, 42, 0.03);
}
.step-title {
    font-weight: 700;
    color: #0F172A;
    font-size: 0.98rem;
    margin-bottom: 0.25rem;
}
.step-body {
    color: #475569;
    font-size: 0.9rem;
    line-height: 1.45;
    margin: 0;
}
.decision-box {
    border-radius: 14px;
    padding: 1.2rem 1.35rem;
    margin-bottom: 0.9rem;
    border: 1px solid transparent;
}
.decision-approve {
    background: #ECFDF5;
    border-color: #6EE7B7;
    color: #065F46;
}
.decision-review {
    background: #FFFBEB;
    border-color: #FCD34D;
    color: #92400E;
}
.decision-reject {
    background: #FFF1F2;
    border-color: #FDA4AF;
    color: #9F1239;
}
.decision-title {
    font-size: 1.25rem;
    font-weight: 800;
    margin: 0 0 0.3rem 0;
}
.decision-desc {
    font-size: 0.95rem;
    margin: 0;
    line-height: 1.45;
}
.metric-explain {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 0.9rem 1rem;
    height: 100%;
}
.metric-explain-val {
    font-size: 1.45rem;
    font-weight: 800;
    color: #0F766E;
    margin: 0.1rem 0 0.25rem 0;
}
.metric-explain-label {
    font-size: 0.85rem;
    font-weight: 700;
    color: #0F172A;
}
.metric-explain-sub {
    font-size: 0.84rem;
    color: #475569;
    line-height: 1.4;
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_bundles() -> tuple[dict, dict]:
    clf_bundle = joblib.load(APP_DIR / "model_credit_risk_xgb.joblib")
    reg_bundle = joblib.load(APP_DIR / "model_interest_rate_reg.joblib")
    return clf_bundle, reg_bundle


@st.cache_data
def load_clean_data() -> pd.DataFrame:
    df = pd.read_csv(APP_DIR / "credit_risk_dataset.csv").drop_duplicates()
    mask = (df["person_age"] <= 100) & (
        df["person_emp_length"].isna() | (df["person_emp_length"] <= 60)
    )
    return df[mask].reset_index(drop=True)


# ------------------------------------------------------------------------------
# Domain Labels & Presets
# ------------------------------------------------------------------------------
CLF_PRESETS = {
    "Low Risk": {
        "person_age": 34,
        "person_income": 95000,
        "person_home_ownership": "OWN",
        "person_emp_length": 9.0,
        "loan_intent": "VENTURE",
        "loan_grade": "A",
        "loan_amnt": 8000,
        "loan_int_rate": 7.5,
        "loan_percent_income": 0.08,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 10,
    },
    "Medium Risk": {
        "person_age": 23,
        "person_income": 83000,
        "person_home_ownership": "RENT",
        "person_emp_length": 7.0,
        "loan_intent": "EDUCATION",
        "loan_grade": "B",
        "loan_amnt": 25300,
        "loan_int_rate": 10.99,
        "loan_percent_income": 0.30,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 3,
    },
    "High Risk": {
        "person_age": 22,
        "person_income": 22000,
        "person_home_ownership": "RENT",
        "person_emp_length": 0.5,
        "loan_intent": "MEDICAL",
        "loan_grade": "E",
        "loan_amnt": 10000,
        "loan_int_rate": 17.8,
        "loan_percent_income": 0.45,
        "cb_person_default_on_file": "Y",
        "cb_person_cred_hist_length": 2,
    },
}

HOME_OPTS = ["RENT", "MORTGAGE", "OWN", "OTHER"]
HOME_LABELS = {
    "RENT": "Sewa / Kos (RENT)",
    "MORTGAGE": "KPR / Cicilan Rumah (MORTGAGE)",
    "OWN": "Milik Sendiri (OWN)",
    "OTHER": "Lainnya (OTHER)",
}

INTENT_OPTS = [
    "EDUCATION",
    "MEDICAL",
    "VENTURE",
    "PERSONAL",
    "DEBTCONSOLIDATION",
    "HOMEIMPROVEMENT",
]
INTENT_LABELS = {
    "EDUCATION": "Pendidikan (EDUCATION)",
    "MEDICAL": "Kesehatan / Medis (MEDICAL)",
    "VENTURE": "Modal Usaha (VENTURE)",
    "PERSONAL": "Kebutuhan Pribadi (PERSONAL)",
    "DEBTCONSOLIDATION": "Gabung / Tutup Utang (DEBTCONSOLIDATION)",
    "HOMEIMPROVEMENT": "Renovasi Rumah (HOMEIMPROVEMENT)",
}

GRADE_OPTS = ["A", "B", "C", "D", "E", "F", "G"]
GRADE_LABELS = {
    "A": "Grade A — Paling Aman",
    "B": "Grade B — Risiko Rendah",
    "C": "Grade C — Risiko Sedang",
    "D": "Grade D — Risiko Tinggi",
    "E": "Grade E — Risiko Sangat Tinggi",
    "F": "Grade F — Risiko Berat",
    "G": "Grade G — Risiko Ekstrem",
}

DEFAULT_OPTS = ["N", "Y"]
DEFAULT_LABELS = {
    "N": "Bersih / Tidak Pernah Gagal Bayar (N)",
    "Y": "Pernah Gagal Bayar Sebelumnya (Y)",
}


def _style_ax(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_facecolor("#FFFFFF")


def _init_session_state(clf_bundle: dict) -> None:
    if "policy_threshold" not in st.session_state:
        st.session_state["policy_threshold"] = round(float(clf_bundle["threshold"]), 2)

    defaults = CLF_PRESETS["Medium Risk"]
    key_map = {
        "clf_age": defaults["person_age"],
        "clf_income": defaults["person_income"],
        "clf_home": defaults["person_home_ownership"],
        "clf_emp": defaults["person_emp_length"],
        "clf_intent": defaults["loan_intent"],
        "clf_grade": defaults["loan_grade"],
        "clf_amnt": defaults["loan_amnt"],
        "clf_rate": defaults["loan_int_rate"],
        "clf_pct": defaults["loan_percent_income"],
        "clf_default": defaults["cb_person_default_on_file"],
        "clf_hist": defaults["cb_person_cred_hist_length"],
        "reg_age": defaults["person_age"],
        "reg_income": defaults["person_income"],
        "reg_home": defaults["person_home_ownership"],
        "reg_emp": defaults["person_emp_length"],
        "reg_intent": defaults["loan_intent"],
        "reg_grade": defaults["loan_grade"],
        "reg_amnt": defaults["loan_amnt"],
        "reg_pct": defaults["loan_percent_income"],
        "reg_default": defaults["cb_person_default_on_file"],
        "reg_hist": defaults["cb_person_cred_hist_length"],
    }
    for k, v in key_map.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _apply_clf_preset(name: str) -> None:
    p = CLF_PRESETS[name]
    st.session_state["clf_age"] = p["person_age"]
    st.session_state["clf_income"] = p["person_income"]
    st.session_state["clf_home"] = p["person_home_ownership"]
    st.session_state["clf_emp"] = p["person_emp_length"]
    st.session_state["clf_intent"] = p["loan_intent"]
    st.session_state["clf_grade"] = p["loan_grade"]
    st.session_state["clf_amnt"] = p["loan_amnt"]
    st.session_state["clf_rate"] = p["loan_int_rate"]
    st.session_state["clf_pct"] = p["loan_percent_income"]
    st.session_state["clf_default"] = p["cb_person_default_on_file"]
    st.session_state["clf_hist"] = p["cb_person_cred_hist_length"]


def _apply_reg_preset(name: str) -> None:
    p = CLF_PRESETS[name]
    st.session_state["reg_age"] = p["person_age"]
    st.session_state["reg_income"] = p["person_income"]
    st.session_state["reg_home"] = p["person_home_ownership"]
    st.session_state["reg_emp"] = p["person_emp_length"]
    st.session_state["reg_intent"] = p["loan_intent"]
    st.session_state["reg_grade"] = p["loan_grade"]
    st.session_state["reg_amnt"] = p["loan_amnt"]
    st.session_state["reg_pct"] = p["loan_percent_income"]
    st.session_state["reg_default"] = p["cb_person_default_on_file"]
    st.session_state["reg_hist"] = p["cb_person_cred_hist_length"]


def _set_threshold(val: float) -> None:
    st.session_state["policy_threshold"] = round(float(val), 2)


def _contextual_risk_notes(row: pd.Series) -> list[str]:
    notes = []
    if row["loan_percent_income"] >= 0.30:
        notes.append(
            f"⚠️ **Beban cicilan berat:** Pinjaman memakan **{row['loan_percent_income']:.0%}** pendapatan tahunan (di atas batas aman 30%)."
        )
    else:
        notes.append(
            f"✅ **Beban cicilan sehat:** Pinjaman hanya **{row['loan_percent_income']:.0%}** dari pendapatan tahunan (< 30%)."
        )

    if row["loan_grade"] in {"D", "E", "F", "G"}:
        notes.append(
            f"⚠️ **Grade berisiko ({row['loan_grade']}):** Secara historis Grade D–G memiliki tingkat gagal bayar 59%–98%."
        )
    elif row["loan_grade"] in {"A", "B"}:
        notes.append(
            f"✅ **Grade baik ({row['loan_grade']}):** Pemohon memiliki peringkat kredit awal yang kuat."
        )

    if row["cb_person_default_on_file"] == "Y":
        notes.append("⚠️ **Riwayat gagal bayar:** Pemohon pernah tercatat gagal bayar sebelumnya.")
    else:
        notes.append("✅ **Rekam jejak kredit bersih:** Tidak ada catatan gagal bayar sebelumnya.")

    if row["person_home_ownership"] == "RENT":
        notes.append("ℹ️ **Status hunian Sewa/Kos (RENT):** Risiko historisnya lebih tinggi dibanding pemilik rumah (`OWN` / `MORTGAGE`).")
    elif row["person_home_ownership"] == "OWN":
        notes.append("✅ **Milik Rumah Sendiri (OWN):** Menjadi faktor pelindung terkuat terhadap risiko gagal bayar.")

    if row["loan_int_rate"] >= 14.0:
        notes.append(f"⚠️ **Suku bunga tinggi ({row['loan_int_rate']:.2f}%):** Cicilan bunga besar memperberat kemampuan bayar.")
    return notes


def main() -> None:
    try:
        clf_bundle, reg_bundle = load_bundles()
        df_clean = load_clean_data()
    except FileNotFoundError as exc:
        st.error(
            f"File model atau dataset belum ditemukan ({exc}). Jalankan notebook `mini-project-credit-risk.ipynb` terlebih dahulu."
        )
        return

    _init_session_state(clf_bundle)

    tuned_test_row = clf_bundle["test_metrics"][-1]
    reg_winner_name = reg_bundle["winner_name"]
    ds_stats = clf_bundle["dataset_stats"]

    # --------------------------------------------------------------------------
    # Hero Banner
    # --------------------------------------------------------------------------
    st.markdown(
        f"""
<div class="hero-banner">
  <div class="hero-title">💳 Smart Credit Analyst AI</div>
  <div class="hero-subtitle">
    Simulasi <b>Asisten Analis Kredit Bank</b> berbasis Machine Learning: menilai apakah pemohon pinjaman
    berisiko <b>Gagal Bayar (Classification)</b> sekaligus menghitung <b>Suku Bunga Wajar (Multiple Regression)</b>.
  </div>
  <div class="pill-row">
    <span class="hero-pill">📊 {ds_stats['clean_rows']:,} Data Pemohon Kaggle</span>
    <span class="hero-pill">🤖 17 Algoritma Dibandingkan</span>
    <span class="hero-pill">🎯 Akurasi Ranking Default (ROC-AUC): {tuned_test_row['ROC_AUC']*100:.2f}%</span>
    <span class="hero-pill">📈 Akurasi Estimasi Bunga (R²): {reg_bundle['test_r2']*100:.2f}%</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------------------------
    # Beginner-Friendly Sidebar
    # --------------------------------------------------------------------------
    with st.sidebar:
        st.header("🎚️ Ketatnya Kebijakan Bank")
        st.caption(
            "Geser batas probabilitas gagal bayar untuk menentukan kapan AI menolak pengajuan:"
        )
        policy_thresh = st.slider(
            "Batas Toleransi Risiko (Threshold)",
            min_value=0.10,
            max_value=0.90,
            step=0.01,
            key="policy_threshold",
            help="Semakin KECIL nilainya, bank semakin KETAT (lebih cepat menolak pemohon berisiko).",
        )
        st.caption("Pilih Cepat Mode Kebijakan:")
        sb1, sb2, sb3 = st.columns(3)
        sb1.button(
            "Seimbang\n(0.60)",
            on_click=_set_threshold,
            args=(clf_bundle["threshold"],),
            width="stretch",
        )
        sb2.button(
            "Standar\n(0.50)",
            on_click=_set_threshold,
            args=(0.50,),
            width="stretch",
        )
        sb3.button(
            "Ketat\n(0.38)",
            on_click=_set_threshold,
            args=(clf_bundle["high_recall_threshold"],),
            width="stretch",
        )
        st.info(
            "**Cara Membaca Threshold:**\n\n"
            f"- Jika risiko pemohon **>= {policy_thresh:.0%}** → **Ditolak / Audit**.\n"
            f"- Jika risiko antara **{min(float(clf_bundle['high_recall_threshold']), policy_thresh):.0%} – {policy_thresh:.0%}** → **Review Manual**.\n"
            "- Di bawahnya → **Disetujui**."
        )

    tab_about, tab_sim, tab_rate, tab_batch, tab_eval = st.tabs(
        [
            "🏠 Tentang Project (Penjelasan Mudah)",
            "🔮 Simulasi Kredit (Coba AI)",
            "📈 Kalkulator Suku Bunga (Regression)",
            "📂 Cek Banyak Nasabah (Upload CSV)",
            "📊 EDA & Perbandingan 17 Model",
        ]
    )

    # ==========================================================================
    # TAB 1: Beginner-Friendly Project Guide
    # ==========================================================================
    with tab_about:
        st.subheader("Apa yang Sedang Kita Bangun?")
        st.write(
            "Bayangkan kamu bekerja sebagai **Analis Kredit di sebuah Bank**. Setiap hari ada orang datang "
            "mengajukan pinjaman dengan data usia, gaji, status rumah, tujuan pinjaman, dan riwayat kreditnya. "
            "Kamu harus menjawab **2 pertanyaan utama** — di proyek ini, kedua pertanyaan itu dijawab oleh **AI (Supervised Learning)** "
            "yang belajar dari **`32,409` riwayat pinjaman nyata** dari Kaggle:"
        )

        q1, q2 = st.columns(2)
        with q1:
            st.markdown(
                f"""
<div class="info-card card-accent-teal">
  <h4 style="margin-top:0;color:#0F766E;">1️⃣ Track A — Classification (Prediksi Aman vs Macet)</h4>
  <p style="color:#334155;">
    <b>Pertanyaan Bisnis:</b> <i>"Apakah pemohon ini bakal lunas (0) atau gagal bayar (1)?"</i><br><br>
    • <b>Kenapa penting?</b> Salah meloloskan nasabah macet merugikan uang pokok bank.<br>
    • <b>Model Juara:</b> <code>XGBoost Classifier</code> (dituning dengan Optuna dari 8 model).<br>
    • <b>Hasil Test Set:</b> ROC-AUC <b>{tuned_test_row['ROC_AUC']:.4f}</b>, F1-Score <b>{tuned_test_row['F1']:.4f}</b>.
  </p>
</div>
""",
                unsafe_allow_html=True,
            )
        with q2:
            st.markdown(
                f"""
<div class="info-card card-accent-blue">
  <h4 style="margin-top:0;color:#0369A1;">2️⃣ Track B — Multiple Regression (Estimasi Suku Bunga)</h4>
  <p style="color:#334155;">
    <b>Pertanyaan Bisnis:</b> <i>"Berapa % suku bunga yang wajar sesuai profil risiko pemohon?"</i><br><br>
    • <b>Prinsip Adil:</b> Dilatih <b>tanpa</b> melihat masa depan (<code>loan_status</code> dibuang agar bebas <i>leakage</i>).<br>
    • <b>Model Juara:</b> <code>{reg_winner_name}</code> (dipilih dari 9 model termasuk Linear Regression & KNN).<br>
    • <b>Hasil Test Set:</b> R² <b>{reg_bundle['test_r2']:.4f}</b>, meleset rata-rata hanya <b>±{reg_bundle['test_mae']:.2f}%</b> poin bunga.
  </p>
</div>
""",
                unsafe_allow_html=True,
            )

        st.markdown("### 6 Tahapan Bootcamp yang Dikerjakan di Project Ini")
        s_col1, s_col2 = st.columns(2)
        steps_left = [
            (
                "1. Data Understanding",
                f"Memeriksa {ds_stats['raw_rows']:,} baris data mentah Kaggle: ditemukan 21.82% kasus gagal bayar, 165 duplikat, nilai kosong di lama kerja (2.75%) & suku bunga (9.56%), serta anomali usia 144 tahun.",
            ),
            (
                "2. Data Cleaning",
                f"Menghapus 165 duplikat persis dan 7 baris mustahil (usia >100 atau masa kerja >60 tahun) sehingga tersisa {ds_stats['clean_rows']:,} baris bersih tanpa membocorkan nilai median imputasi.",
            ),
            (
                "3. Feature Engineering",
                "Membuat 4 rasio keuangan baru: Rasio Gaji/Pinjaman, Lama Kerja/Usia, Riwayat Kredit/Usia, dan Flag Beban Berat (cicilan >=30% gaji, yang risiko macetnya melonjak 4x lipat!).",
            ),
        ]
        steps_right = [
            (
                "4. Exploratory Data Analysis (EDA)",
                "Menemukan pola kunci: Grade A hanya 9.96% macet vs Grade G 98.44%; penyewa kos (RENT) 31.61% macet vs pemilik rumah (OWN) 7.49%.",
            ),
            (
                "5. Data Modeling & Evaluations",
                "Membandingkan 8 model Classification & 9 model Multiple Regression (Linear, Ridge, Lasso, KNN, Decision Tree, Random Forest, XGBoost) dengan 5-Fold CV & Bootstrap 95% CI.",
            ),
            (
                "6. Aplikasi Interaktif Streamlit",
                "Membungkus kedua model menjadi aplikasi web interaktif siap pakai: bisa simulasi 1 nasabah, hitung suku bunga wajar, cek batch CSV, dan lihat dashboard evaluasi.",
            ),
        ]
        with s_col1:
            for title, body in steps_left:
                st.markdown(
                    f'<div class="step-card"><div class="step-title">{title}</div><p class="step-body">{body}</p></div>',
                    unsafe_allow_html=True,
                )
        with s_col2:
            for title, body in steps_right:
                st.markdown(
                    f'<div class="step-card"><div class="step-title">{title}</div><p class="step-body">{body}</p></div>',
                    unsafe_allow_html=True,
                )

        st.markdown("### Arti Angka Performa (Bahasa Sederhana)")
        m1, m2, m3, m4 = st.columns(4)
        metric_cards = [
            (
                m1,
                "ROC-AUC (Track A)",
                f"{tuned_test_row['ROC_AUC']*100:.2f}%",
                "Kemampuan AI mengurutkan siapa yang paling berisiko dibanding yang aman (100% = sempurna).",
            ),
            (
                m2,
                "F1-Score (Track A)",
                f"{tuned_test_row['F1']*100:.2f}%",
                "Keseimbangan antara akurasi penolakan (Precision 95.4%) dan daya tangkap pemacet (Recall 75.6%).",
            ),
            (
                m3,
                "R² Score (Track B)",
                f"{reg_bundle['test_r2']*100:.2f}%",
                "Seberapa besar variasi suku bunga bank yang berhasil dijelaskan oleh model regresi.",
            ),
            (
                m4,
                "MAE Bunga (Track B)",
                f"±{reg_bundle['test_mae']:.2f}%",
                "Rata-rata selisih tebakan suku bunga AI dibanding suku bunga asli di data test.",
            ),
        ]
        for col, label, val, desc in metric_cards:
            col.markdown(
                f"""
<div class="metric-explain">
  <div class="metric-explain-label">{label}</div>
  <div class="metric-explain-val">{val}</div>
  <div class="metric-explain-sub">{desc}</div>
</div>
""",
                unsafe_allow_html=True,
            )

        with st.expander("📖 Kamus Istilah ML (Biar Gampang Jawab Pertanyaan Mentor)"):
            st.markdown(
                """
- **Classification vs Multiple Regression:** *Classification* memprediksi kategori (Lunas vs Gagal Bayar), sedangkan *Multiple Regression* memprediksi angka kontinu (% suku bunga) dari banyak variabel input sekaligus.
- **Data Leakage:** Kesalahan membocorkan informasi masa depan atau statistik data uji ke proses latihan model (misal melakukan scaling sebelum split, atau memakai `loan_status` untuk memprediksi suku bunga). Di proyek ini **semua sudah dicegah**.
- **Overfitting:** Model "menghafal" data latih (contohnya `Decision Tree` tanpa batasan kedalaman mendapat skor latih `1.0000`, tetapi turun saat menghadapi data baru).
- **Threshold Tuning:** Menggeser batas keputusan probabilitas (bukan selalu `50%`) supaya sesuai kerugian bisnis bank antara meloloskan pemacet (*False Negative*) vs menolak nasabah baik (*False Positive*).
"""
            )

    # ==========================================================================
    # TAB 2: Interactive Credit Risk Predictor
    # ==========================================================================
    with tab_sim:
        st.subheader("Simulasi Penilaian Kredit Pemohon")
        st.caption(
            "Klik salah satu contoh profil di bawah atau ubah angkanya sendiri — AI langsung menghitung risiko gagal bayar dan rekomendasi keputusannya."
        )

        p1, p2, p3 = st.columns(3)
        p1.button(
            "🟢 Budi — Profil Aman (Low Risk)",
            on_click=_apply_clf_preset,
            args=("Low Risk",),
            width="stretch",
        )
        p2.button(
            "🟡 Sinta — Perlu Review (Medium Risk)",
            on_click=_apply_clf_preset,
            args=("Medium Risk",),
            width="stretch",
        )
        p3.button(
            "🔴 Andi — Risiko Tinggi (High Risk)",
            on_click=_apply_clf_preset,
            args=("High Risk",),
            width="stretch",
        )

        col_left, col_right = st.columns([1.2, 1.0])
        with col_left:
            c1, c2 = st.columns(2)
            with c1:
                age = st.number_input(
                    "Usia Pemohon (tahun)",
                    18,
                    95,
                    key="clf_age",
                    help="Usia debitur saat mengajukan pinjaman.",
                )
                income = st.number_input(
                    "Pendapatan Tahunan ($)",
                    4000,
                    2000000,
                    step=1000,
                    key="clf_income",
                    help="Total penghasilan pemohon dalam satu tahun.",
                )
                home = st.selectbox(
                    "Status Tempat Tinggal",
                    HOME_OPTS,
                    format_func=lambda x: HOME_LABELS[x],
                    key="clf_home",
                )
                emp_len = st.number_input(
                    "Lama Bekerja (tahun)",
                    0.0,
                    55.0,
                    step=0.5,
                    key="clf_emp",
                )
                intent = st.selectbox(
                    "Tujuan Pinjaman",
                    INTENT_OPTS,
                    format_func=lambda x: INTENT_LABELS[x],
                    key="clf_intent",
                )
                default_hist = st.selectbox(
                    "Riwayat Gagal Bayar Sebelumnya",
                    DEFAULT_OPTS,
                    format_func=lambda x: DEFAULT_LABELS[x],
                    key="clf_default",
                )
            with c2:
                grade = st.selectbox(
                    "Peringkat Kredit (Loan Grade)",
                    GRADE_OPTS,
                    format_func=lambda x: GRADE_LABELS[x],
                    key="clf_grade",
                )
                loan_amnt = st.number_input(
                    "Nominal Pinjaman ($)",
                    500,
                    100000,
                    step=500,
                    key="clf_amnt",
                )
                auto_pct = st.checkbox(
                    "Hitung otomatis porsi pinjaman terhadap gaji",
                    value=False,
                    key="clf_auto_pct",
                )
                if auto_pct:
                    pct_income = round(min(1.0, float(loan_amnt) / max(float(income), 1.0)), 2)
                    st.caption(f"Rasio otomatis terhitung: **{pct_income:.0%} dari gaji**")
                else:
                    pct_income = st.slider(
                        "Porsi Pinjaman / Gaji Tahunan",
                        0.01,
                        0.95,
                        step=0.01,
                        key="clf_pct",
                        help="Contoh: 0.30 artinya pinjaman sebesar 30% dari gaji tahunan.",
                    )

                use_reg_rate = st.checkbox(
                    "Gunakan estimasi suku bunga dari Model Regresi",
                    value=False,
                    key="clf_use_reg_rate",
                )
                reg_probe_df = pd.DataFrame(
                    [
                        {
                            "person_age": age,
                            "person_income": income,
                            "person_home_ownership": home,
                            "person_emp_length": emp_len,
                            "loan_intent": intent,
                            "loan_grade": grade,
                            "loan_amnt": loan_amnt,
                            "loan_percent_income": pct_income,
                            "cb_person_default_on_file": default_hist,
                            "cb_person_cred_hist_length": st.session_state["clf_hist"],
                        }
                    ]
                )
                fair_rate_est = float(
                    prediksi_suku_bunga(reg_probe_df[REG_RAW_COLS], reg_bundle)[
                        "pred_loan_int_rate"
                    ].iloc[0]
                )
                if use_reg_rate:
                    int_rate = fair_rate_est
                    st.caption(f"Suku bunga otomatis dari Track B: **{int_rate:.2f}%**")
                else:
                    int_rate = st.number_input(
                        "Suku Bunga Pinjaman (%)",
                        4.0,
                        26.0,
                        step=0.1,
                        key="clf_rate",
                    )
                cred_hist = st.number_input(
                    "Panjang Riwayat Kredit (tahun)",
                    1,
                    40,
                    key="clf_hist",
                )

        input_clf_df = pd.DataFrame(
            [
                {
                    "person_age": age,
                    "person_income": income,
                    "person_home_ownership": home,
                    "person_emp_length": emp_len,
                    "loan_intent": intent,
                    "loan_grade": grade,
                    "loan_amnt": loan_amnt,
                    "loan_int_rate": int_rate,
                    "loan_percent_income": pct_income,
                    "cb_person_default_on_file": default_hist,
                    "cb_person_cred_hist_length": cred_hist,
                }
            ]
        )

        scored_clf = prediksi_risiko_kredit(
            input_clf_df[CLF_RAW_COLS], clf_bundle, custom_threshold=policy_thresh
        )
        res_row = scored_clf.iloc[0]
        fe_row = CreditFeatureEngineer().transform(input_clf_df).iloc[0]
        prob_def = float(res_row["prob_default"])
        tier = res_row["risk_tier"]

        with col_right:
            st.markdown("### Keputusan & Analisis AI")
            if tier == "Low Risk":
                box_cls = "decision-approve"
                headline = "✅ DISETUJUI (Low Risk)"
            elif tier == "Medium Risk":
                box_cls = "decision-review"
                headline = "⚠️ PERLU REVIEW MANUAL (Medium Risk)"
            else:
                box_cls = "decision-reject"
                headline = "❌ DITOLAK / AUDIT MENDALAM (High Risk)"

            approx_out_of_100 = int(round(prob_def * 100))
            st.markdown(
                f"""
<div class="decision-box {box_cls}">
  <div class="decision-title">{headline}</div>
  <p class="decision-desc">
    Dari 100 pemohon dengan profil serupa, AI memperkirakan sekitar <b>{approx_out_of_100} orang</b> berisiko gagal bayar.
  </p>
</div>
""",
                unsafe_allow_html=True,
            )

            m1, m2, m3 = st.columns(3)
            m1.metric("Probabilitas Default", f"{prob_def * 100:.2f}%")
            m2.metric("Batas Tolak Bank", f"{policy_thresh * 100:.0f}%")
            m3.metric("Estimasi Bunga Wajar", f"{fair_rate_est:.2f}%")
            st.progress(min(max(prob_def, 0.0), 1.0))

            st.markdown("#### Kenapa AI Memutuskan Ini?")
            for note in _contextual_risk_notes(res_row):
                st.write(note)

            st.markdown("#### Rasio Keuangan Otomatis (Feature Engineering)")
            f1, f2 = st.columns(2)
            f1.metric("Pendapatan / Pinjaman", f"{fe_row['income_to_loan_ratio']:.2f}x")
            f2.metric("Riwayat Kredit / Usia", f"{fe_row['cred_hist_to_age_ratio']:.2f}")
            f3, f4 = st.columns(2)
            f3.metric("Lama Kerja / Usia", f"{fe_row['emp_to_age_ratio']:.2f}")
            f4.metric(
                "Beban Cicilan >= 30%",
                "Ya (Risiko Tinggi)" if fe_row["high_burden_flag"] == 1 else "Tidak (Aman)",
            )

    # ==========================================================================
    # TAB 3: Multiple Regression Interest Rate Calculator
    # ==========================================================================
    with tab_rate:
        st.subheader("Kalkulator Estimasi Suku Bunga Pinjaman (Track B — Multiple Regression)")
        st.write(
            f"Menebak suku bunga tahunan yang adil menggunakan **{reg_winner_name}** berdasarkan profil pemohon "
            "(tanpa memakai informasi gagal bayar di masa depan)."
        )

        rp1, rp2, rp3 = st.columns(3)
        rp1.button(
            "🟢 Budi — Grade A (Aman)",
            on_click=_apply_reg_preset,
            args=("Low Risk",),
            key="btn_reg_low",
            width="stretch",
        )
        rp2.button(
            "🟡 Sinta — Grade B (Menengah)",
            on_click=_apply_reg_preset,
            args=("Medium Risk",),
            key="btn_reg_med",
            width="stretch",
        )
        rp3.button(
            "🔴 Andi — Grade E (Berisiko)",
            on_click=_apply_reg_preset,
            args=("High Risk",),
            key="btn_reg_high",
            width="stretch",
        )

        r_left, r_right = st.columns([1.15, 1.0])
        with r_left:
            rc1, rc2 = st.columns(2)
            with rc1:
                r_age = st.number_input("Usia Pemohon (tahun) ", 18, 95, key="reg_age")
                r_income = st.number_input(
                    "Pendapatan Tahunan ($) ", 4000, 2000000, step=1000, key="reg_income"
                )
                r_home = st.selectbox(
                    "Status Tempat Tinggal ",
                    HOME_OPTS,
                    format_func=lambda x: HOME_LABELS[x],
                    key="reg_home",
                )
                r_emp = st.number_input("Lama Bekerja (tahun) ", 0.0, 55.0, step=0.5, key="reg_emp")
                r_intent = st.selectbox(
                    "Tujuan Pinjaman ",
                    INTENT_OPTS,
                    format_func=lambda x: INTENT_LABELS[x],
                    key="reg_intent",
                )
            with rc2:
                r_grade = st.selectbox(
                    "Peringkat Kredit (Grade) ",
                    GRADE_OPTS,
                    format_func=lambda x: GRADE_LABELS[x],
                    key="reg_grade",
                )
                r_amnt = st.number_input("Nominal Pinjaman ($) ", 500, 100000, step=500, key="reg_amnt")
                r_pct = st.slider(
                    "Porsi Pinjaman / Gaji ", 0.01, 0.95, step=0.01, key="reg_pct"
                )
                r_default = st.selectbox(
                    "Riwayat Gagal Bayar ",
                    DEFAULT_OPTS,
                    format_func=lambda x: DEFAULT_LABELS[x],
                    key="reg_default",
                )
                r_hist = st.number_input("Panjang Riwayat Kredit (tahun) ", 1, 40, key="reg_hist")

        reg_in_df = pd.DataFrame(
            [
                {
                    "person_age": r_age,
                    "person_income": r_income,
                    "person_home_ownership": r_home,
                    "person_emp_length": r_emp,
                    "loan_intent": r_intent,
                    "loan_grade": r_grade,
                    "loan_amnt": r_amnt,
                    "loan_percent_income": r_pct,
                    "cb_person_default_on_file": r_default,
                    "cb_person_cred_hist_length": r_hist,
                }
            ]
        )
        reg_pred_row = prediksi_suku_bunga(reg_in_df[REG_RAW_COLS], reg_bundle).iloc[0]
        hist_grade_avg = reg_bundle["grade_mean_rates"].get(r_grade, np.nan)

        with r_right:
            st.markdown("### Hasil Estimasi Suku Bunga")
            rm1, rm2, rm3 = st.columns(3)
            rm1.metric("Prediksi Bunga AI", f"{reg_pred_row['pred_loan_int_rate']:.2f}%")
            rm2.metric(
                "Rentang Wajar (±MAE)",
                f"{reg_pred_row['rate_low']:.2f}% – {reg_pred_row['rate_high']:.2f}%",
            )
            rm3.metric(f"Rata-rata Grade {r_grade}", f"{hist_grade_avg:.2f}%")

            reg_comp_df = pd.DataFrame(reg_bundle["model_comparison"])
            sub_r = reg_comp_df[reg_comp_df["Model"] != "Dummy (Mean)"].sort_values("Val_R2")
            fig_r, ax_r = plt.subplots(figsize=(6.8, 4.0))
            fig_r.patch.set_facecolor("#FFFFFF")
            colors_r = [
                "#0F766E" if ("Random Forest" in m or "XGBoost" in m) else "#64748B"
                for m in sub_r["Model"]
            ]
            ax_r.barh(sub_r["Model"], sub_r["Val_R2"], color=colors_r)
            ax_r.set_xlim(0.75, 0.96)
            for idx_b, val_b in enumerate(sub_r["Val_R2"]):
                ax_r.text(val_b + 0.003, idx_b, f"{val_b:.4f}", va="center", fontsize=8.5, fontweight="bold")
            _style_ax(ax_r)
            ax_r.set_xlabel("Validation R² (Semakin Tinggi = Tebakan Bunga Makin Akurat)")
            ax_r.set_title("Perbandingan 8 Algoritma Regresi (Track B)", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig_r)
            plt.close(fig_r)

    # ==========================================================================
    # TAB 4: Batch CSV Scoring
    # ==========================================================================
    with tab_batch:
        st.subheader("Cek Sekaligus Banyak Nasabah (Upload CSV)")
        st.caption(
            "Upload file CSV atau klik tombol sampel di kanan untuk langsung menilai 20 pemohon sekaligus dengan Track A + Track B."
        )

        b1, b2 = st.columns([2, 1])
        uploaded = b1.file_uploader(
            "Upload file CSV pemohon kredit",
            type=["csv"],
        )
        if b2.button("🎲 Coba 20 Nasabah Acak", width="stretch"):
            st.session_state["batch_df"] = df_clean.sample(20, random_state=42).drop(
                columns=["loan_status"], errors="ignore"
            )

        if uploaded is not None:
            st.session_state["batch_df"] = pd.read_csv(uploaded)

        if "batch_df" in st.session_state:
            batch_raw = st.session_state["batch_df"].copy()
            try:
                scored_b = prediksi_risiko_kredit(
                    batch_raw, clf_bundle, custom_threshold=policy_thresh
                )
                if all(c in batch_raw.columns for c in REG_RAW_COLS):
                    reg_b = prediksi_suku_bunga(batch_raw[REG_RAW_COLS], reg_bundle)
                    scored_b["pred_fair_int_rate"] = reg_b["pred_loan_int_rate"]

                bm1, bm2, bm3, bm4 = st.columns(4)
                bm1.metric("Total Pemohon", f"{len(scored_b):,}")
                bm2.metric("Disetujui (Low Risk)", f"{(scored_b['risk_tier'] == 'Low Risk').sum():,}")
                bm3.metric("Review Manual (Medium)", f"{(scored_b['risk_tier'] == 'Medium Risk').sum():,}")
                bm4.metric("Ditolak / Audit (High)", f"{(scored_b['risk_tier'] == 'High Risk').sum():,}")

                fig_b, ax_b = plt.subplots(figsize=(8.5, 3.3))
                fig_b.patch.set_facecolor("#FFFFFF")
                sns.histplot(scored_b["prob_default"], bins=15, color="#0284C7", ax=ax_b)
                ax_b.axvline(
                    policy_thresh,
                    color="#E11D48",
                    linestyle="--",
                    lw=2,
                    label=f"Batas Tolak ({policy_thresh:.0%})",
                )
                _style_ax(ax_b)
                ax_b.set_title("Sebaran Risiko Gagal Bayar pada Data Batch", fontweight="bold")
                ax_b.set_xlabel("Probabilitas Default (0 = Sangat Aman, 1 = Pasti Gagal Bayar)")
                ax_b.legend()
                plt.tight_layout()
                st.pyplot(fig_b)
                plt.close(fig_b)

                st.dataframe(scored_b, width="stretch")
                csv_bytes = scored_b.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Unduh Hasil Penilaian (.csv)",
                    data=csv_bytes,
                    file_name="hasil_prediksi_kredit.csv",
                    mime="text/csv",
                )
            except ValueError as err:
                st.error(f"Validasi kolom gagal: {err}")

    # ==========================================================================
    # TAB 5: EDA & 17-Model Comparison Dashboard
    # ==========================================================================
    with tab_eval:
        st.subheader("Eksplorasi Data Interaktif & Leaderboard 17 Model")

        f_col1, f_col2, f_col3 = st.columns(3)
        sel_grades = f_col1.multiselect("Filter Loan Grade", GRADE_OPTS, default=GRADE_OPTS)
        sel_homes = f_col2.multiselect("Filter Status Hunian", HOME_OPTS, default=HOME_OPTS)
        sel_intents = f_col3.multiselect("Filter Tujuan Pinjaman", INTENT_OPTS, default=INTENT_OPTS)

        eda_df = df_clean[
            df_clean["loan_grade"].isin(sel_grades)
            & df_clean["person_home_ownership"].isin(sel_homes)
            & df_clean["loan_intent"].isin(sel_intents)
        ]

        e1, e2, e3 = st.columns(3)
        e1.metric("Jumlah Data Terfilter", f"{len(eda_df):,}")
        e2.metric(
            "Tingkat Gagal Bayar (Default Rate)",
            f"{eda_df['loan_status'].mean() * 100:.2f}%" if len(eda_df) else "0.00%",
        )
        e3.metric(
            "Rata-rata Suku Bunga",
            f"{eda_df['loan_int_rate'].mean():.2f}%" if len(eda_df) else "0.00%",
        )

        if len(eda_df) > 0:
            fig_eda, axes_eda = plt.subplots(1, 2, figsize=(14, 4.3))
            fig_eda.patch.set_facecolor("#FFFFFF")
            g_sum = (
                eda_df.groupby("loan_grade", observed=False)
                .agg(
                    default_pct=("loan_status", lambda s: s.mean() * 100),
                    mean_rate=("loan_int_rate", "mean"),
                )
                .reindex([g for g in GRADE_OPTS if g in eda_df["loan_grade"].unique()])
                .reset_index()
            )
            axes_eda[0].bar(g_sum["loan_grade"], g_sum["default_pct"], color="#E11D48")
            _style_ax(axes_eda[0])
            axes_eda[0].set_title("Default Rate (%) per Loan Grade", fontweight="bold")
            axes_eda[0].set_ylabel("Default Rate (%)")

            axes_eda[1].bar(g_sum["loan_grade"], g_sum["mean_rate"], color="#0F766E")
            _style_ax(axes_eda[1])
            axes_eda[1].set_title("Rata-rata Suku Bunga (%) per Loan Grade", fontweight="bold")
            axes_eda[1].set_ylabel("Mean Interest Rate (%)")
            plt.tight_layout()
            st.pyplot(fig_eda)
            plt.close(fig_eda)

        st.divider()
        ev1, ev2 = st.tabs(
            ["Track A — Classification (8 Model)", "Track B — Multiple Regression (9 Model)"]
        )

        with ev1:
            st.info(
                "**Intisari Track A:** `XGBoost` mengungguli seluruh 8 model (`ROC-AUC = 0.9564` di Test Set), "
                "sedangkan `Decision Tree` tanpa regularisasi mengalami *overfitting* (`Train Acc = 1.0000` vs `Val Acc = 0.8866`)."
            )
            st.dataframe(pd.DataFrame(clf_bundle["model_comparison"]).round(4), width="stretch")
            st.markdown("#### Bootstrap 95% Confidence Interval — Test Set (1,000 Resamples)")
            st.dataframe(pd.DataFrame(clf_bundle["bootstrap_ci"]).round(4), width="stretch")

            fi_clf = pd.DataFrame(clf_bundle["feature_importances"]).sort_values("Importance", ascending=True)
            fig_fic, ax_fic = plt.subplots(figsize=(9, 4.8))
            fig_fic.patch.set_facecolor("#FFFFFF")
            ax_fic.barh(fi_clf["Feature"], fi_clf["Importance"], color="#0F766E")
            _style_ax(ax_fic)
            ax_fic.set_title("Top 15 Faktor Penentu Gagal Bayar — Tuned XGBoost", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig_fic)
            plt.close(fig_fic)

        with ev2:
            st.info(
                f"**Intisari Track B:** `Multiple Linear Regression`, `Ridge`, dan `Lasso` mencapai `Val R² = 0.8761`, "
                f"sementara **{reg_winner_name}** menurunkan error ke `Test R² = {reg_bundle['test_r2']:.4f}` (`Test MAE = {reg_bundle['test_mae']:.4f}%`)."
            )
            st.dataframe(pd.DataFrame(reg_bundle["model_comparison"]).round(4), width="stretch")
            st.markdown(f"#### Bootstrap 95% Confidence Interval — {reg_winner_name}")
            st.dataframe(pd.DataFrame(reg_bundle["bootstrap_ci"]).round(4), width="stretch")

            fi_reg = pd.DataFrame(reg_bundle["feature_importances"]).sort_values("Importance", ascending=True)
            fig_fir, ax_fir = plt.subplots(figsize=(9, 4.2))
            fig_fir.patch.set_facecolor("#FFFFFF")
            ax_fir.barh(fi_reg["Feature"], fi_reg["Importance"], color="#0284C7")
            _style_ax(ax_fir)
            ax_fir.set_title(f"Top 10 Faktor Penentu Suku Bunga — {reg_winner_name}", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig_fir)
            plt.close(fig_fir)


if __name__ == "__main__":
    main()
