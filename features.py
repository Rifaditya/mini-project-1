"""Shared feature engineering, preprocessing pipelines, and safe inference helpers.

Used by both `mini-project-credit-risk.ipynb` and `app.py` so joblib artifacts
load cleanly across environments without `__main__` pickling issues.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

CLF_RAW_COLS = [
    "person_age",
    "person_income",
    "person_home_ownership",
    "person_emp_length",
    "loan_intent",
    "loan_grade",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_default_on_file",
    "cb_person_cred_hist_length",
]

REG_RAW_COLS = [
    "person_age",
    "person_income",
    "person_home_ownership",
    "person_emp_length",
    "loan_intent",
    "loan_grade",
    "loan_amnt",
    "loan_percent_income",
    "cb_person_default_on_file",
    "cb_person_cred_hist_length",
]

ENGINEERED_COLS = [
    "cred_hist_to_age_ratio",
    "income_to_loan_ratio",
    "emp_to_age_ratio",
    "high_burden_flag",
]

CLF_NUM_COLS = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
] + ENGINEERED_COLS

REG_NUM_COLS = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_percent_income",
    "cb_person_cred_hist_length",
] + ENGINEERED_COLS

ORD_COLS = ["loan_grade", "cb_person_default_on_file"]
NOM_COLS = ["person_home_ownership", "loan_intent"]


class CreditFeatureEngineer(BaseEstimator, TransformerMixin):
    """Stateless row-wise credit risk feature engineering transformer."""

    CREATED_FEATURES = ENGINEERED_COLS

    def fit(self, X: pd.DataFrame, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df_out = X.copy()
        df_out["cred_hist_to_age_ratio"] = (
            df_out["cb_person_cred_hist_length"] / df_out["person_age"]
        )
        df_out["income_to_loan_ratio"] = df_out["person_income"] / (
            df_out["loan_amnt"] + 1.0
        )
        # Preserve NaN when person_emp_length is missing so SimpleImputer handles it
        df_out["emp_to_age_ratio"] = (
            df_out["person_emp_length"] / df_out["person_age"]
        )
        df_out["high_burden_flag"] = (df_out["loan_percent_income"] >= 0.30).astype(
            int
        )
        return df_out

    def get_feature_names_out(self, input_features=None):
        base = (
            list(input_features)
            if input_features is not None
            else list(getattr(self, "feature_names_in_", []))
        )
        return np.asarray(base + self.CREATED_FEATURES, dtype=object)


def _build_preprocessor(num_cols: list[str]) -> ColumnTransformer:
    """Construct leak-free ColumnTransformer for numeric, ordinal, and nominal columns."""
    num_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    ord_pipe = OrdinalEncoder(
        categories=[["A", "B", "C", "D", "E", "F", "G"], ["N", "Y"]],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    nom_pipe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("ord", ord_pipe, ORD_COLS),
            ("nom", nom_pipe, NOM_COLS),
        ],
        remainder="drop",
    )


def build_clf_preprocessor() -> ColumnTransformer:
    """Preprocessor for Track A (Classification predicting loan_status)."""
    return _build_preprocessor(CLF_NUM_COLS)


def build_reg_preprocessor() -> ColumnTransformer:
    """Preprocessor for Track B (Multiple Regression predicting loan_int_rate)."""
    return _build_preprocessor(REG_NUM_COLS)


def prediksi_risiko_kredit(
    df_baru: pd.DataFrame,
    bundle: dict,
    custom_threshold: float | None = None,
) -> pd.DataFrame:
    """Validate input columns and score default risk on new loan applications."""
    expected_cols = bundle["expected_columns"]
    missing_cols = [c for c in expected_cols if c not in df_baru.columns]
    if missing_cols:
        raise ValueError(f"Kolom wajib tidak lengkap: {missing_cols}")

    X_input = df_baru[expected_cols].copy()
    probs = bundle["model"].predict_proba(X_input)[:, 1]
    opt_th = (
        float(custom_threshold)
        if custom_threshold is not None
        else float(bundle["threshold"])
    )
    low_th = min(float(bundle.get("high_recall_threshold", 0.30)), opt_th)

    risk_tiers = []
    recommendations = []
    for p in probs:
        if p >= opt_th:
            risk_tiers.append("High Risk")
            recommendations.append("Reject / Deep Audit")
        elif p >= low_th:
            risk_tiers.append("Medium Risk")
            recommendations.append("Manual Review")
        else:
            risk_tiers.append("Low Risk")
            recommendations.append("Approve")

    out = df_baru.copy()
    out["prob_default"] = np.round(probs, 4)
    out["pred_default"] = (probs >= opt_th).astype(int)
    out["risk_tier"] = risk_tiers
    out["recommendation"] = recommendations
    return out


def prediksi_suku_bunga(df_baru: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Validate input columns and estimate fair loan interest rate (%) via regression."""
    expected_cols = bundle["expected_columns"]
    missing_cols = [c for c in expected_cols if c not in df_baru.columns]
    if missing_cols:
        raise ValueError(f"Kolom wajib untuk regresi tidak lengkap: {missing_cols}")

    X_input = df_baru[expected_cols].copy()
    preds = bundle["model"].predict(X_input)
    mae_margin = float(bundle.get("test_mae", 0.80))

    out = df_baru.copy()
    out["pred_loan_int_rate"] = np.round(preds, 2)
    out["rate_low"] = np.round(np.maximum(0.0, preds - mae_margin), 2)
    out["rate_high"] = np.round(preds + mae_margin, 2)
    return out
