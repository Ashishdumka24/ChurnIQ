"""
============================================================
ChurnIQ
Customer Churn Prediction & Business Intelligence System
============================================================
Advanced Dashboard Analytics Module

Purpose:
- Dataset quality analysis
- Model performance analysis
- Executive business insights
- Contract-wise churn analysis
- Tenure-wise churn analysis
- Monthly charge analysis
- Payment method analysis
- Internet service analysis
- Demographic analysis
- Churn driver analysis
- Revenue/value-at-risk analysis
- Retention opportunity analysis

Compatible with:
- app.py
- prediction_pipeline.py
- segmentation.py
- recommendations.py

The module is intentionally defensive so that different
customer datasets can be analyzed without crashing the app.
============================================================
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"
MODEL_DIR = BASE_DIR / "Models"


# ============================================================
# GENERAL HELPERS
# ============================================================

def _normalize_name(value):
    """
    Normalize a column name for flexible matching.
    """
    return (
        str(value)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace("/", "")
    )


def _find_column(df, names):
    """
    Find a dataframe column using flexible column-name matching.
    """

    if df is None or df.empty:
        return None

    normalized_columns = {
        _normalize_name(column): column
        for column in df.columns
    }

    for name in names:

        normalized_name = _normalize_name(name)

        if normalized_name in normalized_columns:
            return normalized_columns[normalized_name]

    return None


def _find_columns(df, candidates):
    """
    Return all matching columns from candidate names.
    """

    found = []

    if df is None or df.empty:
        return found

    for candidate in candidates:

        column = _find_column(
            df,
            [candidate],
        )

        if column and column not in found:
            found.append(column)

    return found


def _numeric_series(df, column):
    """
    Safely convert a dataframe column to numeric.
    """

    if (
        df is None
        or column is None
        or column not in df.columns
    ):
        return pd.Series(
            dtype=float
        )

    return pd.to_numeric(
        df[column],
        errors="coerce",
    )


def _clean_text_series(df, column):
    """
    Safely clean text values.
    """

    if (
        df is None
        or column is None
        or column not in df.columns
    ):
        return pd.Series(
            dtype="object"
        )

    return (
        df[column]
        .astype(str)
        .str.strip()
    )


def _churn_mask(df):
    """
    Identify churned customers.

    Supports:
    Yes / No
    True / False
    1 / 0
    Churn / Retained
    Churned / Active
    """

    if df is None or df.empty:
        return pd.Series(
            dtype=bool
        )

    prediction_col = _find_column(
        df,
        [
            "Prediction",
            "Churn Prediction",
            "Predicted Churn",
            "Churn",
        ],
    )

    if prediction_col is None:
        return pd.Series(
            False,
            index=df.index,
        )

    values = (
        df[prediction_col]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return values.isin(
        {
            "yes",
            "true",
            "1",
            "churn",
            "churned",
            "high risk",
            "high-risk",
            "critical",
        }
    )


def _churn_status_series(df):
    """
    Create a standardized churn status column.
    """

    if df is None or df.empty:
        return pd.Series(
            dtype="object"
        )

    prediction_col = _find_column(
        df,
        [
            "Prediction",
            "Churn Prediction",
            "Predicted Churn",
            "Churn",
        ],
    )

    if prediction_col is None:
        return pd.Series(
            "Unknown",
            index=df.index,
        )

    values = (
        df[prediction_col]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return values.map(
        lambda value:
            "Churn"
            if value in {
                "yes",
                "true",
                "1",
                "churn",
                "churned",
            }
            else (
                "Retained"
                if value in {
                    "no",
                    "false",
                    "0",
                    "retain",
                    "retained",
                    "active",
                }
                else "Unknown"
            )
    )


def _probability_series(df):
    """
    Return churn probability as a percentage.

    Handles both:
        0.75 -> 75%
        75   -> 75%
    """

    probability_col = _find_column(
        df,
        [
            "Churn Probability",
            "Churn Probability (%)",
            "Probability",
            "Predicted Probability",
            "Churn Risk Probability",
        ],
    )

    if probability_col is None:
        return pd.Series(
            np.nan,
            index=df.index,
            dtype=float,
        )

    probability = pd.to_numeric(
        df[probability_col],
        errors="coerce",
    )

    valid = probability.dropna()

    if not valid.empty:

        # Probabilities between 0 and 1
        # are converted into percentages.
        if (
            float(valid.min()) >= 0
            and float(valid.max()) <= 1
        ):
            probability = probability * 100

    return probability.clip(
        lower=0,
        upper=100,
    )


def _percentage(value):
    """
    Safely convert a value to percentage.
    """

    try:

        value = float(value)

        if 0 <= value <= 1:
            return value * 100

        return value

    except Exception:

        return 0.0


# ============================================================
# DATA QUALITY
# ============================================================

def get_data_quality(df):
    """
    Calculate dataset quality indicators.

    Returns:
        dict
    """

    if df is None or df.empty:

        return {
            "quality_score": 0.0,
            "missing_cells": 0,
            "duplicate_rows": 0,
            "rows": 0,
            "columns": 0,
            "missing_percentage": 0.0,
            "duplicate_percentage": 0.0,
            "complete_rows": 0,
            "complete_row_percentage": 0.0,
            "numeric_columns": 0,
            "categorical_columns": 0,
        }

    rows = len(df)
    columns = len(df.columns)

    total_cells = rows * columns

    missing_cells = int(
        df.isna().sum().sum()
    )

    duplicate_rows = int(
        df.duplicated().sum()
    )

    missing_percentage = (
        missing_cells
        / total_cells
        * 100
        if total_cells
        else 0.0
    )

    duplicate_percentage = (
        duplicate_rows
        / rows
        * 100
        if rows
        else 0.0
    )

    complete_rows = int(
        df.notna()
        .all(axis=1)
        .sum()
    )

    complete_row_percentage = (
        complete_rows
        / rows
        * 100
        if rows
        else 0.0
    )

    numeric_columns = len(
        df.select_dtypes(
            include=np.number
        ).columns
    )

    categorical_columns = (
        columns - numeric_columns
    )

    # Balanced quality calculation.
    missing_score = max(
        0.0,
        100.0 - missing_percentage * 2,
    )

    duplicate_score = max(
        0.0,
        100.0 - duplicate_percentage * 2,
    )

    completeness_score = (
        complete_row_percentage
    )

    quality_score = (
        missing_score * 0.40
        + duplicate_score * 0.20
        + completeness_score * 0.40
    )

    quality_score = float(
        np.clip(
            quality_score,
            0,
            100,
        )
    )

    return {
        "quality_score": round(
            quality_score,
            2,
        ),
        "missing_cells": missing_cells,
        "duplicate_rows": duplicate_rows,
        "rows": rows,
        "columns": columns,
        "missing_percentage": round(
            missing_percentage,
            2,
        ),
        "duplicate_percentage": round(
            duplicate_percentage,
            2,
        ),
        "complete_rows": complete_rows,
        "complete_row_percentage": round(
            complete_row_percentage,
            2,
        ),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
    }


# ============================================================
# MODEL PERFORMANCE
# ============================================================

def _load_model_results():
    """
    Load model_results.csv from Models/.
    """

    candidates = [
        MODEL_DIR / "model_results.csv",
        DATA_DIR / "model_results.csv",
    ]

    for path in candidates:

        if not path.exists():
            continue

        try:

            df = pd.read_csv(path)

            if not df.empty:
                return df

        except Exception:
            continue

    return pd.DataFrame()


def get_model_performance():
    """
    Return model comparison results.

    Normalizes common metric columns when possible.
    """

    results = _load_model_results()

    if results.empty:
        return pd.DataFrame()

    results = results.copy()

    # Normalize model-name column.
    model_col = _find_column(
        results,
        [
            "Model",
            "Algorithm",
            "Classifier",
            "Model Name",
        ],
    )

    if model_col and model_col != "Model":

        results = results.rename(
            columns={
                model_col: "Model"
            }
        )

    metric_aliases = {
        "Accuracy": [
            "Accuracy",
            "accuracy",
        ],
        "Precision": [
            "Precision",
            "precision",
        ],
        "Recall": [
            "Recall",
            "recall",
        ],
        "F1 Score": [
            "F1 Score",
            "F1",
            "f1",
            "F1-Score",
        ],
        "ROC-AUC": [
            "ROC-AUC",
            "ROC AUC",
            "ROC_AUC",
            "AUC",
            "roc_auc",
        ],
    }

    for standard_name, aliases in metric_aliases.items():

        column = _find_column(
            results,
            aliases,
        )

        if (
            column
            and column != standard_name
        ):

            results = results.rename(
                columns={
                    column: standard_name
                }
            )

    # Convert known metric columns to numeric.
    for column in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC",
    ]:

        if column in results.columns:

            results[column] = pd.to_numeric(
                results[column],
                errors="coerce",
            )

    return results


def get_best_model_metrics():
    """
    Return the best-performing model.

    Primary ranking:
        ROC-AUC
    Fallback:
        F1 Score
        Accuracy
    """

    performance = get_model_performance()

    if performance.empty:
        return {}

    ranking_columns = [
        "ROC-AUC",
        "F1 Score",
        "Accuracy",
        "Recall",
        "Precision",
    ]

    ranking_column = None

    for column in ranking_columns:

        if column in performance.columns:

            values = pd.to_numeric(
                performance[column],
                errors="coerce",
            )

            if values.notna().any():

                ranking_column = column
                performance[column] = values
                break

    if ranking_column is None:
        return performance.iloc[0].to_dict()

    performance = performance.sort_values(
        by=ranking_column,
        ascending=False,
        na_position="last",
    )

    return performance.iloc[0].to_dict()


# ============================================================
# EXECUTIVE INSIGHTS
# ============================================================

def generate_executive_insights(df):
    """
    Generate automatic business insights from prediction data.
    """

    insights = []

    if df is None or df.empty:
        return insights

    working = df.copy()

    status = _churn_status_series(
        working
    )

    probability = _probability_series(
        working
    )

    total = len(working)

    churn_count = int(
        (status == "Churn").sum()
    )

    churn_rate = (
        churn_count
        / total
        * 100
        if total
        else 0.0
    )

    valid_probability = probability.dropna()

    if not valid_probability.empty:

        avg_probability = float(
            valid_probability.mean()
        )

        high_probability = int(
            (probability >= 70).sum()
        )

        insights.append(
            f"Average churn probability is "
            f"{avg_probability:.1f}% across "
            f"{total:,} customers."
        )

        if high_probability:

            insights.append(
                f"{high_probability:,} customers "
                f"have a churn probability of 70% "
                f"or higher and should receive "
                f"priority retention attention."
            )

    if churn_count:

        insights.append(
            f"{churn_rate:.1f}% of customers "
            f"are currently predicted to churn."
        )

    risk_col = _find_column(
        working,
        [
            "Risk Level",
            "Customer Risk Segment",
            "Risk Category",
        ],
    )

    if risk_col:

        risks = (
            working[risk_col]
            .astype(str)
            .str.lower()
        )

        high_risk = int(
            risks.str.contains(
                "high|critical",
                regex=True,
                na=False,
            ).sum()
        )

        if high_risk:

            insights.append(
                f"{high_risk:,} customers are "
                f"classified as high or critical risk."
            )

    contract_col = _find_column(
        working,
        [
            "Contract",
            "Contract Type",
        ],
    )

    if contract_col:

        contract_status = pd.DataFrame(
            {
                "Contract": working[
                    contract_col
                ].astype(str),
                "Churn": (
                    status == "Churn"
                ).astype(int),
            }
        )

        contract_rates = (
            contract_status
            .groupby("Contract")["Churn"]
            .mean()
            * 100
        )

        if not contract_rates.empty:

            highest_contract = (
                contract_rates.idxmax()
            )

            highest_rate = float(
                contract_rates.max()
            )

            insights.append(
                f"{highest_contract} customers "
                f"show the highest contract-level "
                f"churn rate at {highest_rate:.1f}%."
            )

    tenure_col = _find_column(
        working,
        [
            "tenure",
            "Tenure",
            "Customer Tenure",
            "Tenure Months",
        ],
    )

    if tenure_col:

        tenure = _numeric_series(
            working,
            tenure_col,
        )

        if tenure.notna().any():

            avg_tenure = float(
                tenure.mean()
            )

            insights.append(
                f"Average customer tenure is "
                f"{avg_tenure:.1f} months."
            )

    monthly_col = _find_column(
        working,
        [
            "MonthlyCharges",
            "Monthly Charges",
            "Monthly Charge",
        ],
    )

    if monthly_col:

        charges = _numeric_series(
            working,
            monthly_col,
        )

        if charges.notna().any():

            avg_charge = float(
                charges.mean()
            )

            insights.append(
                f"Average monthly customer "
                f"charge is {avg_charge:,.2f}."
            )

    return insights[:10]


# ============================================================
# CONTRACT ANALYSIS
# ============================================================

def get_contract_analysis(df):
    """
    Analyze churn by contract type.

    Returns columns:
        Contract
        Customers
        Churned Customers
        Churn Rate
        Average Probability
        Revenue at Risk
    """

    if df is None or df.empty:
        return pd.DataFrame()

    contract_col = _find_column(
        df,
        [
            "Contract",
            "Contract Type",
        ],
    )

    if contract_col is None:
        return pd.DataFrame()

    working = df.copy()

    working["_Churn"] = (
        _churn_status_series(working)
        == "Churn"
    ).astype(int)

    working["_Probability"] = (
        _probability_series(working)
    )

    charge_col = _find_column(
        working,
        [
            "MonthlyCharges",
            "Monthly Charges",
            "Monthly Charge",
        ],
    )

    if charge_col:
        working["_MonthlyCharge"] = (
            _numeric_series(
                working,
                charge_col,
            )
        )
    else:
        working["_MonthlyCharge"] = 0.0

    grouped = (
        working
        .groupby(contract_col, dropna=False)
        .agg(
            Customers=("_Churn", "size"),
            Churned_Customers=(
                "_Churn",
                "sum",
            ),
            Average_Probability=(
                "_Probability",
                "mean",
            ),
            Revenue_at_Risk=(
                "_MonthlyCharge",
                lambda x: float(
                    x[
                        working.loc[
                            x.index,
                            "_Churn"
                        ]
                        == 1
                    ].sum()
                ),
            ),
        )
        .reset_index()
    )

    grouped = grouped.rename(
        columns={
            contract_col: "Contract",
            "Churned_Customers":
                "Churned Customers",
            "Average_Probability":
                "Average Probability",
            "Revenue_at_Risk":
                "Revenue at Risk",
        }
    )

    grouped["Churn Rate"] = (
        grouped["Churned Customers"]
        / grouped["Customers"]
        * 100
    )

    grouped["Average Probability"] = (
        grouped["Average Probability"]
        .fillna(0)
        .round(2)
    )

    grouped["Churn Rate"] = (
        grouped["Churn Rate"]
        .round(2)
    )

    grouped["Revenue at Risk"] = (
        grouped["Revenue at Risk"]
        .round(2)
    )

    return grouped.sort_values(
        "Churn Rate",
        ascending=False,
    ).reset_index(drop=True)


# ============================================================
# TENURE ANALYSIS
# ============================================================

def get_tenure_analysis(df):
    """
    Analyze churn across tenure groups.

    Returns:
        Tenure Group
        Customers
        Churned Customers
        Churn Rate
        Average Probability
    """

    if df is None or df.empty:
        return pd.DataFrame()

    tenure_col = _find_column(
        df,
        [
            "tenure",
            "Tenure",
            "Customer Tenure",
            "Tenure Months",
        ],
    )

    if tenure_col is None:
        return pd.DataFrame()

    working = df.copy()

    working["_Tenure"] = _numeric_series(
        working,
        tenure_col,
    )

    working["_Churn"] = (
        _churn_status_series(working)
        == "Churn"
    ).astype(int)

    working["_Probability"] = (
        _probability_series(working)
    )

    working = working[
        working["_Tenure"].notna()
    ].copy()

    if working.empty:
        return pd.DataFrame()

    bins = [
        -np.inf,
        6,
        12,
        24,
        36,
        60,
        np.inf,
    ]

    labels = [
        "0–6 months",
        "7–12 months",
        "13–24 months",
        "25–36 months",
        "37–60 months",
        "60+ months",
    ]

    working["Tenure Group"] = pd.cut(
        working["_Tenure"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    analysis = (
        working
        .groupby(
            "Tenure Group",
            observed=False,
        )
        .agg(
            Customers=("_Churn", "size"),
            Churned_Customers=(
                "_Churn",
                "sum",
            ),
            Average_Probability=(
                "_Probability",
                "mean",
            ),
        )
        .reset_index()
    )

    analysis = analysis.rename(
        columns={
            "Churned_Customers":
                "Churned Customers",
            "Average_Probability":
                "Average Probability",
        }
    )

    analysis["Churn Rate"] = (
        analysis["Churned Customers"]
        / analysis["Customers"]
        * 100
    )

    analysis["Churn Rate"] = (
        analysis["Churn Rate"]
        .fillna(0)
        .round(2)
    )

    analysis["Average Probability"] = (
        analysis["Average Probability"]
        .fillna(0)
        .round(2)
    )

    return analysis


# ============================================================
# MONTHLY CHARGE ANALYSIS
# ============================================================

def get_monthly_charge_analysis(df):
    """
    Analyze churn according to monthly charges.

    Creates business-friendly charge bands.
    """

    if df is None or df.empty:
        return pd.DataFrame()

    charge_col = _find_column(
        df,
        [
            "MonthlyCharges",
            "Monthly Charges",
            "Monthly Charge",
        ],
    )

    if charge_col is None:
        return pd.DataFrame()

    working = df.copy()

    working["_MonthlyCharge"] = (
        _numeric_series(
            working,
            charge_col,
        )
    )

    working["_Churn"] = (
        _churn_status_series(working)
        == "Churn"
    ).astype(int)

    working["_Probability"] = (
        _probability_series(working)
    )

    working = working[
        working["_MonthlyCharge"].notna()
    ].copy()

    if working.empty:
        return pd.DataFrame()

    # Use dataset-aware quantile bands when possible.
    try:

        working["Charge Band"] = pd.qcut(
            working["_MonthlyCharge"],
            q=4,
            labels=[
                "Low",
                "Medium",
                "High",
                "Very High",
            ],
            duplicates="drop",
        )

    except Exception:

        working["Charge Band"] = "All Customers"

    analysis = (
        working
        .groupby(
            "Charge Band",
            observed=False,
        )
        .agg(
            Customers=("_Churn", "size"),
            Churned_Customers=(
                "_Churn",
                "sum",
            ),
            Average_Monthly_Charge=(
                "_MonthlyCharge",
                "mean",
            ),
            Average_Probability=(
                "_Probability",
                "mean",
            ),
        )
        .reset_index()
    )

    analysis = analysis.rename(
        columns={
            "Churned_Customers":
                "Churned Customers",
            "Average_Monthly_Charge":
                "Average Monthly Charge",
            "Average_Probability":
                "Average Probability",
        }
    )

    analysis["Churn Rate"] = (
        analysis["Churned Customers"]
        / analysis["Customers"]
        * 100
    )

    analysis["Churn Rate"] = (
        analysis["Churn Rate"]
        .fillna(0)
        .round(2)
    )

    analysis["Average Monthly Charge"] = (
        analysis["Average Monthly Charge"]
        .round(2)
    )

    analysis["Average Probability"] = (
        analysis["Average Probability"]
        .fillna(0)
        .round(2)
    )

    return analysis


# ============================================================
# GENERIC CATEGORY ANALYSIS
# ============================================================

def _category_analysis(
    df,
    column_candidates,
    category_name,
):
    """
    Generic churn analysis for categorical variables.
    """

    if df is None or df.empty:
        return pd.DataFrame()

    column = _find_column(
        df,
        column_candidates,
    )

    if column is None:
        return pd.DataFrame()

    working = df.copy()

    working["_Category"] = (
        working[column]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )

    working["_Churn"] = (
        _churn_status_series(working)
        == "Churn"
    ).astype(int)

    working["_Probability"] = (
        _probability_series(working)
    )

    analysis = (
        working
        .groupby(
            "_Category",
            dropna=False,
        )
        .agg(
            Customers=("_Churn", "size"),
            Churned_Customers=(
                "_Churn",
                "sum",
            ),
            Average_Probability=(
                "_Probability",
                "mean",
            ),
        )
        .reset_index()
    )

    analysis = analysis.rename(
        columns={
            "_Category": category_name,
            "Churned_Customers":
                "Churned Customers",
            "Average_Probability":
                "Average Probability",
        }
    )

    analysis["Churn Rate"] = (
        analysis["Churned Customers"]
        / analysis["Customers"]
        * 100
    )

    analysis["Churn Rate"] = (
        analysis["Churn Rate"]
        .fillna(0)
        .round(2)
    )

    analysis["Average Probability"] = (
        analysis["Average Probability"]
        .fillna(0)
        .round(2)
    )

    return analysis.sort_values(
        "Churn Rate",
        ascending=False,
    ).reset_index(drop=True)


# ============================================================
# PAYMENT METHOD ANALYSIS
# ============================================================

def get_payment_method_analysis(df):
    """
    Analyze churn by payment method.
    """

    return _category_analysis(
        df,
        [
            "PaymentMethod",
            "Payment Method",
        ],
        "Payment Method",
    )


# ============================================================
# INTERNET SERVICE ANALYSIS
# ============================================================

def get_internet_service_analysis(df):
    """
    Analyze churn by internet service.
    """

    return _category_analysis(
        df,
        [
            "InternetService",
            "Internet Service",
        ],
        "Internet Service",
    )


# ============================================================
# GENDER ANALYSIS
# ============================================================

def get_gender_analysis(df):
    """
    Analyze churn by gender.
    """

    return _category_analysis(
        df,
        [
            "gender",
            "Gender",
        ],
        "Gender",
    )


# ============================================================
# SENIOR CITIZEN ANALYSIS
# ============================================================

def get_senior_citizen_analysis(df):
    """
    Analyze churn by senior-citizen status.
    """

    result = _category_analysis(
        df,
        [
            "SeniorCitizen",
            "Senior Citizen",
            "Senior",
        ],
        "Senior Citizen",
    )

    if not result.empty:

        result["Senior Citizen"] = (
            result["Senior Citizen"]
            .replace(
                {
                    "1": "Yes",
                    "0": "No",
                    "1.0": "Yes",
                    "0.0": "No",
                }
            )
        )

    return result


# ============================================================
# SERVICE ANALYSIS
# ============================================================

def get_service_analysis(df):
    """
    Analyze common service columns.

    Returns a combined long-format dataframe.
    """

    if df is None or df.empty:
        return pd.DataFrame()

    service_candidates = {
        "Phone Service": [
            "PhoneService",
            "Phone Service",
        ],
        "Multiple Lines": [
            "MultipleLines",
            "Multiple Lines",
        ],
        "Online Security": [
            "OnlineSecurity",
            "Online Security",
        ],
        "Online Backup": [
            "OnlineBackup",
            "Online Backup",
        ],
        "Device Protection": [
            "DeviceProtection",
            "Device Protection",
        ],
        "Tech Support": [
            "TechSupport",
            "Tech Support",
        ],
        "Streaming TV": [
            "StreamingTV",
            "Streaming TV",
        ],
        "Streaming Movies": [
            "StreamingMovies",
            "Streaming Movies",
        ],
    }

    frames = []

    for service_name, candidates in (
        service_candidates.items()
    ):

        analysis = _category_analysis(
            df,
            candidates,
            "Category",
        )

        if analysis.empty:
            continue

        analysis.insert(
            0,
            "Service",
            service_name,
        )

        frames.append(
            analysis
        )

    if not frames:
        return pd.DataFrame()

    return pd.concat(
        frames,
        ignore_index=True,
    )


# ============================================================
# CHURN DRIVER ANALYSIS
# ============================================================

def get_churn_drivers(df):
    """
    Identify categorical variables with the strongest
    difference in churn rate between their categories.

    Returns:
        Driver
        Category
        Customers
        Churn Rate
        Average Probability
        Impact Score
    """

    if df is None or df.empty:
        return pd.DataFrame()

    candidate_columns = [
        (
            "Contract",
            [
                "Contract",
                "Contract Type",
            ],
        ),
        (
            "Payment Method",
            [
                "PaymentMethod",
                "Payment Method",
            ],
        ),
        (
            "Internet Service",
            [
                "InternetService",
                "Internet Service",
            ],
        ),
        (
            "Gender",
            [
                "gender",
                "Gender",
            ],
        ),
        (
            "Senior Citizen",
            [
                "SeniorCitizen",
                "Senior Citizen",
            ],
        ),
        (
            "Partner",
            [
                "Partner",
            ],
        ),
        (
            "Dependents",
            [
                "Dependents",
            ],
        ),
        (
            "Paperless Billing",
            [
                "PaperlessBilling",
                "Paperless Billing",
            ],
        ),
    ]

    frames = []

    for driver_name, candidates in candidate_columns:

        analysis = _category_analysis(
            df,
            candidates,
            "Category",
        )

        if analysis.empty:
            continue

        baseline = (
            (
                _churn_status_series(df)
                == "Churn"
            )
            .mean()
            * 100
        )

        analysis["Driver"] = driver_name

        analysis["Impact Score"] = (
            analysis["Churn Rate"]
            - baseline
        ).abs()

        frames.append(
            analysis[
                [
                    "Driver",
                    "Category",
                    "Customers",
                    "Churn Rate",
                    "Average Probability",
                    "Impact Score",
                ]
            ]
        )

    if not frames:
        return pd.DataFrame()

    drivers = pd.concat(
        frames,
        ignore_index=True,
    )

    return drivers.sort_values(
        "Impact Score",
        ascending=False,
    ).reset_index(drop=True)


# ============================================================
# REVENUE / VALUE AT RISK
# ============================================================

def get_revenue_at_risk(df):
    """
    Calculate estimated monthly revenue at risk.

    Uses:
        MonthlyCharges

    and predicted churn status.
    """

    if df is None or df.empty:
        return {
            "monthly_revenue_at_risk": 0.0,
            "annual_revenue_at_risk": 0.0,
            "churned_customers": 0,
            "average_charge_at_risk": 0.0,
        }

    charge_col = _find_column(
        df,
        [
            "MonthlyCharges",
            "Monthly Charges",
            "Monthly Charge",
        ],
    )

    if charge_col is None:

        return {
            "monthly_revenue_at_risk": 0.0,
            "annual_revenue_at_risk": 0.0,
            "churned_customers": int(
                _churn_mask(df).sum()
            ),
            "average_charge_at_risk": 0.0,
        }

    charges = _numeric_series(
        df,
        charge_col,
    )

    churn = _churn_mask(df)

    at_risk_charges = charges[
        churn
    ].dropna()

    monthly_risk = float(
        at_risk_charges.sum()
    )

    annual_risk = (
        monthly_risk * 12
    )

    average_charge = (
        float(
            at_risk_charges.mean()
        )
        if not at_risk_charges.empty
        else 0.0
    )

    return {
        "monthly_revenue_at_risk": round(
            monthly_risk,
            2,
        ),
        "annual_revenue_at_risk": round(
            annual_risk,
            2,
        ),
        "churned_customers": int(
            churn.sum()
        ),
        "average_charge_at_risk": round(
            average_charge,
            2,
        ),
    }


# ============================================================
# RETENTION OPPORTUNITY
# ============================================================

def get_retention_opportunity(df):
    """
    Estimate retention opportunity.

    High opportunity:
        Probability >= 70%

    Medium opportunity:
        40% <= Probability < 70%

    Low opportunity:
        Probability < 40%
    """

    if df is None or df.empty:

        return {
            "high_opportunity": 0,
            "medium_opportunity": 0,
            "low_opportunity": 0,
            "total_at_risk": 0,
            "opportunity_rate": 0.0,
        }

    probability = _probability_series(
        df
    )

    high = int(
        (probability >= 70).sum()
    )

    medium = int(
        (
            (probability >= 40)
            & (probability < 70)
        ).sum()
    )

    low = int(
        (
            probability < 40
        ).sum()
    )

    total_at_risk = (
        high + medium
    )

    opportunity_rate = (
        total_at_risk
        / len(df)
        * 100
        if len(df)
        else 0.0
    )

    return {
        "high_opportunity": high,
        "medium_opportunity": medium,
        "low_opportunity": low,
        "total_at_risk": total_at_risk,
        "opportunity_rate": round(
            opportunity_rate,
            2,
        ),
    }


# ============================================================
# CUSTOMER VALUE ANALYSIS
# ============================================================

def get_customer_value_analysis(df):
    """
    Analyze high-value customers and churn exposure.

    Uses:
        MonthlyCharges
        TotalCharges
        Churn status
        Churn probability
    """

    if df is None or df.empty:
        return pd.DataFrame()

    working = df.copy()

    monthly_col = _find_column(
        working,
        [
            "MonthlyCharges",
            "Monthly Charges",
            "Monthly Charge",
        ],
    )

    total_col = _find_column(
        working,
        [
            "TotalCharges",
            "Total Charges",
            "Total Charge",
        ],
    )

    if monthly_col:

        working["Monthly Value"] = (
            _numeric_series(
                working,
                monthly_col,
            )
        )

    else:

        working["Monthly Value"] = np.nan

    if total_col:

        working["Total Value"] = (
            _numeric_series(
                working,
                total_col,
            )
        )

    else:

        working["Total Value"] = np.nan

    working["Churn Status"] = (
        _churn_status_series(working)
    )

    working["Churn Probability"] = (
        _probability_series(working)
    )

    if working["Monthly Value"].notna().any():

        threshold = float(
            working[
                "Monthly Value"
            ].quantile(0.75)
        )

        working["Customer Value Tier"] = np.where(
            working["Monthly Value"] >= threshold,
            "High Value",
            "Standard Value",
        )

    else:

        working["Customer Value Tier"] = (
            "Unknown"
        )

    return working


# ============================================================
# SUMMARY KPI FUNCTION
# ============================================================

def get_advanced_kpis(df):
    """
    Return the main business KPIs for the dashboard.
    """

    if df is None or df.empty:

        return {
            "total_customers": 0,
            "predicted_churn": 0,
            "churn_rate": 0.0,
            "average_probability": 0.0,
            "high_risk_customers": 0,
            "average_tenure": 0.0,
            "average_monthly_charge": 0.0,
            "monthly_revenue_at_risk": 0.0,
            "annual_revenue_at_risk": 0.0,
            "high_value_at_risk": 0,
        }

    total = len(df)

    status = _churn_status_series(
        df
    )

    churn = int(
        (status == "Churn").sum()
    )

    probability = _probability_series(
        df
    )

    risk_col = _find_column(
        df,
        [
            "Risk Level",
            "Customer Risk Segment",
            "Risk Category",
        ],
    )

    if risk_col:

        risk_values = (
            df[risk_col]
            .astype(str)
            .str.lower()
        )

        high_risk = int(
            risk_values.str.contains(
                "high|critical",
                regex=True,
                na=False,
            ).sum()
        )

    else:

        high_risk = int(
            (probability >= 70).sum()
        )

    tenure_col = _find_column(
        df,
        [
            "tenure",
            "Tenure",
            "Customer Tenure",
            "Tenure Months",
        ],
    )

    if tenure_col:

        average_tenure = float(
            _numeric_series(
                df,
                tenure_col,
            ).mean()
        )

    else:

        average_tenure = 0.0

    charge_col = _find_column(
        df,
        [
            "MonthlyCharges",
            "Monthly Charges",
            "Monthly Charge",
        ],
    )

    if charge_col:

        average_charge = float(
            _numeric_series(
                df,
                charge_col,
            ).mean()
        )

    else:

        average_charge = 0.0

    risk = get_revenue_at_risk(
        df
    )

    # High-value customers at risk.
    high_value_at_risk = 0

    if charge_col:

        charges = _numeric_series(
            df,
            charge_col,
        )

        threshold = (
            charges.quantile(0.75)
            if charges.notna().any()
            else np.nan
        )

        if pd.notna(threshold):

            high_value_at_risk = int(
                (
                    (charges >= threshold)
                    & (
                        probability >= 70
                    )
                ).sum()
            )

    return {
        "total_customers": total,
        "predicted_churn": churn,
        "churn_rate": round(
            churn / total * 100,
            2,
        ),
        "average_probability": round(
            float(
                probability.mean()
            )
            if probability.notna().any()
            else 0.0,
            2,
        ),
        "high_risk_customers": high_risk,
        "average_tenure": round(
            average_tenure,
            2,
        ),
        "average_monthly_charge": round(
            average_charge,
            2,
        ),
        "monthly_revenue_at_risk":
            risk[
                "monthly_revenue_at_risk"
            ],
        "annual_revenue_at_risk":
            risk[
                "annual_revenue_at_risk"
            ],
        "high_value_at_risk":
            high_value_at_risk,
    }


# ============================================================
# DATASET COLUMN PROFILE
# ============================================================

def get_column_profile(df):
    """
    Generate a detailed column-level data profile.
    """

    if df is None or df.empty:
        return pd.DataFrame()

    rows = []

    for column in df.columns:

        series = df[column]

        rows.append(
            {
                "Column": column,
                "Data Type": str(
                    series.dtype
                ),
                "Missing": int(
                    series.isna().sum()
                ),
                "Missing %": round(
                    series.isna().mean()
                    * 100,
                    2,
                ),
                "Unique Values": int(
                    series.nunique(
                        dropna=True
                    )
                ),
                "Unique %": round(
                    series.nunique(
                        dropna=True
                    )
                    / len(df)
                    * 100,
                    2,
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# DASHBOARD FILTERING
# ============================================================

def apply_filters(
    df,
    filters=None,
):
    """
    Apply optional dashboard filters.

    Example:

        filters = {
            "Contract": "Month-to-month",
            "Gender": "Female",
            "Internet Service": "Fiber optic",
        }
    """

    if df is None or df.empty:
        return pd.DataFrame()

    if not filters:
        return df.copy()

    result = df.copy()

    for requested_column, selected_value in (
        filters.items()
    ):

        if (
            selected_value is None
            or selected_value == "All"
        ):
            continue

        column = _find_column(
            result,
            [requested_column],
        )

        if column is None:
            continue

        if isinstance(
            selected_value,
            (list, tuple, set),
        ):

            result = result[
                result[column]
                .astype(str)
                .isin(
                    [
                        str(value)
                        for value in selected_value
                    ]
                )
            ]

        else:

            result = result[
                result[column]
                .astype(str)
                == str(selected_value)
            ]

    return result


# ============================================================
# EXPORTABLE ANALYTICS SUMMARY
# ============================================================

def create_analytics_summary(df):
    """
    Create a compact business analytics summary dataframe.
    """

    kpis = get_advanced_kpis(
        df
    )

    return pd.DataFrame(
        [
            {
                "Metric": "Total Customers",
                "Value": kpis[
                    "total_customers"
                ],
            },
            {
                "Metric": "Predicted Churn",
                "Value": kpis[
                    "predicted_churn"
                ],
            },
            {
                "Metric": "Churn Rate (%)",
                "Value": kpis[
                    "churn_rate"
                ],
            },
            {
                "Metric":
                    "Average Churn Probability (%)",
                "Value": kpis[
                    "average_probability"
                ],
            },
            {
                "Metric":
                    "High/Critical Risk Customers",
                "Value": kpis[
                    "high_risk_customers"
                ],
            },
            {
                "Metric":
                    "Average Tenure (Months)",
                "Value": kpis[
                    "average_tenure"
                ],
            },
            {
                "Metric":
                    "Average Monthly Charge",
                "Value": kpis[
                    "average_monthly_charge"
                ],
            },
            {
                "Metric":
                    "Monthly Revenue at Risk",
                "Value": kpis[
                    "monthly_revenue_at_risk"
                ],
            },
            {
                "Metric":
                    "Annual Revenue at Risk",
                "Value": kpis[
                    "annual_revenue_at_risk"
                ],
            },
            {
                "Metric":
                    "High-Value Customers at Risk",
                "Value": kpis[
                    "high_value_at_risk"
                ],
            },
        ]
    )


# ============================================================
# MODULE TEST
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 60
    )

    print(
        "ChurnIQ Dashboard Analytics"
    )

    print(
        "=" * 60
    )

    print(
        "Module loaded successfully."
    )

    print(
        f"BASE_DIR: {BASE_DIR}"
    )

    print(
        f"DATA_DIR: {DATA_DIR}"
    )

    print(
        f"MODEL_DIR: {MODEL_DIR}"
    )

    model_results = (
        get_model_performance()
    )

    if model_results.empty:

        print(
            "Model results: not available."
        )

    else:

        print(
            f"Model results loaded: "
            f"{len(model_results)} models."
        )

    print(
        "=" * 60
    )