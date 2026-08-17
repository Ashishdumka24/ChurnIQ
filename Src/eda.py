"""
ChurnIQ - Exploratory Data Analysis
Phase 3: EDA & Analytics

This module:
- Works with CSV/XLS/XLSX-loaded DataFrames
- Detects common churn column names
- Produces safe statistical summaries
- Generates dashboard-ready analytics
- Never modifies the original DataFrame
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# COLUMN DETECTION
# ---------------------------------------------------------------------

COLUMN_ALIASES = {
    "customer_id": [
        "customerid",
        "customer_id",
        "customer id",
        "clientid",
        "client_id",
        "accountid",
        "account_id",
        "accountnumber",
        "account_number",
    ],
    "churn": [
        "churn",
        "ischurn",
        "is_churn",
        "churned",
        "exited",
        "exit",
        "attrition",
        "customerstatus",
        "customer_status",
    ],
    "tenure": [
        "tenure",
        "tenuremonths",
        "tenure_months",
        "months",
        "monthsactive",
        "months_active",
    ],
    "monthly_charges": [
        "monthlycharges",
        "monthly_charges",
        "monthlycharge",
        "monthly_charge",
        "monthlycost",
        "monthly_cost",
        "monthlyfee",
        "monthly_fee",
    ],
    "total_charges": [
        "totalcharges",
        "total_charges",
        "totalcharge",
        "total_charge",
        "totalcost",
        "total_cost",
        "lifetimevalue",
        "lifetime_value",
    ],
    "contract": [
        "contract",
        "contracttype",
        "contract_type",
        "plan",
        "plantype",
        "plan_type",
        "subscriptiontype",
        "subscription_type",
    ],
    "payment_method": [
        "paymentmethod",
        "payment_method",
        "paymenttype",
        "payment_type",
        "payment",
        "paymethod",
        "pay_method",
    ],
}


def normalize_column_name(column):
    """Normalize a column name for comparison."""
    if column is None:
        return ""

    value = str(column).strip().lower()

    for char in [
        " ",
        "_",
        "-",
        "/",
        "\\",
        ".",
        "(",
        ")",
        "[",
        "]",
    ]:
        value = value.replace(char, "")

    return value


def detect_column(df, aliases):
    """Find the actual DataFrame column matching aliases."""
    normalized = {
        normalize_column_name(column): column
        for column in df.columns
    }

    for alias in aliases:
        key = normalize_column_name(alias)

        if key in normalized:
            return normalized[key]

    return None


def detect_important_columns(df):
    """Detect common customer/churn fields."""
    if df is None or df.empty:
        return {}

    return {
        name: detect_column(df, aliases)
        for name, aliases in COLUMN_ALIASES.items()
    }


# ---------------------------------------------------------------------
# BASIC DATA PROFILE
# ---------------------------------------------------------------------

def get_basic_profile(df):
    """Return general dataset information."""

    if df is None:
        raise ValueError("DataFrame cannot be None.")

    if df.empty:
        raise ValueError("DataFrame is empty.")

    missing_cells = int(df.isna().sum().sum())

    total_cells = int(df.shape[0] * df.shape[1])

    missing_percentage = (
        missing_cells / total_cells * 100
        if total_cells
        else 0
    )

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_cells": missing_cells,
        "missing_percentage": round(
            missing_percentage,
            2
        ),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_columns": int(
            len(
                df.select_dtypes(
                    include=np.number
                ).columns
            )
        ),
        "categorical_columns": int(
            len(
                df.select_dtypes(
                    include=[
                        "object",
                        "category",
                        "bool",
                    ]
                ).columns
            )
        ),
    }


# ---------------------------------------------------------------------
# CHURN TARGET
# ---------------------------------------------------------------------

def normalize_churn_values(series):
    """
    Convert common churn labels to 0/1.

    Unknown values remain NaN.
    """

    if series is None:
        return pd.Series(dtype="float64")

    # Numeric binary target
    if pd.api.types.is_numeric_dtype(series):

        numeric = pd.to_numeric(
            series,
            errors="coerce"
        )

        unique = set(
            numeric.dropna().unique()
        )

        if unique.issubset({0, 1}):
            return numeric.astype("float")

    normalized = (
        series.astype("string")
        .str.strip()
        .str.lower()
    )

    positive = {
        "yes",
        "y",
        "true",
        "1",
        "churn",
        "churned",
        "exited",
        "exit",
        "left",
        "attrition",
        "attrited",
    }

    negative = {
        "no",
        "n",
        "false",
        "0",
        "retained",
        "retain",
        "active",
        "stayed",
        "stay",
    }

    result = normalized.map(
        lambda value:
        1 if value in positive
        else 0 if value in negative
        else np.nan
    )

    return result.astype("float")


def get_churn_analysis(df):
    """
    Calculate churn distribution and rates.
    """

    columns = detect_important_columns(df)

    churn_column = columns.get("churn")

    if not churn_column:
        return {
            "available": False,
            "message": (
                "No churn/attrition target column "
                "was detected."
            ),
        }

    churn = normalize_churn_values(
        df[churn_column]
    )

    valid = churn.dropna()

    if valid.empty:
        return {
            "available": False,
            "message": (
                "The detected churn column does not "
                "contain recognizable churn values."
            ),
        }

    churned = int((valid == 1).sum())
    retained = int((valid == 0).sum())
    total = churned + retained

    churn_rate = (
        churned / total * 100
        if total
        else 0
    )

    retention_rate = (
        retained / total * 100
        if total
        else 0
    )

    distribution = pd.DataFrame(
        {
            "Status": [
                "Churned",
                "Retained",
            ],
            "Customers": [
                churned,
                retained,
            ],
            "Percentage": [
                round(churn_rate, 2),
                round(retention_rate, 2),
            ],
        }
    )

    return {
        "available": True,
        "column": churn_column,
        "total_customers": total,
        "churned_customers": churned,
        "retained_customers": retained,
        "churn_rate": round(churn_rate, 2),
        "retention_rate": round(
            retention_rate,
            2
        ),
        "distribution": distribution,
        "target": churn,
    }


# ---------------------------------------------------------------------
# MISSING DATA
# ---------------------------------------------------------------------

def get_missing_value_analysis(df):
    """Return missing-value statistics."""

    missing = df.isna().sum()

    result = pd.DataFrame(
        {
            "Column": missing.index,
            "Missing Values": missing.values,
        }
    )

    result["Missing Percentage"] = np.where(
        len(df) > 0,
        result["Missing Values"] / len(df) * 100,
        0,
    )

    result = result.sort_values(
        "Missing Values",
        ascending=False,
    ).reset_index(drop=True)

    return result


# ---------------------------------------------------------------------
# NUMERICAL ANALYSIS
# ---------------------------------------------------------------------

def get_numeric_summary(df):
    """Return numerical descriptive statistics."""

    numeric_df = df.select_dtypes(
        include=np.number
    )

    if numeric_df.empty:
        return pd.DataFrame()

    summary = (
        numeric_df
        .describe()
        .T
        .reset_index()
        .rename(
            columns={
                "index": "Feature"
            }
        )
    )

    return summary


def get_categorical_summary(df):
    """Return useful categorical statistics."""

    categorical_columns = df.select_dtypes(
        include=[
            "object",
            "category",
            "bool",
        ]
    ).columns

    records = []

    for column in categorical_columns:

        series = df[column].dropna()

        records.append(
            {
                "Feature": column,
                "Unique Values": int(
                    series.nunique()
                ),
                "Most Common": (
                    series.mode().iloc[0]
                    if not series.mode().empty
                    else "N/A"
                ),
                "Most Common Count": (
                    int(
                        series.value_counts()
                        .iloc[0]
                    )
                    if not series.empty
                    else 0
                ),
            }
        )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------
# CONTRACT ANALYSIS
# ---------------------------------------------------------------------

def get_churn_by_column(
    df,
    column_name,
    min_group_size=1,
):
    """
    Calculate churn rate by any categorical column.

    Returns a dashboard-ready DataFrame.
    """

    if not column_name:
        return pd.DataFrame()

    if column_name not in df.columns:
        return pd.DataFrame()

    columns = detect_important_columns(df)

    churn_column = columns.get("churn")

    if not churn_column:
        return pd.DataFrame()

    working = df[
        [column_name, churn_column]
    ].copy()

    working["_churn_binary"] = (
        normalize_churn_values(
            working[churn_column]
        )
    )

    working = working.dropna(
        subset=["_churn_binary"]
    )

    if working.empty:
        return pd.DataFrame()

    working[column_name] = (
        working[column_name]
        .astype("string")
        .fillna("Unknown")
    )

    grouped = (
        working
        .groupby(column_name)
        .agg(
            Customers=(
                "_churn_binary",
                "size"
            ),
            Churned=(
                "_churn_binary",
                "sum"
            ),
        )
        .reset_index()
    )

    grouped = grouped[
        grouped["Customers"] >= min_group_size
    ]

    grouped["Churn Rate"] = np.where(
        grouped["Customers"] > 0,
        grouped["Churned"]
        / grouped["Customers"]
        * 100,
        0,
    )

    grouped["Retention Rate"] = (
        100 - grouped["Churn Rate"]
    )

    grouped["Churn Rate"] = grouped[
        "Churn Rate"
    ].round(2)

    grouped["Retention Rate"] = grouped[
        "Retention Rate"
    ].round(2)

    return grouped.sort_values(
        "Churn Rate",
        ascending=False,
    ).reset_index(drop=True)


def get_contract_analysis(df):
    """Churn analysis by contract type."""

    columns = detect_important_columns(df)

    return get_churn_by_column(
        df,
        columns.get("contract"),
    )


def get_payment_analysis(df):
    """Churn analysis by payment method."""

    columns = detect_important_columns(df)

    return get_churn_by_column(
        df,
        columns.get("payment_method"),
    )


# ---------------------------------------------------------------------
# TENURE ANALYSIS
# ---------------------------------------------------------------------

def get_tenure_analysis(df):
    """Analyze churn across tenure groups."""

    columns = detect_important_columns(df)

    tenure_column = columns.get("tenure")
    churn_column = columns.get("churn")

    if not tenure_column or not churn_column:
        return pd.DataFrame()

    tenure = pd.to_numeric(
        df[tenure_column],
        errors="coerce",
    )

    churn = normalize_churn_values(
        df[churn_column]
    )

    working = pd.DataFrame(
        {
            "Tenure": tenure,
            "Churn": churn,
        }
    ).dropna()

    if working.empty:
        return pd.DataFrame()

    # Adaptive tenure bins.
    max_tenure = working["Tenure"].max()

    if max_tenure <= 12:
        bins = [-np.inf, 3, 6, 9, 12, np.inf]
        labels = [
            "0-3",
            "4-6",
            "7-9",
            "10-12",
            "12+",
        ]

    elif max_tenure <= 36:
        bins = [
            -np.inf,
            6,
            12,
            18,
            24,
            36,
            np.inf,
        ]
        labels = [
            "0-6",
            "7-12",
            "13-18",
            "19-24",
            "25-36",
            "36+",
        ]

    else:
        bins = [
            -np.inf,
            6,
            12,
            24,
            36,
            48,
            60,
            np.inf,
        ]
        labels = [
            "0-6",
            "7-12",
            "13-24",
            "25-36",
            "37-48",
            "49-60",
            "60+",
        ]

    working["Tenure Group"] = pd.cut(
        working["Tenure"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    result = (
        working
        .groupby(
            "Tenure Group",
            observed=False
        )
        .agg(
            Customers=("Churn", "size"),
            Churned=("Churn", "sum"),
        )
        .reset_index()
    )

    result["Churn Rate"] = np.where(
        result["Customers"] > 0,
        result["Churned"]
        / result["Customers"]
        * 100,
        0,
    )

    result["Churn Rate"] = result[
        "Churn Rate"
    ].round(2)

    return result


# ---------------------------------------------------------------------
# CORRELATION
# ---------------------------------------------------------------------

def get_correlation_matrix(df):
    """Return correlation matrix for numerical features."""

    numeric_df = df.select_dtypes(
        include=np.number
    )

    if numeric_df.shape[1] < 2:
        return pd.DataFrame()

    return numeric_df.corr(
        numeric_only=True
    )


# ---------------------------------------------------------------------
# OUTLIER ANALYSIS
# ---------------------------------------------------------------------

def get_outlier_summary(df):
    """Calculate IQR-based outlier counts."""

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    records = []

    for column in numeric_columns:

        series = pd.to_numeric(
            df[column],
            errors="coerce"
        ).dropna()

        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        count = int(
            (
                (series < lower)
                | (series > upper)
            ).sum()
        )

        records.append(
            {
                "Feature": column,
                "Outliers": count,
                "Outlier Percentage": round(
                    count / len(series) * 100,
                    2,
                ),
            }
        )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------
# BUSINESS INSIGHTS
# ---------------------------------------------------------------------

def generate_eda_insights(df):
    """
    Generate simple, data-driven business insights.
    """

    insights = []

    churn_analysis = get_churn_analysis(df)

    if not churn_analysis.get("available"):
        return [
            "A recognizable churn target was not detected."
        ]

    churn_rate = churn_analysis["churn_rate"]

    if churn_rate >= 30:
        insights.append(
            f"Overall churn is high at {churn_rate:.1f}%. "
            "Retention should be treated as a major business priority."
        )

    elif churn_rate >= 20:
        insights.append(
            f"Overall churn is {churn_rate:.1f}%, "
            "indicating a meaningful retention opportunity."
        )

    else:
        insights.append(
            f"Overall churn is relatively contained at "
            f"{churn_rate:.1f}%."
        )

    columns = detect_important_columns(df)

    contract_column = columns.get("contract")

    if contract_column:

        contract = get_churn_by_column(
            df,
            contract_column,
        )

        if not contract.empty:

            highest = contract.iloc[0]

            insights.append(
                f"The highest observed churn rate is "
                f"{highest['Churn Rate']:.1f}% for "
                f"{highest[contract_column]}."
            )

    tenure = get_tenure_analysis(df)

    if not tenure.empty:

        highest_tenure = tenure.iloc[
            tenure["Churn Rate"].argmax()
        ]

        insights.append(
            f"The tenure group with the highest churn "
            f"rate is {highest_tenure['Tenure Group']} "
            f"at {highest_tenure['Churn Rate']:.1f}%."
        )

    return insights


# ---------------------------------------------------------------------
# COMPLETE EDA PACKAGE
# ---------------------------------------------------------------------

def run_eda(df):
    """
    Run the complete EDA process.

    Returns a dictionary that can later be consumed
    directly by Streamlit dashboard modules.
    """

    if df is None:
        raise ValueError(
            "Cannot run EDA on a None DataFrame."
        )

    if df.empty:
        raise ValueError(
            "Cannot run EDA on an empty DataFrame."
        )

    return {
        "profile": get_basic_profile(df),
        "detected_columns": detect_important_columns(df),
        "churn": get_churn_analysis(df),
        "missing_values": get_missing_value_analysis(df),
        "numeric_summary": get_numeric_summary(df),
        "categorical_summary": get_categorical_summary(df),
        "contract_analysis": get_contract_analysis(df),
        "payment_analysis": get_payment_analysis(df),
        "tenure_analysis": get_tenure_analysis(df),
        "correlation": get_correlation_matrix(df),
        "outliers": get_outlier_summary(df),
        "insights": generate_eda_insights(df),
    }


# ---------------------------------------------------------------------
# FILE-BASED TEST
# ---------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("ChurnIQ - EDA Test")
    print("=" * 60)

    try:

        dataset_path = Path(
            "Data/telco_churn_clean.csv"
        )

        if not dataset_path.exists():

            raise FileNotFoundError(
                f"Dataset not found: {dataset_path}"
            )

        df = pd.read_csv(
            dataset_path
        )

        results = run_eda(df)

        print("\nDATASET PROFILE")
        print("-" * 40)

        for key, value in results[
            "profile"
        ].items():

            print(
                f"{key}: {value}"
            )

        print("\nDETECTED COLUMNS")
        print("-" * 40)

        for key, value in results[
            "detected_columns"
        ].items():

            print(
                f"{key}: {value}"
            )

        print("\nCHURN ANALYSIS")
        print("-" * 40)

        churn = results["churn"]

        if churn.get("available"):

            print(
                f"Customers: "
                f"{churn['total_customers']}"
            )

            print(
                f"Churned: "
                f"{churn['churned_customers']}"
            )

            print(
                f"Churn Rate: "
                f"{churn['churn_rate']}%"
            )

            print(
                f"Retention Rate: "
                f"{churn['retention_rate']}%"
            )

        else:

            print(
                churn.get(
                    "message",
                    "Churn analysis unavailable."
                )
            )

        print("\nBUSINESS INSIGHTS")
        print("-" * 40)

        for insight in results["insights"]:

            print(f"- {insight}")

        print("\nEDA COMPLETED SUCCESSFULLY.")

    except Exception as error:

        print("\nEDA TEST FAILED")
        print("-" * 40)
        print(f"Error: {error}")