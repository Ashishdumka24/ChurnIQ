"""
ChurnIQ - Dataset Cleaning
Supports CSV, XLS and XLSX datasets.
"""

from pathlib import Path
import pandas as pd
import numpy as np

from dataset_validator import (
    load_dataset,
    detect_columns,
)


def clean_column_names(df):
    """Clean column names without changing their meaning."""
    df = df.copy()

    cleaned_columns = []

    for column in df.columns:
        name = str(column).strip()
        name = " ".join(name.split())
        cleaned_columns.append(name)

    df.columns = cleaned_columns

    return df


def remove_empty_data(df):
    """Remove completely empty rows and columns."""
    df = df.copy()

    df = df.dropna(axis=0, how="all")
    df = df.dropna(axis=1, how="all")

    return df


def remove_duplicates(df):
    """Remove duplicate customer records/rows."""
    df = df.copy()

    before = len(df)

    df = df.drop_duplicates()

    removed = before - len(df)

    return df, removed


def clean_text_columns(df):
    """Standardize whitespace in text columns."""
    df = df.copy()

    for column in df.select_dtypes(
        include=["object", "category"]
    ).columns:

        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
        )

        df[column] = df[column].replace(
            {
                "": pd.NA,
                "nan": pd.NA,
                "None": pd.NA,
                "NULL": pd.NA,
                "null": pd.NA,
                "N/A": pd.NA,
                "n/a": pd.NA,
            }
        )

    return df


def convert_numeric_columns(df):
    """
    Convert columns that are predominantly numeric.

    Currency symbols and commas are removed safely.
    """
    df = df.copy()

    for column in df.columns:

        if df[column].dtype not in ["object", "string"]:
            continue

        original = df[column].astype("string").str.strip()

        cleaned = (
            original
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.replace("₹", "", regex=False)
            .str.replace("€", "", regex=False)
            .str.replace("£", "", regex=False)
        )

        numeric = pd.to_numeric(
            cleaned,
            errors="coerce"
        )

        non_null = original.notna().sum()

        if non_null == 0:
            continue

        numeric_ratio = (
            numeric.notna().sum() / non_null
        )

        if numeric_ratio >= 0.90:
            df[column] = numeric

    return df


def clean_common_telco_columns(df):
    """
    Perform safe cleaning for commonly encountered
    telecom/customer churn columns.

    These transformations are only applied when the
    corresponding column exists.
    """
    df = df.copy()

    detected = detect_columns(df)

    # Total charges frequently arrives as text because of
    # blank values in the original Telco dataset.
    total_charges = detected.get("total_charges")

    if total_charges and total_charges in df.columns:

        df[total_charges] = pd.to_numeric(
            df[total_charges]
            .astype("string")
            .str.strip()
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False),
            errors="coerce",
        )

    # Tenure should be numeric where possible.
    tenure = detected.get("tenure")

    if tenure and tenure in df.columns:

        df[tenure] = pd.to_numeric(
            df[tenure],
            errors="coerce"
        )

    # Monthly charges should be numeric where possible.
    monthly = detected.get("monthly_charges")

    if monthly and monthly in df.columns:

        df[monthly] = pd.to_numeric(
            df[monthly]
            .astype("string")
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.replace("₹", "", regex=False)
            .str.replace("€", "", regex=False)
            .str.replace("£", "", regex=False),
            errors="coerce",
        )

    return df


def handle_missing_values(df):
    """
    Handle missing values conservatively.

    Numerical columns:
        median

    Categorical columns:
        mode, or 'Unknown' if no mode exists
    """
    df = df.copy()

    for column in df.columns:

        if df[column].isna().sum() == 0:
            continue

        if pd.api.types.is_numeric_dtype(df[column]):

            median = df[column].median()

            if pd.notna(median):
                df[column] = df[column].fillna(median)

        else:

            mode = df[column].mode(dropna=True)

            if not mode.empty:
                fill_value = mode.iloc[0]
            else:
                fill_value = "Unknown"

            df[column] = df[column].fillna(fill_value)

    return df


def clean_churn_target(df):
    """
    Standardize common churn target values when possible.

    Output is kept as 0/1 for binary churn targets.
    """
    df = df.copy()

    detected = detect_columns(df)
    churn_column = detected.get("churn")

    if not churn_column or churn_column not in df.columns:
        return df

    series = df[churn_column]

    # Already numeric binary target
    if pd.api.types.is_numeric_dtype(series):

        unique_values = set(
            pd.Series(series.dropna()).unique()
        )

        if unique_values.issubset({0, 1}):
            df[churn_column] = series.astype(int)

        return df

    normalized = (
        series.astype("string")
        .str.strip()
        .str.lower()
    )

    positive_values = {
        "yes",
        "y",
        "true",
        "1",
        "churn",
        "churned",
        "exited",
        "exit",
        "left",
        "attrited",
        "attrition",
    }

    negative_values = {
        "no",
        "n",
        "false",
        "0",
        "retained",
        "retain",
        "active",
        "stayed",
        "not churned",
        "not_churned",
    }

    mapped = normalized.map(
        lambda value: (
            1 if value in positive_values
            else 0 if value in negative_values
            else np.nan
        )
    )

    recognized_ratio = (
        mapped.notna().sum() /
        max(normalized.notna().sum(), 1)
    )

    # Only replace the original target if the values are
    # clearly binary churn labels.
    if recognized_ratio >= 0.90:
        df[churn_column] = mapped.astype("Int64")

    return df


def clean_dataset(df):
    """
    Complete cleaning pipeline.

    Returns
    -------
    cleaned_df, report
    """

    if df is None:
        raise ValueError("No dataset was provided.")

    if df.empty:
        raise ValueError("The dataset is empty.")

    original_rows = len(df)
    original_columns = len(df.columns)
    original_missing = int(df.isna().sum().sum())

    df = df.copy()

    df = clean_column_names(df)

    df = remove_empty_data(df)

    df, duplicates_removed = remove_duplicates(df)

    df = clean_text_columns(df)

    df = convert_numeric_columns(df)

    df = clean_common_telco_columns(df)

    df = clean_churn_target(df)

    df = handle_missing_values(df)

    final_missing = int(df.isna().sum().sum())

    report = {
        "original_rows": original_rows,
        "original_columns": original_columns,
        "final_rows": len(df),
        "final_columns": len(df.columns),
        "duplicates_removed": duplicates_removed,
        "missing_values_before": original_missing,
        "missing_values_after": final_missing,
    }

    return df, report


def clean_file(input_path, output_path):
    """
    Load and clean a CSV/XLS/XLSX file and save it as CSV.
    """

    df = load_dataset(input_path)

    cleaned_df, report = clean_dataset(df)

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    cleaned_df.to_csv(
        output_path,
        index=False
    )

    return cleaned_df, report


def create_clean_dataset(
    input_path="Data/Telco-Customer-Churn.csv",
    output_path="Data/telco_churn_clean.csv",
):
    """Create the project's cleaned dataset."""

    return clean_file(
        input_path,
        output_path
    )


if __name__ == "__main__":

    input_file = "Data/Telco-Customer-Churn.csv"
    output_file = "Data/telco_churn_clean.csv"

    print("ChurnIQ Dataset Cleaning")
    print("-" * 40)

    try:

        cleaned_df, report = create_clean_dataset(
            input_file,
            output_file
        )

        print("Cleaning completed successfully.")
        print()
        print(f"Original rows : {report['original_rows']}")
        print(f"Final rows    : {report['final_rows']}")
        print(f"Columns       : {report['final_columns']}")
        print(
            f"Duplicates removed : "
            f"{report['duplicates_removed']}"
        )
        print(
            f"Missing values before : "
            f"{report['missing_values_before']}"
        )
        print(
            f"Missing values after  : "
            f"{report['missing_values_after']}"
        )
        print()
        print(f"Saved to: {output_file}")

    except Exception as error:

        print("Cleaning failed.")
        print(f"Error: {error}")