"""
ChurnIQ - Dataset Validator
Supports CSV, XLS and XLSX files.
Designed to work with different customer churn datasets.
"""

from pathlib import Path
import pandas as pd
import numpy as np


SUPPORTED_EXTENSIONS = {".csv", ".xls", ".xlsx"}

# Common variations of important churn/customer columns
COLUMN_ALIASES = {
    "customer_id": [
        "customerid",
        "customer_id",
        "customer id",
        "customer",
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
        "status",
    ],
    "tenure": [
        "tenure",
        "tenuremonths",
        "tenure_months",
        "months",
        "monthsactive",
        "months_active",
        "customer_tenure",
    ],
    "monthly_charges": [
        "monthlycharges",
        "monthly_charges",
        "monthly charge",
        "monthlycharge",
        "monthlycost",
        "monthly_cost",
        "monthlyfee",
        "monthly_fee",
    ],
    "total_charges": [
        "totalcharges",
        "total_charges",
        "total charge",
        "totalcharge",
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
        "payment",
        "paymenttype",
        "payment_type",
        "paymethod",
        "pay_method",
    ],
}


def normalize_column_name(column):
    """Convert a column name into a comparison-friendly format."""
    if column is None:
        return ""

    value = str(column).strip().lower()

    for char in [" ", "-", "_", "/", "\\", ".", "(", ")", "[", "]"]:
        value = value.replace(char, "")

    return value


def find_matching_column(columns, aliases):
    """Find the first column matching a list of aliases."""
    normalized_columns = {
        normalize_column_name(col): col for col in columns
    }

    for alias in aliases:
        normalized_alias = normalize_column_name(alias)

        if normalized_alias in normalized_columns:
            return normalized_columns[normalized_alias]

    return None


def detect_columns(df):
    """
    Detect important columns using common column-name variations.
    Returns the actual dataset column names.
    """
    detected = {}

    for standard_name, aliases in COLUMN_ALIASES.items():
        detected[standard_name] = find_matching_column(
            df.columns,
            aliases
        )

    return detected


def load_dataset(file_path):
    """
    Load CSV, XLS or XLSX automatically.

    Parameters
    ----------
    file_path : str or Path
        Path to uploaded dataset.

    Returns
    -------
    pandas.DataFrame
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file was not found: {path}"
        )

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            "Unsupported file format. "
            "ChurnIQ supports CSV, XLS and XLSX files."
        )

    if extension == ".csv":
        # Try UTF-8 first, then common fallback encodings.
        encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252"]

        last_error = None

        for encoding in encodings:
            try:
                df = pd.read_csv(
                    path,
                    encoding=encoding,
                    low_memory=False
                )

                if len(df.columns) > 0:
                    return df

            except (UnicodeDecodeError, pd.errors.ParserError) as error:
                last_error = error

        raise ValueError(
            f"Unable to read the CSV file. {last_error}"
        )

    if extension == ".xls":
        try:
            return pd.read_excel(path, engine="xlrd")
        except ImportError:
            raise ImportError(
                "XLS files require the 'xlrd' package. "
                "Install it using: pip install xlrd"
            )

    if extension == ".xlsx":
        try:
            return pd.read_excel(path, engine="openpyxl")
        except ImportError:
            raise ImportError(
                "XLSX files require the 'openpyxl' package. "
                "Install it using: pip install openpyxl"
            )

    raise ValueError("Unable to determine dataset format.")


def basic_quality_check(df):
    """Perform basic data-quality checks."""

    if df is None:
        return {
            "valid": False,
            "message": "No dataset was provided."
        }

    if not isinstance(df, pd.DataFrame):
        return {
            "valid": False,
            "message": "Invalid dataset object."
        }

    if df.empty:
        return {
            "valid": False,
            "message": "The uploaded dataset is empty."
        }

    if len(df.columns) == 0:
        return {
            "valid": False,
            "message": "The dataset contains no columns."
        }

    # Completely empty rows
    empty_rows = int(df.isna().all(axis=1).sum())

    # Duplicate rows
    duplicate_rows = int(df.duplicated().sum())

    # Missing cells
    missing_cells = int(df.isna().sum().sum())

    total_cells = int(df.shape[0] * df.shape[1])

    missing_percentage = (
        (missing_cells / total_cells) * 100
        if total_cells > 0
        else 0
    )

    return {
        "valid": True,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_cells": missing_cells,
        "missing_percentage": round(missing_percentage, 2),
        "duplicate_rows": duplicate_rows,
        "empty_rows": empty_rows,
    }


def validate_churn_dataset(df):
    """
    Validate whether a dataset contains enough information
    for ChurnIQ analysis.
    """

    quality = basic_quality_check(df)

    if not quality["valid"]:
        return {
            **quality,
            "churn_ready": False,
            "detected_columns": {},
            "missing_required": ["Dataset"],
        }

    detected = detect_columns(df)

    # Churn target is the most important requirement.
    churn_found = detected.get("churn") is not None

    # A customer identifier is useful but not mandatory.
    customer_id_found = detected.get("customer_id") is not None

    # At least one useful customer feature is expected.
    feature_candidates = [
        detected.get("tenure"),
        detected.get("monthly_charges"),
        detected.get("total_charges"),
        detected.get("contract"),
        detected.get("payment_method"),
    ]

    useful_features = [
        column for column in feature_candidates
        if column is not None
    ]

    missing_required = []

    if not churn_found:
        missing_required.append("Churn/Attrition target column")

    churn_ready = churn_found and len(useful_features) >= 1

    warnings = []

    if not customer_id_found:
        warnings.append(
            "No customer ID column was detected. "
            "Customer-level identification may be limited."
        )

    if len(useful_features) == 0:
        warnings.append(
            "No common customer behavior or billing features "
            "were detected."
        )

    if quality["missing_percentage"] > 30:
        warnings.append(
            "The dataset contains more than 30% missing values."
        )

    if quality["duplicate_rows"] > 0:
        warnings.append(
            f"{quality['duplicate_rows']} duplicate rows detected."
        )

    return {
        **quality,
        "churn_ready": churn_ready,
        "detected_columns": detected,
        "missing_required": missing_required,
        "warnings": warnings,
        "feature_count": len(useful_features),
    }


def prepare_dataframe(df):
    """
    Perform safe basic preparation without changing the
    business meaning of the dataset.
    """

    if df is None:
        raise ValueError("No dataframe supplied.")

    prepared = df.copy()

    # Remove completely empty rows and columns.
    prepared = prepared.dropna(
        axis=0,
        how="all"
    )

    prepared = prepared.dropna(
        axis=1,
        how="all"
    )

    # Remove duplicate rows.
    prepared = prepared.drop_duplicates()

    # Clean column names while preserving readable names.
    prepared.columns = [
        str(column).strip()
        for column in prepared.columns
    ]

    # Convert common numeric-looking columns.
    for column in prepared.columns:

        if prepared[column].dtype == "object":

            cleaned = (
                prepared[column]
                .astype(str)
                .str.strip()
                .replace(
                    {
                        "": np.nan,
                        "nan": np.nan,
                        "None": np.nan,
                        "NULL": np.nan,
                        "null": np.nan,
                    }
                )
            )

            # Remove currency symbols and commas for numeric detection.
            numeric_candidate = (
                cleaned
                .str.replace(",", "", regex=False)
                .str.replace("$", "", regex=False)
                .str.replace("₹", "", regex=False)
                .str.replace("€", "", regex=False)
                .str.replace("£", "", regex=False)
            )

            numeric_values = pd.to_numeric(
                numeric_candidate,
                errors="coerce"
            )

            non_null_count = cleaned.notna().sum()

            if non_null_count > 0:
                numeric_ratio = (
                    numeric_values.notna().sum() /
                    non_null_count
                )

                if numeric_ratio >= 0.90:
                    prepared[column] = numeric_values
                else:
                    prepared[column] = cleaned

    return prepared


def get_dataset_summary(df):
    """Return a compact summary for the Streamlit dashboard."""

    if df is None or df.empty:
        return {}

    quality = basic_quality_check(df)
    detected = detect_columns(df)

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    return {
        "rows": quality["rows"],
        "columns": quality["columns"],
        "missing_cells": quality["missing_cells"],
        "missing_percentage": quality["missing_percentage"],
        "duplicate_rows": quality["duplicate_rows"],
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "detected_columns": detected,
    }


def validate_file(file_path):
    """
    Main validation function used by the application.
    """

    try:
        df = load_dataset(file_path)

        validation = validate_churn_dataset(df)

        return {
            "success": True,
            "dataframe": df,
            "validation": validation,
            "summary": get_dataset_summary(df),
            "error": None,
        }

    except Exception as error:
        return {
            "success": False,
            "dataframe": None,
            "validation": None,
            "summary": None,
            "error": str(error),
        }


if __name__ == "__main__":
    print("ChurnIQ Dataset Validator")
    print("Supported formats: CSV, XLS, XLSX")