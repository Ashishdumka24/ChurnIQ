"""
============================================================
CHURNIQ - UNIVERSAL PREDICTION PIPELINE
============================================================

Customer Churn Prediction & Business Intelligence System

Designed for:
    - Telecom
    - Banking
    - SaaS
    - E-commerce
    - Subscription businesses
    - Insurance
    - Retail
    - FinTech
    - EdTech
    - Healthcare businesses
    - General customer databases

PIPELINE:

    Uploaded Dataset
          |
          v
    Data Validation
          |
          v
    Automatic Column Detection
          |
          v
    Data Cleaning
          |
          v
    Feature Engineering
          |
          v
    Target Detection
          |
          +-----------------------------+
          |                             |
          v                             v
    Churn Target Found          No Churn Target
          |                             |
          v                             v
    Adaptive ML Model           Behavioral Risk Engine
          |                             |
          +-------------+---------------+
                        |
                        v
                Churn Probability
                        |
                        v
                  Risk Score
                        |
                        v
                   Risk Level
                        |
                        v
                 Churn Prediction

OUTPUT COLUMNS:

    Churn Prediction
    Churn Probability
    Risk Score
    Risk Level
    Prediction Method

============================================================
"""

from pathlib import Path
import warnings
import re

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# OPTIONAL SCIKIT-LEARN IMPORTS
# ============================================================

SKLEARN_AVAILABLE = True

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
except Exception:
    SKLEARN_AVAILABLE = False


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "Models"
    / "best_churn_model.pkl"
)

PREPROCESSING_PATH = (
    BASE_DIR
    / "Models"
    / "preprocessing.pkl"
)


# ============================================================
# GLOBAL COLUMN ALIASES
# ============================================================

TARGET_ALIASES = [
    "churn",
    "churned",
    "churn_flag",
    "churn_status",
    "churn_label",
    "churn_value",
    "customer_churn",
    "attrition",
    "attrition_flag",
    "attrition_status",
    "left",
    "customer_left",
    "cancelled",
    "canceled",
    "cancellation",
    "is_cancelled",
    "is_canceled",
    "subscription_cancelled",
    "subscription_canceled",
    "inactive",
    "is_inactive",
    "retained",
    "retention",
]


CUSTOMER_ID_ALIASES = [
    "customerid",
    "customer_id",
    "customer",
    "customer_number",
    "customer_no",
    "clientid",
    "client_id",
    "client_number",
    "accountid",
    "account_id",
    "account_number",
    "user_id",
    "userid",
    "member_id",
    "subscriber_id",
    "subscription_id",
]


TENURE_ALIASES = [
    "tenure",
    "tenure_months",
    "months_with_company",
    "months_as_customer",
    "customer_tenure",
    "relationship_length",
    "relationship_months",
    "membership_months",
    "subscription_months",
    "duration",
]


RECENCY_ALIASES = [
    "days_since_last_activity",
    "days_since_last_login",
    "days_since_last_purchase",
    "days_since_last_transaction",
    "days_inactive",
    "inactive_days",
    "days_since_activity",
    "recency",
    "last_activity_days",
]


USAGE_ALIASES = [
    "usage",
    "monthly_usage",
    "total_usage",
    "login_count",
    "logins",
    "sessions",
    "session_count",
    "transactions",
    "transaction_count",
    "orders",
    "order_count",
    "purchase_count",
    "frequency",
    "engagement",
    "engagement_score",
    "activity",
    "activity_score",
]


SUPPORT_ALIASES = [
    "support_tickets",
    "ticket_count",
    "tickets",
    "complaints",
    "complaint_count",
    "customer_complaints",
    "service_calls",
    "support_calls",
    "issues",
    "issue_count",
]


PAYMENT_ALIASES = [
    "late_payments",
    "late_payment_count",
    "payment_delays",
    "payment_delay",
    "overdue",
    "overdue_amount",
    "missed_payments",
    "payment_failures",
    "failed_payments",
    "delinquent",
]


SPEND_ALIASES = [
    "monthly_charges",
    "monthly_charge",
    "monthly_spend",
    "monthly_revenue",
    "revenue",
    "revenue_per_month",
    "total_spend",
    "total_revenue",
    "amount_spent",
    "purchase_value",
    "customer_value",
    "clv",
    "customer_lifetime_value",
    "arpu",
]


# ============================================================
# BASIC HELPERS
# ============================================================

def _normalize_column_name(column):
    """
    Convert a column name into a comparison-friendly format.
    """

    value = str(column).strip().lower()

    value = re.sub(
        r"[^a-z0-9]+",
        "_",
        value
    )

    value = re.sub(
        r"_+",
        "_",
        value
    )

    return value.strip("_")


def _normalized_columns(df):
    """
    Return mapping:

        normalized_name -> original_name
    """

    mapping = {}

    for column in df.columns:

        normalized = _normalize_column_name(
            column
        )

        mapping[normalized] = column

    return mapping


def _find_column(
    df,
    aliases,
):
    """
    Find a real DataFrame column using aliases.
    """

    mapping = _normalized_columns(df)

    # Exact alias match
    for alias in aliases:

        normalized_alias = _normalize_column_name(
            alias
        )

        if normalized_alias in mapping:
            return mapping[
                normalized_alias
            ]

    # Partial match
    for normalized_name, original in mapping.items():

        for alias in aliases:

            normalized_alias = _normalize_column_name(
                alias
            )

            if (
                normalized_alias in normalized_name
                or normalized_name in normalized_alias
            ):
                return original

    return None


def _find_columns(
    df,
    aliases,
):
    """
    Find multiple columns related to a concept.
    """

    found = []

    mapping = _normalized_columns(df)

    for normalized_name, original in mapping.items():

        for alias in aliases:

            normalized_alias = _normalize_column_name(
                alias
            )

            if (
                normalized_name == normalized_alias
                or normalized_alias in normalized_name
                or normalized_name in normalized_alias
            ):

                if original not in found:
                    found.append(original)

                break

    return found


# ============================================================
# TARGET DETECTION
# ============================================================

def detect_target_column(df):
    """
    Automatically detect churn / attrition / cancellation
    target column.
    """

    if df is None or df.empty:
        return None

    # Exact aliases first
    target = _find_column(
        df,
        TARGET_ALIASES
    )

    if target is not None:
        return target

    # More intelligent keyword detection
    keywords = [
        "churn",
        "attrition",
        "cancel",
        "cancellation",
        "left",
    ]

    for column in df.columns:

        normalized = _normalize_column_name(
            column
        )

        for keyword in keywords:

            if keyword in normalized:
                return column

    return None


# ============================================================
# CUSTOMER ID DETECTION
# ============================================================

def detect_customer_id_column(df):
    """
    Detect customer identifier.
    """

    if df is None or df.empty:
        return None

    return _find_column(
        df,
        CUSTOMER_ID_ALIASES
    )


# ============================================================
# DATA CLEANING
# ============================================================

def clean_dataset(df):
    """
    Universal customer-data cleaning.

    Handles:
        - Duplicate rows
        - Duplicate columns
        - Empty columns
        - Blank strings
        - Missing values
        - Numeric conversion where appropriate
    """

    if df is None:
        return pd.DataFrame()

    if not isinstance(
        df,
        pd.DataFrame
    ):
        df = pd.DataFrame(df)

    working = df.copy()

    # Remove completely empty rows
    working = working.dropna(
        how="all"
    )

    # Remove duplicate rows
    working = working.drop_duplicates()

    # Remove duplicate column names
    working = working.loc[
        :,
        ~working.columns.duplicated()
    ]

    # Normalize blank strings
    for column in working.columns:

        if (
            working[column].dtype == "object"
            or str(
                working[column].dtype
            ).startswith("string")
        ):

            working[column] = (
                working[column]
                .astype(str)
                .str.strip()
                .replace(
                    {
                        "": np.nan,
                        "nan": np.nan,
                        "None": np.nan,
                        "NULL": np.nan,
                        "null": np.nan,
                        "N/A": np.nan,
                        "n/a": np.nan,
                        "NA": np.nan,
                    }
                )
            )

    # Try numeric conversion where it makes sense
    for column in working.columns:

        if working[column].dtype == "object":

            converted = pd.to_numeric(
                working[column],
                errors="coerce"
            )

            valid_ratio = (
                converted.notna().mean()
            )

            if valid_ratio >= 0.85:

                working[column] = converted

    return working.reset_index(
        drop=True
    )


# ============================================================
# TARGET NORMALIZATION
# ============================================================

def normalize_target(series):
    """
    Convert common churn target formats to:

        0 = retained
        1 = churn

    Supports:

        0 / 1
        Yes / No
        True / False
        Churn / Retained
        Churned / Active
        Left / Stayed
        Cancelled / Active
        Attrited / Existing
    """

    values = series.copy()

    # Numeric
    numeric = pd.to_numeric(
        values,
        errors="coerce"
    )

    numeric_ratio = numeric.notna().mean()

    if numeric_ratio >= 0.95:

        unique = set(
            numeric.dropna().unique()
        )

        if unique.issubset(
            {0, 1}
        ):

            return numeric.fillna(0).astype(int)

        # Generic binary numeric
        if len(unique) == 2:

            sorted_values = sorted(
                unique
            )

            mapping = {
                sorted_values[0]: 0,
                sorted_values[1]: 1,
            }

            return (
                numeric
                .map(mapping)
                .fillna(0)
                .astype(int)
            )

    # Text
    text = (
        values
        .astype(str)
        .str.strip()
        .str.lower()
    )

    churn_words = {
        "1",
        "yes",
        "y",
        "true",
        "churn",
        "churned",
        "left",
        "cancelled",
        "canceled",
        "inactive",
        "attrited",
        "attrition",
        "lost",
        "terminated",
        "closed",
        "exit",
    }

    retain_words = {
        "0",
        "no",
        "n",
        "false",
        "retained",
        "retain",
        "active",
        "existing",
        "stayed",
        "stay",
        "current",
        "loyal",
        "not churned",
    }

    result = pd.Series(
        np.nan,
        index=values.index,
        dtype="float"
    )

    for index, value in text.items():

        if value in churn_words:

            result.loc[index] = 1

        elif value in retain_words:

            result.loc[index] = 0

    # Generic binary text
    if result.notna().sum() == 0:

        unique = [
            value
            for value in text.unique()
            if value not in {
                "nan",
                "none",
                ""
            }
        ]

        if len(unique) == 2:

            result = text.map(
                {
                    unique[0]: 0,
                    unique[1]: 1,
                }
            )

    return result.fillna(0).astype(int)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer_features(df):
    """
    Universal business feature engineering.

    The function does not assume a specific industry.
    """

    working = df.copy()

    # --------------------------------------------------------
    # Missing-value indicators
    # --------------------------------------------------------

    for column in list(
        working.columns
    ):

        missing_ratio = (
            working[column].isna().mean()
        )

        if (
            missing_ratio > 0
            and missing_ratio < 1
        ):

            working[
                f"{column}_missing"
            ] = working[
                column
            ].isna().astype(int)

    # --------------------------------------------------------
    # Tenure
    # --------------------------------------------------------

    tenure_column = _find_column(
        working,
        TENURE_ALIASES
    )

    if tenure_column:

        numeric = pd.to_numeric(
            working[tenure_column],
            errors="coerce"
        )

        working[
            "__cq_tenure"
        ] = numeric

    # --------------------------------------------------------
    # Recency / inactivity
    # --------------------------------------------------------

    recency_column = _find_column(
        working,
        RECENCY_ALIASES
    )

    if recency_column:

        numeric = pd.to_numeric(
            working[recency_column],
            errors="coerce"
        )

        working[
            "__cq_recency"
        ] = numeric

    # --------------------------------------------------------
    # Usage
    # --------------------------------------------------------

    usage_columns = _find_columns(
        working,
        USAGE_ALIASES
    )

    if usage_columns:

        usage_values = []

        for column in usage_columns:

            numeric = pd.to_numeric(
                working[column],
                errors="coerce"
            )

            usage_values.append(
                numeric
            )

        if usage_values:

            working[
                "__cq_usage"
            ] = pd.concat(
                usage_values,
                axis=1
            ).mean(axis=1)

    # --------------------------------------------------------
    # Support / complaints
    # --------------------------------------------------------

    support_columns = _find_columns(
        working,
        SUPPORT_ALIASES
    )

    if support_columns:

        support_values = []

        for column in support_columns:

            support_values.append(
                pd.to_numeric(
                    working[column],
                    errors="coerce"
                )
            )

        working[
            "__cq_support"
        ] = pd.concat(
            support_values,
            axis=1
        ).mean(axis=1)

    # --------------------------------------------------------
    # Payment problems
    # --------------------------------------------------------

    payment_columns = _find_columns(
        working,
        PAYMENT_ALIASES
    )

    if payment_columns:

        payment_values = []

        for column in payment_columns:

            payment_values.append(
                pd.to_numeric(
                    working[column],
                    errors="coerce"
                )
            )

        working[
            "__cq_payment_risk"
        ] = pd.concat(
            payment_values,
            axis=1
        ).mean(axis=1)

    # --------------------------------------------------------
    # Spend / revenue
    # --------------------------------------------------------

    spend_columns = _find_columns(
        working,
        SPEND_ALIASES
    )

    if spend_columns:

        spend_values = []

        for column in spend_columns:

            spend_values.append(
                pd.to_numeric(
                    working[column],
                    errors="coerce"
                )
            )

        working[
            "__cq_value"
        ] = pd.concat(
            spend_values,
            axis=1
        ).mean(axis=1)

    return working


# ============================================================
# MODEL LOADING
# ============================================================

def load_model(
    model_path=MODEL_PATH,
    preprocessing_path=PREPROCESSING_PATH,
):
    """
    Safely load the existing ChurnIQ model.

    Failure here does NOT mean the entire application fails.
    The adaptive model / behavioral engine can take over.
    """

    model = None
    metadata = {}
    model_name = "Existing ChurnIQ Model"
    preprocessing = None

    try:

        model_path = Path(
            model_path
        )

        if not model_path.exists():
            return (
                None,
                metadata,
                model_name,
                preprocessing,
            )

        payload = joblib.load(
            model_path
        )

        if isinstance(
            payload,
            dict
        ):

            model = payload.get(
                "model"
            )

            metadata = payload.get(
                "metadata",
                {}
            )

            model_name = payload.get(
                "model_name",
                model_name
            )

            preprocessing = payload.get(
                "preprocessing",
                payload.get(
                    "preprocessor"
                )
            )

        else:

            model = payload

        if preprocessing is None:

            preprocessing_path = Path(
                preprocessing_path
            )

            if preprocessing_path.exists():

                try:

                    preprocessing_payload = (
                        joblib.load(
                            preprocessing_path
                        )
                    )

                    if isinstance(
                        preprocessing_payload,
                        dict
                    ):

                        preprocessing = (
                            preprocessing_payload.get(
                                "preprocessor",
                                preprocessing_payload.get(
                                    "transformer",
                                    preprocessing_payload.get(
                                        "encoder"
                                    )
                                )
                            )
                        )

                        for key, value in (
                            preprocessing_payload.items()
                        ):

                            if key not in metadata:
                                metadata[key] = value

                    else:

                        preprocessing = (
                            preprocessing_payload
                        )

                except Exception:
                    preprocessing = None

        if not hasattr(
            model,
            "predict"
        ):

            model = None

    except Exception:

        model = None
        metadata = {}
        preprocessing = None

    return (
        model,
        metadata,
        model_name,
        preprocessing,
    )


# ============================================================
# EXISTING MODEL FEATURE DETECTION
# ============================================================

def _get_feature_columns(metadata):
    """
    Retrieve existing model feature columns.
    """

    if not isinstance(
        metadata,
        dict
    ):
        return []

    possible_keys = [
        "feature_columns",
        "features",
        "training_features",
        "input_features",
        "feature_names",
    ]

    for key in possible_keys:

        value = metadata.get(
            key
        )

        if isinstance(
            value,
            (
                list,
                tuple,
                np.ndarray
            )
        ):

            return list(value)

    return []


# ============================================================
# EXISTING MODEL COMPATIBILITY
# ============================================================

def existing_model_is_compatible(
    df,
    metadata,
):
    """
    Check whether the old trained model can process
    the uploaded dataset.

    IMPORTANT:
    This never throws the old missing-column error.
    """

    features = _get_feature_columns(
        metadata
    )

    if not features:
        return False

    return all(
        feature in df.columns
        for feature in features
    )


# ============================================================
# UNIVERSAL ML MODEL
# ============================================================

def train_adaptive_model(
    df,
    target_column,
):
    """
    Train a model directly from the uploaded company dataset.

    This is used when the dataset contains an actual
    churn/attrition/cancellation target.
    """

    if not SKLEARN_AVAILABLE:
        return None, None, None

    if (
        target_column is None
        or target_column not in df.columns
    ):
        return None, None, None

    working = df.copy()

    target = normalize_target(
        working[target_column]
    )

    valid_target = target.notna()

    working = working.loc[
        valid_target
    ].copy()

    target = target.loc[
        valid_target
    ]

    if len(working) < 20:
        return None, None, None

    if target.nunique() < 2:
        return None, None, None

    # --------------------------------------------------------
    # Remove target
    # --------------------------------------------------------

    X = working.drop(
        columns=[target_column],
        errors="ignore"
    )

    # Remove obvious identifiers
    customer_id = detect_customer_id_column(
        X
    )

    if customer_id:

        X = X.drop(
            columns=[customer_id],
            errors="ignore"
        )

    # Remove very high-cardinality object columns
    # such as unique IDs / transaction strings
    high_cardinality = []

    for column in X.columns:

        if (
            X[column].dtype == "object"
            or str(
                X[column].dtype
            ).startswith("string")
        ):

            unique_ratio = (
                X[column].nunique(
                    dropna=True
                )
                /
                max(
                    len(X),
                    1
                )
            )

            if unique_ratio > 0.95:

                high_cardinality.append(
                    column
                )

    if high_cardinality:

        X = X.drop(
            columns=high_cardinality,
            errors="ignore"
        )

    if X.empty:
        return None, None, None

    numeric_columns = [
        column
        for column in X.columns
        if pd.api.types.is_numeric_dtype(
            X[column]
        )
    ]

    categorical_columns = [
        column
        for column in X.columns
        if column not in numeric_columns
    ]

    transformers = []

    if numeric_columns:

        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    )
                ),
                (
                    "scaler",
                    StandardScaler()
                ),
            ]
        )

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_columns
            )
        )

    if categorical_columns:

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                ),
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        )

    if not transformers:
        return None, None, None

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    # --------------------------------------------------------
    # Train/test split
    # --------------------------------------------------------

    try:

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                target,
                test_size=0.20,
                random_state=42,
                stratify=target
            )
        )

    except Exception:

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                target,
                test_size=0.20,
                random_state=42
            )
        )

    # --------------------------------------------------------
    # Primary model
    # --------------------------------------------------------

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=250,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1,
                    min_samples_leaf=2
                )
            ),
        ]
    )

    try:

        model.fit(
            X_train,
            y_train
        )

    except Exception:

        # Logistic regression fallback
        model = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced"
                    )
                ),
            ]
        )

        try:

            model.fit(
                X_train,
                y_train
            )

        except Exception:

            return None, None, None

    # --------------------------------------------------------
    # Model performance
    # --------------------------------------------------------

    auc = None

    try:

        if hasattr(
            model,
            "predict_proba"
        ):

            test_probability = (
                model.predict_proba(
                    X_test
                )[:, 1]
            )

            if len(
                np.unique(y_test)
            ) == 2:

                auc = float(
                    roc_auc_score(
                        y_test,
                        test_probability
                    )
                )

    except Exception:
        auc = None

    metadata = {
        "feature_columns": list(
            X.columns
        ),
        "target_column": target_column,
        "training_rows": len(
            X_train
        ),
        "test_rows": len(
            X_test
        ),
        "model_type": type(
            model.named_steps[
                "classifier"
            ]
        ).__name__,
        "roc_auc": auc,
        "adaptive_model": True,
    }

    return (
        model,
        metadata,
        auc,
    )


# ============================================================
# NORMALIZATION HELPERS
# ============================================================

def _minmax_series(
    series,
    inverse=False,
):
    """
    Convert a numeric signal to 0-1.

    inverse=True means:
        low values = high risk
    """

    values = pd.to_numeric(
        series,
        errors="coerce"
    )

    if values.notna().sum() == 0:

        return pd.Series(
            0.5,
            index=series.index
        )

    median = values.median()

    values = values.fillna(
        median
    )

    minimum = values.min()
    maximum = values.max()

    if maximum == minimum:

        result = pd.Series(
            0.5,
            index=series.index
        )

    else:

        result = (
            values - minimum
        ) / (
            maximum - minimum
        )

    if inverse:
        result = 1 - result

    return result.clip(
        0,
        1
    )


# ============================================================
# BEHAVIORAL RISK ENGINE
# ============================================================

def behavioral_risk_engine(df):
    """
    Universal churn-risk engine for datasets without
    an explicit churn target.

    IMPORTANT:

    This is a behavioral risk estimate, NOT a trained
    churn classifier.

    It allows ChurnIQ to analyze unlabeled company
    databases without pretending that a true ML target
    exists.
    """

    working = df.copy()

    risk = pd.Series(
        0.45,
        index=working.index,
        dtype=float
    )

    signals_used = []

    # --------------------------------------------------------
    # RECENCY / INACTIVITY
    # --------------------------------------------------------

    recency_columns = _find_columns(
        working,
        RECENCY_ALIASES
    )

    for column in recency_columns:

        signal = _minmax_series(
            working[column]
        )

        risk += (
            signal * 0.25
        )

        signals_used.append(
            column
        )

    # --------------------------------------------------------
    # USAGE
    # Low usage = higher risk
    # --------------------------------------------------------

    usage_columns = _find_columns(
        working,
        USAGE_ALIASES
    )

    for column in usage_columns:

        signal = _minmax_series(
            working[column],
            inverse=True
        )

        risk += (
            signal * 0.15
        )

        signals_used.append(
            column
        )

    # --------------------------------------------------------
    # TENURE
    # Very low tenure can indicate onboarding risk
    # --------------------------------------------------------

    tenure_column = _find_column(
        working,
        TENURE_ALIASES
    )

    if tenure_column:

        signal = _minmax_series(
            working[tenure_column],
            inverse=True
        )

        risk += (
            signal * 0.10
        )

        signals_used.append(
            tenure_column
        )

    # --------------------------------------------------------
    # SUPPORT / COMPLAINTS
    # More problems = higher risk
    # --------------------------------------------------------

    support_columns = _find_columns(
        working,
        SUPPORT_ALIASES
    )

    for column in support_columns:

        signal = _minmax_series(
            working[column]
        )

        risk += (
            signal * 0.15
        )

        signals_used.append(
            column
        )

    # --------------------------------------------------------
    # PAYMENT PROBLEMS
    # --------------------------------------------------------

    payment_columns = _find_columns(
        working,
        PAYMENT_ALIASES
    )

    for column in payment_columns:

        signal = _minmax_series(
            working[column]
        )

        risk += (
            signal * 0.20
        )

        signals_used.append(
            column
        )

    # --------------------------------------------------------
    # Clamp
    # --------------------------------------------------------

    risk = risk.clip(
        0.01,
        0.99
    )

    # If no recognized business signals exist,
    # create a neutral but varied score using
    # available numeric information.
    if not signals_used:

        numeric_columns = [
            column
            for column in working.columns
            if pd.api.types.is_numeric_dtype(
                working[column]
            )
        ]

        if numeric_columns:

            numeric_data = (
                working[
                    numeric_columns
                ]
                .apply(
                    pd.to_numeric,
                    errors="coerce"
                )
            )

            numeric_data = (
                numeric_data
                .replace(
                    [np.inf, -np.inf],
                    np.nan
                )
            )

            if not numeric_data.empty:

                normalized = (
                    numeric_data
                    .rank(
                        pct=True
                    )
                    .mean(
                        axis=1
                    )
                    .fillna(
                        0.5
                    )
                )

                risk = (
                    0.25
                    +
                    normalized * 0.50
                ).clip(
                    0.05,
                    0.95
                )

    return (
        risk.to_numpy(),
        signals_used
    )


# ============================================================
# PROBABILITY
# ============================================================

def get_churn_probability(
    model,
    X,
):
    """
    Safely obtain churn probability.
    """

    if model is None:

        return np.zeros(
            len(X)
        )

    # --------------------------------------------------------
    # predict_proba
    # --------------------------------------------------------

    if hasattr(
        model,
        "predict_proba"
    ):

        try:

            probabilities = np.asarray(
                model.predict_proba(X),
                dtype=float
            )

            if probabilities.ndim == 2:

                if probabilities.shape[1] >= 2:
                    return probabilities[:, 1]

                if probabilities.shape[1] == 1:
                    return probabilities[:, 0]

            if probabilities.ndim == 1:
                return probabilities

        except Exception:
            pass

    # --------------------------------------------------------
    # decision function
    # --------------------------------------------------------

    if hasattr(
        model,
        "decision_function"
    ):

        try:

            scores = np.asarray(
                model.decision_function(X),
                dtype=float
            )

            scores = np.clip(
                scores,
                -50,
                50
            )

            return (
                1
                /
                (
                    1
                    +
                    np.exp(-scores)
                )
            )

        except Exception:
            pass

    # --------------------------------------------------------
    # Prediction fallback
    # --------------------------------------------------------

    try:

        predictions = np.asarray(
            model.predict(X)
        )

        return np.where(
            predictions == 1,
            1.0,
            0.0
        )

    except Exception:

        return np.zeros(
            len(X)
        )


# ============================================================
# RISK LEVEL
# ============================================================

def calculate_risk_level(
    probability
):
    """
    Convert probability into business risk.
    """

    probability = float(
        probability
    )

    if probability >= 0.75:
        return "Critical"

    if probability >= 0.50:
        return "High"

    if probability >= 0.25:
        return "Medium"

    return "Low"


# ============================================================
# RISK SCORE
# ============================================================

def calculate_risk_score(
    probability
):
    """
    Convert probability to 0-100 score.
    """

    probability = float(
        probability
    )

    return round(
        max(
            0,
            min(
                100,
                probability * 100
            )
        ),
        2
    )


# ============================================================
# ADAPTIVE PREDICTION
# ============================================================

def predict_customers(
    df,
    model_path=MODEL_PATH,
):
    """
    Universal ChurnIQ prediction function.

    This function NEVER requires IBM Telco columns.

    Prediction priority:

        1. Existing compatible ChurnIQ model
        2. Adaptive ML model trained from uploaded labels
        3. Behavioral risk engine
    """

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if df is None:

        return pd.DataFrame()

    if not isinstance(
        df,
        pd.DataFrame
    ):

        df = pd.DataFrame(df)

    if df.empty:

        return df.copy()

    original_df = df.copy()

    # --------------------------------------------------------
    # Clean
    # --------------------------------------------------------

    working = clean_dataset(
        original_df
    )

    if working.empty:

        return original_df.copy()

    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------

    engineered = engineer_features(
        working
    )

    # --------------------------------------------------------
    # Target detection
    # --------------------------------------------------------

    target_column = detect_target_column(
        engineered
    )

    # ========================================================
    # METHOD 1
    # Existing model
    # ========================================================

    (
        existing_model,
        existing_metadata,
        existing_model_name,
        existing_preprocessing,
    ) = load_model(
        model_path
    )

    if (
        existing_model is not None
        and existing_model_is_compatible(
            engineered,
            existing_metadata
        )
    ):

        feature_columns = _get_feature_columns(
            existing_metadata
        )

        X = engineered[
            feature_columns
        ].copy()

        try:

            is_pipeline = (
                hasattr(
                    existing_model,
                    "named_steps"
                )
                or
                hasattr(
                    existing_model,
                    "steps"
                )
            )

            if (
                not is_pipeline
                and existing_preprocessing is not None
                and hasattr(
                    existing_preprocessing,
                    "transform"
                )
            ):

                X = existing_preprocessing.transform(
                    X
                )

            predictions = existing_model.predict(
                X
            )

            probabilities = get_churn_probability(
                existing_model,
                X
            )

            method = (
                "Existing ChurnIQ ML Model"
            )

        except Exception:

            predictions = None
            probabilities = None

    else:

        predictions = None
        probabilities = None
        method = None

    # ========================================================
    # METHOD 2
    # Adaptive ML
    # ========================================================

    if (
        predictions is None
        and target_column is not None
    ):

        try:

            (
                adaptive_model,
                adaptive_metadata,
                auc,
            ) = train_adaptive_model(
                engineered,
                target_column
            )

        except Exception:

            adaptive_model = None
            adaptive_metadata = None
            auc = None

        if adaptive_model is not None:

            try:

                # Build prediction features
                feature_columns = (
                    adaptive_metadata[
                        "feature_columns"
                    ]
                )

                X = engineered[
                    feature_columns
                ].copy()

                predictions = (
                    adaptive_model.predict(
                        X
                    )
                )

                probabilities = (
                    get_churn_probability(
                        adaptive_model,
                        X
                    )
                )

                method = (
                    "Adaptive ML Model"
                )

            except Exception:

                predictions = None
                probabilities = None

    # ========================================================
    # METHOD 3
    # Behavioral Risk Engine
    # ========================================================

    if (
        predictions is None
        or probabilities is None
    ):

        (
            probabilities,
            signals_used,
        ) = behavioral_risk_engine(
            engineered
        )

        predictions = np.where(
            probabilities >= 0.50,
            1,
            0
        )

        method = (
            "Behavioral Risk Engine"
        )

    # --------------------------------------------------------
    # Safety
    # --------------------------------------------------------

    probabilities = np.asarray(
        probabilities,
        dtype=float
    )

    probabilities = np.clip(
        probabilities,
        0,
        1
    )

    predictions = np.asarray(
        predictions
    )

    # Fix unexpected lengths
    if len(probabilities) != len(original_df):

        probabilities = np.resize(
            probabilities,
            len(original_df)
        )

    if len(predictions) != len(original_df):

        predictions = np.resize(
            predictions,
            len(original_df)
        )

    # ========================================================
    # RESULT
    # ========================================================

    result = original_df.copy()

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    normalized_predictions = []

    for prediction in predictions:

        if isinstance(
            prediction,
            str
        ):

            value = (
                prediction
                .strip()
                .lower()
            )

            if value in {
                "1",
                "yes",
                "true",
                "churn",
                "churned",
                "left",
                "cancelled",
                "canceled",
                "inactive",
                "attrited",
            }:

                normalized_predictions.append(
                    "Churn"
                )

            else:

                normalized_predictions.append(
                    "Retained"
                )

        else:

            try:

                normalized_predictions.append(
                    "Churn"
                    if float(
                        prediction
                    ) >= 0.5
                    else "Retained"
                )

            except Exception:

                normalized_predictions.append(
                    "Retained"
                )

    result[
        "Churn Prediction"
    ] = normalized_predictions

    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    result[
        "Churn Probability"
    ] = (
        probabilities * 100
    ).round(2)

    # --------------------------------------------------------
    # Risk score
    # --------------------------------------------------------

    result[
        "Risk Score"
    ] = [
        calculate_risk_score(
            probability
        )
        for probability in probabilities
    ]

    # --------------------------------------------------------
    # Risk level
    # --------------------------------------------------------

    result[
        "Risk Level"
    ] = [
        calculate_risk_level(
            probability
        )
        for probability in probabilities
    ]

    # --------------------------------------------------------
    # Prediction method
    # --------------------------------------------------------

    result[
        "Prediction Method"
    ] = method

    # --------------------------------------------------------
    # Customer ID
    # --------------------------------------------------------

    customer_id = detect_customer_id_column(
        result
    )

    if customer_id is None:

        result[
            "ChurnIQ Customer ID"
        ] = [
            f"CQ-{index + 1:05d}"
            for index in range(
                len(result)
            )
        ]

    return result


# ============================================================
# APP COMPATIBILITY WRAPPER
# ============================================================

def predict_customer_churn(
    df,
    model_path=MODEL_PATH,
):
    """
    Compatibility function used by app.py.

    Keep this name unchanged.
    """

    return predict_customers(
        df=df,
        model_path=model_path
    )


# ============================================================
# SINGLE CUSTOMER PREDICTION
# ============================================================

def predict_single_customer(
    customer_data,
    model_path=MODEL_PATH,
):
    """
    Predict one customer.
    """

    if isinstance(
        customer_data,
        dict
    ):

        df = pd.DataFrame(
            [customer_data]
        )

    elif isinstance(
        customer_data,
        pd.Series
    ):

        df = pd.DataFrame(
            [customer_data.to_dict()]
        )

    elif isinstance(
        customer_data,
        pd.DataFrame
    ):

        df = customer_data.copy()

    else:

        raise TypeError(
            "Customer data must be "
            "a dictionary, Series or DataFrame."
        )

    return predict_customers(
        df,
        model_path
    )


# ============================================================
# SUMMARY
# ============================================================

def get_prediction_summary(
    prediction_df,
):
    """
    Generate dashboard-ready statistics.
    """

    if prediction_df is None:

        return {
            "total_customers": 0,
            "predicted_churn": 0,
            "predicted_retained": 0,
            "predicted_churn_rate": 0.0,
            "average_risk_score": 0.0,
            "average_churn_probability": 0.0,
            "critical_risk": 0,
            "high_risk": 0,
            "medium_risk": 0,
            "low_risk": 0,
        }

    if not isinstance(
        prediction_df,
        pd.DataFrame
    ):

        prediction_df = pd.DataFrame(
            prediction_df
        )

    if prediction_df.empty:

        return {
            "total_customers": 0,
            "predicted_churn": 0,
            "predicted_retained": 0,
            "predicted_churn_rate": 0.0,
            "average_risk_score": 0.0,
            "average_churn_probability": 0.0,
            "critical_risk": 0,
            "high_risk": 0,
            "medium_risk": 0,
            "low_risk": 0,
        }

    total = len(
        prediction_df
    )

    churn_count = int(
        (
            prediction_df[
                "Churn Prediction"
            ]
            == "Churn"
        ).sum()
    )

    retained_count = int(
        (
            prediction_df[
                "Churn Prediction"
            ]
            == "Retained"
        ).sum()
    )

    churn_rate = (
        churn_count
        /
        total
        *
        100
        if total
        else 0
    )

    average_probability = float(
        pd.to_numeric(
            prediction_df[
                "Churn Probability"
            ],
            errors="coerce"
        ).mean()
    )

    average_risk = float(
        pd.to_numeric(
            prediction_df[
                "Risk Score"
            ],
            errors="coerce"
        ).mean()
    )

    risk_counts = (
        prediction_df[
            "Risk Level"
        ]
        .value_counts()
        .to_dict()
    )

    return {
        "total_customers":
            total,

        "predicted_churn":
            churn_count,

        "predicted_retained":
            retained_count,

        "predicted_churn_rate":
            round(
                churn_rate,
                2
            ),

        "average_risk_score":
            round(
                average_risk,
                2
            ),

        "average_churn_probability":
            round(
                average_probability,
                2
            ),

        "critical_risk":
            int(
                risk_counts.get(
                    "Critical",
                    0
                )
            ),

        "high_risk":
            int(
                risk_counts.get(
                    "High",
                    0
                )
            ),

        "medium_risk":
            int(
                risk_counts.get(
                    "Medium",
                    0
                )
            ),

        "low_risk":
            int(
                risk_counts.get(
                    "Low",
                    0
                )
            ),
    }


# ============================================================
# RISK SUMMARY COMPATIBILITY
# ============================================================

def generate_risk_summary(
    prediction_df
):
    """
    Compatibility helper for dashboard code.
    """

    return get_prediction_summary(
        prediction_df
    )


# ============================================================
# DATASET ANALYSIS
# ============================================================

def analyze_dataset(
    df
):
    """
    Return automatic dataset intelligence.

    Useful for ChurnIQ dashboard / Data Quality page.
    """

    if df is None:

        return {
            "rows": 0,
            "columns": 0,
            "target_column": None,
            "customer_id_column": None,
            "numeric_columns": [],
            "categorical_columns": [],
            "missing_values": 0,
            "duplicate_rows": 0,
            "prediction_mode":
                "Behavioral Risk Engine",
        }

    working = clean_dataset(
        df
    )

    target = detect_target_column(
        working
    )

    customer_id = (
        detect_customer_id_column(
            working
        )
    )

    numeric_columns = [
        column
        for column in working.columns
        if pd.api.types.is_numeric_dtype(
            working[column]
        )
    ]

    categorical_columns = [
        column
        for column in working.columns
        if column not in numeric_columns
    ]

    if target:

        prediction_mode = (
            "Adaptive ML Model"
        )

    else:

        prediction_mode = (
            "Behavioral Risk Engine"
        )

    return {
        "rows":
            len(working),

        "columns":
            len(working.columns),

        "target_column":
            target,

        "customer_id_column":
            customer_id,

        "numeric_columns":
            numeric_columns,

        "categorical_columns":
            categorical_columns,

        "missing_values":
            int(
                working.isna()
                .sum()
                .sum()
            ),

        "duplicate_rows":
            int(
                df.duplicated()
                .sum()
            ),

        "prediction_mode":
            prediction_mode,
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print(
        "CHURNIQ - UNIVERSAL PREDICTION PIPELINE TEST"
    )
    print("=" * 70)

    dataset_path = (
        BASE_DIR
        / "Data"
        / "telco_churn_clean.csv"
    )

    if not dataset_path.exists():

        print(
            "Default dataset not found."
        )

    else:

        df = pd.read_csv(
            dataset_path
        )

        print()
        print(
            f"Dataset rows    : {len(df)}"
        )

        print(
            f"Dataset columns : {len(df.columns)}"
        )

        information = analyze_dataset(
            df
        )

        print()
        print(
            f"Target column   : "
            f"{information['target_column']}"
        )

        print(
            f"Customer ID     : "
            f"{information['customer_id_column']}"
        )

        print(
            f"Prediction mode : "
            f"{information['prediction_mode']}"
        )

        print()
        print(
            "Running ChurnIQ analysis..."
        )

        predictions = predict_customers(
            df
        )

        summary = get_prediction_summary(
            predictions
        )

        print()
        print("=" * 70)
        print(
            "CHURNIQ ANALYSIS COMPLETE"
        )
        print("=" * 70)

        print(
            f"Customers       : "
            f"{summary['total_customers']}"
        )

        print(
            f"Predicted churn : "
            f"{summary['predicted_churn']}"
        )

        print(
            f"Churn rate      : "
            f"{summary['predicted_churn_rate']}%"
        )

        print(
            f"Average risk    : "
            f"{summary['average_risk_score']}"
        )

        print(
            f"Critical        : "
            f"{summary['critical_risk']}"
        )

        print(
            f"High            : "
            f"{summary['high_risk']}"
        )

        print(
            f"Medium          : "
            f"{summary['medium_risk']}"
        )

        print(
            f"Low             : "
            f"{summary['low_risk']}"
        )

        print()
        print(
            predictions[
                [
                    "Churn Prediction",
                    "Churn Probability",
                    "Risk Score",
                    "Risk Level",
                    "Prediction Method",
                ]
            ]
            .head(10)
            .to_string(
                index=False
            )
        )

        print()
        print("=" * 70)
        print(
            "PIPELINE TEST COMPLETE"
        )
        print("=" * 70)