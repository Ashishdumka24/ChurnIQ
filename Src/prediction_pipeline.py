"""
ChurnIQ - Universal Prediction & Customer Intelligence Pipeline

Designed for:
- Telco
- Banking
- SaaS
- E-commerce
- Subscription businesses
- Insurance
- Retail
- Telecom
- Any customer-level CSV/XLS/XLSX dataset

Strategy
--------
1. Inspect uploaded customer data.
2. Automatically detect customer ID / churn / attrition columns.
3. Normalize common column naming variations.
4. Use the trained ChurnIQ model when the uploaded data is compatible.
5. When the uploaded company dataset is structurally different,
   generate a transparent universal customer-risk score.
6. Never require IBM/Telco column names from uploaded datasets.
7. Preserve the original uploaded data.
8. Return churn/risk intelligence in a common ChurnIQ format.
"""

from pathlib import Path
import warnings
import re

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR / "Models" / "best_churn_model.pkl"
)

PREPROCESSING_PATH = (
    BASE_DIR / "Models" / "preprocessing.pkl"
)


# ============================================================
# COMMON COLUMN PATTERNS
# ============================================================

ID_PATTERNS = [
    "customerid",
    "customer_id",
    "clientid",
    "client_id",
    "userid",
    "user_id",
    "accountid",
    "account_id",
    "accountnumber",
    "account_number",
    "customerkey",
    "customer_key",
    "memberid",
    "member_id",
    "subscriberid",
    "subscriber_id",
]

CHURN_PATTERNS = [
    "churn",
    "churned",
    "churnflag",
    "churn_flag",
    "churnstatus",
    "churn_status",
    "attrition",
    "attrited",
    "attritionflag",
    "attrition_flag",
    "exited",
    "exitflag",
    "exit_flag",
    "customerstatus",
    "customer_status",
    "customerstate",
    "customer_state",
    "leftcompany",
    "left_company",
    "cancelled",
    "canceled",
    "is_churn",
    "ischurn",
]

TENURE_PATTERNS = [
    "tenure",
    "tenuremonths",
    "tenure_months",
    "monthsactive",
    "months_active",
    "customerage",
    "relationshipmonths",
    "relationship_months",
]

CONTRACT_PATTERNS = [
    "contract",
    "contracttype",
    "contract_type",
    "subscriptiontype",
    "subscription_type",
    "plan",
    "plantype",
    "plan_type",
]

PAYMENT_PATTERNS = [
    "paymentmethod",
    "payment_method",
    "billingmethod",
    "billing_method",
    "paymenttype",
    "payment_type",
]

MONTHLY_VALUE_PATTERNS = [
    "monthlycharges",
    "monthly_charges",
    "monthlycharge",
    "monthly_charge",
    "monthlyspend",
    "monthly_spend",
    "monthlyrevenue",
    "monthly_revenue",
    "monthlyamount",
    "monthly_amount",
    "subscriptionprice",
    "subscription_price",
    "mrr",
]

TOTAL_VALUE_PATTERNS = [
    "totalcharges",
    "total_charges",
    "totalcharge",
    "total_charge",
    "totalspend",
    "total_spend",
    "totalrevenue",
    "total_revenue",
    "totalamount",
    "total_amount",
    "lifetimevalue",
    "lifetime_value",
    "ltv",
]

SUPPORT_PATTERNS = [
    "support",
    "tickets",
    "ticketcount",
    "ticket_count",
    "complaints",
    "complaintcount",
    "complaint_count",
    "issues",
    "issuecount",
    "issue_count",
]

USAGE_PATTERNS = [
    "usage",
    "usagehours",
    "usage_hours",
    "logins",
    "login_count",
    "logincount",
    "sessions",
    "sessioncount",
    "session_count",
    "activity",
    "activityscore",
    "activity_score",
]

ENGAGEMENT_PATTERNS = [
    "engagement",
    "engagementscore",
    "engagement_score",
    "satisfaction",
    "satisfactionscore",
    "satisfaction_score",
    "nps",
    "csat",
]

DISCOUNT_PATTERNS = [
    "discount",
    "discountpercent",
    "discount_percent",
    "discountamount",
    "discount_amount",
    "offer",
    "promotion",
    "promo",
]


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_column_name(column):
    """
    Normalize a column name for intelligent matching.
    """

    value = str(column).strip().lower()

    value = re.sub(
        r"[^a-z0-9]+",
        "",
        value,
    )

    return value


def _match_column(
    columns,
    patterns,
):
    """
    Find the best matching column from a list of patterns.
    """

    normalized = {
        column: normalize_column_name(column)
        for column in columns
    }

    # Exact match first
    for column, value in normalized.items():

        if value in patterns:
            return column

    # Partial match second
    for column, value in normalized.items():

        for pattern in patterns:

            if pattern in value:
                return column

    return None


# ============================================================
# DATASET PROFILING
# ============================================================

def profile_dataset(df):
    """
    Automatically understand an uploaded customer dataset.
    """

    if df is None:
        raise ValueError(
            "Dataset cannot be None."
        )

    if not isinstance(
        df,
        pd.DataFrame,
    ):
        raise TypeError(
            "Dataset must be a pandas DataFrame."
        )

    if df.empty:
        raise ValueError(
            "Dataset is empty."
        )

    columns = list(df.columns)

    profile = {
        "customer_id": _match_column(
            columns,
            ID_PATTERNS,
        ),

        "churn_column": _match_column(
            columns,
            CHURN_PATTERNS,
        ),

        "tenure": _match_column(
            columns,
            TENURE_PATTERNS,
        ),

        "contract": _match_column(
            columns,
            CONTRACT_PATTERNS,
        ),

        "payment": _match_column(
            columns,
            PAYMENT_PATTERNS,
        ),

        "monthly_value": _match_column(
            columns,
            MONTHLY_VALUE_PATTERNS,
        ),

        "total_value": _match_column(
            columns,
            TOTAL_VALUE_PATTERNS,
        ),

        "support": _match_column(
            columns,
            SUPPORT_PATTERNS,
        ),

        "usage": _match_column(
            columns,
            USAGE_PATTERNS,
        ),

        "engagement": _match_column(
            columns,
            ENGAGEMENT_PATTERNS,
        ),

        "discount": _match_column(
            columns,
            DISCOUNT_PATTERNS,
        ),
    }

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=[
            "object",
            "category",
            "string",
            "bool",
        ]
    ).columns.tolist()

    profile["numeric_columns"] = numeric_columns
    profile["categorical_columns"] = categorical_columns

    profile["rows"] = len(df)
    profile["columns"] = len(df.columns)

    return profile


# ============================================================
# TARGET CONVERSION
# ============================================================

def convert_churn_target(series):
    """
    Convert common churn labels to 0/1.
    """

    if pd.api.types.is_numeric_dtype(series):

        numeric = pd.to_numeric(
            series,
            errors="coerce",
        )

        unique_values = set(
            numeric.dropna().unique()
        )

        if unique_values.issubset(
            {0, 1}
        ):
            return numeric

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
        "left",
        "exited",
        "exit",
        "attrition",
        "attrited",
        "cancelled",
        "canceled",
        "inactive",
        "lost",
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
        "not churned",
        "not_churned",
    }

    result = normalized.map(
        lambda value:
        1
        if value in positive
        else 0
        if value in negative
        else np.nan
    )

    return result


# ============================================================
# MODEL LOADING
# ============================================================

def load_model(
    model_path=MODEL_PATH,
    preprocessing_path=PREPROCESSING_PATH,
):
    """
    Load the saved ChurnIQ model.
    """

    model_path = Path(
        model_path
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}"
        )

    payload = joblib.load(
        model_path
    )

    metadata = {}
    model_name = "Unknown Model"
    preprocessing = None

    if isinstance(
        payload,
        dict,
    ):

        model = payload.get(
            "model"
        )

        if model is None:
            raise ValueError(
                "Saved model does not contain "
                "a trained model."
            )

        metadata = payload.get(
            "metadata",
            {},
        )

        model_name = payload.get(
            "model_name",
            "Unknown Model",
        )

        preprocessing = payload.get(
            "preprocessing",
            payload.get(
                "preprocessor"
            ),
        )

    else:

        model = payload

    if not hasattr(
        model,
        "predict",
    ):
        raise TypeError(
            "Loaded object is not a valid "
            "prediction model."
        )

    # Load standalone preprocessing metadata
    if (
        preprocessing is None
        and Path(
            preprocessing_path
        ).exists()
    ):

        try:

            payload = joblib.load(
                preprocessing_path
            )

            if isinstance(
                payload,
                dict,
            ):

                preprocessing = payload.get(
                    "preprocessor",
                    payload.get(
                        "transformer"
                    ),
                )

                saved_metadata = payload.get(
                    "metadata",
                    {},
                )

                if isinstance(
                    saved_metadata,
                    dict,
                ):

                    for key, value in saved_metadata.items():

                        if key not in metadata:
                            metadata[key] = value

            else:

                preprocessing = payload

        except Exception:

            preprocessing = None

    return (
        model,
        metadata,
        model_name,
        preprocessing,
    )


# ============================================================
# TRAINING FEATURE DETECTION
# ============================================================

def get_training_features(metadata):
    """
    Retrieve features used during training.
    """

    if not isinstance(
        metadata,
        dict,
    ):
        return []

    for key in [
        "feature_columns",
        "features",
        "training_features",
        "input_features",
        "feature_names",
    ]:

        value = metadata.get(
            key
        )

        if isinstance(
            value,
            (
                list,
                tuple,
                np.ndarray,
            ),
        ):

            return list(value)

    return []


# ============================================================
# COMPATIBILITY CHECK
# ============================================================

def check_model_compatibility(
    df,
    metadata,
):
    """
    Determine whether an uploaded dataset can directly
    use the trained ChurnIQ model.
    """

    required = get_training_features(
        metadata
    )

    if not required:
        return False

    available = set(
        df.columns
    )

    return all(
        feature in available
        for feature in required
    )


# ============================================================
# DIRECT MODEL PREDICTION
# ============================================================

def run_model_prediction(
    df,
    model,
    metadata,
):
    """
    Run the original trained ChurnIQ model when
    the uploaded dataset is compatible.
    """

    features = get_training_features(
        metadata
    )

    if not features:
        raise ValueError(
            "Training feature metadata is unavailable."
        )

    X = df[
        features
    ].copy()

    # Pipeline models already contain preprocessing.
    try:

        predictions = model.predict(
            X
        )

    except Exception as error:

        raise RuntimeError(
            f"Model could not process data: {error}"
        ) from error

    probabilities = None

    if hasattr(
        model,
        "predict_proba",
    ):

        try:

            probabilities = model.predict_proba(
                X
            )[:, 1]

        except Exception:
            probabilities = None

    if probabilities is None:

        if hasattr(
            model,
            "decision_function",
        ):

            try:

                scores = model.decision_function(
                    X
                )

                scores = np.asarray(
                    scores,
                    dtype=float,
                )

                scores = np.clip(
                    scores,
                    -50,
                    50,
                )

                probabilities = (
                    1
                    /
                    (
                        1
                        +
                        np.exp(-scores)
                    )
                )

            except Exception:

                probabilities = None

    if probabilities is None:

        probabilities = np.where(
            np.asarray(
                predictions
            ) == 1,
            1.0,
            0.0,
        )

    return (
        predictions,
        probabilities,
        "Machine Learning Model",
    )


# ============================================================
# UNIVERSAL RISK ENGINE
# ============================================================

def _numeric_score(
    series,
    reverse=False,
):
    """
    Convert a numeric customer metric into a 0-1 risk signal.

    Higher values normally represent higher risk.

    reverse=True means lower values represent higher risk.
    """

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    if numeric.notna().sum() < 2:
        return pd.Series(
            0.5,
            index=series.index,
        )

    low = numeric.quantile(
        0.05
    )

    high = numeric.quantile(
        0.95
    )

    if high <= low:
        return pd.Series(
            0.5,
            index=series.index,
        )

    normalized = (
        numeric - low
    ) / (
        high - low
    )

    normalized = normalized.clip(
        0,
        1,
    )

    if reverse:
        normalized = 1 - normalized

    return normalized.fillna(
        normalized.median()
        if normalized.notna().any()
        else 0.5
    )


def _categorical_risk(
    series,
):
    """
    Convert categorical customer signals into
    a basic risk estimate.

    This is intentionally conservative.
    """

    values = (
        series.astype("string")
        .str.strip()
        .str.lower()
    )

    risk = pd.Series(
        0.5,
        index=series.index,
        dtype=float,
    )

    high_risk_words = [
        "inactive",
        "cancel",
        "cancelled",
        "canceled",
        "complaint",
        "complaints",
        "poor",
        "low",
        "bad",
        "month-to-month",
        "monthly",
        "declined",
        "negative",
        "unhappy",
        "dissatisfied",
        "expired",
        "overdue",
        "late",
    ]

    low_risk_words = [
        "active",
        "annual",
        "yearly",
        "premium",
        "loyal",
        "good",
        "excellent",
        "satisfied",
        "engaged",
    ]

    for word in high_risk_words:

        risk.loc[
            values.str.contains(
                word,
                na=False,
            )
        ] = np.maximum(
            risk.loc[
                values.str.contains(
                    word,
                    na=False,
                )
            ],
            0.75,
        )

    for word in low_risk_words:

        risk.loc[
            values.str.contains(
                word,
                na=False,
            )
        ] = np.minimum(
            risk.loc[
                values.str.contains(
                    word,
                    na=False,
                )
            ],
            0.25,
        )

    return risk


def universal_risk_prediction(
    df,
    profile,
):
    """
    Generate a universal business-risk estimate for
    datasets that cannot directly use the trained model.

    This is NOT presented as a trained ML prediction.

    It combines available customer signals such as:

    - Tenure
    - Usage
    - Engagement
    - Support issues
    - Monthly value
    - Contract
    - Payment
    - Discount
    - Satisfaction
    """

    risk_components = []
    weights = []

    # --------------------------------------------------------
    # TENURE
    # --------------------------------------------------------

    if profile.get(
        "tenure"
    ):

        score = _numeric_score(
            df[
                profile["tenure"]
            ],
            reverse=True,
        )

        risk_components.append(
            score
        )

        weights.append(
            1.5
        )

    # --------------------------------------------------------
    # USAGE
    # --------------------------------------------------------

    if profile.get(
        "usage"
    ):

        score = _numeric_score(
            df[
                profile["usage"]
            ],
            reverse=True,
        )

        risk_components.append(
            score
        )

        weights.append(
            1.5
        )

    # --------------------------------------------------------
    # ENGAGEMENT
    # --------------------------------------------------------

    if profile.get(
        "engagement"
    ):

        score = _numeric_score(
            df[
                profile["engagement"]
            ],
            reverse=True,
        )

        risk_components.append(
            score
        )

        weights.append(
            2.0
        )

    # --------------------------------------------------------
    # SUPPORT
    # --------------------------------------------------------

    if profile.get(
        "support"
    ):

        score = _numeric_score(
            df[
                profile["support"]
            ],
            reverse=False,
        )

        risk_components.append(
            score
        )

        weights.append(
            1.5
        )

    # --------------------------------------------------------
    # MONTHLY VALUE
    # --------------------------------------------------------

    if profile.get(
        "monthly_value"
    ):

        score = _numeric_score(
            df[
                profile["monthly_value"]
            ],
            reverse=False,
        )

        risk_components.append(
            score
        )

        weights.append(
            0.75
        )

    # --------------------------------------------------------
    # DISCOUNT
    # --------------------------------------------------------

    if profile.get(
        "discount"
    ):

        score = _numeric_score(
            df[
                profile["discount"]
            ],
            reverse=False,
        )

        risk_components.append(
            score
        )

        weights.append(
            0.5
        )

    # --------------------------------------------------------
    # CONTRACT
    # --------------------------------------------------------

    if profile.get(
        "contract"
    ):

        score = _categorical_risk(
            df[
                profile["contract"]
            ]
        )

        risk_components.append(
            score
        )

        weights.append(
            1.5
        )

    # --------------------------------------------------------
    # PAYMENT
    # --------------------------------------------------------

    if profile.get(
        "payment"
    ):

        score = _categorical_risk(
            df[
                profile["payment"]
            ]
        )

        risk_components.append(
            score
        )

        weights.append(
            0.5
        )

    # --------------------------------------------------------
    # NO RECOGNIZED BUSINESS FEATURES
    # --------------------------------------------------------

    if not risk_components:

        # Use a neutral-but-useful distribution based
        # on row position only to avoid every customer
        # receiving exactly the same score.
        #
        # This is explicitly an estimated risk score,
        # not a trained ML prediction.

        numeric_columns = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        numeric_columns = [
            column
            for column in numeric_columns
            if column
            != profile.get(
                "churn_column"
            )
        ]

        if numeric_columns:

            scores = []

            for column in numeric_columns[:5]:

                scores.append(
                    _numeric_score(
                        df[column]
                    )
                )

            combined = pd.concat(
                scores,
                axis=1,
            ).mean(
                axis=1
            )

        else:

            combined = pd.Series(
                0.5,
                index=df.index,
            )

    else:

        combined = pd.concat(
            risk_components,
            axis=1,
        )

        weights_array = np.asarray(
            weights,
            dtype=float,
        )

        combined = (
            combined
            .mul(
                weights_array,
                axis=1,
            )
            .sum(
                axis=1
            )
            /
            weights_array.sum()
        )

    combined = combined.clip(
        0,
        1,
    )

    # --------------------------------------------------------
    # Add small controlled variation
    # --------------------------------------------------------

    if combined.nunique() <= 1:

        numeric_columns = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        if numeric_columns:

            variation = _numeric_score(
                df[
                    numeric_columns[0]
                ]
            )

            combined = (
                combined * 0.65
                +
                variation * 0.35
            )

    probabilities = combined.to_numpy(
        dtype=float
    )

    predictions = np.where(
        probabilities >= 0.50,
        1,
        0,
    )

    return (
        predictions,
        probabilities,
        "Universal Customer Risk Engine",
    )


# ============================================================
# RISK LEVEL
# ============================================================

def calculate_risk_level(
    probability
):
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
    probability = float(
        probability
    )

    return round(
        max(
            0,
            min(
                100,
                probability * 100,
            ),
        ),
        2,
    )


# ============================================================
# MAIN PREDICTION FUNCTION
# ============================================================

def predict_customers(
    df,
    model_path=MODEL_PATH,
):
    """
    Universal ChurnIQ prediction function.

    Behaviour:

    Compatible dataset
        ↓
    Existing trained ML model

    Incompatible company dataset
        ↓
    Universal Customer Risk Engine

    Existing churn column
        ↓
    Used for analysis / validation

    No churn column
        ↓
    Risk estimation still works
    """

    if df is None:
        raise ValueError(
            "Prediction dataset cannot be None."
        )

    if not isinstance(
        df,
        pd.DataFrame,
    ):
        raise TypeError(
            "Prediction dataset must be a "
            "pandas DataFrame."
        )

    if df.empty:
        raise ValueError(
            "Prediction dataset is empty."
        )

    working = df.copy()

    # Clean column names very lightly.
    # Original names are preserved.
    working.columns = [
        str(column).strip()
        for column in working.columns
    ]

    profile = profile_dataset(
        working
    )

    # --------------------------------------------------------
    # Attempt trained ML model
    # --------------------------------------------------------

    used_method = None

    try:

        (
            model,
            metadata,
            model_name,
            preprocessing,
        ) = load_model(
            model_path
        )

        compatible = check_model_compatibility(
            working,
            metadata
        )

    except Exception:

        model = None
        metadata = {}
        model_name = None
        preprocessing = None
        compatible = False

    if compatible:

        try:

            (
                predictions,
                probabilities,
                used_method,
            ) = run_model_prediction(
                working,
                model,
                metadata,
            )

        except Exception:

            (
                predictions,
                probabilities,
                used_method,
            ) = universal_risk_prediction(
                working,
                profile,
            )

    else:

        (
            predictions,
            probabilities,
            used_method,
        ) = universal_risk_prediction(
            working,
            profile,
        )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    result = working.copy()

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    predictions = np.asarray(
        predictions
    )

    probabilities = np.clip(
        probabilities,
        0,
        1,
    )

    if len(
        probabilities
    ) != len(result):

        raise ValueError(
            "Risk probabilities do not match "
            "the number of customers."
        )

    if len(
        predictions
    ) != len(result):

        raise ValueError(
            "Predictions do not match "
            "the number of customers."
        )

    result[
        "Churn Prediction"
    ] = np.where(
        predictions == 1,
        "Churn",
        "Retained",
    )

    result[
        "Churn Probability"
    ] = (
        probabilities * 100
    ).round(2)

    result[
        "Risk Score"
    ] = [
        calculate_risk_score(
            value
        )
        for value in probabilities
    ]

    result[
        "Risk Level"
    ] = [
        calculate_risk_level(
            value
        )
        for value in probabilities
    ]

    # --------------------------------------------------------
    # Intelligence metadata
    # --------------------------------------------------------

    result.attrs[
        "churniq_method"
    ] = used_method

    result.attrs[
        "dataset_profile"
    ] = profile

    result.attrs[
        "model_name"
    ] = model_name

    result.attrs[
        "has_actual_churn"
    ] = bool(
        profile.get(
            "churn_column"
        )
    )

    return result


# ============================================================
# APP COMPATIBILITY
# ============================================================

def predict_customer_churn(
    df,
    model_path=MODEL_PATH,
):
    """
    Compatibility wrapper for app.py.
    """

    return predict_customers(
        df=df,
        model_path=model_path,
    )


# ============================================================
# SUMMARY
# ============================================================

def get_prediction_summary(
    prediction_df
):
    """
    Dashboard-ready prediction statistics.
    """

    if prediction_df is None:
        raise ValueError(
            "Prediction DataFrame is None."
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

    prediction_column = (
        "Churn Prediction"
    )

    probability_column = (
        "Churn Probability"
    )

    risk_column = (
        "Risk Level"
    )

    score_column = (
        "Risk Score"
    )

    predicted_churn = int(
        (
            prediction_df[
                prediction_column
            ]
            == "Churn"
        ).sum()
    )

    predicted_retained = int(
        (
            prediction_df[
                prediction_column
            ]
            == "Retained"
        ).sum()
    )

    total = len(
        prediction_df
    )

    churn_rate = (
        predicted_churn
        /
        total
        *
        100
        if total
        else 0
    )

    average_probability = float(
        prediction_df[
            probability_column
        ].mean()
    )

    average_risk = float(
        prediction_df[
            score_column
        ].mean()
    )

    risk_counts = (
        prediction_df[
            risk_column
        ]
        .value_counts()
        .to_dict()
    )

    return {
        "total_customers": total,

        "predicted_churn":
            predicted_churn,

        "predicted_retained":
            predicted_retained,

        "predicted_churn_rate":
            round(
                churn_rate,
                2,
            ),

        "average_risk_score":
            round(
                average_risk,
                2,
            ),

        "average_churn_probability":
            round(
                average_probability,
                2,
            ),

        "critical_risk":
            int(
                risk_counts.get(
                    "Critical",
                    0,
                )
            ),

        "high_risk":
            int(
                risk_counts.get(
                    "High",
                    0,
                )
            ),

        "medium_risk":
            int(
                risk_counts.get(
                    "Medium",
                    0,
                )
            ),

        "low_risk":
            int(
                risk_counts.get(
                    "Low",
                    0,
                )
            ),
    }


# ============================================================
# DATASET INTELLIGENCE
# ============================================================

def get_dataset_intelligence(
    df
):
    """
    Return information about what ChurnIQ detected.
    """

    profile = profile_dataset(
        df
    )

    return {
        "rows":
            profile["rows"],

        "columns":
            profile["columns"],

        "customer_id":
            profile["customer_id"],

        "churn_column":
            profile["churn_column"],

        "tenure_column":
            profile["tenure"],

        "contract_column":
            profile["contract"],

        "payment_column":
            profile["payment"],

        "monthly_value_column":
            profile["monthly_value"],

        "support_column":
            profile["support"],

        "usage_column":
            profile["usage"],

        "engagement_column":
            profile["engagement"],

        "detected_numeric_columns":
            len(
                profile[
                    "numeric_columns"
                ]
            ),

        "detected_categorical_columns":
            len(
                profile[
                    "categorical_columns"
                ]
            ),

        "actual_churn_available":
            bool(
                profile[
                    "churn_column"
                ]
            ),
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print(
        "ChurnIQ - Universal Prediction Pipeline Test"
    )
    print("=" * 70)

    dataset_path = (
        BASE_DIR
        / "Data"
        / "telco_churn_clean.csv"
    )

    if not dataset_path.exists():

        print(
            "Dataset not found:"
        )

        print(
            dataset_path
        )

    else:

        try:

            df = pd.read_csv(
                dataset_path
            )

            print()
            print(
                f"Dataset: {len(df):,} rows"
            )

            intelligence = (
                get_dataset_intelligence(
                    df
                )
            )

            print()
            print(
                "Detected dataset intelligence:"
            )

            for key, value in (
                intelligence.items()
            ):

                print(
                    f"  {key}: {value}"
                )

            result = predict_customers(
                df
            )

            summary = (
                get_prediction_summary(
                    result
                )
            )

            print()
            print(
                "Prediction method:"
            )

            print(
                result.attrs.get(
                    "churniq_method"
                )
            )

            print()
            print(
                "SUMMARY"
            )

            print(
                summary
            )

            print()
            print(
                result[
                    [
                        "Churn Prediction",
                        "Churn Probability",
                        "Risk Score",
                        "Risk Level",
                    ]
                ]
                .head(10)
                .to_string(
                    index=False
                )
            )

            print()
            print(
                "UNIVERSAL PREDICTION TEST PASSED"
            )

        except Exception as error:

            print()
            print(
                "PREDICTION PIPELINE FAILED"
            )

            print(
                f"{type(error).__name__}: "
                f"{error}"
            )