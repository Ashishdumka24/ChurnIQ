"""
ChurnIQ - Feature Engineering
Phase 4

Purpose:
- Detect churn target automatically
- Separate features and target
- Handle numeric and categorical features
- Handle missing values
- Encode categorical variables safely
- Scale numerical variables
- Prevent data leakage
- Create a reusable preprocessing pipeline
- Support future CSV/XLS/XLSX uploads
- Keep preprocessing compatible with model training
  and prediction
"""

from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

from dataset_validator import (
    detect_columns,
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_MODEL_DIR = Path("Models")
DEFAULT_PREPROCESSOR_PATH = (
    DEFAULT_MODEL_DIR / "preprocessing.pkl"
)

DEFAULT_TEST_SIZE = 0.20
DEFAULT_RANDOM_STATE = 42


# ============================================================
# TARGET DETECTION
# ============================================================

def find_churn_column(df):
    """
    Detect the churn/attrition target column.

    Returns
    -------
    str or None
    """

    if df is None or df.empty:
        return None

    detected = detect_columns(df)

    return detected.get("churn")


# ============================================================
# TARGET CONVERSION
# ============================================================

def convert_target_to_binary(df, target_column):
    """
    Convert common churn labels into binary values.

    Churn = 1
    Retained = 0

    Unknown target values become NaN.
    """

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' "
            "does not exist."
        )

    target = df[target_column].copy()

    # Numeric binary target
    if pd.api.types.is_numeric_dtype(target):

        numeric = pd.to_numeric(
            target,
            errors="coerce",
        )

        unique_values = set(
            numeric.dropna().unique()
        )

        if unique_values.issubset({0, 1}):
            return numeric.astype("float")

    normalized = (
        target.astype("string")
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
        "attrition",
        "attrited",
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
        "stay",
        "not churned",
        "not_churned",
    }

    result = normalized.map(
        lambda value:
        1 if value in positive_values
        else 0 if value in negative_values
        else np.nan
    )

    return result.astype("float")


# ============================================================
# DATA TYPE PREPARATION
# ============================================================

def prepare_feature_types(df):
    """
    Safely convert columns that are clearly numeric.

    This is intentionally conservative so that categorical
    information is not accidentally destroyed.
    """

    result = df.copy()

    for column in result.columns:

        if result[column].dtype not in [
            "object",
            "string",
            "category",
        ]:
            continue

        series = (
            result[column]
            .astype("string")
            .str.strip()
        )

        cleaned = (
            series
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.replace("₹", "", regex=False)
            .str.replace("€", "", regex=False)
            .str.replace("£", "", regex=False)
        )

        numeric = pd.to_numeric(
            cleaned,
            errors="coerce",
        )

        non_null = series.notna().sum()

        if non_null == 0:
            continue

        numeric_ratio = (
            numeric.notna().sum() / non_null
        )

        # Only convert when the column is overwhelmingly numeric.
        if numeric_ratio >= 0.90:
            result[column] = numeric

    return result


# ============================================================
# REMOVE USELESS FEATURES
# ============================================================

def identify_useless_columns(
    df,
    target_column,
):
    """
    Identify columns that should not be used for ML.

    Removes:
    - Completely empty columns
    - Constant columns
    - Columns containing the target
    - Obvious raw index columns
    """

    useless = []

    for column in df.columns:

        if column == target_column:
            continue

        # Completely empty
        if df[column].isna().all():
            useless.append(column)
            continue

        # Constant
        if df[column].nunique(
            dropna=False
        ) <= 1:
            useless.append(column)
            continue

        normalized = (
            str(column)
            .strip()
            .lower()
            .replace(" ", "")
            .replace("_", "")
            .replace("-", "")
        )

        # Raw dataframe/index artifacts
        if normalized in {
            "unnamed0",
            "index",
            "rowindex",
            "rownumber",
        }:
            useless.append(column)

    return sorted(
        set(useless)
    )


# ============================================================
# IDENTIFIER HANDLING
# ============================================================

def identify_identifier_columns(
    df,
    target_column,
):
    """
    Identify likely customer identifiers.

    Customer IDs should normally not be used as predictive
    features because they represent identity rather than
    customer behavior.
    """

    identifiers = []

    detected = detect_columns(df)

    customer_id = detected.get(
        "customer_id"
    )

    if (
        customer_id
        and customer_id in df.columns
        and customer_id != target_column
    ):
        identifiers.append(customer_id)

    for column in df.columns:

        if column == target_column:
            continue

        normalized = (
            str(column)
            .strip()
            .lower()
            .replace(" ", "")
            .replace("_", "")
            .replace("-", "")
        )

        if any(
            keyword in normalized
            for keyword in [
                "customerid",
                "clientid",
                "accountid",
                "accountnumber",
                "customerkey",
            ]
        ):
            identifiers.append(column)

    return sorted(
        set(identifiers)
    )


# ============================================================
# FEATURE SELECTION
# ============================================================

def select_features(
    df,
    target_column,
    include_identifiers=False,
):
    """
    Select ML features while preserving useful customer
    behavior information.
    """

    if target_column not in df.columns:
        raise ValueError(
            "Target column not found."
        )

    features = df.drop(
        columns=[target_column]
    ).copy()

    useless = identify_useless_columns(
        df,
        target_column,
    )

    if useless:
        features = features.drop(
            columns=useless,
            errors="ignore",
        )

    if not include_identifiers:

        identifiers = identify_identifier_columns(
            df,
            target_column,
        )

        if identifiers:
            features = features.drop(
                columns=identifiers,
                errors="ignore",
            )

    if features.shape[1] == 0:
        raise ValueError(
            "No usable features were found "
            "after feature selection."
        )

    return features


# ============================================================
# PREPROCESSOR
# ============================================================

def build_preprocessor(X):
    """
    Build the preprocessing pipeline.

    Numeric:
        median imputation
        standard scaling

    Categorical:
        most-frequent imputation
        one-hot encoding
    """

    numeric_features = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=[
            "object",
            "category",
            "string",
            "bool",
        ]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    # Handle sklearn versions that support
    # sparse_output and older versions that use sparse.
    try:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True,
        )
    except TypeError:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse=True,
        )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                encoder,
            ),
        ]
    )

    transformers = []

    if numeric_features:
        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            )
        )

    if categorical_features:
        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            )
        )

    if not transformers:
        raise ValueError(
            "No numerical or categorical features "
            "were detected."
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    return preprocessor


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_training_data(
    df,
    test_size=DEFAULT_TEST_SIZE,
    random_state=DEFAULT_RANDOM_STATE,
    include_identifiers=False,
):
    """
    Prepare raw/cleaned data for model training.

    Returns:
        X_train
        X_test
        y_train
        y_test
        preprocessor
        metadata
    """

    if df is None:
        raise ValueError(
            "Training dataframe cannot be None."
        )

    if df.empty:
        raise ValueError(
            "Training dataframe is empty."
        )

    working = df.copy()

    # Safely prepare numeric-looking fields.
    working = prepare_feature_types(
        working
    )

    target_column = find_churn_column(
        working
    )

    if not target_column:
        raise ValueError(
            "Could not detect a churn/attrition "
            "target column."
        )

    # Convert target.
    target = convert_target_to_binary(
        working,
        target_column,
    )

    # Remove rows with unknown target labels.
    valid_target = target.notna()

    working = working.loc[
        valid_target
    ].copy()

    target = target.loc[
        valid_target
    ].astype(int)

    if target.nunique() < 2:
        raise ValueError(
            "The churn target must contain at least "
            "two classes."
        )

    X = select_features(
        working,
        target_column,
        include_identifiers=include_identifiers,
    )

    # Reset indexes to prevent accidental alignment issues.
    X = X.reset_index(drop=True)
    y = target.reset_index(drop=True)

    # Verify class counts.
    class_counts = y.value_counts()

    if len(class_counts) < 2:
        raise ValueError(
            "Both churn and non-churn classes "
            "are required."
        )

    # Stratification is preferred, but very small classes
    # cannot always support it.
    min_class_count = int(
        class_counts.min()
    )

    stratify = (
        y
        if min_class_count >= 2
        else None
    )

    try:

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify,
            )
        )

    except ValueError:

        # Safe fallback for unusual/small datasets.
        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=None,
            )
        )

    preprocessor = build_preprocessor(
        X_train
    )

    metadata = {
        "target_column": target_column,
        "feature_columns": X.columns.tolist(),
        "numeric_features": (
            X.select_dtypes(
                include=np.number
            ).columns.tolist()
        ),
        "categorical_features": (
            X.select_dtypes(
                include=[
                    "object",
                    "category",
                    "string",
                    "bool",
                ]
            ).columns.tolist()
        ),
        "identifier_columns": (
            identify_identifier_columns(
                working,
                target_column,
            )
        ),
        "removed_columns": (
            identify_useless_columns(
                working,
                target_column,
            )
        ),
        "total_rows": int(len(working)),
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "class_distribution": {
            int(key): int(value)
            for key, value in y.value_counts().items()
        },
        "random_state": random_state,
        "test_size": test_size,
    }

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
        metadata,
    )


# ============================================================
# FIT AND TRANSFORM
# ============================================================

def fit_preprocessor(
    X_train,
    preprocessor=None,
):
    """Fit preprocessing only on training data."""

    if X_train is None or X_train.empty:
        raise ValueError(
            "Training features are empty."
        )

    if preprocessor is None:
        preprocessor = build_preprocessor(
            X_train
        )

    X_train_processed = (
        preprocessor.fit_transform(
            X_train
        )
    )

    return (
        preprocessor,
        X_train_processed,
    )


def transform_features(
    preprocessor,
    X,
):
    """
    Transform new data using an already-fitted
    preprocessing pipeline.
    """

    if preprocessor is None:
        raise ValueError(
            "A fitted preprocessor is required."
        )

    if X is None:
        raise ValueError(
            "Features cannot be None."
        )

    return preprocessor.transform(
        X
    )


# ============================================================
# FEATURE NAMES
# ============================================================

def get_transformed_feature_names(
    preprocessor,
):
    """
    Retrieve feature names after preprocessing.
    """

    if preprocessor is None:
        return []

    try:
        names = (
            preprocessor
            .get_feature_names_out()
        )

        return [
            str(name)
            for name in names
        ]

    except Exception:
        return []


# ============================================================
# SAVE / LOAD PREPROCESSOR
# ============================================================

def save_preprocessor(
    preprocessor,
    output_path=DEFAULT_PREPROCESSOR_PATH,
    metadata=None,
):
    """
    Save preprocessing pipeline and metadata together.

    This prevents later prediction code from using a
    different preprocessing configuration.
    """

    if preprocessor is None:
        raise ValueError(
            "Cannot save an empty preprocessor."
        )

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "preprocessor": preprocessor,
        "metadata": metadata or {},
        "version": 1,
    }

    joblib.dump(
        payload,
        output_path,
    )

    return output_path


def load_preprocessor(
    input_path=DEFAULT_PREPROCESSOR_PATH,
):
    """Load a previously saved preprocessing pipeline."""

    input_path = Path(
        input_path
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Preprocessor not found: "
            f"{input_path}"
        )

    payload = joblib.load(
        input_path
    )

    # New format
    if isinstance(payload, dict):

        if "preprocessor" in payload:
            return payload["preprocessor"]

    # Backward compatibility with a directly
    # serialized sklearn preprocessor.
    return payload


# ============================================================
# COMPLETE FEATURE ENGINEERING PIPELINE
# ============================================================

def run_feature_engineering(
    df,
    save_preprocessor_path=DEFAULT_PREPROCESSOR_PATH,
    test_size=DEFAULT_TEST_SIZE,
    random_state=DEFAULT_RANDOM_STATE,
):
    """
    Complete feature-engineering workflow.

    Important:
    The preprocessor is fitted ONLY on X_train.
    This prevents data leakage.
    """

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
        metadata,
    ) = prepare_training_data(
        df,
        test_size=test_size,
        random_state=random_state,
    )

    preprocessor, X_train_processed = (
        fit_preprocessor(
            X_train,
            preprocessor,
        )
    )

    X_test_processed = (
        transform_features(
            preprocessor,
            X_test,
        )
    )

    feature_names = (
        get_transformed_feature_names(
            preprocessor
        )
    )

    metadata[
        "transformed_feature_count"
    ] = int(
        len(feature_names)
    )

    metadata[
        "transformed_feature_names"
    ] = feature_names

    save_preprocessor(
        preprocessor,
        save_preprocessor_path,
        metadata,
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_processed": X_train_processed,
        "X_test_processed": X_test_processed,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "metadata": metadata,
    }


# ============================================================
# CREATE PROCESSED CSV
# ============================================================

def create_processed_dataset(
    df,
    output_path="Data/telco_churn_processed.csv",
):
    """
    Create a human-readable processed dataset.

    This file is for inspection/reporting.

    IMPORTANT:
    The actual model training should use the fitted
    preprocessing pipeline rather than this CSV.
    """

    if df is None or df.empty:
        raise ValueError(
            "Cannot process an empty dataset."
        )

    working = df.copy()

    working = prepare_feature_types(
        working
    )

    target_column = find_churn_column(
        working
    )

    if not target_column:
        raise ValueError(
            "Churn target could not be detected."
        )

    target = convert_target_to_binary(
        working,
        target_column,
    )

    valid = target.notna()

    working = working.loc[
        valid
    ].copy()

    target = target.loc[
        valid
    ].astype(int)

    working[target_column] = target

    features = select_features(
        working,
        target_column,
    )

    # Keep original target separately.
    processed = features.copy()

    processed[
        "Churn"
    ] = target.values

    # Human-readable one-hot encoding.
    categorical_columns = (
        processed
        .select_dtypes(
            include=[
                "object",
                "category",
                "string",
                "bool",
            ]
        )
        .columns
        .tolist()
    )

    if "Churn" in categorical_columns:
        categorical_columns.remove(
            "Churn"
        )

    if categorical_columns:

        processed = pd.get_dummies(
            processed,
            columns=categorical_columns,
            dtype=int,
        )

    # Fill numerical missing values.
    numeric_columns = (
        processed
        .select_dtypes(
            include=np.number
        )
        .columns
    )

    for column in numeric_columns:

        if column == "Churn":
            continue

        median = processed[
            column
        ].median()

        if pd.notna(median):
            processed[
                column
            ] = processed[
                column
            ].fillna(median)

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    processed.to_csv(
        output_path,
        index=False,
    )

    return processed


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 65)
    print("ChurnIQ - Feature Engineering Test")
    print("=" * 65)

    try:

        dataset_path = Path(
            "Data/telco_churn_clean.csv"
        )

        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: "
                f"{dataset_path}"
            )

        df = pd.read_csv(
            dataset_path
        )

        print(
            f"\nOriginal dataset: "
            f"{df.shape[0]} rows × "
            f"{df.shape[1]} columns"
        )

        result = run_feature_engineering(
            df
        )

        print("\nTARGET")
        print("-" * 40)

        print(
            result["metadata"][
                "target_column"
            ]
        )

        print("\nFEATURES")
        print("-" * 40)

        print(
            f"Original features: "
            f"{len(result['metadata']['feature_columns'])}"
        )

        print(
            f"Transformed features: "
            f"{result['metadata']['transformed_feature_count']}"
        )

        print("\nDATA SPLIT")
        print("-" * 40)

        print(
            f"Training rows: "
            f"{len(result['X_train'])}"
        )

        print(
            f"Testing rows: "
            f"{len(result['X_test'])}"
        )

        print("\nCLASS DISTRIBUTION")
        print("-" * 40)

        print(
            result["metadata"][
                "class_distribution"
            ]
        )

        print("\nPROCESSED DATASET")
        print("-" * 40)

        processed = create_processed_dataset(
            df
        )

        print(
            f"Saved processed dataset: "
            f"{processed.shape[0]} rows × "
            f"{processed.shape[1]} columns"
        )

        print("\nSUCCESS")
        print("-" * 40)

        print(
            "Feature engineering completed "
            "successfully."
        )

        print(
            "Preprocessor saved to: "
            "Models/preprocessing.pkl"
        )

        print(
            "Processed dataset saved to: "
            "Data/telco_churn_processed.csv"
        )

    except Exception as error:

        print("\nFEATURE ENGINEERING FAILED")
        print("-" * 40)
        print(
            f"{type(error).__name__}: "
            f"{error}"
        )