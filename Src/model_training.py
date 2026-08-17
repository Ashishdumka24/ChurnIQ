"""
ChurnIQ - Machine Learning Model Training
Phase 5

Trains and compares multiple classification models.

Models:
- Logistic Regression
- Decision Tree
- Random Forest
- Extra Trees
- Gradient Boosting
- HistGradientBoosting

The preprocessing pipeline is fitted ONLY on the training data
and saved separately for use by the prediction pipeline.
"""

from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
)

from sklearn.pipeline import Pipeline

from feature_engineering import (
    prepare_training_data,
    fit_preprocessor,
    save_preprocessor,
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path(
    "Data/telco_churn_clean.csv"
)

MODEL_DIR = Path(
    "Models"
)

BEST_MODEL_PATH = (
    MODEL_DIR / "best_churn_model.pkl"
)

PREPROCESSOR_PATH = (
    MODEL_DIR / "preprocessing.pkl"
)

RESULTS_PATH = (
    MODEL_DIR / "model_results.csv"
)

RANDOM_STATE = 42

TEST_SIZE = 0.20


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def get_models():
    """
    Return all candidate classification models.
    """

    models = {

        "Logistic Regression":
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),

        "Decision Tree":
            DecisionTreeClassifier(
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),

        "Extra Trees":
            ExtraTreesClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),

        "Gradient Boosting":
            GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                min_samples_split=5,
                min_samples_leaf=3,
                random_state=RANDOM_STATE,
            ),

        "HistGradient Boosting":
            HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.05,
                max_leaf_nodes=15,
                min_samples_leaf=10,
                l2_regularization=1.0,
                random_state=RANDOM_STATE,
            ),
    }

    return models


# ============================================================
# DATA CONVERSION
# ============================================================

def make_dense_if_required(matrix):
    """
    Convert sparse matrices to dense arrays when required
    by estimators such as HistGradientBoosting.
    """

    if hasattr(matrix, "toarray"):
        return matrix.toarray()

    return matrix


# ============================================================
# MODEL EVALUATION
# ============================================================

def calculate_metrics(
    model,
    X_test,
    y_test,
):
    """
    Calculate all important classification metrics.
    """

    predictions = model.predict(
        X_test
    )

    # Probability / decision score
    probabilities = None

    if hasattr(
        model,
        "predict_proba"
    ):
        try:
            probabilities = (
                model.predict_proba(
                    X_test
                )[:, 1]
            )
        except Exception:
            probabilities = None

    if probabilities is None and hasattr(
        model,
        "decision_function"
    ):
        try:
            probabilities = (
                model.decision_function(
                    X_test
                )
            )
        except Exception:
            probabilities = None

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    if probabilities is not None:

        try:
            roc_auc = roc_auc_score(
                y_test,
                probabilities,
            )
        except Exception:
            roc_auc = 0.0

    else:
        roc_auc = 0.0

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    )

    tn, fp, fn, tp = matrix.ravel()

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


# ============================================================
# CROSS VALIDATION
# ============================================================

def calculate_cross_validation(
    model,
    X,
    y,
):
    """
    Perform stratified cross-validation.

    Uses F1 as the primary CV metric because churn
    datasets are frequently imbalanced.
    """

    class_counts = y.value_counts()

    if class_counts.empty:
        return 0.0, 0.0

    smallest_class = int(
        class_counts.min()
    )

    # Cannot perform meaningful CV with only one
    # example in a class.
    if smallest_class < 2:
        return 0.0, 0.0

    folds = min(
        5,
        smallest_class,
    )

    if folds < 2:
        return 0.0, 0.0

    cv = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    try:

        scores = cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring="f1",
            n_jobs=-1,
        )

        return (
            float(scores.mean()),
            float(scores.std()),
        )

    except Exception:

        return 0.0, 0.0


# ============================================================
# MODEL PIPELINE
# ============================================================

def build_model_pipeline(
    preprocessor,
    model,
):
    """
    Combine preprocessing and model into ONE pipeline.

    This guarantees that training and inference use
    exactly the same transformations.
    """

    return Pipeline(
        steps=[
            (
                "preprocessor",
                clone(preprocessor),
            ),
            (
                "model",
                model,
            ),
        ]
    )


# ============================================================
# MODEL TRAINING
# ============================================================

def train_models(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor,
):
    """
    Train every candidate model and compare results.
    """

    models = get_models()

    results = []

    trained_models = {}

    for model_name, model in models.items():

        print()
        print(
            f"Training: {model_name}"
        )

        pipeline = build_model_pipeline(
            preprocessor,
            model,
        )

        try:

            # Fit pipeline on training data only.
            pipeline.fit(
                X_train,
                y_train,
            )

            metrics = calculate_metrics(
                pipeline,
                X_test,
                y_test,
            )

            cv_mean, cv_std = (
                calculate_cross_validation(
                    pipeline,
                    X_train,
                    y_train,
                )
            )

            result = {
                "Model": model_name,
                "Accuracy": round(
                    metrics["accuracy"],
                    4,
                ),
                "Precision": round(
                    metrics["precision"],
                    4,
                ),
                "Recall": round(
                    metrics["recall"],
                    4,
                ),
                "F1 Score": round(
                    metrics["f1_score"],
                    4,
                ),
                "ROC-AUC": round(
                    metrics["roc_auc"],
                    4,
                ),
                "CV F1 Mean": round(
                    cv_mean,
                    4,
                ),
                "CV F1 Std": round(
                    cv_std,
                    4,
                ),
                "True Negative": metrics[
                    "true_negative"
                ],
                "False Positive": metrics[
                    "false_positive"
                ],
                "False Negative": metrics[
                    "false_negative"
                ],
                "True Positive": metrics[
                    "true_positive"
                ],
                "Status": "Success",
            }

            results.append(
                result
            )

            trained_models[
                model_name
            ] = pipeline

            print(
                f"  Accuracy : "
                f"{metrics['accuracy']:.4f}"
            )

            print(
                f"  Precision: "
                f"{metrics['precision']:.4f}"
            )

            print(
                f"  Recall   : "
                f"{metrics['recall']:.4f}"
            )

            print(
                f"  F1 Score : "
                f"{metrics['f1_score']:.4f}"
            )

            print(
                f"  ROC-AUC  : "
                f"{metrics['roc_auc']:.4f}"
            )

            print(
                f"  CV F1    : "
                f"{cv_mean:.4f}"
            )

        except Exception as error:

            print(
                f"  FAILED: {error}"
            )

            results.append(
                {
                    "Model": model_name,
                    "Accuracy": 0.0,
                    "Precision": 0.0,
                    "Recall": 0.0,
                    "F1 Score": 0.0,
                    "ROC-AUC": 0.0,
                    "CV F1 Mean": 0.0,
                    "CV F1 Std": 0.0,
                    "True Negative": 0,
                    "False Positive": 0,
                    "False Negative": 0,
                    "True Positive": 0,
                    "Status": f"Failed: {error}",
                }
            )

    results_df = pd.DataFrame(
        results
    )

    return (
        results_df,
        trained_models,
    )


# ============================================================
# BEST MODEL SELECTION
# ============================================================

def select_best_model(
    results_df,
    trained_models,
):
    """
    Select the best model using a balanced scoring strategy.

    Priority:
    1. F1 Score
    2. ROC-AUC
    3. Recall
    4. Accuracy

    This is more appropriate for churn prediction than
    selecting solely by accuracy.
    """

    if results_df.empty:
        raise RuntimeError(
            "No model results were produced."
        )

    successful = results_df[
        results_df["Status"]
        == "Success"
    ].copy()

    if successful.empty:
        raise RuntimeError(
            "All model training attempts failed."
        )

    successful = successful.sort_values(
        by=[
            "F1 Score",
            "ROC-AUC",
            "Recall",
            "Accuracy",
        ],
        ascending=False,
    )

    best_name = successful.iloc[
        0
    ]["Model"]

    best_model = trained_models[
        best_name
    ]

    return (
        best_name,
        best_model,
        successful,
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_model_results(
    results_df,
    output_path=RESULTS_PATH,
):
    """Save model comparison results."""

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    return output_path


# ============================================================
# SAVE BEST MODEL
# ============================================================

def save_best_model(
    model,
    model_name,
    metadata,
    output_path=BEST_MODEL_PATH,
):
    """
    Save the complete model pipeline.

    The pipeline contains:
        preprocessing
        trained model

    Therefore prediction does not need to recreate
    preprocessing manually.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "model": model,
        "model_name": model_name,
        "metadata": metadata,
        "version": 1,
    }

    joblib.dump(
        payload,
        output_path,
    )

    return output_path


# ============================================================
# COMPLETE TRAINING WORKFLOW
# ============================================================

def run_training(
    data_path=DATA_PATH,
):
    """
    Complete Phase 5 training workflow.
    """

    data_path = Path(
        data_path
    )

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: "
            f"{data_path}"
        )

    print()
    print("=" * 65)
    print("ChurnIQ - Machine Learning Training")
    print("=" * 65)

    print()
    print(
        f"Loading dataset: {data_path}"
    )

    df = pd.read_csv(
        data_path
    )

    if df.empty:
        raise ValueError(
            "The dataset is empty."
        )

    print(
        f"Dataset shape: "
        f"{df.shape[0]} rows × "
        f"{df.shape[1]} columns"
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
        metadata,
    ) = prepare_training_data(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print()
    print("Target column:")
    print(
        f"  {metadata['target_column']}"
    )

    print()
    print(
        f"Training rows: "
        f"{len(X_train)}"
    )

    print(
        f"Testing rows: "
        f"{len(X_test)}"
    )

    print()
    print(
        "Training class distribution:"
    )

    print(
        y_train.value_counts()
        .sort_index()
        .to_dict()
    )

    # Save the standalone preprocessing object.
    #
    # This is fitted only on X_train.
    fitted_preprocessor, _ = (
        fit_preprocessor(
            X_train,
            preprocessor,
        )
    )

    save_preprocessor(
        fitted_preprocessor,
        PREPROCESSOR_PATH,
        metadata,
    )

    print()
    print(
        "Preprocessor saved:"
    )
    print(
        f"  {PREPROCESSOR_PATH}"
    )

    # Train models.
    results_df, trained_models = (
        train_models(
            X_train,
            X_test,
            y_train,
            y_test,
            preprocessor,
        )
    )

    # Save comparison.
    save_model_results(
        results_df,
        RESULTS_PATH,
    )

    print()
    print(
        "Model comparison saved:"
    )
    print(
        f"  {RESULTS_PATH}"
    )

    # Select best.
    (
        best_name,
        best_model,
        successful_results,
    ) = select_best_model(
        results_df,
        trained_models,
    )

    # Add final metadata.
    metadata = dict(metadata)

    metadata[
        "best_model"
    ] = best_name

    metadata[
        "selection_metric"
    ] = (
        "F1 Score → ROC-AUC → Recall → Accuracy"
    )

    metadata[
        "test_metrics"
    ] = (
        successful_results
        .loc[
            successful_results["Model"]
            == best_name
        ]
        .iloc[0]
        .to_dict()
    )

    save_best_model(
        best_model,
        best_name,
        metadata,
        BEST_MODEL_PATH,
    )

    print()
    print("=" * 65)
    print("MODEL COMPARISON")
    print("=" * 65)

    display_columns = [
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC",
        "CV F1 Mean",
        "Status",
    ]

    print(
        results_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 65)
    print(
        f"BEST MODEL: {best_name}"
    )
    print("=" * 65)

    best_row = successful_results.iloc[
        0
    ]

    print(
        f"F1 Score : "
        f"{best_row['F1 Score']:.4f}"
    )

    print(
        f"ROC-AUC  : "
        f"{best_row['ROC-AUC']:.4f}"
    )

    print(
        f"Recall   : "
        f"{best_row['Recall']:.4f}"
    )

    print(
        f"Accuracy : "
        f"{best_row['Accuracy']:.4f}"
    )

    print()
    print("Saved files:")
    print(
        f"  ✓ {BEST_MODEL_PATH}"
    )
    print(
        f"  ✓ {PREPROCESSOR_PATH}"
    )
    print(
        f"  ✓ {RESULTS_PATH}"
    )

    print()
    print(
        "Phase 5 machine-learning training "
        "completed successfully."
    )

    return {
        "best_model": best_model,
        "best_model_name": best_name,
        "results": results_df,
        "metadata": metadata,
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    try:

        run_training()

    except Exception as error:

        print()
        print("=" * 65)
        print("MODEL TRAINING FAILED")
        print("=" * 65)
        print(
            f"{type(error).__name__}: "
            f"{error}"
        )
        print()
        print(
            "Fix the reported error before "
            "moving to Phase 6."
        )