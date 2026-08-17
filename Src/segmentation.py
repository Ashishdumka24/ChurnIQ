"""
============================================================
ChurnIQ
Advanced Customer Intelligence & Risk Segmentation
Phase 12
============================================================
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ============================================================
# COLUMN HELPERS
# ============================================================

def _find_column(
    df: pd.DataFrame,
    names: List[str],
) -> Optional[str]:
    """Find a dataframe column using normalized names."""

    if df is None or df.empty:
        return None

    normalized = {
        str(col)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", ""): col
        for col in df.columns
    }

    for name in names:
        key = (
            str(name)
            .strip()
            .lower()
            .replace(" ", "")
            .replace("_", "")
            .replace("-", "")
        )

        if key in normalized:
            return normalized[key]

    return None


def _numeric_series(
    df: pd.DataFrame,
    column: Optional[str],
) -> pd.Series:
    """Safely convert a column to numeric."""

    if column is None:
        return pd.Series(
            np.nan,
            index=df.index,
            dtype=float,
        )

    return pd.to_numeric(
        df[column],
        errors="coerce",
    )


# ============================================================
# PROBABILITY NORMALIZATION
# ============================================================

def _normalize_probability(
    series: pd.Series,
) -> pd.Series:
    """
    Normalize churn probability to percentage scale 0-100.
    Supports values such as:
        0.82
        82
        82%
    """

    values = (
        series.astype(str)
        .str.replace("%", "", regex=False)
        .str.strip()
    )

    values = pd.to_numeric(
        values,
        errors="coerce",
    )

    valid = values.dropna()

    if not valid.empty:
        # If probability appears to be 0-1.
        if valid.max() <= 1.0:
            values = values * 100

    return values.clip(
        lower=0,
        upper=100,
    )


# ============================================================
# RISK SCORE
# ============================================================

def calculate_risk_score(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Calculate a standardized customer risk score from 0-100.

    Main signal:
        Churn Probability

    Additional signals:
        Tenure
        Monthly Charges
        Contract
        Payment Method
        Tech Support
    """

    score = pd.Series(
        0.0,
        index=df.index,
    )

    # --------------------------------------------------------
    # Churn probability
    # --------------------------------------------------------

    probability_col = _find_column(
        df,
        [
            "Churn Probability",
            "ChurnProbability",
            "Probability",
            "Churn Prob",
        ],
    )

    if probability_col:

        probability = _normalize_probability(
            df[probability_col]
        )

        score += (
            probability
            .fillna(0)
            .clip(0, 100)
            * 0.70
        )

    # --------------------------------------------------------
    # Tenure risk
    # --------------------------------------------------------

    tenure_col = _find_column(
        df,
        [
            "tenure",
            "Tenure",
            "Tenure Months",
        ],
    )

    if tenure_col:

        tenure = _numeric_series(
            df,
            tenure_col,
        )

        tenure_risk = (
            100
            - (
                tenure.clip(
                    lower=0,
                    upper=72,
                )
                / 72
                * 100
            )
        )

        score += (
            tenure_risk
            .fillna(50)
            * 0.10
        )

    # --------------------------------------------------------
    # Monthly charges
    # --------------------------------------------------------

    monthly_col = _find_column(
        df,
        [
            "MonthlyCharges",
            "Monthly Charges",
            "MonthlyCharge",
        ],
    )

    if monthly_col:

        monthly = _numeric_series(
            df,
            monthly_col,
        )

        if monthly.notna().any():

            min_value = monthly.min()
            max_value = monthly.max()

            if max_value > min_value:

                charge_risk = (
                    (
                        monthly
                        - min_value
                    )
                    /
                    (
                        max_value
                        - min_value
                    )
                    * 100
                )

                score += (
                    charge_risk
                    .fillna(50)
                    * 0.08
                )

    # --------------------------------------------------------
    # Contract risk
    # --------------------------------------------------------

    contract_col = _find_column(
        df,
        [
            "Contract",
        ],
    )

    if contract_col:

        contract = (
            df[contract_col]
            .astype(str)
            .str.lower()
        )

        contract_risk = pd.Series(
            50.0,
            index=df.index,
        )

        contract_risk.loc[
            contract.str.contains(
                "month",
                na=False,
            )
        ] = 100

        contract_risk.loc[
            contract.str.contains(
                "one year|one-year|1 year",
                na=False,
            )
        ] = 45

        contract_risk.loc[
            contract.str.contains(
                "two year|two-year|2 year",
                na=False,
            )
        ] = 15

        score += (
            contract_risk
            * 0.07
        )

    # --------------------------------------------------------
    # Payment method risk
    # --------------------------------------------------------

    payment_col = _find_column(
        df,
        [
            "PaymentMethod",
            "Payment Method",
        ],
    )

    if payment_col:

        payment = (
            df[payment_col]
            .astype(str)
            .str.lower()
        )

        payment_risk = pd.Series(
            40.0,
            index=df.index,
        )

        payment_risk.loc[
            payment.str.contains(
                "electronic check",
                na=False,
            )
        ] = 100

        score += (
            payment_risk
            * 0.05
        )

    # --------------------------------------------------------
    # Final normalization
    # --------------------------------------------------------

    return score.clip(
        lower=0,
        upper=100,
    ).round(2)


# ============================================================
# RISK LEVEL
# ============================================================

def assign_risk_level(
    score: pd.Series,
) -> pd.Series:
    """Convert numerical risk score into business categories."""

    return pd.cut(
        score,
        bins=[
            -np.inf,
            29.99,
            59.99,
            79.99,
            np.inf,
        ],
        labels=[
            "Low",
            "Medium",
            "High",
            "Critical",
        ],
    ).astype(str)


# ============================================================
# CUSTOMER VALUE
# ============================================================

def calculate_customer_value(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Estimate customer value using available revenue fields.

    Priority:
        Total Charges
        Monthly Charges
    """

    total_col = _find_column(
        df,
        [
            "TotalCharges",
            "Total Charges",
            "TotalCharge",
        ],
    )

    monthly_col = _find_column(
        df,
        [
            "MonthlyCharges",
            "Monthly Charges",
        ],
    )

    if total_col:

        total = _numeric_series(
            df,
            total_col,
        ).fillna(0)

        return total.round(2)

    if monthly_col:

        monthly = _numeric_series(
            df,
            monthly_col,
        ).fillna(0)

        tenure_col = _find_column(
            df,
            [
                "tenure",
                "Tenure",
                "Tenure Months",
            ],
        )

        if tenure_col:

            tenure = _numeric_series(
                df,
                tenure_col,
            ).fillna(0)

            return (
                monthly
                * tenure
            ).round(2)

        return monthly.round(2)

    return pd.Series(
        0.0,
        index=df.index,
    )


# ============================================================
# STRATEGIC SEGMENT
# ============================================================

def assign_strategic_segment(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Create actionable business segments.

    Segments:
        Critical Retention
        High Value At Risk
        Growth Opportunity
        Stable Valuable
        Loyal Customer
        Standard Customer
    """

    risk_score = pd.to_numeric(
        df["Customer Risk Score"],
        errors="coerce",
    ).fillna(0)

    probability = pd.to_numeric(
        df.get(
            "Churn Probability",
            pd.Series(
                0,
                index=df.index,
            ),
        ),
        errors="coerce",
    ).fillna(0)

    customer_value = pd.to_numeric(
        df.get(
            "Customer Value",
            pd.Series(
                0,
                index=df.index,
            ),
        ),
        errors="coerce",
    ).fillna(0)

    if customer_value.max() > customer_value.min():

        value_percentile = (
            customer_value
            .rank(
                pct=True,
            )
            * 100
        )

    else:

        value_percentile = pd.Series(
            50,
            index=df.index,
        )

    segment = pd.Series(
        "Standard Customer",
        index=df.index,
        dtype="object",
    )

    # Critical customers.
    segment.loc[
        risk_score >= 80
    ] = "Critical Retention"

    # High-value customers at risk.
    segment.loc[
        (
            (risk_score >= 60)
            &
            (risk_score < 80)
            &
            (value_percentile >= 70)
        )
    ] = "High Value At Risk"

    # Growth opportunity.
    segment.loc[
        (
            (risk_score < 60)
            &
            (probability < 40)
            &
            (value_percentile >= 75)
        )
    ] = "Growth Opportunity"

    # Stable valuable.
    segment.loc[
        (
            (risk_score < 40)
            &
            (value_percentile >= 70)
        )
    ] = "Stable Valuable"

    # Loyal customers.
    segment.loc[
        (
            (risk_score < 30)
            &
            (probability < 30)
        )
    ] = "Loyal Customer"

    return segment


# ============================================================
# CUSTOMER INTELLIGENCE
# ============================================================

def create_customer_intelligence(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the complete customer intelligence layer.

    Returns the original dataframe plus:
        Customer Risk Score
        Risk Level
        Customer Value
        Strategic Segment
        Priority Rank
        Retention Priority
    """

    if df is None:
        return pd.DataFrame()

    if not isinstance(
        df,
        pd.DataFrame,
    ):
        raise TypeError(
            "create_customer_intelligence expects a DataFrame."
        )

    if df.empty:
        return df.copy()

    result = df.copy()

    # --------------------------------------------------------
    # Normalize probability
    # --------------------------------------------------------

    probability_col = _find_column(
        result,
        [
            "Churn Probability",
            "ChurnProbability",
            "Probability",
        ],
    )

    if probability_col:

        result["Churn Probability"] = (
            _normalize_probability(
                result[probability_col]
            )
            .fillna(0)
            .round(2)
        )

    else:

        result["Churn Probability"] = 0.0

    # --------------------------------------------------------
    # Risk score
    # --------------------------------------------------------

    result[
        "Customer Risk Score"
    ] = calculate_risk_score(
        result
    )

    # --------------------------------------------------------
    # Risk level
    # --------------------------------------------------------

    result[
        "Risk Level"
    ] = assign_risk_level(
        result[
            "Customer Risk Score"
        ]
    )

    # --------------------------------------------------------
    # Customer value
    # --------------------------------------------------------

    result[
        "Customer Value"
    ] = calculate_customer_value(
        result
    )

    # --------------------------------------------------------
    # Strategic segment
    # --------------------------------------------------------

    result[
        "Strategic Segment"
    ] = assign_strategic_segment(
        result
    )

    # --------------------------------------------------------
    # Retention priority
    # --------------------------------------------------------

    result[
        "Retention Priority"
    ] = np.select(
        [
            result[
                "Customer Risk Score"
            ] >= 80,

            result[
                "Customer Risk Score"
            ] >= 60,

            result[
                "Customer Risk Score"
            ] >= 40,
        ],
        [
            "Immediate",
            "High",
            "Monitor",
        ],
        default="Low",
    )

    # --------------------------------------------------------
    # Priority rank
    # --------------------------------------------------------

    result[
        "Priority Rank"
    ] = (
        result[
            "Customer Risk Score"
        ]
        .rank(
            method="dense",
            ascending=False,
        )
        .astype(int)
    )

    # --------------------------------------------------------
    # Estimated revenue at risk
    # --------------------------------------------------------

    result[
        "Revenue At Risk"
    ] = (
        result["Customer Value"]
        *
        result["Churn Probability"]
        / 100
    ).round(2)

    return result


# ============================================================
# SEGMENT SUMMARY
# ============================================================

def get_segment_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Return strategic segment statistics."""

    if df is None or df.empty:
        return pd.DataFrame()

    segment_col = _find_column(
        df,
        [
            "Strategic Segment",
            "Customer Segment",
        ],
    )

    if segment_col is None:
        return pd.DataFrame()

    grouped = (
        df.groupby(
            segment_col,
            dropna=False,
        )
        .agg(
            Customers=(
                segment_col,
                "size",
            ),
            Avg_Risk_Score=(
                "Customer Risk Score",
                "mean",
            ),
            Avg_Churn_Probability=(
                "Churn Probability",
                "mean",
            ),
            Customer_Value=(
                "Customer Value",
                "sum",
            ),
            Revenue_At_Risk=(
                "Revenue At Risk",
                "sum",
            ),
        )
        .reset_index()
    )

    grouped = grouped.rename(
        columns={
            segment_col: "Segment",
        }
    )

    grouped[
        "Avg_Risk_Score"
    ] = grouped[
        "Avg_Risk_Score"
    ].round(2)

    grouped[
        "Avg_Churn_Probability"
    ] = grouped[
        "Avg_Churn_Probability"
    ].round(2)

    grouped[
        "Customer_Value"
    ] = grouped[
        "Customer_Value"
    ].round(2)

    grouped[
        "Revenue_At_Risk"
    ] = grouped[
        "Revenue_At_Risk"
    ].round(2)

    return grouped.sort_values(
        "Avg_Risk_Score",
        ascending=False,
    ).reset_index(
        drop=True
    )


# ============================================================
# HIGH-RISK CUSTOMERS
# ============================================================

def get_high_risk_customers(
    df: pd.DataFrame,
    threshold: float = 60,
) -> pd.DataFrame:
    """Return customers above the requested risk threshold."""

    if df is None or df.empty:
        return pd.DataFrame()

    score_col = _find_column(
        df,
        [
            "Customer Risk Score",
            "Risk Score",
        ],
    )

    if score_col is None:
        return pd.DataFrame()

    scores = pd.to_numeric(
        df[score_col],
        errors="coerce",
    )

    high_risk = df.loc[
        scores >= threshold
    ].copy()

    return high_risk.sort_values(
        score_col,
        ascending=False,
    ).reset_index(
        drop=True
    )


# ============================================================
# RISK SUMMARY
# ============================================================

def get_risk_summary(
    df: pd.DataFrame,
) -> Dict[str, float]:
    """Return high-level risk statistics."""

    if df is None or df.empty:
        return {
            "total_customers": 0,
            "low_risk": 0,
            "medium_risk": 0,
            "high_risk": 0,
            "critical_risk": 0,
            "average_risk_score": 0.0,
            "average_churn_probability": 0.0,
            "revenue_at_risk": 0.0,
        }

    risk_col = _find_column(
        df,
        [
            "Risk Level",
            "Customer Risk Segment",
        ],
    )

    score_col = _find_column(
        df,
        [
            "Customer Risk Score",
            "Risk Score",
        ],
    )

    probability_col = _find_column(
        df,
        [
            "Churn Probability",
        ],
    )

    revenue_col = _find_column(
        df,
        [
            "Revenue At Risk",
        ],
    )

    if risk_col:

        risks = (
            df[risk_col]
            .astype(str)
            .str.lower()
        )

        low = int(
            risks.eq("low").sum()
        )

        medium = int(
            risks.eq("medium").sum()
        )

        high = int(
            risks.eq("high").sum()
        )

        critical = int(
            risks.eq("critical").sum()
        )

    else:

        low = medium = high = critical = 0

    return {
        "total_customers": len(df),

        "low_risk": low,

        "medium_risk": medium,

        "high_risk": high,

        "critical_risk": critical,

        "average_risk_score": (
            float(
                pd.to_numeric(
                    df[score_col],
                    errors="coerce",
                ).mean()
            )
            if score_col
            else 0.0
        ),

        "average_churn_probability": (
            float(
                pd.to_numeric(
                    df[probability_col],
                    errors="coerce",
                ).mean()
            )
            if probability_col
            else 0.0
        ),

        "revenue_at_risk": (
            float(
                pd.to_numeric(
                    df[revenue_col],
                    errors="coerce",
                ).sum()
            )
            if revenue_col
            else 0.0
        ),
    }


# ============================================================
# MODULE TEST
# ============================================================

if __name__ == "__main__":

    demo = pd.DataFrame(
        {
            "customerID": [
                "C001",
                "C002",
                "C003",
                "C004",
            ],
            "Churn Probability": [
                0.91,
                0.72,
                0.24,
                0.08,
            ],
            "tenure": [
                2,
                8,
                48,
                60,
            ],
            "MonthlyCharges": [
                90,
                85,
                60,
                45,
            ],
            "TotalCharges": [
                180,
                680,
                2880,
                2700,
            ],
            "Contract": [
                "Month-to-month",
                "Month-to-month",
                "One year",
                "Two year",
            ],
        }
    )

    output = create_customer_intelligence(
        demo
    )

    print("\nCustomer Intelligence:")
    print(
        output[
            [
                "customerID",
                "Churn Probability",
                "Customer Risk Score",
                "Risk Level",
                "Customer Value",
                "Strategic Segment",
                "Retention Priority",
                "Revenue At Risk",
            ]
        ]
    )

    print("\nSegment Summary:")
    print(
        get_segment_summary(
            output
        )
    )

    print("\nRisk Summary:")
    print(
        get_risk_summary(
            output
        )
    )

    print("\nHigh Risk Customers:")
    print(
        get_high_risk_customers(
            output
        )
    )