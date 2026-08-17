"""
============================================================
ChurnIQ
Advanced Business Recommendations & Retention Intelligence
Phase 12
============================================================
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ============================================================
# HELPERS
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


def _numeric(
    df: pd.DataFrame,
    column: Optional[str],
) -> pd.Series:

    if column is None:

        return pd.Series(
            0.0,
            index=df.index,
        )

    return pd.to_numeric(
        df[column],
        errors="coerce",
    ).fillna(0)


# ============================================================
# INDIVIDUAL CUSTOMER RECOMMENDATION
# ============================================================

def generate_customer_recommendation(
    row: pd.Series,
) -> Dict[str, str]:
    """Generate an actionable retention recommendation."""

    risk = str(
        row.get(
            "Risk Level",
            "",
        )
    ).lower()

    segment = str(
        row.get(
            "Strategic Segment",
            "",
        )
    ).lower()

    probability = pd.to_numeric(
        row.get(
            "Churn Probability",
            0,
        ),
        errors="coerce",
    )

    if pd.isna(probability):
        probability = 0

    contract = str(
        row.get(
            "Contract",
            "",
        )
    ).lower()

    payment = str(
        row.get(
            "PaymentMethod",
            row.get(
                "Payment Method",
                "",
            ),
        )
    ).lower()

    tenure = pd.to_numeric(
        row.get(
            "tenure",
            0,
        ),
        errors="coerce",
    )

    if pd.isna(tenure):
        tenure = 0

    # --------------------------------------------------------
    # Critical
    # --------------------------------------------------------

    if risk == "critical":

        if (
            "month" in contract
            and contract.strip()
        ):

            return {
                "Retention Priority": "Immediate",
                "Recommended Action":
                    "Offer a personalized contract upgrade or retention incentive.",
                "Retention Recommendation":
                    "Immediate intervention with a targeted retention offer.",
            }

        return {
            "Retention Priority": "Immediate",
            "Recommended Action":
                "Contact customer immediately with a personalized retention offer.",
            "Retention Recommendation":
                "High-touch retention intervention is recommended.",
        }

    # --------------------------------------------------------
    # High
    # --------------------------------------------------------

    if risk == "high":

        if (
            "electronic check"
            in payment
        ):

            return {
                "Retention Priority": "High",
                "Recommended Action":
                    "Review payment experience and offer an easier payment option.",
                "Retention Recommendation":
                    "Improve payment convenience and provide targeted engagement.",
            }

        if tenure <= 6:

            return {
                "Retention Priority": "High",
                "Recommended Action":
                    "Launch an early-tenure onboarding and engagement campaign.",
                "Retention Recommendation":
                    "Strengthen onboarding to prevent early customer churn.",
            }

        return {
            "Retention Priority": "High",
            "Recommended Action":
                "Provide personalized service outreach and retention incentives.",
            "Retention Recommendation":
                "Proactive customer engagement is recommended.",
        }

    # --------------------------------------------------------
    # Medium
    # --------------------------------------------------------

    if risk == "medium":

        if "growth" in segment:

            return {
                "Retention Priority": "Monitor",
                "Recommended Action":
                    "Introduce relevant upgrades and personalized offers.",
                "Retention Recommendation":
                    "Use engagement and cross-sell opportunities to strengthen loyalty.",
            }

        return {
            "Retention Priority": "Monitor",
            "Recommended Action":
                "Monitor behavior and provide proactive customer engagement.",
            "Retention Recommendation":
                "Customer should remain under regular retention monitoring.",
        }

    # --------------------------------------------------------
    # Low risk
    # --------------------------------------------------------

    if (
        "loyal" in segment
        or "stable" in segment
    ):

        return {
            "Retention Priority": "Low",
            "Recommended Action":
                "Maintain service quality and reward customer loyalty.",
            "Retention Recommendation":
                "Focus on loyalty, advocacy and long-term value.",
        }

    return {
        "Retention Priority": "Low",
        "Recommended Action":
            "Maintain engagement and monitor future churn signals.",
        "Retention Recommendation":
            "Continue standard customer engagement.",
    }


# ============================================================
# BULK RECOMMENDATIONS
# ============================================================

def create_recommendations(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add advanced retention recommendations to every customer.
    """

    if df is None:
        return pd.DataFrame()

    if not isinstance(
        df,
        pd.DataFrame,
    ):
        raise TypeError(
            "create_recommendations expects a DataFrame."
        )

    if df.empty:
        return df.copy()

    result = df.copy()

    recommendations = (
        result.apply(
            generate_customer_recommendation,
            axis=1,
            result_type="expand",
        )
    )

    for column in [
        "Retention Priority",
        "Recommended Action",
        "Retention Recommendation",
    ]:

        if column in recommendations.columns:

            result[column] = (
                recommendations[column]
            )

    # --------------------------------------------------------
    # Revenue at risk
    # --------------------------------------------------------

    if "Revenue At Risk" not in result.columns:

        value_col = _find_column(
            result,
            [
                "Customer Value",
                "TotalCharges",
                "Total Charges",
            ],
        )

        probability_col = _find_column(
            result,
            [
                "Churn Probability",
            ],
        )

        if (
            value_col
            and probability_col
        ):

            value = _numeric(
                result,
                value_col,
            )

            probability = _numeric(
                result,
                probability_col,
            )

            if probability.max() <= 1:
                probability = (
                    probability
                    * 100
                )

            result[
                "Revenue At Risk"
            ] = (
                value
                * probability
                / 100
            ).round(2)

        else:

            result[
                "Revenue At Risk"
            ] = 0.0

    # --------------------------------------------------------
    # Retention urgency score
    # --------------------------------------------------------

    score_col = _find_column(
        result,
        [
            "Customer Risk Score",
            "Risk Score",
        ],
    )

    if score_col:

        risk_score = _numeric(
            result,
            score_col,
        )

        revenue_risk = _numeric(
            result,
            "Revenue At Risk",
        )

        if revenue_risk.max() > 0:

            value_factor = (
                revenue_risk
                / revenue_risk.max()
                * 100
            )

        else:

            value_factor = pd.Series(
                0,
                index=result.index,
            )

        result[
            "Retention Urgency Score"
        ] = (
            risk_score * 0.70
            +
            value_factor * 0.30
        ).clip(
            0,
            100,
        ).round(2)

    else:

        result[
            "Retention Urgency Score"
        ] = 0.0

    # --------------------------------------------------------
    # Recommended channel
    # --------------------------------------------------------

    result[
        "Recommended Channel"
    ] = np.select(
        [
            result[
                "Retention Priority"
            ].eq("Immediate"),

            result[
                "Retention Priority"
            ].eq("High"),

            result[
                "Retention Priority"
            ].eq("Monitor"),
        ],
        [
            "Personal Outreach",
            "Targeted Campaign",
            "Email / In-App",
        ],
        default="Standard Engagement",
    )

    return result


# ============================================================
# BUSINESS SUMMARY
# ============================================================

def get_business_summary(
    df: pd.DataFrame,
) -> Dict[str, float]:
    """
    Generate executive-level business risk summary.
    """

    if df is None or df.empty:

        return {
            "Total Customers": 0,
            "Critical Customers": 0,
            "High Risk Customers": 0,
            "Medium Risk Customers": 0,
            "Low Risk Customers": 0,
            "Monthly Revenue At Risk": 0.0,
            "Total Revenue At Risk": 0.0,
            "Average Churn Probability": 0.0,
            "Average Risk Score": 0.0,
        }

    risk_col = _find_column(
        df,
        [
            "Risk Level",
            "Customer Risk Segment",
        ],
    )

    probability_col = _find_column(
        df,
        [
            "Churn Probability",
        ],
    )

    score_col = _find_column(
        df,
        [
            "Customer Risk Score",
            "Risk Score",
        ],
    )

    revenue_col = _find_column(
        df,
        [
            "Revenue At Risk",
        ],
    )

    monthly_col = _find_column(
        df,
        [
            "MonthlyCharges",
            "Monthly Charges",
        ],
    )

    if risk_col:

        risk = (
            df[risk_col]
            .astype(str)
            .str.lower()
        )

        critical = int(
            risk.eq("critical").sum()
        )

        high = int(
            risk.eq("high").sum()
        )

        medium = int(
            risk.eq("medium").sum()
        )

        low = int(
            risk.eq("low").sum()
        )

    else:

        critical = high = medium = low = 0

    if revenue_col:

        revenue_at_risk = float(
            _numeric(
                df,
                revenue_col,
            ).sum()
        )

    else:

        revenue_at_risk = 0.0

    # Monthly recurring value at risk.
    monthly_revenue_at_risk = 0.0

    if monthly_col and probability_col:

        monthly = _numeric(
            df,
            monthly_col,
        )

        probability = _numeric(
            df,
            probability_col,
        )

        if probability.max() <= 1:

            probability = (
                probability
                * 100
            )

        monthly_revenue_at_risk = float(
            (
                monthly
                * probability
                / 100
            ).sum()
        )

    return {
        "Total Customers": len(df),

        "Critical Customers": critical,

        "High Risk Customers": high,

        "Medium Risk Customers": medium,

        "Low Risk Customers": low,

        "Monthly Revenue At Risk":
            round(
                monthly_revenue_at_risk,
                2,
            ),

        "Total Revenue At Risk":
            round(
                revenue_at_risk,
                2,
            ),

        "Average Churn Probability":
            round(
                float(
                    _numeric(
                        df,
                        probability_col,
                    ).mean()
                )
                if probability_col
                else 0.0,
                2,
            ),

        "Average Risk Score":
            round(
                float(
                    _numeric(
                        df,
                        score_col,
                    ).mean()
                )
                if score_col
                else 0.0,
                2,
            ),
    }


# ============================================================
# PRIORITY ACTION PLAN
# ============================================================

def get_priority_action_plan(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate a compact business action plan grouped by
    retention priority.
    """

    if df is None or df.empty:
        return pd.DataFrame()

    priority_col = _find_column(
        df,
        [
            "Retention Priority",
        ],
    )

    if priority_col is None:
        return pd.DataFrame()

    revenue_col = _find_column(
        df,
        [
            "Revenue At Risk",
        ],
    )

    probability_col = _find_column(
        df,
        [
            "Churn Probability",
        ],
    )

    grouped = (
        df.groupby(
            priority_col,
            dropna=False,
        )
        .agg(
            Customers=(
                priority_col,
                "size",
            ),
        )
        .reset_index()
    )

    grouped = grouped.rename(
        columns={
            priority_col: "Priority",
        }
    )

    if revenue_col:

        revenue = (
            df.groupby(
                priority_col,
                dropna=False,
            )[revenue_col]
            .sum()
            .reset_index(
                name="Revenue At Risk"
            )
        )

        revenue = revenue.rename(
            columns={
                priority_col: "Priority",
            }
        )

        grouped = grouped.merge(
            revenue,
            on="Priority",
            how="left",
        )

    if probability_col:

        probability = (
            df.groupby(
                priority_col,
                dropna=False,
            )[probability_col]
            .mean()
            .reset_index(
                name="Average Churn Probability"
            )
        )

        probability = probability.rename(
            columns={
                priority_col: "Priority",
            }
        )

        grouped = grouped.merge(
            probability,
            on="Priority",
            how="left",
        )

    priority_order = {
        "Immediate": 1,
        "High": 2,
        "Monitor": 3,
        "Low": 4,
    }

    grouped["_order"] = (
        grouped["Priority"]
        .map(priority_order)
        .fillna(99)
    )

    grouped = (
        grouped
        .sort_values("_order")
        .drop(
            columns="_order"
        )
        .reset_index(
            drop=True
        )
    )

    return grouped


# ============================================================
# EXECUTIVE RECOMMENDATIONS
# ============================================================

def generate_executive_recommendations(
    df: pd.DataFrame,
) -> List[str]:
    """Generate high-level recommendations for executives."""

    if df is None or df.empty:
        return []

    summary = get_business_summary(
        df
    )

    recommendations = []

    critical = summary[
        "Critical Customers"
    ]

    high = summary[
        "High Risk Customers"
    ]

    revenue = summary[
        "Monthly Revenue At Risk"
    ]

    avg_probability = summary[
        "Average Churn Probability"
    ]

    # --------------------------------------------------------
    # Critical action
    # --------------------------------------------------------

    if critical > 0:

        recommendations.append(
            f"Prioritize immediate outreach to "
            f"{critical:,} critical-risk customers."
        )

    # --------------------------------------------------------
    # High-risk action
    # --------------------------------------------------------

    if high > 0:

        recommendations.append(
            f"Launch targeted retention campaigns "
            f"for {high:,} high-risk customers."
        )

    # --------------------------------------------------------
    # Revenue action
    # --------------------------------------------------------

    if revenue > 0:

        recommendations.append(
            f"Focus retention resources on the "
            f"estimated {revenue:,.2f} monthly revenue at risk."
        )

    # --------------------------------------------------------
    # Overall churn
    # --------------------------------------------------------

    if avg_probability >= 60:

        recommendations.append(
            "Overall churn probability is elevated; "
            "increase proactive customer engagement."
        )

    elif avg_probability >= 35:

        recommendations.append(
            "Maintain active churn monitoring and "
            "segment customers by behavioral risk."
        )

    else:

        recommendations.append(
            "Overall churn risk is relatively controlled; "
            "focus on loyalty and customer-value expansion."
        )

    # --------------------------------------------------------
    # Default
    # --------------------------------------------------------

    if not recommendations:

        recommendations.append(
            "Continue monitoring customer risk and "
            "maintain proactive retention programs."
        )

    return recommendations


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
                91,
                72,
                24,
                8,
            ],
            "Customer Risk Score": [
                90,
                70,
                25,
                10,
            ],
            "Risk Level": [
                "Critical",
                "High",
                "Low",
                "Low",
            ],
            "Strategic Segment": [
                "Critical Retention",
                "High Value At Risk",
                "Loyal Customer",
                "Stable Valuable",
            ],
            "MonthlyCharges": [
                90,
                85,
                60,
                45,
            ],
            "Customer Value": [
                180,
                680,
                2880,
                2700,
            ],
        }
    )

    output = create_recommendations(
        demo
    )

    print("\nRecommendations:")
    print(
        output[
            [
                "customerID",
                "Risk Level",
                "Retention Priority",
                "Recommended Action",
                "Recommended Channel",
                "Revenue At Risk",
                "Retention Urgency Score",
            ]
        ]
    )

    print("\nBusiness Summary:")
    print(
        get_business_summary(
            output
        )
    )

    print("\nAction Plan:")
    print(
        get_priority_action_plan(
            output
        )
    )

    print("\nExecutive Recommendations:")

    for item in generate_executive_recommendations(
        output
    ):
        print("-", item)