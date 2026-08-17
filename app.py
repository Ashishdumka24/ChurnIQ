"""
============================================================
ChurnIQ
Customer Churn Prediction & Business Intelligence System
Phase 10 - Streamlit Dashboard
============================================================
"""

from pathlib import Path
import sys
import io

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "Src"
DATA_DIR = BASE_DIR / "Data"
MODEL_DIR = BASE_DIR / "Models"
STYLE_PATH = BASE_DIR / "style.css"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ChurnIQ | Customer Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# LOAD EXTERNAL CSS
# ============================================================

def load_css():
    if STYLE_PATH.exists():
        with open(STYLE_PATH, "r", encoding="utf-8") as f:
            st.markdown(
                f"<style>{f.read()}</style>",
                unsafe_allow_html=True,
            )


load_css()


# ============================================================
# OPTIONAL MODULE IMPORTS
# ============================================================

PREDICTION_AVAILABLE = False
PREDICTION_ERROR = None

try:

    from prediction_pipeline_backup import (
        predict_customer_churn,
    )

    PREDICTION_AVAILABLE = True

except Exception as exc:

    PREDICTION_ERROR = exc


SEGMENTATION_AVAILABLE = False
SEGMENTATION_ERROR = None

try:

    from segmentation import (
        create_customer_intelligence,
        get_segment_summary,
        get_high_risk_customers,
    )

    SEGMENTATION_AVAILABLE = True

except Exception as exc:

    SEGMENTATION_ERROR = exc


RECOMMENDATIONS_AVAILABLE = False
RECOMMENDATIONS_ERROR = None

try:

    from recommendations import (
        create_recommendations,
        get_business_summary,
    )

    RECOMMENDATIONS_AVAILABLE = True

except Exception as exc:

    RECOMMENDATIONS_ERROR = exc


ANALYTICS_AVAILABLE = False
ANALYTICS_ERROR = None

try:

    from dashboard_analytics import (
        get_data_quality,
        get_model_performance,
        get_best_model_metrics,
        generate_executive_insights,
        get_contract_analysis,
        get_tenure_analysis,
        get_monthly_charge_analysis,
    )

    ANALYTICS_AVAILABLE = True

except Exception as exc:

    ANALYTICS_ERROR = exc

DATABASE_AVAILABLE = False
DATABASE_ERROR = None

try:
    from database import (
        test_connection,
        initialize_database,
        get_customer_count,
    )

    DATABASE_AVAILABLE = True

except Exception as exc:
    DATABASE_ERROR = exc


# ============================================================
# SESSION STATE
# ============================================================

if "current_data" not in st.session_state:
    st.session_state.current_data = None

if "data_source" not in st.session_state:
    st.session_state.data_source = None

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "intelligence_result" not in st.session_state:
    st.session_state.intelligence_result = None

if "recommendation_result" not in st.session_state:
    st.session_state.recommendation_result = None


# ============================================================
# DATA LOADING
# ============================================================

def read_uploaded_file(uploaded_file):

    if uploaded_file is None:
        return None

    filename = uploaded_file.name.lower()

    try:

        if filename.endswith(".csv"):

            return pd.read_csv(
                uploaded_file
            )

        if filename.endswith(".xlsx"):

            return pd.read_excel(
                uploaded_file,
                engine="openpyxl",
            )

        if filename.endswith(".xls"):

            return pd.read_excel(
                uploaded_file,
                engine="xlrd",
            )

        raise ValueError(
            "Unsupported file format. "
            "Please upload CSV, XLS or XLSX."
        )

    except Exception as exc:

        raise ValueError(
            f"Could not read '{uploaded_file.name}': "
            f"{exc}"
        ) from exc


def load_default_dataset():

    candidates = [
        DATA_DIR / "telco_churn_clean.csv",
        DATA_DIR / "Telco-Customer-Churn.csv",
        DATA_DIR / "telco_churn_processed.csv",
    ]

    for path in candidates:

        if not path.exists():
            continue

        try:

            df = pd.read_csv(
                path
            )

            if not df.empty:

                return (
                    df,
                    path.name,
                )

        except Exception:
            continue

    return (
        pd.DataFrame(),
        None,
    )


def ensure_default_data():

    if (
        st.session_state.current_data
        is None
    ):

        df, source = (
            load_default_dataset()
        )

        st.session_state.current_data = df
        st.session_state.data_source = source


ensure_default_data()


# ============================================================
# HELPERS
# ============================================================

def find_column(
    df,
    names,
):

    if df is None or df.empty:
        return None

    normalized = {
        str(column)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", ""):
            column
        for column in df.columns
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


def money(value):

    try:

        return f"{float(value):,.2f}"

    except Exception:

        return "0.00"


def prediction_summary(
    df
):

    if df is None or df.empty:

        return {
            "total": 0,
            "churn": 0,
            "retained": 0,
            "churn_rate": 0.0,
            "avg_probability": 0.0,
            "high_risk": 0,
            "medium_risk": 0,
            "low_risk": 0,
        }

    total = len(df)

    prediction_col = find_column(
        df,
        [
            "Prediction",
            "Churn Prediction",
        ],
    )

    probability_col = find_column(
        df,
        [
            "Churn Probability",
        ],
    )

    risk_col = find_column(
        df,
        [
            "Risk Level",
            "Customer Risk Segment",
        ],
    )

    churn = 0

    if prediction_col:

        values = (
            df[prediction_col]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        churn = int(
            values.isin(
                {
                    "yes",
                    "true",
                    "1",
                    "churn",
                    "churned",
                }
            ).sum()
        )

    retained = max(
        total - churn,
        0,
    )

    if probability_col:

        probability = pd.to_numeric(
            df[probability_col],
            errors="coerce",
        )

        avg_probability = (
            float(
                probability.mean()
            )
            if probability.notna().any()
            else 0.0
        )

    else:

        avg_probability = 0.0

    high = 0
    medium = 0
    low = 0

    if risk_col:

        risks = (
            df[risk_col]
            .astype(str)
            .str.lower()
        )

        high = int(
            risks.str.contains(
                "high|critical",
                regex=True,
                na=False,
            ).sum()
        )

        medium = int(
            risks.str.contains(
                "medium",
                regex=True,
                na=False,
            ).sum()
        )

        low = int(
            risks.str.contains(
                "low",
                regex=True,
                na=False,
            ).sum()
        )

    return {

        "total":
            total,

        "churn":
            churn,

        "retained":
            retained,

        "churn_rate":
            (
                churn / total * 100
                if total
                else 0.0
            ),

        "avg_probability":
            avg_probability,

        "high_risk":
            high,

        "medium_risk":
            medium,

        "low_risk":
            low,
    }


def run_prediction():

    if not PREDICTION_AVAILABLE:

        raise RuntimeError(
            "Prediction pipeline could not be imported."
        )

    data = st.session_state.current_data

    if data is None or data.empty:

        raise ValueError(
            "No dataset is loaded."
        )

    with st.spinner(
        "Running ChurnIQ machine-learning analysis..."
    ):

        predictions = predict_customer_churn(
            data
        )

    if not isinstance(
        predictions,
        pd.DataFrame,
    ):

        raise TypeError(
            "Prediction pipeline did not return a DataFrame."
        )

    st.session_state.prediction_result = (
        predictions
    )

    # --------------------------------------------------------
    # Customer intelligence
    # --------------------------------------------------------

    intelligence = predictions

    if SEGMENTATION_AVAILABLE:

        try:

            intelligence = (
                create_customer_intelligence(
                    predictions
                )
            )

            st.session_state.intelligence_result = (
                intelligence
            )

        except Exception:
            st.session_state.intelligence_result = (
                predictions
            )

    # --------------------------------------------------------
    # Recommendations
    # --------------------------------------------------------

    recommendations = intelligence

    if RECOMMENDATIONS_AVAILABLE:

        try:

            recommendations = (
                create_recommendations(
                    intelligence
                )
            )

            st.session_state.recommendation_result = (
                recommendations
            )

        except Exception:

            st.session_state.recommendation_result = (
                intelligence
            )

    else:

        st.session_state.recommendation_result = (
            intelligence
        )

    return recommendations


def get_result_data():

    if (
        st.session_state.recommendation_result
        is not None
    ):

        return (
            st.session_state.recommendation_result
        )

    if (
        st.session_state.intelligence_result
        is not None
    ):

        return (
            st.session_state.intelligence_result
        )

    return (
        st.session_state.prediction_result
    )


def download_csv(
    df,
    filename,
    label,
):

    if df is None or df.empty:
        return

    data = df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label=label,
        data=data,
        file_name=filename,
        mime="text/csv",
        key=f"download_{filename}",
    )

# ============================================================
# SIDEBAR TOGGLE
# ============================================================

if "sidebar_hidden" not in st.session_state:
    st.session_state.sidebar_hidden = False

toggle_label = (
    "☰ Show Sidebar"
    if st.session_state.sidebar_hidden
    else "☰ Hide Sidebar"
)

if st.button(
    toggle_label,
    key="sidebar_toggle",
):
    st.session_state.sidebar_hidden = (
        not st.session_state.sidebar_hidden
    )
    st.rerun()

if st.session_state.sidebar_hidden:
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] {
            display: none !important;
        }

        [data-testid="stSidebarCollapsedControl"] {
            display: none !important;
        }

        [data-testid="collapsedControl"] {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )    

# ============================================================
# SIDEBAR
# ============================================================


pages = [
    "🏠 Executive Overview",
    "📂 Dataset Center",
    "📊 Churn Analysis",
    "👥 Customer Segments",
    "⚠️ Risk Center",
    "🧠 Prediction Center",
    "🧪 Model Performance",
    "🔎 Customer Explorer",
    "💡 Recommendations",
    "📥 Reports",
    "👨‍💻 About",
]

page = st.sidebar.radio(
    "Navigation",
    pages,
    key="main_navigation",
)


# ============================================================
# MYSQL STATUS
# ============================================================

if DATABASE_AVAILABLE:

    try:
        mysql_connected = test_connection()
    except Exception:
        mysql_connected = False

    if mysql_connected:

        st.sidebar.markdown(
            """
            <span class="status-good">
            ● MySQL Connected
            </span>
            """,
            unsafe_allow_html=True,
        )

        try:
            customer_count = get_customer_count()

            st.sidebar.caption(
                f"MySQL Records: {customer_count:,}"
            )

        except Exception:
            pass

    else:

        st.sidebar.markdown(
            """
            <span class="status-warning">
            ● MySQL Offline
            </span>
            """,
            unsafe_allow_html=True,
        )

else:

    st.sidebar.markdown(
        """
        <span class="status-warning">
        ● MySQL Module Unavailable
        </span>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# CURRENT DATA
# ============================================================

df = st.session_state.current_data

if df is None:

    df = pd.DataFrame()

if not isinstance(
    df,
    pd.DataFrame,
):

    df = pd.DataFrame()


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
<div class="churniq-hero">

<h1>📊 ChurnIQ</h1>

<p>
Customer Churn Prediction & Business Intelligence System
</p>

<p>
Transform customer data into predictive insights,
risk intelligence, segmentation and actionable
retention strategies.
</p>

</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# EXECUTIVE OVERVIEW
# ============================================================

if page == "🏠 Executive Overview":

    st.markdown(
        '<div class="section-title">Executive Overview</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-caption">'
        "A real-time view of customer health, churn risk and business exposure."
        "</div>",
        unsafe_allow_html=True,
    )

    if df.empty:

        st.warning(
            "No dataset is currently available."
        )

        st.stop()

    result = get_result_data()

    if result is not None:

        metrics = prediction_summary(
            result
        )

    else:

        metrics = {
            "total": len(df),
            "churn": 0,
            "retained": 0,
            "churn_rate": 0,
            "avg_probability": 0,
            "high_risk": 0,
            "medium_risk": 0,
            "low_risk": 0,
        }

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(
        4
    )

    with c1:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">
                    TOTAL CUSTOMERS
                </div>
                <div class="metric-value">
                    {metrics["total"]:,}
                </div>
                <div class="metric-subtitle">
                    Customers in current dataset
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">
                    PREDICTED CHURN
                </div>
                <div class="metric-value">
                    {metrics["churn"]:,}
                </div>
                <div class="metric-subtitle">
                    {metrics["churn_rate"]:.1f}% predicted churn rate
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">
                    AVERAGE CHURN PROBABILITY
                </div>
                <div class="metric-value">
                    {metrics["avg_probability"]:.1f}%
                </div>
                <div class="metric-subtitle">
                    Model-generated probability
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">
                    HIGH / CRITICAL RISK
                </div>
                <div class="metric-value">
                    {metrics["high_risk"]:,}
                </div>
                <div class="metric-subtitle">
                    Priority customers
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


    # --------------------------------------------------------
    # ACTION
    # --------------------------------------------------------

    if result is None:

        st.info(
            "Run the ML analysis to populate the predictive dashboard."
        )

        if st.button(
            "🚀 Run Churn Analysis",
            type="primary",
            key="overview_run_analysis",
        ):

            try:

                run_prediction()

                st.rerun()

            except Exception as exc:

                st.error(
                    "Prediction could not be completed."
                )

                st.exception(exc)

    else:

        # ----------------------------------------------------
        # CHARTS
        # ----------------------------------------------------

        left, right = st.columns(
            2
        )

        with left:

            labels = [
                "Retained",
                "Predicted Churn",
            ]

            values = [
                metrics["retained"],
                metrics["churn"],
            ]

            chart_df = pd.DataFrame(
                {
                    "Status": labels,
                    "Customers": values,
                }
            )

            fig = px.pie(
                chart_df,
                names="Status",
                values="Customers",
                hole=0.62,
                title="Customer Retention Outlook",
            )

            fig.update_layout(
                height=390,
                margin=dict(
                    l=10,
                    r=10,
                    t=60,
                    b=10,
                ),
                legend_title="",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                key="overview_retention_pie",
            )

        with right:

            risk_df = pd.DataFrame(
                {
                    "Risk": [
                        "Low",
                        "Medium",
                        "High / Critical",
                    ],
                    "Customers": [
                        metrics["low_risk"],
                        metrics["medium_risk"],
                        metrics["high_risk"],
                    ],
                }
            )

            fig = px.bar(
                risk_df,
                x="Risk",
                y="Customers",
                text="Customers",
                title="Customer Risk Distribution",
            )

            fig.update_layout(
                height=390,
                margin=dict(
                    l=10,
                    r=10,
                    t=60,
                    b=10,
                ),
                xaxis_title="",
                yaxis_title="Customers",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                key="overview_risk_bar",
            )

        # ----------------------------------------------------
        # EXECUTIVE INSIGHTS
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">Executive Insights</div>',
            unsafe_allow_html=True,
        )

        if ANALYTICS_AVAILABLE:

            try:

                insights = (
                    generate_executive_insights(
                        result
                    )
                )

            except Exception:

                insights = []

        else:

            insights = []

        if not insights:

            insights = [
                f"{metrics['churn_rate']:.1f}% of customers "
                "are currently predicted to churn.",
                f"{metrics['high_risk']:,} customers "
                "require priority attention.",
                f"Average churn probability is "
                f"{metrics['avg_probability']:.1f}%.",
            ]

        for insight in insights[:6]:

            st.markdown(
                f"""
                <div class="insight-card">
                    💡 {insight}
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# DATASET CENTER
# ============================================================

elif page == "📂 Dataset Center":

    st.markdown(
        '<div class="section-title">Dataset Center</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-caption">'
        "Upload, inspect and manage your customer dataset."
        "</div>",
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # UPLOAD
    # --------------------------------------------------------

    col1, col2 = st.columns([2, 1])

    with col1:

        st.markdown(
            """
            <div class="upload-box">
                <h3>📤 Upload Customer Dataset</h3>
                <p>
                Upload CSV, XLS or XLSX customer data.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        page_upload = st.file_uploader(
            "Choose a dataset",
            type=["csv", "xls", "xlsx"],
            key="dataset_center_uploader",
            label_visibility="collapsed",
        )

        # ----------------------------------------------------
        # NEW FILE UPLOADED
        # ----------------------------------------------------

        if page_upload is not None:

            upload_signature = (
                page_upload.name,
                page_upload.size,
            )

            last_signature = st.session_state.get(
                "last_upload_signature"
            )

            # Only process a genuinely new upload
            if last_signature != upload_signature:

                try:

                    new_df = read_uploaded_file(
                        page_upload
                    )

                    if new_df is None or new_df.empty:

                        st.error(
                            "The uploaded dataset is empty."
                        )

                    else:

                        # ------------------------------------
                        # SAVE UPLOADED DATASET
                        # ------------------------------------

                        st.session_state.current_data = (
                            new_df.copy()
                        )

                        st.session_state.data_source = (
                            page_upload.name
                        )

                        st.session_state.last_upload_signature = (
                            upload_signature
                        )

                        # ------------------------------------
                        # CLEAR OLD ANALYSIS
                        # ------------------------------------

                        st.session_state.prediction_result = None

                        st.session_state.intelligence_result = None

                        st.session_state.recommendation_result = None

                        st.success(
                            f"Successfully loaded "
                            f"{len(new_df):,} customers "
                            f"from {page_upload.name}."
                        )

                        st.rerun()

                except Exception as exc:

                    st.error(
                        f"Could not load dataset: {exc}"
                    )

    # --------------------------------------------------------
    # ACTIVE DATASET
    # --------------------------------------------------------

    active_df = st.session_state.get(
        "current_data"
    )

    if active_df is None:

        active_df = pd.DataFrame()

    # --------------------------------------------------------
    # DATASET STATUS
    # --------------------------------------------------------

    with col2:

        if not active_df.empty:

            st.metric(
                "Rows",
                f"{len(active_df):,}",
            )

            st.metric(
                "Columns",
                f"{len(active_df.columns):,}",
            )

            st.metric(
                "Missing Cells",
                f"{int(active_df.isna().sum().sum()):,}",
            )

            st.caption(
                f"📌 Active dataset: "
                f"{st.session_state.get('data_source') or 'Loaded Dataset'}"
            )

        else:

            st.info(
                "No dataset loaded."
            )

    # --------------------------------------------------------
    # REMOVE / RESTORE CONTROLS
    # --------------------------------------------------------

    st.markdown("---")

    action_col1, action_col2, action_col3 = st.columns(
        [1, 1, 2]
    )

    
    # ========================================================
    # RESTORE DEFAULT DATASET
    # ========================================================

    with action_col2:

        if st.button(
            "🔄 Restore Default Dataset",
            key="restore_default_dataset",
            use_container_width=True,
        ):

            try:

                default_path = (
                    BASE_DIR
                    / "Data"
                    / "telco_churn_clean.csv"
                )

                if not default_path.exists():

                    st.error(
                        "Default ChurnIQ dataset was not found."
                    )

                else:

                    default_df = pd.read_csv(
                        default_path
                    )

                    if default_df.empty:

                        st.error(
                            "Default dataset is empty."
                        )

                    else:

                        # ------------------------------------
                        # RESTORE DEFAULT DATA
                        # ------------------------------------

                        st.session_state.current_data = (
                            default_df.copy()
                        )

                        st.session_state.data_source = (
                            "ChurnIQ Default Dataset"
                        )

                        st.session_state.last_upload_signature = (
                            None
                        )

                        # Clear analysis
                        st.session_state.prediction_result = None

                        st.session_state.intelligence_result = None

                        st.session_state.recommendation_result = None

                        st.success(
                            "Default dataset restored."
                        )

                        st.rerun()

            except Exception as exc:

                st.error(
                    f"Could not restore default dataset: {exc}"
                )

    # ========================================================
    # DATASET PREVIEW
    # ========================================================

    active_df = st.session_state.get(
        "current_data"
    )

    if active_df is None:

        active_df = pd.DataFrame()

    if not active_df.empty:

        st.markdown("---")

        st.subheader(
            "Dataset Preview"
        )

        st.dataframe(
            active_df.head(100),
            use_container_width=True,
            hide_index=True,
        )

        # ----------------------------------------------------
        # DATASET INFORMATION
        # ----------------------------------------------------

        st.subheader(
            "Dataset Information"
        )

        info_df = pd.DataFrame(
            {
                "Column": active_df.columns,

                "Data Type": [
                    str(
                        active_df[column].dtype
                    )
                    for column in active_df.columns
                ],

                "Missing": [
                    int(
                        active_df[column]
                        .isna()
                        .sum()
                    )
                    for column in active_df.columns
                ],

                "Unique Values": [
                    int(
                        active_df[column]
                        .nunique(
                            dropna=True
                        )
                    )
                    for column in active_df.columns
                ],
            }
        )

        st.dataframe(
            info_df,
            use_container_width=True,
            hide_index=True,
        )

        # ----------------------------------------------------
        # DATA QUALITY
        # ----------------------------------------------------

        if ANALYTICS_AVAILABLE:

            quality = get_data_quality(
                active_df
            )

            st.markdown(
                "### Data Quality"
            )

            q1, q2, q3, q4 = st.columns(4)

            q1.metric(
                "Quality Score",
                f"{quality['quality_score']:.1f}%",
            )

            q2.metric(
                "Missing Cells",
                quality["missing_cells"],
            )

            q3.metric(
                "Duplicate Rows",
                quality["duplicate_rows"],
            )

            q4.metric(
                "Rows",
                quality["rows"],
            )

    else:

        st.info(
            "No active dataset. "
            "Upload a new dataset or restore the default dataset."
        )
# ============================================================
# CHURN ANALYSIS
# ============================================================

elif page == "📊 Churn Analysis":

    st.markdown(
        '<div class="section-title">Churn Analysis</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-caption">'
        "Explore predicted churn patterns and customer behavior."
        "</div>",
        unsafe_allow_html=True,
    )

    result = get_result_data()

    if result is None:

        st.info(
            "Run churn analysis first."
        )

        if st.button(
            "🚀 Run Analysis",
            type="primary",
            key="churn_analysis_run",
        ):

            try:

                run_prediction()

                st.rerun()

            except Exception as exc:

                st.error(
                    str(exc)
                )

        st.stop()

    metrics = prediction_summary(
        result
    )

    a, b, c, d = st.columns(
        4
    )

    a.metric(
        "Customers",
        f"{metrics['total']:,}",
    )

    b.metric(
        "Predicted Churn",
        f"{metrics['churn']:,}",
    )

    c.metric(
        "Churn Rate",
        f"{metrics['churn_rate']:.2f}%",
    )

    d.metric(
        "Avg Probability",
        f"{metrics['avg_probability']:.2f}%",
    )

    left, right = st.columns(
        2
    )

    with left:

        prediction_col = find_column(
            result,
            [
                "Prediction",
                "Churn Prediction",
            ],
        )

        if prediction_col:

            counts = (
                result[
                    prediction_col
                ]
                .astype(str)
                .value_counts()
                .reset_index()
            )

            counts.columns = [
                "Prediction",
                "Customers",
            ]

            fig = px.pie(
                counts,
                names="Prediction",
                values="Customers",
                hole=0.58,
                title="Predicted Customer Status",
            )

            fig.update_layout(
                height=420
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                key="churn_prediction_pie",
            )

    with right:

        probability_col = find_column(
            result,
            [
                "Churn Probability",
            ],
        )

        if probability_col:

            probability = pd.to_numeric(
                result[
                    probability_col
                ],
                errors="coerce",
            ).dropna()

            histogram_df = pd.DataFrame(
                {
                    "Churn Probability":
                        probability
                }
            )

            fig = px.histogram(
                histogram_df,
                x="Churn Probability",
                nbins=20,
                title="Churn Probability Distribution",
            )

            fig.update_layout(
                height=420,
                xaxis_title="Probability (%)",
                yaxis_title="Customers",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                key="churn_probability_histogram",
            )


# ============================================================
# CUSTOMER SEGMENTS
# ============================================================

elif page == "👥 Customer Segments":

    st.markdown(
        '<div class="section-title">Customer Segmentation</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-caption">'
        "Understand customer groups and prioritize valuable at-risk customers."
        "</div>",
        unsafe_allow_html=True,
    )

    result = get_result_data()

    if result is None:

        st.info(
            "Run churn analysis to generate customer segments."
        )

        st.stop()

    segment_col = find_column(
        result,
        [
            "Strategic Segment",
            "Customer Segment",
        ],
    )

    if segment_col:

        segment_counts = (
            result[
                segment_col
            ]
            .astype(str)
            .value_counts()
            .reset_index()
        )

        segment_counts.columns = [
            "Segment",
            "Customers",
        ]

        left, right = st.columns(
            2
        )

        with left:

            fig = px.bar(
                segment_counts,
                x="Segment",
                y="Customers",
                text="Customers",
                title="Strategic Customer Segments",
            )

            fig.update_layout(
                height=430,
                xaxis_title="",
                yaxis_title="Customers",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                key="segments_bar",
            )

        with right:

            fig = px.pie(
                segment_counts,
                names="Segment",
                values="Customers",
                hole=0.55,
                title="Segment Composition",
            )

            fig.update_layout(
                height=430
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                key="segments_pie",
            )

        st.dataframe(
            segment_counts,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.warning(
            "Strategic segmentation is not available for this result."
        )


# ============================================================
# RISK CENTER
# ============================================================

elif page == "⚠️ Risk Center":

    st.markdown(
        '<div class="section-title">Risk Center</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-caption">'
        "Identify customers requiring retention attention."
        "</div>",
        unsafe_allow_html=True,
    )

    result = get_result_data()

    if result is None:

        st.info(
            "Run the prediction engine first."
        )

        if st.button(
            "🚀 Run Risk Analysis",
            type="primary",
            key="risk_center_run",
        ):

            try:

                run_prediction()

                st.rerun()

            except Exception as exc:

                st.error(
                    str(exc)
                )

        st.stop()

    metrics = prediction_summary(
        result
    )

    c1, c2, c3, c4 = st.columns(
        4
    )

    c1.metric(
        "Critical",
        metrics["high_risk"],
    )

    c2.metric(
        "Medium",
        metrics["medium_risk"],
    )

    c3.metric(
        "Low",
        metrics["low_risk"],
    )

    c4.metric(
        "Avg Risk",
        f"{metrics['avg_probability']:.1f}%",
    )

    risk_col = find_column(
        result,
        [
            "Risk Level",
            "Customer Risk Segment",
        ],
    )

    if risk_col:

        risk_counts = (
            result[
                risk_col
            ]
            .astype(str)
            .value_counts()
            .reset_index()
        )

        risk_counts.columns = [
            "Risk Level",
            "Customers",
        ]

        fig = px.bar(
            risk_counts,
            x="Risk Level",
            y="Customers",
            text="Customers",
            title="Risk Distribution",
        )

        fig.update_layout(
            height=430
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key="risk_center_distribution",
        )

    score_col = find_column(
        result,
        [
            "Customer Risk Score",
            "Risk Score",
        ],
    )

    if score_col:

        st.subheader(
            "Priority Customers"
        )

        scores = pd.to_numeric(
            result[
                score_col
            ],
            errors="coerce",
        )

        priority = result.loc[
            scores >= 50
        ].copy()

        priority = priority.sort_values(
            by=score_col,
            ascending=False,
        )

        st.dataframe(
            priority.head(200),
            use_container_width=True,
            hide_index=True,
        )

        download_csv(
            priority,
            "churniq_high_risk_customers.csv",
            "📥 Download High-Risk Customers",
        )


# ============================================================
# PREDICTION CENTER
# ============================================================

elif page == "🧠 Prediction Center":

    st.markdown(
        '<div class="section-title">Prediction Center</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-caption">'
        "Run the trained machine-learning model against the active dataset."
        "</div>",
        unsafe_allow_html=True,
    )

    if not PREDICTION_AVAILABLE:

        st.error(
            "Prediction pipeline is unavailable."
        )

        if PREDICTION_ERROR:
            st.exception(
                PREDICTION_ERROR
            )

    else:

        st.success(
            "✓ Trained ChurnIQ prediction pipeline is available."
        )

        st.write(
            f"Active dataset: "
            f"**{st.session_state.data_source or 'Unknown'}**"
        )

        if st.button(
            "🚀 Generate Churn Predictions",
            type="primary",
            key="prediction_center_run",
        ):

            try:

                run_prediction()

                st.success(
                    "Prediction completed successfully."
                )

                st.rerun()

            except Exception as exc:

                st.error(
                    "Prediction failed."
                )

                st.exception(exc)

    result = get_result_data()

    if result is not None:

        st.markdown("---")

        st.subheader(
            "Prediction Output"
        )

        st.dataframe(
            result.head(500),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "🧪 Model Performance":

    st.markdown(
        '<div class="section-title">Model Performance</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-caption">'
        "Compare the actual evaluation metrics generated during model training."
        "</div>",
        unsafe_allow_html=True,
    )

    if not ANALYTICS_AVAILABLE:

        st.error(
            "Dashboard analytics module is unavailable."
        )

    else:

        performance = (
            get_model_performance()
        )

        best = (
            get_best_model_metrics()
        )

        if best:

            st.success(
                f"Best Model: "
                f"{best.get('Model', 'Unknown')}"
            )

            metric_columns = [
                (
                    "Accuracy",
                    best.get(
                        "Accuracy",
                        np.nan,
                    ),
                ),
                (
                    "Precision",
                    best.get(
                        "Precision",
                        np.nan,
                    ),
                ),
                (
                    "Recall",
                    best.get(
                        "Recall",
                        np.nan,
                    ),
                ),
                (
                    "F1 Score",
                    best.get(
                        "F1 Score",
                        np.nan,
                    ),
                ),
                (
                    "ROC-AUC",
                    best.get(
                        "ROC-AUC",
                        np.nan,
                    ),
                ),
            ]

            cols = st.columns(
                len(
                    metric_columns
                )
            )

            for col, (
                name,
                value,
            ) in zip(
                cols,
                metric_columns,
            ):

                with col:

                    if pd.notna(value):

                        col.metric(
                            name,
                            f"{float(value) * 100:.2f}%",
                        )

                    else:

                        col.metric(
                            name,
                            "N/A",
                        )

        if not performance.empty:

            st.markdown("---")

            st.subheader(
                "Model Comparison"
            )

            st.dataframe(
                performance,
                use_container_width=True,
                hide_index=True,
            )

            # ------------------------------------------------
            # Model comparison chart
            # ------------------------------------------------

            chart_metrics = [
                column
                for column in [
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1 Score",
                    "ROC-AUC",
                ]
                if column
                in performance.columns
            ]

            model_col = find_column(
                performance,
                [
                    "Model",
                    "Algorithm",
                ],
            )

            if (
                chart_metrics
                and model_col
            ):

                plot_df = performance[
                    [
                        model_col
                    ]
                    + chart_metrics
                ].copy()

                long_df = plot_df.melt(
                    id_vars=[
                        model_col
                    ],
                    var_name="Metric",
                    value_name="Score",
                )

                long_df["Score"] = (
                    pd.to_numeric(
                        long_df[
                            "Score"
                        ],
                        errors="coerce",
                    )
                    * 100
                )

                fig = px.bar(
                    long_df,
                    x=model_col,
                    y="Score",
                    color="Metric",
                    barmode="group",
                    title="Model Performance Comparison",
                )

                fig.update_layout(
                    height=500,
                    yaxis_title="Score (%)",
                    xaxis_title="Model",
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key="model_comparison_chart",
                )

        else:

            st.warning(
                "Models/model_results.csv was not found "
                "or contains no usable results."
            )


# ============================================================
# CUSTOMER EXPLORER
# ============================================================

elif page == "🔎 Customer Explorer":

    st.markdown(
        '<div class="section-title">Customer Explorer</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-caption">'
        "Search, filter and inspect individual customer records."
        "</div>",
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # SOURCE DATA
    # --------------------------------------------------------

    result = get_result_data()

    explorer_df = (
        result.copy()
        if result is not None
        else df.copy()
    )

    if explorer_df.empty:

        st.warning(
            "No customer data is currently available."
        )

        st.info(
            "Run the ChurnIQ prediction engine first "
            "to unlock advanced customer intelligence."
        )

        st.stop()

    # --------------------------------------------------------
    # COLUMN DETECTION
    # --------------------------------------------------------

    id_column = find_column(
        explorer_df,
        [
            "customerID",
            "CustomerID",
            "Customer ID",
            "Customer_Id",
            "ID",
        ],
    )

    prediction_col = find_column(
        explorer_df,
        [
            "Prediction",
            "Churn Prediction",
            "Predicted Churn",
            "Churn",
        ],
    )

    probability_col = find_column(
        explorer_df,
        [
            "Churn Probability",
            "ChurnProbability",
            "Churn Probability (%)",
            "Probability",
        ],
    )

    risk_col = find_column(
        explorer_df,
        [
            "Risk Level",
            "Customer Risk Segment",
            "Risk",
            "Risk Category",
        ],
    )

    risk_score_col = find_column(
        explorer_df,
        [
            "Customer Risk Score",
            "Risk Score",
        ],
    )

    contract_col = find_column(
        explorer_df,
        [
            "Contract",
            "Contract Type",
        ],
    )

    monthly_charge_col = find_column(
        explorer_df,
        [
            "MonthlyCharges",
            "Monthly Charges",
            "Monthly_Charges",
        ],
    )

    tenure_col = find_column(
        explorer_df,
        [
            "tenure",
            "Tenure",
            "Customer Tenure",
        ],
    )

    recommendation_col = find_column(
        explorer_df,
        [
            "Retention Recommendation",
            "Recommendation",
        ],
    )

    action_col = find_column(
        explorer_df,
        [
            "Recommended Action",
            "Action",
        ],
    )

    priority_col = find_column(
        explorer_df,
        [
            "Retention Priority",
            "Priority",
        ],
    )

    # --------------------------------------------------------
    # CUSTOMER SEARCH
    # --------------------------------------------------------

    st.markdown(
        "### 🔎 Search Customers"
    )

    search_col, reset_col = st.columns(
        [5, 1]
    )

    with search_col:



        search = st.text_input(
            "Search",
            key="advanced_customer_search",
            placeholder=(
                "Search customer ID, contract, "
                "service, risk, prediction..."
            ),
            label_visibility="collapsed",
        )

    with reset_col:

        reset_filters = st.button(
            "↻ Reset",
            key="reset_customer_filters",
            use_container_width=True,
        )

    if reset_filters:

    

        st.rerun()

    # --------------------------------------------------------
    # FILTER PANEL
    # --------------------------------------------------------

    st.markdown(
        "### 🎯 Customer Filters"
    )

    f1, f2, f3, f4 = st.columns(
        4
    )

    # --------------------------------------------------------
    # Risk filter
    # --------------------------------------------------------

    with f1:

        if risk_col:

            risk_values = sorted(
                explorer_df[
                    risk_col
                ]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            selected_risks = st.multiselect(
                "⚠️ Risk Level",
                options=risk_values,
                default=[],
                key="customer_risk_filter",
            )

        else:

            selected_risks = []

            st.caption(
                "Risk data unavailable"
            )

    # --------------------------------------------------------
    # Prediction filter
    # --------------------------------------------------------

    with f2:

        if prediction_col:

            prediction_values = sorted(
                explorer_df[
                    prediction_col
                ]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            selected_predictions = st.multiselect(
                "📊 Churn Prediction",
                options=prediction_values,
                default=[],
                key="customer_prediction_filter",
            )

        else:

            selected_predictions = []

            st.caption(
                "Prediction unavailable"
            )

    # --------------------------------------------------------
    # Contract filter
    # --------------------------------------------------------

    with f3:

        if contract_col:

            contract_values = sorted(
                explorer_df[
                    contract_col
                ]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            selected_contracts = st.multiselect(
                "📄 Contract",
                options=contract_values,
                default=[],
                key="customer_contract_filter",
            )

        else:

            selected_contracts = []

            st.caption(
                "Contract unavailable"
            )

    # --------------------------------------------------------
    # Priority filter
    # --------------------------------------------------------

    with f4:

        if priority_col:

            priority_values = sorted(
                explorer_df[
                    priority_col
                ]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            selected_priorities = st.multiselect(
                "🎯 Retention Priority",
                options=priority_values,
                default=[],
                key="customer_priority_filter",
            )

        else:

            selected_priorities = []

            st.caption(
                "Priority unavailable"
            )

    # --------------------------------------------------------
    # NUMERIC FILTERS
    # --------------------------------------------------------

    st.markdown(
        "### 📈 Numeric Filters"
    )

    n1, n2, n3 = st.columns(
        3
    )

    probability_values = None
    score_values = None
    charge_values = None

    # --------------------------------------------------------
    # Churn probability
    # --------------------------------------------------------

    with n1:

        if probability_col:

            probability_numeric = pd.to_numeric(
                explorer_df[
                    probability_col
                ],
                errors="coerce",
            )

            probability_numeric = (
                probability_numeric
                .dropna()
            )

            if not probability_numeric.empty:

                probability_min = float(
                    probability_numeric.min()
                )

                probability_max = float(
                    probability_numeric.max()
                )

                if probability_min < 0:
                    probability_min = 0.0

                if probability_max > 100:
                    probability_max = 100.0

                probability_values = st.slider(
                    "🧠 Churn Probability (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=(
                        probability_min,
                        probability_max,
                    ),
                    step=1.0,
                    key="customer_probability_filter",
                )

        else:

            st.caption(
                "Churn probability unavailable"
            )

    # --------------------------------------------------------
    # Risk score
    # --------------------------------------------------------

    with n2:

        if risk_score_col:

            score_numeric = pd.to_numeric(
                explorer_df[
                    risk_score_col
                ],
                errors="coerce",
            )

            score_numeric = (
                score_numeric
                .dropna()
            )

            if not score_numeric.empty:

                score_min = float(
                    score_numeric.min()
                )

                score_max = float(
                    score_numeric.max()
                )

                score_values = st.slider(
                    "⚠️ Risk Score",
                    min_value=float(
                        score_min
                    ),
                    max_value=float(
                        score_max
                    ),
                    value=(
                        float(score_min),
                        float(score_max),
                    ),
                    step=1.0,
                    key="customer_score_filter",
                )

        else:

            st.caption(
                "Risk score unavailable"
            )

    # --------------------------------------------------------
    # Monthly charges
    # --------------------------------------------------------

    with n3:

        if monthly_charge_col:

            charge_numeric = pd.to_numeric(
                explorer_df[
                    monthly_charge_col
                ],
                errors="coerce",
            )

            charge_numeric = (
                charge_numeric
                .dropna()
            )

            if not charge_numeric.empty:

                charge_min = float(
                    charge_numeric.min()
                )

                charge_max = float(
                    charge_numeric.max()
                )

                charge_values = st.slider(
                    "💰 Monthly Charges",
                    min_value=float(
                        charge_min
                    ),
                    max_value=float(
                        charge_max
                    ),
                    value=(
                        float(charge_min),
                        float(charge_max),
                    ),
                    step=1.0,
                    key="customer_charge_filter",
                )

        else:

            st.caption(
                "Monthly charges unavailable"
            )

    # --------------------------------------------------------
    # APPLY FILTERS
    # --------------------------------------------------------

    filtered = explorer_df.copy()

    # Search
    if search and search.strip():

        search_text = search.strip()

        mask = pd.Series(
            False,
            index=filtered.index,
        )

        for column in filtered.columns:

            mask = (
                mask
                |
                filtered[column]
                .astype(str)
                .str.contains(
                    search_text,
                    case=False,
                    na=False,
                    regex=False,
                )
            )

        filtered = filtered.loc[
            mask
        ]

    # Risk
    if (
        risk_col
        and selected_risks
    ):

        filtered = filtered[
            filtered[
                risk_col
            ]
            .astype(str)
            .isin(
                selected_risks
            )
        ]

    # Prediction
    if (
        prediction_col
        and selected_predictions
    ):

        filtered = filtered[
            filtered[
                prediction_col
            ]
            .astype(str)
            .isin(
                selected_predictions
            )
        ]

    # Contract
    if (
        contract_col
        and selected_contracts
    ):

        filtered = filtered[
            filtered[
                contract_col
            ]
            .astype(str)
            .isin(
                selected_contracts
            )
        ]

    # Priority
    if (
        priority_col
        and selected_priorities
    ):

        filtered = filtered[
            filtered[
                priority_col
            ]
            .astype(str)
            .isin(
                selected_priorities
            )
        ]

    # Probability
    if (
        probability_col
        and probability_values
    ):

        probability_series = pd.to_numeric(
            filtered[
                probability_col
            ],
            errors="coerce",
        )

        filtered = filtered[
            probability_series.between(
                probability_values[0],
                probability_values[1],
                inclusive="both",
            )
        ]

    # Risk score
    if (
        risk_score_col
        and score_values
    ):

        score_series = pd.to_numeric(
            filtered[
                risk_score_col
            ],
            errors="coerce",
        )

        filtered = filtered[
            score_series.between(
                score_values[0],
                score_values[1],
                inclusive="both",
            )
        ]

    # Monthly charges
    if (
        monthly_charge_col
        and charge_values
    ):

        charge_series = pd.to_numeric(
            filtered[
                monthly_charge_col
            ],
            errors="coerce",
        )

        filtered = filtered[
            charge_series.between(
                charge_values[0],
                charge_values[1],
                inclusive="both",
            )
        ]

    # --------------------------------------------------------
    # RESULTS SUMMARY
    # --------------------------------------------------------

    st.markdown("---")

    total_customers = len(explorer_df)
    filtered_customers = len(filtered)

    r1, r2, r3, r4 = st.columns(
        4
    )

    with r1:

        st.metric(
            "👥 Total Customers",
            f"{total_customers:,}",
        )

    with r2:

        st.metric(
            "🔎 Matching Customers",
            f"{filtered_customers:,}",
        )

    with r3:

        if total_customers:

            match_rate = (
                filtered_customers
                / total_customers
                * 100
            )

        else:

            match_rate = 0.0

        st.metric(
            "📊 Match Rate",
            f"{match_rate:.1f}%",
        )

    with r4:

        if risk_col and not filtered.empty:

            high_risk_count = int(
                filtered[
                    risk_col
                ]
                .astype(str)
                .str.contains(
                    "high|critical",
                    case=False,
                    regex=True,
                    na=False,
                )
                .sum()
            )

        else:

            high_risk_count = 0

        st.metric(
            "⚠️ High/Critical Risk",
            f"{high_risk_count:,}",
        )

    # --------------------------------------------------------
    # NO RESULTS
    # --------------------------------------------------------

    if filtered.empty:

        st.warning(
            "No customers match the selected filters."
        )

        st.info(
            "Try removing one or more filters "
            "or broadening your search."
        )

        st.stop()

    # --------------------------------------------------------
    # CUSTOMER PROFILE
    # --------------------------------------------------------

    st.markdown("---")

    st.markdown(
        "### 👤 Customer Profile"
    )

    if id_column:

        customer_ids = (
            filtered[
                id_column
            ]
            .astype(str)
            .tolist()
        )

        selected_customer = st.selectbox(
            "Select a customer",
            options=customer_ids,
            key="selected_customer_profile",
        )

        selected_rows = filtered[
            filtered[
                id_column
            ]
            .astype(str)
            == str(
                selected_customer
            )
        ]

    else:

        selected_customer = None

        selected_rows = filtered.iloc[
            :1
        ]

    if not selected_rows.empty:

        customer = selected_rows.iloc[
            0
        ]

        p1, p2, p3, p4 = st.columns(
            4
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        with p1:

            if prediction_col:

                prediction_value = str(
                    customer[
                        prediction_col
                    ]
                )

                st.metric(
                    "Prediction",
                    prediction_value,
                )

            else:

                st.metric(
                    "Prediction",
                    "N/A",
                )

        # ----------------------------------------------------
        # Probability
        # ----------------------------------------------------

        with p2:

            if probability_col:

                probability_value = pd.to_numeric(
                    customer[
                        probability_col
                    ],
                    errors="coerce",
                )

                if pd.notna(
                    probability_value
                ):

                    if (
                        probability_value
                        <= 1
                    ):

                        probability_value *= 100

                    st.metric(
                        "Churn Probability",
                        f"{probability_value:.1f}%",
                    )

                else:

                    st.metric(
                        "Churn Probability",
                        "N/A",
                    )

            else:

                st.metric(
                    "Churn Probability",
                    "N/A",
                )

        # ----------------------------------------------------
        # Risk
        # ----------------------------------------------------

        with p3:

            if risk_col:

                st.metric(
                    "Risk Level",
                    str(
                        customer[
                            risk_col
                        ]
                    ),
                )

            else:

                st.metric(
                    "Risk Level",
                    "N/A",
                )

        # ----------------------------------------------------
        # Risk Score
        # ----------------------------------------------------

        with p4:

            if risk_score_col:

                score_value = pd.to_numeric(
                    customer[
                        risk_score_col
                    ],
                    errors="coerce",
                )

                if pd.notna(
                    score_value
                ):

                    st.metric(
                        "Risk Score",
                        f"{score_value:.1f}",
                    )

                else:

                    st.metric(
                        "Risk Score",
                        "N/A",
                    )

            else:

                st.metric(
                    "Risk Score",
                    "N/A",
                )

        # ----------------------------------------------------
        # CUSTOMER DETAILS
        # ----------------------------------------------------

        st.markdown(
            "#### 📋 Customer Details"
        )

        profile_columns = []

        for column in [
            id_column,
            contract_col,
            tenure_col,
            monthly_charge_col,
            prediction_col,
            probability_col,
            risk_col,
            risk_score_col,
            priority_col,
        ]:

            if (
                column
                and column not in profile_columns
            ):

                profile_columns.append(
                    column
                )

        if profile_columns:

            profile_data = pd.DataFrame(
                {
                    "Attribute": profile_columns,
                    "Value": [
                        customer[
                            column
                        ]
                        for column
                        in profile_columns
                    ],
                }
            )

            st.dataframe(
                profile_data,
                use_container_width=True,
                hide_index=True,
            )

        # ----------------------------------------------------
        # RETENTION INTELLIGENCE
        # ----------------------------------------------------

        if (
            recommendation_col
            or action_col
            or priority_col
        ):

            st.markdown(
                "#### 💡 Retention Intelligence"
            )

            ri1, ri2 = st.columns(
                2
            )

            with ri1:

                if priority_col:

                    st.markdown(
                        f"""
                        <div class="insight-card">
                            🎯 <strong>Retention Priority</strong><br>
                            {customer[priority_col]}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                if recommendation_col:

                    st.markdown(
                        f"""
                        <div class="insight-card">
                            💡 <strong>Recommendation</strong><br>
                            {customer[recommendation_col]}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with ri2:

                if action_col:

                    st.markdown(
                        f"""
                        <div class="insight-card">
                            🚀 <strong>Recommended Action</strong><br>
                            {customer[action_col]}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    # --------------------------------------------------------
    # CUSTOMER TABLE
    # --------------------------------------------------------

    st.markdown("---")

    st.markdown(
        "### 📊 Customer Results"
    )

    st.caption(
        f"Showing {min(len(filtered), 500):,} "
        f"of {len(filtered):,} matching customers."
    )

    # --------------------------------------------------------
    # Sort results
    # --------------------------------------------------------

    sort_column = None

    if risk_score_col:

        sort_column = risk_score_col

    elif probability_col:

        sort_column = probability_col

    elif id_column:

        sort_column = id_column

    if sort_column:

        sort_direction = st.selectbox(
            "Sort customers by",
            options=[
                "Highest to Lowest",
                "Lowest to Highest",
            ],
            key="customer_sort_direction",
        )

        sort_numeric = pd.to_numeric(
            filtered[
                sort_column
            ],
            errors="coerce",
        )

        if sort_numeric.notna().sum() > 0:

            sorted_index = sort_numeric.sort_values(
                ascending=(
                    sort_direction
                    == "Lowest to Highest"
                )
            ).index

            display_df = filtered.loc[
                sorted_index
            ]

        else:

            display_df = filtered.sort_values(
                by=sort_column,
                ascending=(
                    sort_direction
                    == "Lowest to Highest"
                ),
            )

    else:

        display_df = filtered

    st.dataframe(
        display_df.head(500),
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # DOWNLOAD FILTERED RESULTS
    # --------------------------------------------------------

    st.markdown(
        "### 📥 Export Results"
    )

    download_csv(
        filtered,
        "churniq_filtered_customer_results.csv",
        "📥 Download Filtered Customers",
    )

# ============================================================
# BUSINESS RECOMMENDATIONS
# ============================================================

elif page == "💡 Recommendations":

    st.markdown(
        '<div class="section-title">Business Recommendations</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-caption">'
        "Convert churn predictions into practical retention actions."
        "</div>",
        unsafe_allow_html=True,
    )

    result = get_result_data()

    if result is None:

        st.info(
            "Run churn analysis first."
        )

        st.stop()

    if RECOMMENDATIONS_AVAILABLE:

        try:

            summary = get_business_summary(
                result
            )

        except Exception:

            summary = {}

    else:

        summary = {}

    if summary:

        c1, c2, c3, c4 = st.columns(
            4
        )

        c1.metric(
            "Critical",
            summary.get(
                "Critical Customers",
                0,
            ),
        )

        c2.metric(
            "High Risk",
            summary.get(
                "High Risk Customers",
                0,
            ),
        )

        c3.metric(
            "Medium Risk",
            summary.get(
                "Medium Risk Customers",
                0,
            ),
        )

        c4.metric(
            "Monthly Value At Risk",
            money(
                summary.get(
                    "Monthly Revenue At Risk",
                    0,
                )
            ),
        )

    recommendation_col = find_column(
        result,
        [
            "Retention Recommendation",
        ],
    )

    action_col = find_column(
        result,
        [
            "Recommended Action",
        ],
    )

    priority_col = find_column(
        result,
        [
            "Retention Priority",
        ],
    )

    display_columns = []

    for column in [
        priority_col,
        action_col,
        recommendation_col,
    ]:

        if column and column not in display_columns:

            display_columns.append(
                column
            )

    # Include customer ID
    id_column = find_column(
        result,
        [
            "customerID",
            "CustomerID",
            "Customer ID",
        ],
    )

    if id_column:

        display_columns.insert(
            0,
            id_column,
        )

    if display_columns:

        st.dataframe(
            result[
                display_columns
            ].head(300),
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "No recommendation columns are available."
        )


# ============================================================
# REPORTS
# ============================================================

elif page == "📥 Reports":

    st.markdown(
        '<div class="section-title">Reports & Business Intelligence</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-caption">'
        "Generate actionable business reports from ChurnIQ predictions, "
        "risk intelligence and customer analytics."
        "</div>",
        unsafe_allow_html=True,
    )

    result = get_result_data()

    # --------------------------------------------------------
    # REPORT STATUS
    # --------------------------------------------------------

    if result is None or result.empty:

        st.info(
            "Run the ChurnIQ prediction engine first to generate "
            "advanced business reports."
        )

        if st.button(
            "🚀 Run Churn Analysis",
            type="primary",
            key="reports_run_analysis",
        ):

            try:

                run_prediction()

                st.success(
                    "Analysis completed successfully."
                )

                st.rerun()

            except Exception as exc:

                st.error(
                    "Prediction could not be completed."
                )

                st.exception(exc)

        st.stop()

    # --------------------------------------------------------
    # REPORT METRICS
    # --------------------------------------------------------

    metrics = prediction_summary(result)

    revenue_col = find_column(
        result,
        [
            "MonthlyCharges",
            "Monthly Charges",
            "Monthly Charge",
        ],
    )

    total_monthly_revenue = 0.0
    monthly_revenue_at_risk = 0.0

    if revenue_col:

        revenue = pd.to_numeric(
            result[revenue_col],
            errors="coerce",
        ).fillna(0)

        total_monthly_revenue = float(
            revenue.sum()
        )

        prediction_col = find_column(
            result,
            [
                "Prediction",
                "Churn Prediction",
            ],
        )

        risk_col = find_column(
            result,
            [
                "Risk Level",
                "Customer Risk Segment",
            ],
        )

        if prediction_col:

            churn_mask = (
                result[prediction_col]
                .astype(str)
                .str.strip()
                .str.lower()
                .isin(
                    {
                        "yes",
                        "true",
                        "1",
                        "churn",
                        "churned",
                    }
                )
            )

            monthly_revenue_at_risk = float(
                revenue.loc[churn_mask].sum()
            )

        elif risk_col:

            risk_mask = (
                result[risk_col]
                .astype(str)
                .str.lower()
                .str.contains(
                    "high|critical",
                    regex=True,
                    na=False,
                )
            )

            monthly_revenue_at_risk = float(
                revenue.loc[risk_mask].sum()
            )

    # --------------------------------------------------------
    # EXECUTIVE KPI CARDS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Executive KPI Summary</div>',
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4, k5 = st.columns(5)

    k1.metric(
        "Total Customers",
        f"{metrics['total']:,}",
    )

    k2.metric(
        "Predicted Churn",
        f"{metrics['churn']:,}",
    )

    k3.metric(
        "Churn Rate",
        f"{metrics['churn_rate']:.1f}%",
    )

    k4.metric(
        "High / Critical Risk",
        f"{metrics['high_risk']:,}",
    )

    k5.metric(
        "Monthly Revenue at Risk",
        f"{money(monthly_revenue_at_risk)}",
    )

    st.markdown("---")

    # --------------------------------------------------------
    # BUSINESS EXPOSURE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Business Exposure</div>',
        unsafe_allow_html=True,
    )

    e1, e2, e3 = st.columns(3)

    with e1:

        st.metric(
            "Total Monthly Charges",
            money(total_monthly_revenue),
        )

    with e2:

        st.metric(
            "Revenue at Risk",
            money(monthly_revenue_at_risk),
        )

    with e3:

        exposure_percentage = (
            (
                monthly_revenue_at_risk
                / total_monthly_revenue
                * 100
            )
            if total_monthly_revenue > 0
            else 0.0
        )

        st.metric(
            "Revenue Exposure",
            f"{exposure_percentage:.1f}%",
        )

    # --------------------------------------------------------
    # CHURN VS RETENTION REPORT
    # --------------------------------------------------------

    st.markdown("---")

    left, right = st.columns(2)

    with left:

        retention_df = pd.DataFrame(
            {
                "Customer Status": [
                    "Retained",
                    "Predicted Churn",
                ],
                "Customers": [
                    metrics["retained"],
                    metrics["churn"],
                ],
            }
        )

        fig = px.pie(
            retention_df,
            names="Customer Status",
            values="Customers",
            hole=0.58,
            title="Customer Retention Outlook",
        )

        fig.update_layout(
            height=420,
            margin=dict(
                l=10,
                r=10,
                t=60,
                b=10,
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key="reports_retention_chart",
        )

    with right:

        risk_df = pd.DataFrame(
            {
                "Risk Level": [
                    "Low",
                    "Medium",
                    "High / Critical",
                ],
                "Customers": [
                    metrics["low_risk"],
                    metrics["medium_risk"],
                    metrics["high_risk"],
                ],
            }
        )

        fig = px.bar(
            risk_df,
            x="Risk Level",
            y="Customers",
            text="Customers",
            title="Customer Risk Exposure",
        )

        fig.update_layout(
            height=420,
            margin=dict(
                l=10,
                r=10,
                t=60,
                b=10,
            ),
            xaxis_title="",
            yaxis_title="Customers",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            key="reports_risk_chart",
        )

    # --------------------------------------------------------
    # HIGH-RISK REPORT
    # --------------------------------------------------------

    st.markdown("---")

    st.markdown(
        '<div class="section-title">High-Risk Customer Report</div>',
        unsafe_allow_html=True,
    )

    score_col = find_column(
        result,
        [
            "Customer Risk Score",
            "Risk Score",
        ],
    )

    risk_col = find_column(
        result,
        [
            "Risk Level",
            "Customer Risk Segment",
        ],
    )

    high_risk_report = result.copy()

    if risk_col:

        high_risk_report = result[
            result[risk_col]
            .astype(str)
            .str.lower()
            .str.contains(
                "high|critical",
                regex=True,
                na=False,
            )
        ].copy()

    elif score_col:

        scores = pd.to_numeric(
            result[score_col],
            errors="coerce",
        )

        high_risk_report = result.loc[
            scores >= 50
        ].copy()

    st.write(
        f"**{len(high_risk_report):,}** customers "
        "are currently identified as priority retention candidates."
    )

    if not high_risk_report.empty:

        display_cols = []

        customer_id_col = find_column(
            high_risk_report,
            [
                "customerID",
                "CustomerID",
                "Customer ID",
            ],
        )

        prediction_col = find_column(
            high_risk_report,
            [
                "Prediction",
                "Churn Prediction",
            ],
        )

        probability_col = find_column(
            high_risk_report,
            [
                "Churn Probability",
            ],
        )

        for column in [
            customer_id_col,
            risk_col,
            score_col,
            prediction_col,
            probability_col,
            revenue_col,
        ]:

            if (
                column
                and column in high_risk_report.columns
                and column not in display_cols
            ):

                display_cols.append(column)

        if display_cols:

            st.dataframe(
                high_risk_report[
                    display_cols
                ].head(200),
                use_container_width=True,
                hide_index=True,
            )

        download_csv(
            high_risk_report,
            "churniq_high_risk_report.csv",
            "⚠️ Download High-Risk Report",
        )

    else:

        st.success(
            "No high-risk customers were identified."
        )

    # --------------------------------------------------------
    # RETENTION OPPORTUNITY REPORT
    # --------------------------------------------------------

    st.markdown("---")

    st.markdown(
        '<div class="section-title">Retention Opportunity Report</div>',
        unsafe_allow_html=True,
    )

    recommendation_col = find_column(
        result,
        [
            "Retention Recommendation",
        ],
    )

    action_col = find_column(
        result,
        [
            "Recommended Action",
        ],
    )

    priority_col = find_column(
        result,
        [
            "Retention Priority",
        ],
    )

    opportunity_columns = []

    for column in [
        customer_id_col
        if "customer_id_col" in locals()
        else None,
        priority_col,
        risk_col,
        score_col,
        probability_col,
        recommendation_col,
        action_col,
        revenue_col,
    ]:

        if (
            column
            and column in result.columns
            and column not in opportunity_columns
        ):

            opportunity_columns.append(
                column
            )

    if opportunity_columns:

        opportunity_df = result[
            opportunity_columns
        ].copy()

        st.dataframe(
            opportunity_df.head(300),
            use_container_width=True,
            hide_index=True,
        )

        download_csv(
            opportunity_df,
            "churniq_retention_opportunities.csv",
            "🎯 Download Retention Opportunities",
        )

    else:

        st.info(
            "Retention recommendation fields are not available."
        )

    # --------------------------------------------------------
    # COMPLETE CUSTOMER ANALYSIS
    # --------------------------------------------------------

    st.markdown("---")

    st.markdown(
        '<div class="section-title">Complete Analytical Report</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "Download the complete customer-level output "
        "including predictions, risk intelligence, segmentation "
        "and retention recommendations."
    )

    download_csv(
        result,
        "churniq_complete_customer_analysis.csv",
        "📥 Download Complete Customer Analysis",
    )

    # --------------------------------------------------------
    # RAW DATASET
    # --------------------------------------------------------

    st.markdown("---")

    st.markdown(
        '<div class="section-title">Current Dataset</div>',
        unsafe_allow_html=True,
    )

    download_csv(
        df,
        "churniq_current_dataset.csv",
        "📥 Download Current Dataset",
    )
    

# ============================================================
# ABOUT CHURNIQ
# ============================================================

if page == "👨‍💻 About":

    st.write("")

    hero_col1, hero_col2, hero_col3 = st.columns(3)

    with hero_col1:
        st.metric(
            "Platform",
            "ChurnIQ"
        )

    with hero_col2:
        st.metric(
            "Focus",
            "Customer Churn"
        )

    with hero_col3:
        st.metric(
            "Technology",
            "Data Science"
        )

    st.divider()

    # ========================================================
    # INTRODUCTION
    # ========================================================

    st.header("🔎 What is ChurnIQ?")

    st.write(
        "ChurnIQ is an end-to-end Customer Churn Prediction and "
        "Business Intelligence System developed to analyze "
        "customer behavior and identify customers who may be "
        "at risk of discontinuing a company's services."
    )

    st.write(
        "The system combines data preparation, exploratory "
        "analysis, feature engineering, machine learning, "
        "model evaluation, customer segmentation, risk analysis "
        "and interactive visualization into a single analytical "
        "platform."
    )

    st.write(
        "Rather than treating machine-learning predictions as "
        "isolated numbers, ChurnIQ connects predictions with "
        "customer intelligence and business recommendations."
    )

    st.info(
        "The core idea behind ChurnIQ is simple: "
        "transform raw customer data into meaningful intelligence "
        "that can support better customer-retention decisions."
    )

    # ========================================================
    # CORE IDEA
    # ========================================================

    st.header("💡 The ChurnIQ Concept")

    st.write(
        "The platform follows a complete analytical journey "
        "from raw data to business action."
    )

    concept_col1, concept_col2, concept_col3, concept_col4 = (
        st.columns(4)
    )

    with concept_col1:

        st.subheader("📥 Data")

        st.write(
            "Customer information is collected and prepared "
            "for analysis."
        )

    with concept_col2:

        st.subheader("🧠 Intelligence")

        st.write(
            "Patterns and relationships are discovered using "
            "Data Science and Machine Learning."
        )

    with concept_col3:

        st.subheader("⚠️ Risk")

        st.write(
            "Customers are analyzed according to predicted "
            "churn risk and business relevance."
        )

    with concept_col4:

        st.subheader("💼 Action")

        st.write(
            "Insights are converted into practical "
            "retention-oriented recommendations."
        )

    st.divider()

    # ========================================================
    # PROJECT OVERVIEW
    # ========================================================

    st.header("📚 Project Overview")

    st.write(
        "Customer churn is one of the important analytical "
        "problems faced by service-based organizations. When "
        "customers discontinue a service, businesses may lose "
        "revenue, customer lifetime value and long-term "
        "relationships."
    )

    st.write(
        "A traditional reporting system generally explains "
        "what has already happened. ChurnIQ takes a predictive "
        "approach by studying historical customer information "
        "and using machine-learning techniques to estimate "
        "future churn risk."
    )

    st.write(
        "The platform is designed to help users move through "
        "the complete Data Science lifecycle. Customer data "
        "is first inspected and prepared. Exploratory analysis "
        "is then used to understand customer behavior. "
        "Predictive features are prepared and machine-learning "
        "models are evaluated."
    )

    st.write(
        "The resulting predictions are then connected to "
        "customer segmentation, risk analysis and business "
        "recommendations."
    )

    st.success(
        "Customer Data → Data Processing → Analysis → "
        "Machine Learning → Prediction → Risk Intelligence → "
        "Business Recommendations"
    )

    # ========================================================
    # PROJECT OBJECTIVES
    # ========================================================

    st.header("🎯 Project Objectives")

    objectives = [
        (
            "Predict Customer Churn",
            "Identify customers who are likely to discontinue "
            "their services."
        ),
        (
            "Understand Customer Behavior",
            "Analyze customer characteristics and patterns "
            "associated with churn."
        ),
        (
            "Improve Data Quality",
            "Inspect datasets for missing values, duplicates, "
            "data types and structural issues."
        ),
        (
            "Engineer Predictive Features",
            "Transform raw customer information into useful "
            "machine-learning features."
        ),
        (
            "Evaluate Models",
            "Compare classification models using appropriate "
            "performance metrics."
        ),
        (
            "Identify Risk",
            "Highlight customers who may require greater "
            "attention."
        ),
        (
            "Segment Customers",
            "Create meaningful customer groups for business "
            "analysis."
        ),
        (
            "Generate Recommendations",
            "Convert analytical results into practical "
            "retention strategies."
        ),
        (
            "Build Business Intelligence",
            "Present analytical information through an "
            "interactive dashboard."
        ),
    ]

    objective_cols = st.columns(3)

    for i, (title, description) in enumerate(objectives):

        with objective_cols[i % 3]:

            st.subheader(title)

            st.write(description)

    # ========================================================
    # HOW THE SYSTEM WORKS
    # ========================================================

    st.header("⚙️ How ChurnIQ Works")

    st.write(
        "ChurnIQ follows a structured workflow in which each "
        "stage contributes to the final customer intelligence "
        "output."
    )

    # --------------------------------------------------------
    # WORKFLOW 01
    # --------------------------------------------------------

    workflow_col1, workflow_col2 = st.columns(2)

    with workflow_col1:

        st.subheader("01 · 📥 Data Collection")

        st.write(
            "Customer information is obtained from a structured "
            "dataset or supported data source. The dataset "
            "provides the foundation for all subsequent "
            "analytical operations."
        )

    with workflow_col2:

        st.subheader("02 · 🔎 Data Validation")

        st.write(
            "The dataset is inspected to understand its "
            "structure, columns, data types, missing values, "
            "duplicate records and general quality."
        )

    # --------------------------------------------------------
    # WORKFLOW 03
    # --------------------------------------------------------

    workflow_col1, workflow_col2 = st.columns(2)

    with workflow_col1:

        st.subheader("03 · 🧹 Data Cleaning")

        st.write(
            "Customer records are prepared for reliable "
            "analysis by handling data-quality issues and "
            "ensuring that the dataset is suitable for "
            "downstream processing."
        )

    with workflow_col2:

        st.subheader("04 · 📊 Exploratory Data Analysis")

        st.write(
            "Customer behavior is explored using statistical "
            "analysis and visualization to discover patterns, "
            "relationships and characteristics associated "
            "with churn."
        )

    # --------------------------------------------------------
    # WORKFLOW 05
    # --------------------------------------------------------

    workflow_col1, workflow_col2 = st.columns(2)

    with workflow_col1:

        st.subheader("05 · 🔧 Feature Engineering")

        st.write(
            "Relevant information is transformed into useful "
            "predictive features that can improve the ability "
            "of machine-learning models to identify churn."
        )

    with workflow_col2:

        st.subheader("06 · 🤖 Model Training")

        st.write(
            "Classification algorithms are trained using "
            "historical customer information and the prepared "
            "feature set."
        )

    # --------------------------------------------------------
    # WORKFLOW 07
    # --------------------------------------------------------

    workflow_col1, workflow_col2 = st.columns(2)

    with workflow_col1:

        st.subheader("07 · 🧪 Model Evaluation")

        st.write(
            "Models are evaluated using classification metrics "
            "such as Accuracy, Precision, Recall, F1-score "
            "and ROC-AUC where applicable."
        )

    with workflow_col2:

        st.subheader("08 · 🔮 Churn Prediction")

        st.write(
            "The trained prediction workflow generates "
            "customer-level churn predictions and probability "
            "information."
        )

    # --------------------------------------------------------
    # WORKFLOW 09
    # --------------------------------------------------------

    workflow_col1, workflow_col2 = st.columns(2)

    with workflow_col1:

        st.subheader("09 · 👥 Customer Intelligence")

        st.write(
            "Prediction results are interpreted through "
            "customer segments, risk categories and analytical "
            "insights."
        )

    with workflow_col2:

        st.subheader("10 · 💡 Business Recommendations")

        st.write(
            "The final stage connects analytical findings "
            "with practical retention-oriented strategies "
            "and business decisions."
        )

    # ========================================================
    # KEY FEATURES
    # ========================================================

    st.header("🚀 Key Features")

    features = [
        (
            "📂 Dataset Center",
            "Upload, inspect and manage customer datasets."
        ),
        (
            "🧹 Data Processing",
            "Prepare customer data for analysis and prediction."
        ),
        (
            "📊 Churn Analytics",
            "Explore customer behavior and churn patterns."
        ),
        (
            "🤖 Churn Prediction",
            "Generate churn predictions and probability scores."
        ),
        (
            "⚠️ Risk Center",
            "Identify high-risk and priority customers."
        ),
        (
            "👥 Customer Segmentation",
            "Create meaningful customer groups."
        ),
        (
            "🔎 Customer Explorer",
            "Search and inspect individual customer records."
        ),
        (
            "🧪 Model Performance",
            "Compare machine-learning model performance."
        ),
        (
            "💡 Recommendations",
            "Generate retention-oriented business insights."
        ),
        (
            "📥 Reports",
            "Review and download analytical information."
        ),
        (
            "📈 Interactive Dashboard",
            "Explore customer intelligence through Streamlit."
        ),
        (
            "🗄️ Database Support",
            "Support structured customer-data management."
        ),
    ]

    feature_cols = st.columns(3)

    for i, (title, description) in enumerate(features):

        with feature_cols[i % 3]:

            st.subheader(title)

            st.write(description)

    # ========================================================
    # TECHNOLOGY STACK
    # ========================================================

    st.header("🛠️ Technology Stack")

    st.write(
        "ChurnIQ combines Python-based Data Science tools "
        "with machine-learning, visualization and dashboard "
        "technologies."
    )

    technologies = [
        (
            "🐍 Python",
            "Core programming language used throughout "
            "the project."
        ),
        (
            "🐼 Pandas",
            "Data loading, manipulation, cleaning and analysis."
        ),
        (
            "🔢 NumPy",
            "Numerical computation and data processing."
        ),
        (
            "🤖 Scikit-learn",
            "Machine-learning preprocessing, classification "
            "and model evaluation."
        ),
        (
            "📊 Matplotlib",
            "Analytical and statistical visualization."
        ),
        (
            "📈 Seaborn",
            "Statistical visualization and exploratory analysis."
        ),
        (
            "📉 Plotly",
            "Interactive analytical charts."
        ),
        (
            "🌐 Streamlit",
            "Interactive dashboard and application interface."
        ),
        (
            "🗄️ MySQL",
            "Structured database technology for customer data."
        ),
    ]

    technology_cols = st.columns(3)

    for i, (technology, description) in enumerate(
        technologies
    ):

        with technology_cols[i % 3]:

            st.subheader(technology)

            st.write(description)

    # ========================================================
    # MACHINE LEARNING
    # ========================================================

    st.header("🤖 Machine Learning in ChurnIQ")

    st.write(
        "Machine Learning forms the predictive foundation of "
        "ChurnIQ. Historical customer information is used to "
        "learn patterns that can help estimate whether a "
        "customer may belong to the churn category."
    )

    ml_col1, ml_col2 = st.columns(2)

    with ml_col1:

        st.subheader("📥 Input Data")

        st.write(
            "Customer attributes provide the information used "
            "by the analytical and predictive workflow."
        )

        st.subheader("⚙️ Preprocessing")

        st.write(
            "Data is prepared through cleaning, transformation "
            "and feature engineering before prediction."
        )

        st.subheader("🧠 Model")

        st.write(
            "Classification models learn patterns from the "
            "prepared historical data."
        )

    with ml_col2:

        st.subheader("🔮 Prediction")

        st.write(
            "The trained pipeline produces customer-level "
            "churn predictions and probability information."
        )

        st.subheader("⚠️ Risk")

        st.write(
            "Predictions can be interpreted as customer-risk "
            "information for further analysis."
        )

        st.subheader("💼 Business Action")

        st.write(
            "Risk information can support customer-retention "
            "analysis and prioritization."
        )

    # ========================================================
    # MODEL EVALUATION
    # ========================================================

    st.header("🧪 Model Evaluation")

    st.write(
        "A machine-learning model should be evaluated from "
        "multiple perspectives rather than relying on a "
        "single metric."
    )

    evaluation = [
        (
            "Accuracy",
            "Measures the proportion of overall predictions "
            "that are correct."
        ),
        (
            "Precision",
            "Measures how reliable positive churn predictions "
            "are."
        ),
        (
            "Recall",
            "Measures how effectively churn customers are "
            "identified."
        ),
        (
            "F1-Score",
            "Balances Precision and Recall."
        ),
        (
            "ROC-AUC",
            "Measures classification discrimination across "
            "different thresholds where applicable."
        ),
    ]

    evaluation_cols = st.columns(3)

    for i, (metric_name, explanation) in enumerate(
        evaluation
    ):

        with evaluation_cols[i % 3]:

            st.subheader(metric_name)

            st.write(explanation)

    # ========================================================
    # CUSTOMER RISK INTELLIGENCE
    # ========================================================

    st.header("⚠️ Customer Risk Intelligence")

    st.write(
        "ChurnIQ is designed to make predictive information "
        "easier to understand from a business perspective."
    )

    risk_col1, risk_col2, risk_col3 = st.columns(3)

    with risk_col1:

        st.subheader("🔴 High Risk")

        st.write(
            "Customers requiring closer attention based on "
            "available prediction and risk information."
        )

    with risk_col2:

        st.subheader("🟠 Medium Risk")

        st.write(
            "Customers who may benefit from monitoring and "
            "further analysis."
        )

    with risk_col3:

        st.subheader("🟢 Lower Risk")

        st.write(
            "Customers with comparatively lower predicted "
            "churn risk."
        )

    st.info(
        "Risk categories should be interpreted together with "
        "customer characteristics and business context rather "
        "than treated as absolute decisions."
    )

    # ========================================================
    # CUSTOMER SEGMENTATION
    # ========================================================

    st.header("👥 Customer Segmentation")

    st.write(
        "Customer segmentation allows the platform to move "
        "beyond a single churn prediction and analyze groups "
        "of customers with similar characteristics or risk "
        "profiles."
    )

    segment_col1, segment_col2 = st.columns(2)

    with segment_col1:

        st.subheader("Why Segment Customers?")

        st.write(
            "Different customer groups can have different "
            "needs, behaviors and retention priorities."
        )

        st.write(
            "Segmentation helps organize customers into "
            "meaningful groups that can be analyzed separately."
        )

    with segment_col2:

        st.subheader("Business Benefit")

        st.write(
            "Segment-level analysis can help businesses "
            "develop more targeted strategies instead of "
            "using the same approach for every customer."
        )

    # ========================================================
    # BUSINESS RECOMMENDATIONS
    # ========================================================

    st.header("💡 From Prediction to Action")

    st.write(
        "The purpose of ChurnIQ is not simply to produce "
        "predictions. The platform is designed to help "
        "translate analytical findings into practical "
        "business thinking."
    )

    action_col1, action_col2, action_col3, action_col4 = (
        st.columns(4)
    )

    with action_col1:

        st.subheader("01 · Identify")

        st.write(
            "Find customers who may represent higher churn risk."
        )

    with action_col2:

        st.subheader("02 · Understand")

        st.write(
            "Study the available customer characteristics "
            "and behavioral patterns."
        )

    with action_col3:

        st.subheader("03 · Prioritize")

        st.write(
            "Focus analytical attention on important "
            "customer groups."
        )

    with action_col4:

        st.subheader("04 · Act")

        st.write(
            "Use insights to support suitable retention "
            "strategies."
        )

    # ========================================================
    # DATA QUALITY
    # ========================================================

    st.header("🧹 Data Quality & Reliability")

    st.write(
        "Reliable analytics depend on reliable data. Poor "
        "quality data can influence analysis, visualization "
        "and machine-learning predictions."
    )

    quality_col1, quality_col2 = st.columns(2)

    with quality_col1:

        st.subheader("🔎 What is Inspected?")

        st.write(
            "Dataset structure, number of rows, number of "
            "columns, missing values, data types, unique "
            "values and duplicate records."
        )

    with quality_col2:

        st.subheader("📊 Why Does It Matter?")

        st.write(
            "Data-quality information helps users understand "
            "whether a dataset is suitable for further "
            "analytical processing."
        )

    # ========================================================
    # CONCEPTS COVERED
    # ========================================================

    st.header("📚 Data Science Concepts Covered")

    concepts = [
        "Data Collection",
        "Data Validation",
        "Data Cleaning",
        "Exploratory Data Analysis",
        "Data Visualization",
        "Feature Engineering",
        "Feature Selection",
        "Classification Algorithms",
        "Model Training",
        "Model Evaluation",
        "Predictive Analytics",
        "Customer Segmentation",
        "Risk Analysis",
        "Business Intelligence",
        "Customer Retention Analytics",
        "Interactive Dashboard Development",
    ]

    concept_cols = st.columns(4)

    for i, concept in enumerate(concepts):

        with concept_cols[i % 4]:

            st.write(
                f"✓ {concept}"
            )

    # ========================================================
    # LEARNING OUTCOMES
    # ========================================================

    st.header("🎓 Learning Outcomes")

    outcomes = [
        "Build a complete customer churn analytics workflow.",
        "Work with real-world customer datasets.",
        "Perform data validation and data cleaning.",
        "Conduct exploratory data analysis.",
        "Understand customer behavior through visualization.",
        "Engineer useful predictive features.",
        "Train classification-based machine-learning models.",
        "Evaluate and compare model performance.",
        "Interpret machine-learning predictions.",
        "Identify customer-risk patterns.",
        "Perform customer segmentation.",
        "Create interactive business dashboards.",
        "Translate analytical results into business insights.",
        "Develop customer-retention strategies.",
        "Understand the relationship between Data Science "
        "and business decision-making.",
    ]

    for outcome in outcomes:

        st.write(
            f"🎓 {outcome}"
        )

    # ========================================================
    # BUSINESS VALUE
    # ========================================================

    st.header("💼 Business Value")

    business_value = [
        (
            "Early Risk Identification",
            "Potentially high-risk customers can be identified "
            "for further analysis."
        ),
        (
            "Customer Understanding",
            "Customer behavior can be studied using data-driven "
            "analytical methods."
        ),
        (
            "Retention Prioritization",
            "Risk information can help prioritize customers "
            "for further attention."
        ),
        (
            "Data-Driven Decisions",
            "Business decisions can be supported by analytical "
            "evidence."
        ),
        (
            "Integrated Intelligence",
            "Prediction, segmentation, risk and analytics are "
            "connected through one platform."
        ),
        (
            "Practical Data Science",
            "The project demonstrates how machine learning "
            "can be applied to a real business problem."
        ),
    ]

    value_cols = st.columns(3)

    for i, (title, description) in enumerate(
        business_value
    ):

        with value_cols[i % 3]:

            st.subheader(title)

            st.write(description)

    # ========================================================
    # DATABASE INTEGRATION
    # ========================================================

    st.header("🗄️ Database Integration")

    st.write(
        "The project architecture supports the use of MySQL "
        "as a structured database technology for customer-data "
        "management."
    )

    st.write(
        "A database layer can provide structured storage for "
        "customer information while the analytical and "
        "machine-learning components process that information "
        "for business intelligence."
    )

    db_col1, db_col2, db_col3 = st.columns(3)

    with db_col1:

        st.metric(
            "Database",
            "MySQL"
        )

    with db_col2:

        st.metric(
            "Purpose",
            "Customer Data"
        )

    with db_col3:

        st.metric(
            "Platform",
            "ChurnIQ"
        )

    # ========================================================
    # PROJECT ARCHITECTURE
    # ========================================================

    st.header("🏗️ Project Architecture")

    st.write(
        "ChurnIQ follows a modular architecture in which "
        "different components are responsible for different "
        "stages of the analytical workflow."
    )

    architecture = [
        (
            "📂 Data",
            "Customer datasets used by the application."
        ),
        (
            "🧹 Data Processing",
            "Cleaning and preparation of customer data."
        ),
        (
            "🔧 Feature Engineering",
            "Preparation of predictive features."
        ),
        (
            "🤖 Model Training",
            "Development of machine-learning models."
        ),
        (
            "🧪 Model Evaluation",
            "Measurement and comparison of model performance."
        ),
        (
            "🔮 Prediction Pipeline",
            "Connection between preprocessing, models and "
            "customer-level prediction."
        ),
        (
            "👥 Segmentation",
            "Customer grouping and segmentation analysis."
        ),
        (
            "📊 Analytics",
            "Business-oriented customer insights."
        ),
        (
            "🌐 Streamlit Application",
            "Interactive dashboard and user interface."
        ),
    ]

    architecture_cols = st.columns(3)

    for i, (title, description) in enumerate(
        architecture
    ):

        with architecture_cols[i % 3]:

            st.subheader(title)

            st.write(description)

    # ========================================================
    # DEVELOPER
    # ========================================================

    st.header("👨‍💻 Developer")

    st.subheader(
        "Ashish Dumka"
    )

    st.write(
        "Data Science Project Developer"
    )

    developer_col1, developer_col2 = st.columns(2)

    with developer_col1:

        st.write(
            "**Project**"
        )

        st.write(
            "Customer Churn Prediction and Business "
            "Intelligence System"
        )

        st.write(
            "**Platform**"
        )

        st.write(
            "ChurnIQ"
        )

    with developer_col2:

        st.write(
            "**Project Type**"
        )

        st.write(
            "College Data Science Project"
        )

        st.write(
            "**Technology Focus**"
        )

        st.write(
            "Data Science, Machine Learning and "
            "Business Intelligence"
        )

    # ========================================================
    # PROJECT SIGNIFICANCE
    # ========================================================

    st.header("🌟 Why ChurnIQ Matters")

    st.write(
        "ChurnIQ demonstrates that a successful Data Science "
        "project involves much more than training a machine-"
        "learning model."
    )

    st.write(
        "A complete analytical solution must understand the "
        "data, prepare it correctly, explore meaningful "
        "patterns, evaluate predictive models and communicate "
        "the results in a way that users can understand."
    )

    st.write(
        "ChurnIQ brings these components together in a single "
        "interactive application. This makes the project both "
        "a machine-learning implementation and a practical "
        "Business Intelligence solution."
    )

    st.success(
        "ChurnIQ connects technical Data Science with practical "
        "business decision-making."
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    st.header("🏆 Project Summary")

    st.write(
        "ChurnIQ is an integrated Customer Churn Prediction "
        "and Business Intelligence System designed to "
        "demonstrate the practical application of Data Science "
        "and Machine Learning to a real-world customer-retention "
        "problem."
    )

    st.write(
        "The platform combines data processing, exploratory "
        "analysis, feature engineering, machine learning, "
        "model evaluation, prediction, customer segmentation, "
        "risk intelligence, visualization and business "
        "recommendations."
    )

    st.write(
        "Its central purpose is to transform customer records "
        "into meaningful information that can help users "
        "understand customer behavior, recognize potential "
        "risk and support data-driven decisions."
    )

    st.info(
        "ChurnIQ — From Customer Data to Business Decisions."
    )

    # ========================================================
    # CLOSING
    # ========================================================

    st.divider()

    st.subheader(
        "📊 Analyze the Data."
    )

    st.subheader(
        "🧠 Understand the Customer."
    )

    st.subheader(
        "⚠️ Identify the Risk."
    )

    st.subheader(
        "💡 Discover the Insight."
    )

    st.subheader(
        "🚀 Support Better Decisions."
    )

    st.write(
        "Customer Churn Prediction & Business Intelligence System"
    )

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
<hr style="margin-top:3rem;">

<div style="
    text-align:center;
    color:#94a3b8;
    font-size:0.82rem;
    padding:0.7rem;
">
    <strong>ChurnIQ</strong>
    &nbsp;•&nbsp;
    Customer Churn Prediction & Business Intelligence
    &nbsp;•&nbsp;
    Data Science Project
</div>
""",
    unsafe_allow_html=True,
)
