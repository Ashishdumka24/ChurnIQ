"""
ChurnIQ - Report diagram generator.

Draws architecture, workflow, pipeline and module-dependency diagrams that
reflect the ACTUAL repository structure (verified against app.py and Src/).

Run:  python report/make_diagrams.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

NAVY = "#1f4e79"
BLUE = "#2e75b6"
LIGHT = "#dce9f5"
GREEN = "#27ae60"
LGREEN = "#d8f0e0"
RED = "#c0392b"
LRED = "#f8dcd8"
ORANGE = "#e67e22"
LORANGE = "#fbe6d2"
GREY = "#5a5a5a"
LGREY = "#ececec"

plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 220,
                     "savefig.bbox": "tight"})


def box(ax, x, y, w, h, text, fc=LIGHT, ec=NAVY, fs=8, bold=False, tc="#111111"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor=fc, edgecolor=ec, linewidth=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal", linespacing=1.35)


def arrow(ax, p1, p2, color=GREY, style="-|>", lw=1.1, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle=style, mutation_scale=11, color=color,
        linewidth=lw, connectionstyle=f"arc3,rad={rad}",
        shrinkA=1, shrinkB=1))


def canvas(w, h, title):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    return fig, ax


def save(fig, name):
    fig.savefig(OUT / name, facecolor="white")
    plt.close(fig)
    print("wrote", name)


# ==================================================================
# 5.1 Layered system architecture
# ==================================================================
fig, ax = canvas(7.6, 6.4, "ChurnIQ Layered System Architecture")

box(ax, 0.3, 8.55, 9.4, 1.15,
    "PRESENTATION LAYER\n"
    "app.py  —  Streamlit dashboard (11 pages)  ·  style.css  ·  Plotly charts",
    fc=LIGHT, ec=NAVY, bold=True, fs=8.5)

box(ax, 0.3, 6.75, 9.4, 1.5,
    "APPLICATION / INTELLIGENCE LAYER\n"
    "dashboard_analytics.py   ·   segmentation.py   ·   recommendations.py",
    fc=LGREEN, ec=GREEN, bold=True, fs=8.5)

box(ax, 0.3, 4.95, 9.4, 1.5,
    "MACHINE LEARNING LAYER\n"
    "prediction_pipeline_backup.py (active)  ·  prediction_pipeline.py\n"
    "model_training.py   ·   feature_engineering.py",
    fc=LORANGE, ec=ORANGE, bold=True, fs=8.5)

box(ax, 0.3, 3.15, 9.4, 1.5,
    "DATA PROCESSING LAYER\n"
    "dataset_validator.py   ·   clean_data.py   ·   eda.py",
    fc=LRED, ec=RED, bold=True, fs=8.5)

box(ax, 0.3, 1.35, 4.5, 1.5,
    "DATA STORAGE\nData/  CSV datasets\nUser uploads (CSV/XLS/XLSX)",
    fc=LGREY, ec=GREY, bold=True, fs=8)

box(ax, 5.2, 1.35, 4.5, 1.5,
    "MODEL STORAGE\nModels/best_churn_model.pkl\npreprocessing.pkl · model_results.csv",
    fc=LGREY, ec=GREY, bold=True, fs=8)

box(ax, 0.3, 0.15, 9.4, 0.85,
    "OPTIONAL PERSISTENCE  —  database.py (MySQL, optional; app degrades gracefully)",
    fc="#ffffff", ec=GREY, fs=8)

for y1, y2 in [(8.55, 8.25), (6.75, 6.45), (4.95, 4.65), (3.15, 2.85)]:
    arrow(ax, (5.0, y1), (5.0, y2), color=NAVY, style="<|-|>")
save(fig, "dia_architecture.png")


# ==================================================================
# 5.2 System workflow
# ==================================================================
fig, ax = canvas(5.4, 8.6, "ChurnIQ End-to-End System Workflow")
steps = [
    ("Customer Dataset\n(CSV / XLS / XLSX upload or default)", LGREY, GREY),
    ("Data Validation\ndataset_validator.py", LRED, RED),
    ("Data Cleaning\nclean_data.py", LRED, RED),
    ("Exploratory Data Analysis\neda.py", LRED, RED),
    ("Feature Engineering\nfeature_engineering.py", LORANGE, ORANGE),
    ("Model Training & Comparison\nmodel_training.py", LORANGE, ORANGE),
    ("Model Persistence\nModels/*.pkl", LORANGE, ORANGE),
    ("Churn Prediction\nprediction_pipeline_backup.py", LIGHT, NAVY),
    ("Customer Intelligence & Segmentation\nsegmentation.py", LGREEN, GREEN),
    ("Business Recommendations\nrecommendations.py", LGREEN, GREEN),
    ("Interactive Dashboard & Reports\napp.py", LIGHT, NAVY),
]
h = 0.66
gap = 0.205
y = 9.55 - h
for text, fc, ec in steps:
    box(ax, 1.15, y, 7.7, h, text, fc=fc, ec=ec, fs=7.8)
    if y - gap > 0.2:
        arrow(ax, (5.0, y), (5.0, y - gap), color=NAVY)
    y -= (h + gap)
save(fig, "dia_workflow.png")


# ==================================================================
# 5.3 Data flow diagram
# ==================================================================
fig, ax = canvas(7.8, 4.6, "ChurnIQ Data Flow Diagram")

box(ax, 0.15, 6.6, 1.85, 1.5, "USER\n\nUploads dataset\nSelects page",
    fc="#ffffff", ec=GREY, bold=True, fs=7.8)
box(ax, 0.15, 1.5, 1.85, 1.5, "Data/\nCSV files", fc=LGREY, ec=GREY,
    bold=True, fs=7.8)

box(ax, 2.7, 6.6, 2.3, 1.5, "1.0\nRead & Validate\nUpload\n(app.py,\ndataset_validator)",
    fc=LRED, ec=RED, fs=7.5)
box(ax, 5.6, 6.6, 2.3, 1.5, "2.0\nClean &\nEngineer Features\n(clean_data,\nfeature_engineering)",
    fc=LORANGE, ec=ORANGE, fs=7.5)
box(ax, 5.6, 3.9, 2.3, 1.5, "3.0\nPredict Churn\n(prediction_\npipeline_backup)",
    fc=LIGHT, ec=NAVY, fs=7.5)
box(ax, 2.7, 3.9, 2.3, 1.5, "4.0\nScore Risk &\nSegment\n(segmentation)",
    fc=LGREEN, ec=GREEN, fs=7.5)
box(ax, 2.7, 1.2, 2.3, 1.5, "5.0\nGenerate\nRecommendations\n(recommendations)",
    fc=LGREEN, ec=GREEN, fs=7.5)
box(ax, 5.6, 1.2, 2.3, 1.5, "6.0\nRender Dashboard\n& CSV Reports\n(app.py)",
    fc=LIGHT, ec=NAVY, fs=7.5)
box(ax, 8.3, 3.9, 1.55, 1.5, "Models/\nbest_churn_\nmodel.pkl",
    fc=LGREY, ec=GREY, bold=True, fs=7.5)

arrow(ax, (2.0, 7.35), (2.7, 7.35))
arrow(ax, (5.0, 7.35), (5.6, 7.35))
arrow(ax, (6.75, 6.6), (6.75, 5.4))
arrow(ax, (8.3, 4.65), (7.9, 4.65))
arrow(ax, (5.6, 4.65), (5.0, 4.65))
arrow(ax, (3.85, 3.9), (3.85, 2.7))
arrow(ax, (5.0, 1.95), (5.6, 1.95))
arrow(ax, (5.6, 1.95), (2.0, 6.6), color=BLUE, rad=0.30)
arrow(ax, (2.0, 2.25), (2.7, 6.75), color=GREY, rad=0.22)
ax.text(0.95, 3.55, "default\ndataset", fontsize=6.8, ha="center", color=GREY)
ax.text(9.05, 5.75, "trained\npipeline", fontsize=6.8, ha="center", color=GREY)
ax.text(2.35, 5.15, "dashboards,\ncharts & CSV\nreports", fontsize=6.8,
        ha="center", color=BLUE)
save(fig, "dia_dataflow.png")


# ==================================================================
# 6 Preprocessing pipeline (ColumnTransformer)
# ==================================================================
fig, ax = canvas(7.6, 4.3, "Preprocessing Pipeline (feature_engineering.build_preprocessor)")

box(ax, 0.3, 7.6, 9.4, 1.4,
    "Input DataFrame  —  19 predictive features\n"
    "(customerID removed as identifier · Churn removed as target)",
    fc=LGREY, ec=GREY, bold=True, fs=8)

box(ax, 0.5, 4.0, 4.3, 2.9,
    "NUMERIC BRANCH  (4 features)\nSeniorCitizen · tenure\nMonthlyCharges · TotalCharges\n\n"
    "SimpleImputer(strategy='median')\n↓\nStandardScaler()",
    fc=LIGHT, ec=NAVY, fs=8)

box(ax, 5.2, 4.0, 4.3, 2.9,
    "CATEGORICAL BRANCH  (15 features)\ngender · Partner · Dependents · Contract\nPaymentMethod · InternetService · …\n\n"
    "SimpleImputer(strategy='most_frequent')\n↓\nOneHotEncoder(handle_unknown='ignore')",
    fc=LGREEN, ec=GREEN, fs=8)

box(ax, 0.3, 2.0, 9.4, 1.4,
    "ColumnTransformer(remainder='drop')\nSingle fitted preprocessing object",
    fc=LORANGE, ec=ORANGE, bold=True, fs=8.5)

box(ax, 0.3, 0.3, 9.4, 1.2,
    "sklearn Pipeline([('preprocessor', …), ('model', …)])  →  Models/best_churn_model.pkl",
    fc=LRED, ec=RED, bold=True, fs=8.5)

arrow(ax, (2.65, 7.6), (2.65, 6.9), color=NAVY)
arrow(ax, (7.35, 7.6), (7.35, 6.9), color=GREEN)
arrow(ax, (2.65, 4.0), (2.65, 3.4), color=NAVY)
arrow(ax, (7.35, 4.0), (7.35, 3.4), color=GREEN)
arrow(ax, (5.0, 2.0), (5.0, 1.5), color=ORANGE)
save(fig, "dia_preprocessing.png")


# ==================================================================
# 8 Prediction fallback strategy
# ==================================================================
fig, ax = canvas(7.6, 5.2, "Three-Tier Prediction Strategy (predict_customers)")

box(ax, 2.6, 8.7, 4.8, 0.95, "Uploaded / Active Dataset", fc=LGREY, ec=GREY,
    bold=True, fs=8.5)
box(ax, 2.6, 7.35, 4.8, 0.95, "clean_dataset()  →  engineer_features()\n→ detect_target_column()",
    fc=LRED, ec=RED, fs=8)

box(ax, 0.35, 5.35, 9.3, 1.35,
    "METHOD 1 — Existing ChurnIQ ML Model\n"
    "Loads Models/best_churn_model.pkl; used only if the saved metadata's\n"
    "feature columns are present in the uploaded data",
    fc=LIGHT, ec=NAVY, bold=True, fs=8)

box(ax, 0.35, 3.35, 9.3, 1.35,
    "METHOD 2 — Adaptive ML Model\n"
    "If the saved model is incompatible but a churn label exists, trains a\n"
    "Logistic Regression / Random Forest pipeline on the uploaded data",
    fc=LGREEN, ec=GREEN, bold=True, fs=8)

box(ax, 0.35, 1.35, 9.3, 1.35,
    "METHOD 3 — Behavioral Risk Engine\n"
    "If no usable label exists, derives a heuristic risk estimate from tenure,\n"
    "recency, usage, support, payment and spend signals (not a trained classifier)",
    fc=LORANGE, ec=ORANGE, bold=True, fs=8)

box(ax, 0.35, 0.15, 9.3, 0.85,
    "Output columns: Churn Prediction · Churn Probability · Risk Score · "
    "Risk Level · Prediction Method",
    fc="#ffffff", ec=GREY, fs=8)

arrow(ax, (5.0, 8.7), (5.0, 8.3), color=NAVY)
arrow(ax, (5.0, 7.35), (5.0, 6.7), color=NAVY)
arrow(ax, (0.9, 5.35), (0.9, 4.7), color=RED)
arrow(ax, (0.9, 3.35), (0.9, 2.7), color=RED)
ax.text(1.35, 5.0, "incompatible", fontsize=7, color=RED)
ax.text(1.35, 3.0, "no model / no label", fontsize=7, color=RED)
arrow(ax, (9.15, 5.35), (9.15, 1.05), color=NAVY, rad=-0.28)
save(fig, "dia_prediction_strategy.png")


# ==================================================================
# 5.7 Module dependency map
# ==================================================================
fig, ax = canvas(7.6, 5.4, "Module Dependency Map (as imported in the repository)")

box(ax, 3.1, 8.6, 3.8, 1.0, "app.py\nStreamlit entry point", fc=LIGHT, ec=NAVY,
    bold=True, fs=8.5)

leaves = [
    (0.15, 6.2, "prediction_\npipeline_backup.py", LORANGE, ORANGE),
    (2.15, 6.2, "segmentation.py", LGREEN, GREEN),
    (4.15, 6.2, "recommendations.py", LGREEN, GREEN),
    (6.15, 6.2, "dashboard_\nanalytics.py", LGREEN, GREEN),
    (8.15, 6.2, "database.py\n(optional)", LGREY, GREY),
]
for x, y, t, fc, ec in leaves:
    box(ax, x, y, 1.75, 1.15, t, fc=fc, ec=ec, fs=7.2)
    arrow(ax, (5.0, 8.6), (x + 0.875, y + 1.15), color=NAVY, rad=0.05)

box(ax, 0.6, 3.7, 2.6, 1.0, "model_training.py", fc=LORANGE, ec=ORANGE, fs=8)
box(ax, 3.7, 3.7, 2.6, 1.0, "feature_engineering.py", fc=LORANGE, ec=ORANGE, fs=8)
box(ax, 6.8, 3.7, 2.6, 1.0, "clean_data.py", fc=LRED, ec=RED, fs=8)
box(ax, 3.7, 1.9, 2.6, 1.0, "dataset_validator.py", fc=LRED, ec=RED, bold=True, fs=8)

arrow(ax, (3.2, 4.2), (3.7, 4.2), color=ORANGE)
arrow(ax, (5.0, 3.7), (5.0, 2.9), color=RED)
arrow(ax, (6.8, 4.0), (6.3, 2.7), color=RED, rad=0.2)

box(ax, 0.6, 0.35, 4.0, 1.0,
    "prediction_pipeline.py\n(present but not imported by app.py)",
    fc="#ffffff", ec=GREY, fs=7.2)
box(ax, 5.4, 0.35, 4.0, 1.0,
    "eda.py\n(standalone analysis module)",
    fc="#ffffff", ec=GREY, fs=7.2)
save(fig, "dia_modules.png")


# ==================================================================
# 10 Dashboard navigation map
# ==================================================================
fig, ax = canvas(7.6, 4.6, "ChurnIQ Dashboard Navigation Structure (app.py sidebar)")

box(ax, 3.4, 8.5, 3.2, 1.1, "Sidebar Navigation\n(st.sidebar.radio)", fc=LIGHT,
    ec=NAVY, bold=True, fs=8.5)

pages = [
    "1. Executive\nOverview", "2. Dataset\nCenter", "3. Churn\nAnalysis",
    "4. Customer\nSegments", "5. Risk\nCenter", "6. Prediction\nCenter",
    "7. Model\nPerformance", "8. Customer\nExplorer", "9. Recommen-\ndations",
    "10. Reports", "11. About",
]
cols, w, hgap = 4, 2.15, 0.28
x0 = (10 - (cols * w + (cols - 1) * hgap)) / 2
y0, hh, vgap = 5.9, 1.15, 0.42
for i, p in enumerate(pages):
    r, c = divmod(i, cols)
    x = x0 + c * (w + hgap)
    y = y0 - r * (hh + vgap)
    box(ax, x, y, w, hh, p, fc=LGREEN if i < 6 else LORANGE,
        ec=GREEN if i < 6 else ORANGE, fs=7.2)
    if r == 0:
        arrow(ax, (5.0, 8.5), (x + w / 2, y + hh), color=NAVY, rad=0.06)

box(ax, 0.5, 0.35, 9.0, 0.95,
    "Shared session state:  current_data · prediction_result · "
    "intelligence_result · recommendation_result",
    fc=LGREY, ec=GREY, fs=8)
save(fig, "dia_dashboard_nav.png")

print("all diagrams written to", OUT)
