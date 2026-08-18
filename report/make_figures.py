"""
ChurnIQ - Report figure generator.

Generates all analytical figures used in the project report directly from
the repository's actual data files, actual model artefacts and actual
Src/ modules. No values are hard-coded.

Run:  python report/make_figures.py
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
SRC = BASE / "Src"
DATA = BASE / "Data"
MODELS = BASE / "Models"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SRC))

PRIMARY = "#1f4e79"
ACCENT = "#c0392b"
GREY = "#7f8c8d"
PALETTE = ["#1f4e79", "#c0392b", "#e67e22", "#27ae60", "#8e44ad", "#16a085"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "axes.edgecolor": "#4d4d4d",
    "axes.grid": True,
    "grid.color": "#d9d9d9",
    "grid.linewidth": 0.5,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def save(fig, name):
    path = OUT / name
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print("wrote", path.name)


def bar_labels(ax, bars, fmt="{:.1f}%", dy=0.6):
    for b in bars:
        ax.annotate(
            fmt.format(b.get_height()),
            (b.get_x() + b.get_width() / 2, b.get_height() + dy),
            ha="center", va="bottom", fontsize=8,
        )


# ------------------------------------------------------------------
# Load actual data
# ------------------------------------------------------------------
raw = pd.read_csv(DATA / "Telco-Customer-Churn.csv")
clean = pd.read_csv(DATA / "telco_churn_clean.csv")
results = pd.read_csv(MODELS / "model_results.csv")

work = raw.copy()
work["TotalChargesNum"] = pd.to_numeric(work["TotalCharges"], errors="coerce")
work["ChurnFlag"] = (work["Churn"] == "Yes").astype(int)


def churn_rate_by(col):
    g = work.groupby(col)["ChurnFlag"].agg(["mean", "count"])
    g["rate"] = g["mean"] * 100
    return g


# ------------------------------------------------------------------
# Fig: churn distribution
# ------------------------------------------------------------------
counts = work["Churn"].value_counts()
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.0))
a1.pie(
    [counts["No"], counts["Yes"]],
    labels=["Retained", "Churned"],
    autopct="%1.2f%%", startangle=90,
    colors=[PRIMARY, ACCENT],
    textprops={"color": "white", "fontsize": 9, "weight": "bold"},
)
a1.set_title("Churn Distribution (Proportion)")
a1.grid(False)
b = a2.bar(["Retained", "Churned"], [counts["No"], counts["Yes"]],
           color=[PRIMARY, ACCENT], width=0.55)
bar_labels(a2, b, "{:,.0f}", dy=60)
a2.set_ylabel("Number of Customers")
a2.set_title("Churn Distribution (Counts)")
a2.set_ylim(0, counts.max() * 1.18)
save(fig, "fig_churn_distribution.png")

# ------------------------------------------------------------------
# Fig: contract
# ------------------------------------------------------------------
c = churn_rate_by("Contract").reindex(["Month-to-month", "One year", "Two year"])
fig, ax = plt.subplots(figsize=(5.4, 3.0))
bars = ax.bar(c.index, c["rate"], color=[ACCENT, "#e67e22", "#27ae60"], width=0.55)
bar_labels(ax, bars)
ax.set_ylabel("Churn Rate (%)")
ax.set_xlabel("Contract Type")
ax.set_title("Churn Rate by Contract Type")
ax.set_ylim(0, c["rate"].max() * 1.25)
save(fig, "fig_contract.png")

# ------------------------------------------------------------------
# Fig: payment method
# ------------------------------------------------------------------
p = churn_rate_by("PaymentMethod").sort_values("rate")
fig, ax = plt.subplots(figsize=(6.4, 3.0))
bars = ax.barh(p.index, p["rate"], color=PRIMARY, height=0.6)
bars[-1].set_color(ACCENT)
for bar in bars:
    ax.annotate(f"{bar.get_width():.1f}%",
                (bar.get_width() + 0.7, bar.get_y() + bar.get_height() / 2),
                va="center", fontsize=8)
ax.set_xlabel("Churn Rate (%)")
ax.set_title("Churn Rate by Payment Method")
ax.set_xlim(0, p["rate"].max() * 1.22)
save(fig, "fig_payment.png")

# ------------------------------------------------------------------
# Fig: tenure groups (matches Src/dashboard_analytics bands)
# ------------------------------------------------------------------
bins = [-0.1, 6, 12, 24, 36, 60, np.inf]
labels = ["0–6", "7–12", "13–24", "25–36", "37–60", "60+"]
work["TenureGroup"] = pd.cut(work["tenure"], bins=bins, labels=labels)
t = work.groupby("TenureGroup", observed=True)["ChurnFlag"].agg(["mean", "count"])
t["rate"] = t["mean"] * 100
fig, ax = plt.subplots(figsize=(6.2, 3.0))
bars = ax.bar(t.index.astype(str), t["rate"], color=PRIMARY, width=0.6)
bars[0].set_color(ACCENT)
bar_labels(ax, bars)
ax.plot(range(len(t)), t["rate"], color="#333333", lw=1.2, marker="o", ms=3.5)
ax.set_ylabel("Churn Rate (%)")
ax.set_xlabel("Tenure Group (months)")
ax.set_title("Churn Rate by Customer Tenure")
ax.set_ylim(0, t["rate"].max() * 1.25)
save(fig, "fig_tenure.png")

# ------------------------------------------------------------------
# Fig: internet service + tech support
# ------------------------------------------------------------------
i = churn_rate_by("InternetService").reindex(["DSL", "Fiber optic", "No"])
s = churn_rate_by("TechSupport").reindex(["No", "Yes", "No internet service"])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.0))
b1 = a1.bar(i.index, i["rate"], color=[PRIMARY, ACCENT, "#27ae60"], width=0.55)
bar_labels(a1, b1)
a1.set_title("Churn Rate by Internet Service")
a1.set_ylabel("Churn Rate (%)")
a1.set_ylim(0, i["rate"].max() * 1.25)
b2 = a2.bar(["No", "Yes", "No internet"], s["rate"],
            color=[ACCENT, "#27ae60", GREY], width=0.55)
bar_labels(a2, b2)
a2.set_title("Churn Rate by Tech Support")
a2.set_ylabel("Churn Rate (%)")
a2.set_ylim(0, s["rate"].max() * 1.25)
save(fig, "fig_service.png")

# ------------------------------------------------------------------
# Fig: demographics
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.7))
for ax, col, title in zip(
    axes,
    ["gender", "SeniorCitizen", "Partner", "Dependents"],
    ["Gender", "Senior Citizen", "Partner", "Dependents"],
):
    d = churn_rate_by(col)
    idx = [str(x) for x in d.index]
    bars = ax.bar(idx, d["rate"], color=PALETTE[:len(d)], width=0.55)
    for bar in bars:
        ax.annotate(f"{bar.get_height():.1f}",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6),
                    ha="center", fontsize=7)
    ax.set_title(title, fontsize=9)
    ax.set_ylim(0, 50)
    ax.tick_params(labelsize=7)
axes[0].set_ylabel("Churn Rate (%)")
fig.suptitle("Churn Rate by Customer Demographics", fontsize=10, fontweight="bold")
save(fig, "fig_demographics.png")

# ------------------------------------------------------------------
# Fig: numeric distributions
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
pairs = [("tenure", "Tenure (months)"),
         ("MonthlyCharges", "Monthly Charges"),
         ("TotalChargesNum", "Total Charges")]
for ax, (col, label) in zip(axes, pairs):
    ax.hist(work.loc[work.ChurnFlag == 0, col].dropna(), bins=30, alpha=0.75,
            label="Retained", color=PRIMARY)
    ax.hist(work.loc[work.ChurnFlag == 1, col].dropna(), bins=30, alpha=0.75,
            label="Churned", color=ACCENT)
    ax.set_xlabel(label)
    ax.tick_params(labelsize=7)
axes[0].set_ylabel("Customers")
axes[0].legend(fontsize=7)
fig.suptitle("Distribution of Numerical Features by Churn Status",
             fontsize=10, fontweight="bold")
save(fig, "fig_numeric_dist.png")

# ------------------------------------------------------------------
# Fig: correlation of numeric features
# ------------------------------------------------------------------
corr = work[["tenure", "MonthlyCharges", "TotalChargesNum",
             "SeniorCitizen", "ChurnFlag"]].corr()
corr.index = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", "Churn"]
corr.columns = corr.index
fig, ax = plt.subplots(figsize=(4.4, 3.6))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(corr)), corr.index, fontsize=8)
for r in range(len(corr)):
    for c2 in range(len(corr)):
        v = corr.iloc[r, c2]
        ax.text(c2, r, f"{v:.2f}", ha="center", va="center", fontsize=7,
                color="white" if abs(v) > 0.55 else "black")
ax.grid(False)
ax.set_title("Correlation Matrix of Numerical Features")
fig.colorbar(im, ax=ax, shrink=0.8)
save(fig, "fig_correlation.png")

# ------------------------------------------------------------------
# Fig: model comparison (actual Models/model_results.csv)
# ------------------------------------------------------------------
metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
x = np.arange(len(results))
w = 0.16
fig, ax = plt.subplots(figsize=(7.4, 3.4))
for k, m in enumerate(metrics):
    ax.bar(x + (k - 2) * w, results[m] * 100, width=w, label=m, color=PALETTE[k])
ax.set_xticks(x, [n.replace(" ", "\n") for n in results["Model"]], fontsize=7.5)
ax.set_ylabel("Score (%)")
ax.set_ylim(0, 100)
ax.set_title("Model Performance Comparison (Models/model_results.csv)")
ax.legend(fontsize=7.5, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.16))
save(fig, "fig_model_comparison.png")

# ------------------------------------------------------------------
# Fig: F1 vs ROC-AUC selection trade-off
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.0, 3.2))
ax.scatter(results["ROC-AUC"] * 100, results["F1 Score"] * 100,
           s=70, color=PRIMARY, zorder=3)
for _, r in results.iterrows():
    ax.annotate(r["Model"], (r["ROC-AUC"] * 100, r["F1 Score"] * 100),
                textcoords="offset points", xytext=(6, 4), fontsize=7.5)
best_f1 = results.loc[results["F1 Score"].idxmax()]
best_auc = results.loc[results["ROC-AUC"].idxmax()]
ax.scatter([best_f1["ROC-AUC"] * 100], [best_f1["F1 Score"] * 100],
           s=140, facecolors="none", edgecolors=ACCENT, lw=1.8, zorder=4)
ax.scatter([best_auc["ROC-AUC"] * 100], [best_auc["F1 Score"] * 100],
           s=140, facecolors="none", edgecolors="#27ae60", lw=1.8, zorder=4)
ax.set_xlabel("ROC-AUC (%)")
ax.set_ylabel("F1 Score (%)")
ax.set_title("Model Selection Trade-off: F1 Score vs ROC-AUC")
save(fig, "fig_model_tradeoff.png")

# ------------------------------------------------------------------
# Fig: confusion matrices of saved model vs dashboard-ranked model
# ------------------------------------------------------------------
def cm_of(row):
    return np.array([[row["True Negative"], row["False Positive"]],
                     [row["False Negative"], row["True Positive"]]])

rf = results[results["Model"] == "Random Forest"].iloc[0]
gb = results[results["Model"] == "Gradient Boosting"].iloc[0]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
for ax, row, title in zip(axes, [rf, gb],
                          ["Random Forest (saved best_churn_model.pkl)",
                           "Gradient Boosting (highest ROC-AUC)"]):
    m = cm_of(row)
    im = ax.imshow(m, cmap="Blues")
    for r in range(2):
        for c2 in range(2):
            ax.text(c2, r, f"{m[r, c2]:,}", ha="center", va="center", fontsize=10,
                    color="white" if m[r, c2] > m.max() * 0.55 else "black")
    ax.set_xticks([0, 1], ["Pred: Retained", "Pred: Churn"], fontsize=8)
    ax.set_yticks([0, 1], ["Actual: Retained", "Actual: Churn"], fontsize=8)
    ax.set_title(title, fontsize=8.5)
    ax.grid(False)
save(fig, "fig_confusion.png")

# ------------------------------------------------------------------
# Pipeline outputs (actual Src modules executed on the clean dataset)
# ------------------------------------------------------------------
from prediction_pipeline_backup import predict_customer_churn  # noqa: E402
from segmentation import create_customer_intelligence, get_segment_summary  # noqa: E402
from recommendations import create_recommendations, get_business_summary  # noqa: E402

pred = predict_customer_churn(clean)
intel = create_customer_intelligence(pred)
rec = create_recommendations(intel)
seg_summary = get_segment_summary(intel)
biz = get_business_summary(rec)

rec.to_csv(OUT.parent / "pipeline_output_sample.csv", index=False)

# Fig: risk levels
order = ["Low", "Medium", "High", "Critical"]
rl = intel["Risk Level"].value_counts().reindex(order).fillna(0)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.0))
colors = ["#27ae60", "#e67e22", "#d35400", ACCENT]
bars = a1.bar(order, rl.values, color=colors, width=0.55)
bar_labels(a1, bars, "{:,.0f}", dy=40)
a1.set_ylabel("Customers")
a1.set_title("Customer Distribution by Risk Level")
a1.set_ylim(0, rl.max() * 1.2)
a2.hist(pd.to_numeric(intel["Churn Probability"], errors="coerce").dropna(),
        bins=40, color=PRIMARY)
a2.set_xlabel("Churn Probability (%)")
a2.set_ylabel("Customers")
a2.set_title("Churn Probability Distribution")
save(fig, "fig_risk_levels.png")

# Fig: strategic segments
ss = seg_summary.sort_values("Customers", ascending=True)
fig, ax = plt.subplots(figsize=(6.6, 3.2))
bars = ax.barh(ss["Segment"], ss["Customers"], color=PRIMARY, height=0.6)
for bar, risk in zip(bars, ss["Avg_Risk_Score"]):
    ax.annotate(f"{bar.get_width():,.0f}  (avg risk {risk:.1f})",
                (bar.get_width() + 40, bar.get_y() + bar.get_height() / 2),
                va="center", fontsize=7.5)
ax.set_xlabel("Number of Customers")
ax.set_title("Strategic Customer Segments Produced by segmentation.py")
ax.set_xlim(0, ss["Customers"].max() * 1.45)
save(fig, "fig_segments.png")

# Fig: revenue at risk by segment
ss2 = seg_summary.sort_values("Revenue_At_Risk", ascending=False)
fig, ax = plt.subplots(figsize=(6.6, 3.0))
bars = ax.bar(ss2["Segment"], ss2["Revenue_At_Risk"], color=ACCENT, width=0.55)
for bar in bars:
    ax.annotate(f"{bar.get_height():,.0f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center", va="bottom", fontsize=7.5)
ax.set_ylabel("Revenue at Risk")
ax.set_xticks(range(len(ss2)), [s.replace(" ", "\n") for s in ss2["Segment"]],
              fontsize=7.5)
ax.set_title("Estimated Revenue at Risk by Strategic Segment")
ax.set_ylim(0, ss2["Revenue_At_Risk"].max() * 1.18)
save(fig, "fig_revenue_risk.png")

# Fig: retention priority / channel
pr = rec["Retention Priority"].value_counts().reindex(
    ["Immediate", "High", "Monitor", "Low"]).fillna(0)
ch = rec["Recommended Channel"].value_counts()
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.0))
bars = a1.bar(pr.index, pr.values, color=[ACCENT, "#d35400", "#e67e22", "#27ae60"],
              width=0.55)
bar_labels(a1, bars, "{:,.0f}", dy=40)
a1.set_ylabel("Customers")
a1.set_title("Retention Priority Assignment")
a1.set_ylim(0, pr.max() * 1.2)
a2.pie(ch.values, labels=[c.replace(" / ", "/\n") for c in ch.index],
       autopct="%1.1f%%", colors=PALETTE[:len(ch)],
       textprops={"fontsize": 7.5})
a2.set_title("Recommended Engagement Channel")
a2.grid(False)
save(fig, "fig_recommendations.png")

# ------------------------------------------------------------------
# Console summary consumed while writing the report
# ------------------------------------------------------------------
print("\n--- pipeline summary ---")
print("prediction method:", pd.unique(pred["Prediction Method"])[0])
print(intel["Risk Level"].value_counts().to_string())
print(seg_summary.to_string())
print(rec["Retention Priority"].value_counts().to_string())
for k, v in biz.items():
    print(f"{k}: {v}")
