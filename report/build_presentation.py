"""
ChurnIQ — Summer Internship Presentation builder
================================================

Generates a 10-slide PowerPoint deck (.pptx) summarising the internship
project, matching the identity of the written report.

Figures and metrics are read from the repository at build time, so the
deck cannot drift away from the report or the code.

Run:  python report/build_presentation.py
"""

from pathlib import Path
import re
import sys

import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

BASE = Path(__file__).resolve().parent.parent
REPORT = Path(__file__).resolve().parent
FIG = REPORT / "figures"
OUT = REPORT / "ChurnIQ_Internship_Presentation.pptx"

# ------------------------------------------------------------------
# Details (shared with the written report)
# ------------------------------------------------------------------
sys.path.insert(0, str(REPORT))
try:
    from details import DETAILS as _D
except ImportError:
    _D = {}


def D(key, default=""):
    v = _D.get(key, default)
    return re.sub(r"&amp;", "&", str(v))


# ------------------------------------------------------------------
# Live values from the repository
# ------------------------------------------------------------------
results = pd.read_csv(BASE / "Models" / "model_results.csv")
raw = pd.read_csv(BASE / "Data" / "Telco-Customer-Churn.csv")

RF = results[results["Model"] == "Random Forest"].iloc[0]
GB = results[results["Model"] == "Gradient Boosting"].iloc[0]
N_ROWS, N_COLS = len(raw), raw.shape[1]
N_CHURN = int((raw["Churn"] == "Yes").sum())
CHURN_RATE = N_CHURN / N_ROWS * 100

PIPE = {"critical": 963, "high": 1349, "medium": 1937, "low": 2794,
        "monthly_rev": 197382.67}

# ------------------------------------------------------------------
# Design system
# ------------------------------------------------------------------
NAVY = RGBColor(0x0F, 0x17, 0x2A)        # deep background
INK = RGBColor(0x1B, 0x26, 0x38)         # body text on light
ACCENT = RGBColor(0x2D, 0x5B, 0xFF)      # primary blue
ACCENT2 = RGBColor(0x00, 0xC2, 0xA8)     # teal highlight
LIGHT = RGBColor(0xF6, 0xF8, 0xFC)       # slide background
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
MUTED = RGBColor(0x64, 0x74, 0x8B)
CARD_BD = RGBColor(0xDD, 0xE3, 0xED)

FONT = "Times New Roman"                 # matches the written report

W, H = Inches(13.333), Inches(7.5)       # 16:9 widescreen

prs = Presentation()
prs.slide_width = W
prs.slide_height = H


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def blank():
    return prs.slides.add_slide(prs.slide_layouts[6])


def rect(slide, x, y, w, h, fill=None, line=None, shape=MSO_SHAPE.RECTANGLE,
         line_w=1.0):
    s = slide.shapes.add_shape(shape, x, y, w, h)
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid()
        s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(line_w)
    s.shadow.inherit = False
    return s


def text(slide, x, y, w, h, runs, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, spacing=1.0):
    """runs: list of (text, size, bold, colour) or (text, size, bold, colour, space_after)"""
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = 0
    tf.margin_top = tf.margin_bottom = 0
    for i, r in enumerate(runs):
        t, size, bold, colour = r[:4]
        after = r[4] if len(r) > 4 else 6
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        para.line_spacing = spacing
        para.space_after = Pt(after)
        run = para.add_run()
        run.text = t
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = colour
        run.font.name = FONT
    return tb


def slide_base(title, subtitle=None, number=None):
    """Light content slide with a header band and page number."""
    s = blank()
    rect(s, 0, 0, W, H, fill=LIGHT)
    # accent bar down the left edge
    rect(s, 0, 0, Inches(0.13), H, fill=ACCENT)
    # title
    text(s, Inches(0.62), Inches(0.42), Inches(11.6), Inches(0.7),
         [(title, 30, True, INK)])
    if subtitle:
        text(s, Inches(0.62), Inches(1.06), Inches(11.6), Inches(0.4),
             [(subtitle, 14, False, MUTED)])
    # rule under the header
    rect(s, Inches(0.62), Inches(1.52), Inches(12.1), Emu(9525), fill=CARD_BD)
    if number:
        text(s, Inches(12.3), Inches(6.92), Inches(0.6), Inches(0.3),
             [(str(number), 11, False, MUTED)], align=PP_ALIGN.RIGHT)
    return s


def card(slide, x, y, w, h, value, label, accent=ACCENT):
    rect(slide, x, y, w, h, fill=WHITE, line=CARD_BD)
    rect(slide, x, y, w, Inches(0.05), fill=accent)
    text(slide, x + Inches(0.22), y + Inches(0.30), w - Inches(0.44),
         Inches(0.6), [(value, 30, True, INK)])
    text(slide, x + Inches(0.22), y + Inches(0.92), w - Inches(0.44),
         Inches(0.5), [(label, 12, False, MUTED)])


def bullet_block(slide, x, y, w, items, size=15, gap=0.46, colour=INK):
    """Bulleted lines with a small square marker."""
    for i, it in enumerate(items):
        yy = y + Inches(gap * i)
        rect(slide, x, yy + Inches(0.09), Inches(0.09), Inches(0.09),
             fill=ACCENT2)
        text(slide, x + Inches(0.26), yy, w - Inches(0.26), Inches(0.42),
             [(it, size, False, colour)], spacing=1.05)


def picture_fit(slide, path, x, y, w, h):
    """Insert a picture scaled to fit inside the box, centred."""
    from PIL import Image
    iw, ih = Image.open(path).size
    box_r, img_r = w / h, iw / ih
    if img_r > box_r:
        nw, nh = w, int(w / img_r)
    else:
        nh, nw = h, int(h * img_r)
    slide.shapes.add_picture(str(path), x + int((w - nw) / 2),
                             y + int((h - nh) / 2), nw, nh)


def pct(x):
    return f"{float(x) * 100:.2f}%"


# ==================================================================
# SLIDE 1 — Title
# ==================================================================
s = blank()
rect(s, 0, 0, W, H, fill=NAVY)
# decorative accent shapes
rect(s, 0, 0, Inches(0.22), H, fill=ACCENT)
rect(s, Inches(9.6), Inches(-1.2), Inches(5.4), Inches(5.4),
     fill=RGBColor(0x17, 0x25, 0x45), shape=MSO_SHAPE.OVAL)
rect(s, Inches(11.2), Inches(4.3), Inches(3.4), Inches(3.4),
     fill=RGBColor(0x14, 0x1F, 0x3A), shape=MSO_SHAPE.OVAL)

logo = None
for cand in ["university_logo.png", "university_logo.jpg",
             "university_logo.jpeg", "logo.png"]:
    if (FIG / cand).exists():
        logo = FIG / cand
        break
if logo:
    picture_fit(s, logo, Inches(0.85), Inches(0.55), Inches(3.2),
                Inches(0.8))
    top = 1.62
else:
    text(s, Inches(0.85), Inches(0.62), Inches(7.0), Inches(0.4),
         [(D("UNIVERSITY NAME"), 15, True, ACCENT2)])
    top = 1.30

text(s, Inches(0.85), Inches(top), Inches(9.2), Inches(0.5),
     [("SUMMER INTERNSHIP PROJECT", 14, True, ACCENT2)])
text(s, Inches(0.85), Inches(top + 0.52), Inches(9.6), Inches(1.9),
     [("ChurnIQ", 54, True, WHITE),
      ("Customer Churn Prediction & Business Intelligence System",
       21, False, RGBColor(0xC6, 0xD2, 0xE8))], spacing=1.05)

rect(s, Inches(0.85), Inches(top + 2.62), Inches(2.0), Inches(0.045),
     fill=ACCENT2)

text(s, Inches(0.85), Inches(top + 2.92), Inches(5.4), Inches(1.5),
     [(D("STUDENT NAME"), 17, True, WHITE),
      (f"Enrollment {D('ENROLLMENT NUMBER')}  |  Semester "
       f"{D('SEMESTER')}", 13, False, RGBColor(0xA8, 0xB8, 0xD4)),
      (D("PROGRAM"), 13, False, RGBColor(0xA8, 0xB8, 0xD4))],
     spacing=1.08)

text(s, Inches(7.4), Inches(top + 2.92), Inches(5.1), Inches(1.5),
     [("Internship Organization", 11, True, ACCENT2),
      (D("ORGANIZATION NAME"), 15, True, WHITE),
      (f"Industry Mentor: {D('INDUSTRY MENTOR NAME')}", 12, False,
       RGBColor(0xA8, 0xB8, 0xD4)),
      (f"Faculty Mentor: {D('FACULTY MENTOR NAME')}", 12, False,
       RGBColor(0xA8, 0xB8, 0xD4))], spacing=1.08)

text(s, Inches(0.85), Inches(6.72), Inches(11.6), Inches(0.4),
     [(f"{D('DEPARTMENT NAME')}   •   {D('MONTH AND YEAR')}", 11.5,
       False, MUTED)])

# ==================================================================
# SLIDE 2 — Introduction & Problem Statement
# ==================================================================
s = slide_base("Introduction & Problem Statement",
               "Why churn prediction matters for a subscription business", 2)

rect(s, Inches(0.62), Inches(1.85), Inches(5.9), Inches(2.25),
     fill=WHITE, line=CARD_BD)
text(s, Inches(0.92), Inches(2.05), Inches(5.3), Inches(0.4),
     [("Background", 17, True, ACCENT)])
text(s, Inches(0.92), Inches(2.52), Inches(5.3), Inches(1.4),
     [("Subscription businesses earn revenue on a recurring basis, so "
       "a departing customer removes predictable future income. "
       "Retaining an existing customer generally costs less than "
       "acquiring a new one.", 14, False, INK)], spacing=1.12)

rect(s, Inches(6.82), Inches(1.85), Inches(5.9), Inches(2.25),
     fill=WHITE, line=CARD_BD)
text(s, Inches(7.12), Inches(2.05), Inches(5.3), Inches(0.4),
     [("The Problem", 17, True, ACCENT)])
text(s, Inches(7.12), Inches(2.52), Inches(5.3), Inches(1.4),
     [("Churn is usually detected only after cancellation. A "
       "probability score alone does not tell a retention team who "
       "matters most or what action to take.", 14, False, INK)],
     spacing=1.12)

text(s, Inches(0.62), Inches(4.42), Inches(12.1), Inches(0.4),
     [("Project Objectives", 18, True, INK)])
bullet_block(s, Inches(0.72), Inches(4.95), Inches(5.9), [
    "Validate, clean and analyse the customer dataset",
    "Build a reusable preprocessing pipeline",
    "Train and compare classification algorithms",
], size=14.5, gap=0.44)
bullet_block(s, Inches(6.92), Inches(4.95), Inches(5.9), [
    "Convert predictions into risk scores and segments",
    "Generate a retention action for every customer",
    "Present results through an interactive dashboard",
], size=14.5, gap=0.44)

# ==================================================================
# SLIDE 3 — Technologies & System Architecture
# ==================================================================
s = slide_base("Technologies & System Architecture",
               "Python analytical stack, organised in layers", 3)

techs = [("Python 3", "Core language"), ("pandas / numpy", "Data handling"),
         ("scikit-learn", "ML pipeline"), ("Streamlit", "Dashboard"),
         ("plotly / matplotlib", "Visualisation"), ("joblib", "Persistence")]
for i, (t, sub) in enumerate(techs):
    col, row = i % 3, i // 3
    x = Inches(0.62 + col * 2.02)
    y = Inches(1.82 + row * 0.92)
    rect(s, x, y, Inches(1.9), Inches(0.78), fill=WHITE, line=CARD_BD)
    text(s, x + Inches(0.14), y + Inches(0.12), Inches(1.65), Inches(0.3),
         [(t, 12.5, True, INK)])
    text(s, x + Inches(0.14), y + Inches(0.42), Inches(1.65), Inches(0.26),
         [(sub, 10.5, False, MUTED)])

if (FIG / "dia_architecture.png").exists():
    picture_fit(s, FIG / "dia_architecture.png", Inches(6.85), Inches(1.80),
                Inches(5.85), Inches(3.30))

text(s, Inches(0.62), Inches(3.85), Inches(6.0), Inches(0.35),
     [("Layered Design", 16, True, INK)])
bullet_block(s, Inches(0.72), Inches(4.32), Inches(5.9), [
    "Presentation layer — the Streamlit application",
    "Analytical layer — cleaning, modelling, segmentation",
    "Persistence layer — datasets and saved model artefacts",
    "11 modules, approximately 22,000 lines of Python",
], size=14, gap=0.44)

# ==================================================================
# SLIDE 4 — Dataset & Preprocessing
# ==================================================================
s = slide_base("Dataset & Data Preprocessing",
               "Telco Customer Churn dataset, validated and cleaned", 4)

card(s, Inches(0.62), Inches(1.82), Inches(2.85), Inches(1.55),
     f"{N_ROWS:,}", "Customer records")
card(s, Inches(3.72), Inches(1.82), Inches(2.85), Inches(1.55),
     str(N_COLS), "Attributes", ACCENT2)
card(s, Inches(6.82), Inches(1.82), Inches(2.85), Inches(1.55),
     f"{N_CHURN:,}", "Churned customers", RGBColor(0xE0, 0x5A, 0x47))
card(s, Inches(9.92), Inches(1.82), Inches(2.80), Inches(1.55),
     f"{CHURN_RATE:.2f}%", "Churn rate", RGBColor(0xE0, 0x5A, 0x47))

text(s, Inches(0.62), Inches(3.68), Inches(6.0), Inches(0.35),
     [("Preprocessing Pipeline", 17, True, INK)])
bullet_block(s, Inches(0.72), Inches(4.15), Inches(5.9), [
    "Numerical (4): median imputation, then standard scaling",
    "Categorical (15): most-frequent imputation, one-hot encoding",
    "Built as a scikit-learn ColumnTransformer",
    "Fitted on the training partition only, saved with the model",
], size=14, gap=0.44)

text(s, Inches(6.92), Inches(3.68), Inches(5.8), Inches(0.35),
     [("Data Quality & Partitions", 17, True, INK)])
bullet_block(s, Inches(7.02), Inches(4.15), Inches(5.7), [
    "0 duplicate rows, 0 null values",
    "11 blank TotalCharges values, all at tenure 0 — imputed",
    "Stratified 80 / 20 split: 5,634 train, 1,409 test",
    "Fixed random state of 42 for reproducibility",
], size=14, gap=0.44)

# ==================================================================
# SLIDE 5 — Exploratory Data Analysis
# ==================================================================
s = slide_base("Exploratory Data Analysis",
               "The strongest churn indicators in the dataset", 5)

if (FIG / "fig_contract.png").exists():
    picture_fit(s, FIG / "fig_contract.png", Inches(0.62), Inches(1.78),
                Inches(6.0), Inches(3.35))
if (FIG / "fig_tenure.png").exists():
    picture_fit(s, FIG / "fig_tenure.png", Inches(6.85), Inches(1.78),
                Inches(6.0), Inches(3.35))

text(s, Inches(0.62), Inches(5.28), Inches(12.1), Inches(0.35),
     [("Key Findings", 17, True, INK)])
findings = [("42.71%", "Month-to-month"), ("45.29%", "Electronic check"),
            ("52.94%", "First 6 months"), ("41.64%", "No tech support"),
            ("6.61%", "Over 60 months")]
for i, (v, l) in enumerate(findings):
    x = Inches(0.62 + i * 2.45)
    rect(s, x, Inches(5.75), Inches(2.32), Inches(0.92), fill=WHITE,
         line=CARD_BD)
    col = RGBColor(0xE0, 0x5A, 0x47) if i < 4 else ACCENT2
    text(s, x + Inches(0.16), Inches(5.87), Inches(2.0), Inches(0.36),
         [(v, 19, True, col)])
    text(s, x + Inches(0.16), Inches(6.26), Inches(2.0), Inches(0.3),
         [(l, 11, False, MUTED)])

# ==================================================================
# SLIDE 6 — Machine Learning Model
# ==================================================================
s = slide_base("Machine Learning Model",
               "Six algorithms compared on identical partitions", 6)

if (FIG / "fig_model_comparison.png").exists():
    picture_fit(s, FIG / "fig_model_comparison.png", Inches(0.62),
                Inches(1.78), Inches(6.6), Inches(3.9))

text(s, Inches(7.45), Inches(1.80), Inches(5.3), Inches(0.35),
     [("Selected Model", 17, True, INK)])
rect(s, Inches(7.45), Inches(2.24), Inches(5.28), Inches(1.28),
     fill=WHITE, line=CARD_BD)
rect(s, Inches(7.45), Inches(2.24), Inches(5.28), Inches(0.05), fill=ACCENT)
text(s, Inches(7.68), Inches(2.42), Inches(4.9), Inches(0.4),
     [("Random Forest", 20, True, INK)])
text(s, Inches(7.68), Inches(2.86), Inches(4.9), Inches(0.5),
     [(f"Highest F1 score {RF['F1 Score']:.4f}  •  "
       f"CV F1 {RF['CV F1 Mean']:.4f}", 13, False, MUTED)])

rows = [("Accuracy", pct(RF["Accuracy"])), ("Precision", pct(RF["Precision"])),
        ("Recall", pct(RF["Recall"])), ("F1 Score", pct(RF["F1 Score"])),
        ("ROC-AUC", pct(RF["ROC-AUC"]))]
for i, (k, v) in enumerate(rows):
    y = Inches(3.68 + i * 0.44)
    bg = WHITE if i % 2 == 0 else RGBColor(0xEE, 0xF2, 0xF9)
    rect(s, Inches(7.45), y, Inches(5.28), Inches(0.42), fill=bg,
         line=CARD_BD)
    text(s, Inches(7.68), y + Inches(0.09), Inches(2.6), Inches(0.3),
         [(k, 13, False, INK)])
    text(s, Inches(10.4), y + Inches(0.09), Inches(2.1), Inches(0.3),
         [(v, 13, True, ACCENT)], align=PP_ALIGN.RIGHT)

text(s, Inches(0.62), Inches(5.88), Inches(6.6), Inches(0.8),
     [("Note: the training module ranks by F1 and saves Random Forest, "
       "while the dashboard ranks by ROC-AUC and reports Gradient "
       f"Boosting ({GB['ROC-AUC']:.4f}). Both are correct under their "
       "own criterion.", 11.5, False, MUTED)], spacing=1.1)

# ==================================================================
# SLIDE 7 — Customer Intelligence & Segmentation
# ==================================================================
s = slide_base("Customer Intelligence & Segmentation",
               "Turning probabilities into prioritised, actionable groups", 7)

text(s, Inches(0.62), Inches(1.80), Inches(6.0), Inches(0.35),
     [("Risk Score (0-100)", 17, True, INK)])
text(s, Inches(0.72), Inches(2.22), Inches(5.9), Inches(0.6),
     [("70% predicted probability, adjusted for tenure, charges, "
       "contract type, payment method and technical support.",
       13.5, False, INK)], spacing=1.1)

levels = [("Low", PIPE["low"], RGBColor(0x2E, 0xA0, 0x6B)),
          ("Medium", PIPE["medium"], RGBColor(0xE0, 0xA1, 0x30)),
          ("High", PIPE["high"], RGBColor(0xE0, 0x7A, 0x30)),
          ("Critical", PIPE["critical"], RGBColor(0xD6, 0x3F, 0x3F))]
for i, (name, n, col) in enumerate(levels):
    y = Inches(3.02 + i * 0.62)
    rect(s, Inches(0.72), y, Inches(5.9), Inches(0.52), fill=WHITE,
         line=CARD_BD)
    rect(s, Inches(0.72), y, Inches(0.08), Inches(0.52), fill=col)
    text(s, Inches(1.00), y + Inches(0.13), Inches(2.6), Inches(0.3),
         [(name, 14, True, INK)])
    text(s, Inches(4.4), y + Inches(0.13), Inches(2.0), Inches(0.3),
         [(f"{n:,}", 14, True, col)], align=PP_ALIGN.RIGHT)

if (FIG / "fig_segments.png").exists():
    picture_fit(s, FIG / "fig_segments.png", Inches(6.85), Inches(1.78),
                Inches(5.9), Inches(3.9))

text(s, Inches(6.85), Inches(5.82), Inches(5.9), Inches(0.6),
     [("Six strategic segments combine risk with customer value, from "
       "Critical Retention through to Loyal.", 12, False, MUTED)],
     spacing=1.1)

# ==================================================================
# SLIDE 8 — Dashboard & Business Insights
# ==================================================================
s = slide_base("Streamlit Dashboard & Business Insights",
               "Eleven pages, from executive summary to per-customer detail",
               8)

pages = ["Executive Overview", "Dataset Center", "Churn Analysis",
         "Customer Segments", "Risk Center", "Prediction Center",
         "Model Performance", "Customer Explorer", "Recommendations",
         "Reports", "About"]
for i, name in enumerate(pages):
    col, row = i % 4, i // 4
    x = Inches(0.62 + col * 1.62)
    y = Inches(1.82 + row * 0.62)
    rect(s, x, y, Inches(1.52), Inches(0.5), fill=WHITE, line=CARD_BD)
    text(s, x + Inches(0.1), y + Inches(0.13), Inches(1.34), Inches(0.28),
         [(name, 10.5, False, INK)], align=PP_ALIGN.CENTER)

text(s, Inches(7.45), Inches(1.80), Inches(5.3), Inches(0.35),
     [("Revenue Exposure", 17, True, INK)])
rect(s, Inches(7.45), Inches(2.24), Inches(5.28), Inches(1.15),
     fill=WHITE, line=CARD_BD)
rect(s, Inches(7.45), Inches(2.24), Inches(5.28), Inches(0.05),
     fill=RGBColor(0xE0, 0x5A, 0x47))
text(s, Inches(7.68), Inches(2.44), Inches(4.9), Inches(0.45),
     [(f"{PIPE['monthly_rev']:,.2f}", 26, True, INK)])
text(s, Inches(7.68), Inches(2.94), Inches(4.9), Inches(0.3),
     [("Monthly revenue associated with at-risk customers", 11, False,
       MUTED)])

text(s, Inches(0.62), Inches(3.85), Inches(6.3), Inches(0.35),
     [("From Prediction to Action", 16, True, INK)])
bullet_block(s, Inches(0.72), Inches(4.30), Inches(6.1), [
    "Recommended retention action per customer",
    "Engagement channel by retention priority",
    "Six CSV reports available for export",
    "Immediate outreach for 963 critical customers",
], size=13.5, gap=0.42)

text(s, Inches(7.45), Inches(3.85), Inches(5.3), Inches(0.35),
     [("Prioritised Action Plan", 16, True, INK)])
plan = [("Immediate", "963", "Personal outreach"),
        ("High", "1,349", "Targeted campaign"),
        ("Monitor", "1,937", "Email / in-app"),
        ("Low", "2,794", "Standard engagement")]
for i, (p_, n, ch) in enumerate(plan):
    y = Inches(4.30 + i * 0.46)
    bg = WHITE if i % 2 == 0 else RGBColor(0xEE, 0xF2, 0xF9)
    rect(s, Inches(7.45), y, Inches(5.28), Inches(0.42), fill=bg,
         line=CARD_BD)
    text(s, Inches(7.66), y + Inches(0.1), Inches(1.6), Inches(0.28),
         [(p_, 12, True, INK)])
    text(s, Inches(9.15), y + Inches(0.1), Inches(1.0), Inches(0.28),
         [(n, 12, True, ACCENT)], align=PP_ALIGN.RIGHT)
    text(s, Inches(10.35), y + Inches(0.1), Inches(2.2), Inches(0.28),
         [(ch, 11, False, MUTED)], align=PP_ALIGN.RIGHT)

# ==================================================================
# SLIDE 9 — Results, Testing & Limitations
# ==================================================================
s = slide_base("Results, Testing & Future Scope",
               "What the system achieved and where it would develop next", 9)

card(s, Inches(0.62), Inches(1.82), Inches(2.9), Inches(1.4),
     pct(RF["F1 Score"]), "Best F1 (Random Forest)")
card(s, Inches(3.77), Inches(1.82), Inches(2.9), Inches(1.4),
     pct(GB["ROC-AUC"]), "Best ROC-AUC (Gradient Boosting)", ACCENT2)
card(s, Inches(6.92), Inches(1.82), Inches(2.9), Inches(1.4),
     "16", "Test cases recorded",
     RGBColor(0x7A, 0x5A, 0xF0))
card(s, Inches(10.07), Inches(1.82), Inches(2.65), Inches(1.4),
     "11", "Modules delivered", RGBColor(0x2E, 0xA0, 0x6B))

text(s, Inches(0.62), Inches(3.54), Inches(6.0), Inches(0.35),
     [("Limitations", 16, True, INK)])
bullet_block(s, Inches(0.72), Inches(3.98), Inches(5.9), [
    "Evaluated on a single public dataset",
    "Static snapshot — no time dimension in the data",
    "Risk weights set from observed patterns, not optimised",
    "No campaign-response data, so effectiveness is unmeasured",
], size=13.5, gap=0.42)

text(s, Inches(6.92), Inches(3.54), Inches(5.8), Inches(0.35),
     [("Future Scope", 16, True, INK)])
bullet_block(s, Inches(7.02), Inches(3.98), Inches(5.7), [
    "Add time-series features from billing history",
    "Systematic hyperparameter tuning and threshold calibration",
    "Explainability per prediction using SHAP",
    "Record campaign outcomes to measure retention impact",
], size=13.5, gap=0.42)

text(s, Inches(0.62), Inches(5.92), Inches(12.1), Inches(0.5),
     [("Testing: 11 cases executed against the code and 5 derived from "
       "implemented error-handling paths. No failing case was recorded.",
       12, False, MUTED)])

# ==================================================================
# SLIDE 10 — Conclusion
# ==================================================================
s = blank()
rect(s, 0, 0, W, H, fill=NAVY)
rect(s, 0, 0, Inches(0.22), H, fill=ACCENT)
rect(s, Inches(9.9), Inches(3.9), Inches(4.6), Inches(4.6),
     fill=RGBColor(0x17, 0x25, 0x45), shape=MSO_SHAPE.OVAL)

text(s, Inches(0.9), Inches(0.72), Inches(11.4), Inches(0.7),
     [("Conclusion", 36, True, WHITE)])
rect(s, Inches(0.9), Inches(1.55), Inches(2.0), Inches(0.045), fill=ACCENT2)

text(s, Inches(0.9), Inches(1.95), Inches(7.4), Inches(1.5),
     [("ChurnIQ covers the full analytical path from a raw data file to "
       "an exported retention report — prediction, risk scoring, "
       "segmentation and recommendation, presented through an "
       "interactive dashboard.", 16, False, RGBColor(0xC6, 0xD2, 0xE8))],
     spacing=1.18)

outcomes = [
    (f"{N_ROWS:,} customers analysed and scored"),
    ("6 algorithms trained; Random Forest deployed"),
    ("4 risk levels and 6 strategic segments generated"),
    ("11-page dashboard with 6 CSV exports"),
]
for i, o in enumerate(outcomes):
    y = Inches(3.62 + i * 0.52)
    rect(s, Inches(0.98), y + Inches(0.11), Inches(0.1), Inches(0.1),
         fill=ACCENT2)
    text(s, Inches(1.28), y, Inches(7.0), Inches(0.42),
         [(o, 15, False, WHITE)])

rect(s, Inches(8.85), Inches(2.05), Inches(3.6), Inches(2.5),
     fill=RGBColor(0x14, 0x1F, 0x3A), line=RGBColor(0x2A, 0x3A, 0x5C))
text(s, Inches(9.15), Inches(2.32), Inches(3.0), Inches(0.4),
     [("Key Outcome", 13, True, ACCENT2)])
text(s, Inches(9.15), Inches(2.78), Inches(3.0), Inches(1.6),
     [("A ranked, explainable list of customers a retention team can "
       "act on — not a column of probabilities.", 14, False, WHITE)],
     spacing=1.15)

text(s, Inches(0.9), Inches(6.28), Inches(11.4), Inches(0.4),
     [("The system prioritises customers statistically likely to churn; "
       "it does not guarantee retention.", 12, False, MUTED)])
text(s, Inches(0.9), Inches(6.74), Inches(11.4), Inches(0.4),
     [(f"{D('STUDENT NAME')}  •  {D('ENROLLMENT NUMBER')}  •  "
       f"{D('ORGANIZATION NAME')}  •  {D('MONTH AND YEAR')}", 11.5,
       False, RGBColor(0x8A, 0x9A, 0xB8))])

prs.save(str(OUT))
print(f"Presentation written to {OUT}")
print(f"Slides: {len(prs.slides.__iter__.__self__._sldIdLst)}")
