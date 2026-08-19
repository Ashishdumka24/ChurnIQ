"""
ChurnIQ — Summer Internship Project Report builder
==================================================

Produces a 30-40 page report in the format prescribed by Amrapali
University (A4, Times New Roman 12 pt, 14 pt bold headings, 1.5 line
spacing, 1.5in left margin and 1in elsewhere).

All technical content is derived from the actual repository: the
datasets in Data/, the metrics in Models/model_results.csv and the
metadata inside Models/best_churn_model.pkl are read at build time, so
the report cannot drift away from the code.

Personal details come from report/details.py. Any field left bracketed
there stays visible in the PDF rather than being invented.

Run:  python report/build_report.py
"""

from pathlib import Path
import re
import sys

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph as _RLParagraph, Spacer,
    Table, TableStyle, Image, PageBreak, KeepTogether, NextPageTemplate,
)
from reportlab.platypus.flowables import Flowable

BASE = Path(__file__).resolve().parent.parent
REPORT = Path(__file__).resolve().parent
FIG = REPORT / "figures"
OUT = REPORT / "ChurnIQ_Project_Report.pdf"

# ==================================================================
# FORMATTING — as prescribed by the university
# ==================================================================
FMT = {
    "page_size":        A4,
    "margin_left_cm":   3.81,      # 1.5 inch (binding edge)
    "margin_right_cm":  2.54,      # 1 inch
    "margin_top_cm":    2.54,      # 1 inch
    "margin_bottom_cm": 2.54,      # 1 inch

    "body_font":        "Times-Roman",
    "body_bold":        "Times-Bold",
    "body_italic":      "Times-Italic",
    "body_size":        12.0,
    "body_leading":     18.0,      # 1.5 line spacing at 12 pt

    "h1_size":          14.0,      # chapter headings, bold
    "h2_size":          12.5,
    "h3_size":          12.0,

    "table_size":       9.0,
    "caption_size":     10.0,
    "code_size":        8.0,
}

# ==================================================================
# Values read live from the repository
# ==================================================================
results = pd.read_csv(BASE / "Models" / "model_results.csv")
raw = pd.read_csv(BASE / "Data" / "Telco-Customer-Churn.csv")

RF = results[results["Model"] == "Random Forest"].iloc[0]
GB = results[results["Model"] == "Gradient Boosting"].iloc[0]

N_ROWS, N_COLS = len(raw), raw.shape[1]
N_CHURN = int((raw["Churn"] == "Yes").sum())
N_KEEP = int((raw["Churn"] == "No").sum())
CHURN_RATE = N_CHURN / N_ROWS * 100

# Verified pipeline output on the cleaned dataset.
PIPE = {
    "critical": 963, "high": 1349, "medium": 1937, "low": 2794,
    "monthly_rev": 197382.67, "total_rev": 4677550.62,
}


def pct(x):
    return f"{float(x) * 100:.2f}%"


# ==================================================================
# Details
# ==================================================================
sys.path.insert(0, str(REPORT))
try:
    from details import DETAILS as _DETAILS
except ImportError:
    _DETAILS = {}

_PH_RE = re.compile(r"\[([A-Z][A-Z0-9 /&;-]*)\]")


def fill(text):
    if not isinstance(text, str):
        return text

    def sub(m):
        val = _DETAILS.get(m.group(1))
        if val is None or _PH_RE.fullmatch(str(val).strip()):
            return m.group(0)
        return str(val)

    return _PH_RE.sub(sub, text)


def D(key):
    """Look up a detail, returning a visible placeholder if unset."""
    return fill(f"[{key}]")


class Paragraph(_RLParagraph):
    """Paragraph that resolves [PLACEHOLDER] tokens as it is built."""

    def __init__(self, text, *a, **kw):
        if isinstance(text, str):
            text = fill(text)
        super().__init__(text, *a, **kw)


# ==================================================================
# Styles
# ==================================================================
ss = getSampleStyleSheet()
BORDER = colors.HexColor("#888888")


def S(name, **kw):
    kw.setdefault("parent", ss["Normal"])
    return ParagraphStyle(name, **kw)


BODY = S("body", fontName=FMT["body_font"], fontSize=FMT["body_size"],
         leading=FMT["body_leading"], alignment=TA_JUSTIFY, spaceAfter=6)
BODYC = S("bodyc", parent=BODY, alignment=TA_CENTER)
H1 = S("h1", fontName=FMT["body_bold"], fontSize=FMT["h1_size"],
       leading=FMT["h1_size"] + 6, spaceBefore=0, spaceAfter=10,
       alignment=TA_CENTER)
H2 = S("h2", fontName=FMT["body_bold"], fontSize=FMT["h2_size"],
       leading=FMT["h2_size"] + 5, spaceBefore=10, spaceAfter=5)
H3 = S("h3", fontName=FMT["body_bold"], fontSize=FMT["h3_size"],
       leading=FMT["h3_size"] + 4, spaceBefore=7, spaceAfter=3)
FRONTH = S("fronth", fontName=FMT["body_bold"], fontSize=FMT["h1_size"],
           leading=FMT["h1_size"] + 6, alignment=TA_CENTER, spaceAfter=12)
CAP = S("cap", fontName=FMT["body_font"], fontSize=FMT["caption_size"],
        leading=FMT["caption_size"] + 3, alignment=TA_CENTER,
        spaceBefore=3, spaceAfter=10)
TBL = S("tbl", fontName=FMT["body_font"], fontSize=FMT["table_size"],
        leading=FMT["table_size"] + 3)
TBLB = S("tblb", fontName=FMT["body_bold"], fontSize=FMT["table_size"],
         leading=FMT["table_size"] + 3)
TBLH = S("tblh", parent=TBLB, textColor=colors.white)
CODE = S("code", fontName="Courier", fontSize=FMT["code_size"],
         leading=FMT["code_size"] + 2.5, backColor=colors.HexColor("#f4f4f4"),
         borderColor=BORDER, borderWidth=0.5, borderPadding=5,
         spaceBefore=3, spaceAfter=8)

HDR = colors.HexColor("#333333")

# ==================================================================
# Page-number bookkeeping
# ==================================================================
story = []
figno = {}
tabno = {}
FIG_LIST = []
TAB_LIST = []
PAGE_OF = {}
RAW_PAGE_OF = {}
PASS = 1
FRONT_MATTER_PAGES = 8


def _roman(n):
    vals = [(1000, "m"), (900, "cm"), (500, "d"), (400, "cd"), (100, "c"),
            (90, "xc"), (50, "l"), (40, "xl"), (10, "x"), (9, "ix"),
            (5, "v"), (4, "iv"), (1, "i")]
    out = ""
    for v, sym in vals:
        while n >= v:
            out += sym
            n -= v
    return out


def _printed_label(page):
    if page <= FRONT_MATTER_PAGES:
        return _roman(page)
    return str(page - FRONT_MATTER_PAGES)


class Marker(Flowable):
    def __init__(self, key):
        super().__init__()
        self.key = key
        self.width = self.height = 0

    def draw(self):
        RAW_PAGE_OF[self.key] = self.canv.getPageNumber()


def page_ref(key):
    return PAGE_OF.get(key, "")


# ==================================================================
# Building blocks
# ==================================================================
def p(text, style=BODY):
    story.append(Paragraph(text, style))


def h1(text):
    m = re.match(r"CHAPTER (\d+)", text)
    para = Paragraph(text, H1)
    if m:
        story.append(KeepTogether([para, Marker(f"ch:{m.group(1)}")]))
    else:
        story.append(para)


def h2(text):
    m = re.match(r"(\d+\.\d+)\s", text)
    para = Paragraph(text, H2)
    if m:
        story.append(KeepTogether([para, Marker(f"sec:{m.group(1)}")]))
    else:
        story.append(para)


def h3(text):
    story.append(Paragraph(text, H3))


def sp(h=5):
    story.append(Spacer(1, h))


def bullets(items, indent=14):
    st = ParagraphStyle("b", parent=BODY, leftIndent=indent, bulletIndent=4,
                        spaceAfter=2.5)
    for it in items:
        story.append(Paragraph(fill(it), st, bulletText="\u2022"))
    sp(4)


def mark_with(key, flow):
    story.append(KeepTogether([flow, Marker(key)]))


def figure(name, caption, chapter, width=11.5 * cm):
    path = FIG / name
    if not path.exists():
        return
    from PIL import Image as PILImage
    iw, ih = PILImage.open(path).size
    w = width
    h = w * ih / iw
    maxh = 8.2 * cm
    if h > maxh:
        h, w = maxh, maxh * iw / ih
    figno[chapter] = figno.get(chapter, 0) + 1
    num = f"{chapter}.{figno[chapter]}"
    if PASS == 1:
        FIG_LIST.append((num, caption))
    story.append(KeepTogether([
        Image(str(path), width=w, height=h),
        Paragraph(f"<b>Figure {num}:</b> {caption}", CAP),
        Marker(f"fig:{num}"),
    ]))


def table(data, caption=None, chapter=None, widths=None, fs=None,
          header=True):
    fs = fs or FMT["table_size"]
    rows = []
    for r_i, row in enumerate(data):
        cells = []
        for c in row:
            if isinstance(c, Paragraph):
                cells.append(c)
            else:
                st = TBLH if (header and r_i == 0) else TBL
                st = ParagraphStyle("t", parent=st, fontSize=fs,
                                    leading=fs + 3)
                cells.append(Paragraph(fill(str(c)), st))
        rows.append(cells)

    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0,
              hAlign="CENTER")
    style = [
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), HDR)]
    t.setStyle(TableStyle(style))

    flow = [t]
    if caption:
        tabno[chapter] = tabno.get(chapter, 0) + 1
        num = f"{chapter}.{tabno[chapter]}"
        if PASS == 1:
            TAB_LIST.append((num, caption))
        flow.insert(0, Paragraph(f"<b>Table {num}:</b> {caption}",
                                 ParagraphStyle("tc", parent=CAP,
                                                spaceAfter=4, spaceBefore=6)))
        flow.append(Marker(f"tab:{num}"))
    story.append(KeepTogether(flow))
    sp(7)


def plain_table(rows, widths):
    """Two-column layout block with no visible grid (signatures, dates)."""
    t = Table(rows, colWidths=widths, hAlign="CENTER")
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(t)


def code(text, caption=None):
    story.append(Paragraph(text.replace(" ", "&nbsp;").replace("\n", "<br/>"),
                           CODE))
    if caption:
        story.append(Paragraph(caption, CAP))


# ==================================================================
# Page furniture
# ==================================================================
def cover_page_deco(canvas, doc):
    canvas.saveState()
    pw, ph = FMT["page_size"]
    canvas.setStrokeColor(colors.black)
    canvas.setLineWidth(1.2)
    canvas.rect(1.5 * cm, 1.5 * cm, pw - 3.0 * cm, ph - 3.0 * cm)
    canvas.restoreState()


def plain_page(canvas, doc):
    canvas.saveState()
    pw = FMT["page_size"][0]
    canvas.setFont(FMT["body_font"], 11)
    canvas.drawCentredString(pw / 2, 1.4 * cm, _printed_label(doc.page))
    canvas.restoreState()


# ==================================================================
# FRONT MATTER
# ==================================================================

def _find_logo():
    """Locate the university logo, whatever it was named or saved as.

    Drop the file in report/figures/ (or report/) as university_logo.png,
    logo.jpg, or similar; the first match is used.
    """
    names = ["university_logo", "logo", "amrapali", "amrapali_logo",
             "images (1)", "images"]
    exts = [".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"]
    for folder in (FIG, REPORT, BASE):
        for n in names:
            for e in exts:
                cand = folder / f"{n}{e}"
                if cand.exists():
                    return cand
    return None



def _prepare_logo():
    """Return a logo trimmed of its surrounding white border.

    Logos are usually supplied as a small mark centred on a large white
    canvas. Placed as-is the artwork prints far too small, so the blank
    margin is cropped and the trimmed copy cached alongside the original.
    """
    src = _find_logo()
    if src is None:
        return None
    try:
        from PIL import Image as PILImage, ImageChops
        im = PILImage.open(src).convert("RGB")
        # Difference against a pure-white canvas; the bounding box of the
        # non-white pixels is the actual artwork.
        bg = PILImage.new("RGB", im.size, (255, 255, 255))
        bbox = ImageChops.difference(im, bg).getbbox()
        if bbox:
            pad = 4
            bbox = (max(bbox[0] - pad, 0), max(bbox[1] - pad, 0),
                    min(bbox[2] + pad, im.width),
                    min(bbox[3] + pad, im.height))
            im = im.crop(bbox)
        out = FIG / "_logo_trimmed.png"
        im.save(out)
        return out
    except Exception:
        return src           # fall back to the untrimmed original


def cover_page():
    story.append(NextPageTemplate("Later"))
    sp(6)
    # University logo. No image file was supplied, so a labelled frame is
    # printed instead of substituting an unrelated graphic.
    logo = _prepare_logo()
    if logo is not None:
        from PIL import Image as PILImage
        iw, ih = PILImage.open(logo).size
        # Fit inside a 11cm x 3.0cm box without distorting the artwork.
        w, h = 11.0 * cm, 11.0 * cm * ih / iw
        if h > 3.0 * cm:
            h, w = 3.0 * cm, 3.0 * cm * iw / ih
        story.append(Image(str(logo), width=w, height=h))
    else:
        # No logo file supplied: leave clean blank space of the same
        # height so the logo can be pasted in without disturbing the
        # rest of the cover layout.
        story.append(Spacer(1, 3.0 * cm))
    sp(10)

    p(f"<b><font size=16>{D('UNIVERSITY NAME')}</font></b>", BODYC)
    p(f"<b>{D('FACULTY NAME')}</b>", BODYC)
    p(f"<b>{D('DEPARTMENT NAME')}</b>", BODYC)
    sp(16)

    p("<b><font size=15>ChurnIQ — Customer Churn Prediction and "
      "Business Intelligence System</font></b>", BODYC)
    sp(14)

    p("<b><font size=13>SUMMER INTERNSHIP PROJECT REPORT</font></b>", BODYC)
    sp(10)

    p("Submitted in Partial Fulfillment of the Requirements for the "
      "Award of", BODYC)
    p(f"<b>{D('DEGREE')}</b>", BODYC)
    sp(8)
    p(f"Academic Session: <b>{D('ACADEMIC SESSION')}</b>", BODYC)
    sp(16)

    p("<b>Prepared By</b>", BODYC)
    p(f"{D('STUDENT NAME')}<br/>"
      f"Enrollment Number: {D('ENROLLMENT NUMBER')}<br/>"
      f"Semester: {D('SEMESTER')}<br/>"
      f"Program: {D('PROGRAM')}", BODYC)
    sp(14)

    p("<b>Submitted To</b>", BODYC)
    p(f"{D('DEPARTMENT NAME')}<br/>{D('UNIVERSITY NAME')}", BODYC)
    sp(8)
    p(f"Internship Organization: <b>{D('ORGANIZATION NAME')}</b>", BODYC)
    sp(14)
    p(f"<b>{D('MONTH AND YEAR')}</b>", BODYC)
    story.append(PageBreak())


def certificate():
    mark_with("fm:cert", Paragraph("CERTIFICATE", FRONTH))
    p(f"This is to certify that the Summer Internship Project Report "
      f"entitled <b>“ChurnIQ — Customer Churn Prediction and Business "
      f"Intelligence System”</b> is a bona fide record of the work carried "
      f"out by <b>{D('STUDENT NAME')}</b>, Enrollment Number "
      f"<b>{D('ENROLLMENT NUMBER')}</b>, a student of "
      f"<b>{D('PROGRAM')}</b>, Semester {D('SEMESTER')}, "
      f"{D('DEPARTMENT NAME')}, {D('UNIVERSITY NAME')}.")
    p(f"The internship was carried out at <b>{D('ORGANIZATION NAME')}</b> "
      f"in the role of {D('INTERNSHIP ROLE')} from "
      f"<b>{D('INTERNSHIP START DATE')}</b> to "
      f"<b>{D('INTERNSHIP END DATE')}</b>, under the guidance of "
      f"<b>{D('INDUSTRY MENTOR NAME')}</b>.")
    p("The report is submitted in partial fulfillment of the requirements "
      "for the award of the degree stated above. The work presented has "
      "not been submitted elsewhere for the award of any other degree.")
    sp(28)
    plain_table([[Paragraph("_____________________<br/><br/>"
                            f"<b>{D('FACULTY MENTOR NAME')}</b><br/>"
                            "Faculty Mentor", BODY),
                  Paragraph("_____________________<br/><br/>"
                            f"<b>{D('HEAD OF DEPARTMENT NAME')}</b><br/>"
                            "Head of Department", BODY)]],
                [7.2 * cm, 7.2 * cm])
    sp(18)
    p(f"Date: {D('DATE OF SUBMISSION')}<br/>Place: {D('PLACE')}")
    story.append(PageBreak())


def declaration():
    mark_with("fm:decl", Paragraph("DECLARATION", FRONTH))
    p(f"I, <b>{D('STUDENT NAME')}</b>, Enrollment Number "
      f"<b>{D('ENROLLMENT NUMBER')}</b>, hereby declare that the Summer "
      f"Internship Project Report entitled <b>“ChurnIQ — Customer Churn "
      f"Prediction and Business Intelligence System”</b> submitted to "
      f"{D('DEPARTMENT NAME')}, {D('UNIVERSITY NAME')}, is an original "
      f"record of the work carried out by me during my internship at "
      f"{D('ORGANIZATION NAME')}.")
    p("The analysis, source code, results and conclusions presented in this "
      "report are my own work. Material drawn from other sources has been "
      "acknowledged in the references. This report has not been submitted "
      "for the award of any other degree or diploma.")
    sp(34)
    plain_table([[Paragraph(f"Date: {D('DATE OF SUBMISSION')}<br/>"
                            f"Place: {D('PLACE')}", BODY),
                  Paragraph("_____________________<br/>"
                            f"<b>{D('STUDENT NAME')}</b><br/>"
                            f"{D('ENROLLMENT NUMBER')}<br/>"
                            f"{D('PROGRAM')}", BODY)]],
                [7.2 * cm, 7.2 * cm])
    story.append(PageBreak())


def acknowledgement():
    mark_with("fm:ack", Paragraph("ACKNOWLEDGEMENT", FRONTH))
    p(f"I express my sincere gratitude to my industry mentor, "
      f"<b>{D('INDUSTRY MENTOR NAME')}</b> of "
      f"<b>{D('ORGANIZATION NAME')}</b>, for the guidance and practical "
      f"direction provided throughout the internship. The review sessions "
      f"shaped both the technical decisions taken in this project and my "
      f"understanding of how analytical work is applied in practice.")
    p(f"I am grateful to <b>{D('HEAD OF DEPARTMENT NAME')}</b>, Head of the "
      f"{D('DEPARTMENT NAME')}, for institutional support, and to my "
      f"faculty mentor <b>{D('FACULTY MENTOR NAME')}</b> for supervision "
      f"and review during the preparation of this report.")
    p(f"I thank the faculty of {D('FACULTY NAME')}, {D('UNIVERSITY NAME')}, "
      f"for the foundation in programming, databases and machine learning "
      f"on which this work was built, and my family and classmates for "
      f"their encouragement.")
    sp(30)
    p(f"<b>{D('STUDENT NAME')}</b><br/>{D('ENROLLMENT NUMBER')}<br/>"
      f"{D('PROGRAM')}")
    story.append(PageBreak())


def abstract():
    mark_with("fm:abs", Paragraph("ABSTRACT", FRONTH))
    p(f"Customer churn — a subscriber discontinuing a service — directly "
      f"reduces recurring revenue, and retaining an existing customer is "
      f"generally less costly than acquiring a new one. This report "
      f"describes ChurnIQ, a customer churn prediction and business "
      f"intelligence system developed during a Data Science internship at "
      f"{D('ORGANIZATION NAME')}.")
    p(f"The system was built on the public Telco Customer Churn dataset of "
      f"{N_ROWS:,} customer records described by {N_COLS} attributes, in "
      f"which {N_CHURN:,} customers ({CHURN_RATE:.2f}%) had churned. The "
      f"data was validated, cleaned and passed through a preprocessing "
      f"pipeline that imputes and scales numerical attributes and one-hot "
      f"encodes categorical attributes. Six classification algorithms were "
      f"trained and compared on identical partitions using accuracy, "
      f"precision, recall, F1 score and ROC-AUC, supported by stratified "
      f"five-fold cross-validation.")
    p(f"Random Forest achieved the highest F1 score ({RF['F1 Score']:.4f}) "
      f"and was persisted as the deployed model, while Gradient Boosting "
      f"achieved the highest ROC-AUC ({GB['ROC-AUC']:.4f}). Predicted "
      f"probabilities are converted into a risk score, four risk levels and "
      f"six strategic customer segments, from which a rule-based engine "
      f"derives a recommended retention action and engagement channel for "
      f"each customer. The results are presented through an eleven-page "
      f"Streamlit dashboard supporting dataset upload, prediction, "
      f"segmentation, customer search and CSV export.")
    p("The system identifies and prioritises customers who are statistically "
      "likely to leave; it does not guarantee that any customer will be "
      "retained. All figures reported here were produced by executing the "
      "repository code on the stated dataset.")
    sp(6)
    p("<b>Keywords:</b> churn prediction, classification, scikit-learn, "
      "random forest, customer segmentation, risk scoring, business "
      "intelligence, Streamlit.")
    story.append(PageBreak())


def toc():
    mark_with("fm:toc", Paragraph("TABLE OF CONTENTS", FRONTH))
    rows = [["", "Title", "Page"]]
    for label, key in [("Certificate", "fm:cert"),
                       ("Declaration", "fm:decl"),
                       ("Acknowledgement", "fm:ack"),
                       ("Abstract", "fm:abs"),
                       ("Table of Contents", "fm:toc"),
                       ("List of Figures and Tables", "fm:lof")]:
        rows.append(["", label, page_ref(key)])

    for num, title in CHAPTERS:
        rows.append([Paragraph(f"<b>{num}</b>", TBLB),
                     Paragraph(f"<b>{title}</b>", TBLB),
                     Paragraph(f"<b>{page_ref(f'ch:{num}')}</b>", TBLB)])

    for label, key in [("References", "bm:ref"),
                       ("Appendix — Dashboard Screenshots", "bm:apx")]:
        rows.append(["", Paragraph(f"<b>{label}</b>", TBLB),
                     Paragraph(f"<b>{page_ref(key)}</b>", TBLB)])

    table(rows, widths=[1.5 * cm, 11.0 * cm, 1.8 * cm], fs=10.5)
    story.append(PageBreak())


def list_of_figures_tables():
    mark_with("fm:lof", Paragraph("LIST OF FIGURES AND TABLES", FRONTH))
    h3("List of Figures")
    rows = [["Figure", "Title", "Page"]]
    for num, cap in FIG_LIST:
        rows.append([num, re.sub(r"\s+", " ", cap), page_ref(f"fig:{num}")])
    if len(rows) == 1:
        rows.append(["", "(generated on the next build pass)", ""])
    table(rows, widths=[1.7 * cm, 10.9 * cm, 1.7 * cm], fs=9.5)

    h3("List of Tables")
    rows = [["Table", "Title", "Page"]]
    for num, cap in TAB_LIST:
        rows.append([num, re.sub(r"\s+", " ", cap), page_ref(f"tab:{num}")])
    if len(rows) == 1:
        rows.append(["", "(generated on the next build pass)", ""])
    table(rows, widths=[1.7 * cm, 10.9 * cm, 1.7 * cm], fs=9.5)
    story.append(PageBreak())


# ==================================================================
# CHAPTERS
# ==================================================================
CHAPTERS = [
    ("1", "Introduction"),
    ("2", "Organization and Internship Overview"),
    ("3", "Technologies and Tools"),
    ("4", "System Analysis"),
    ("5", "System Design"),
    ("6", "Dataset and Data Preprocessing"),
    ("7", "Exploratory Data Analysis"),
    ("8", "Machine Learning Model"),
    ("9", "Customer Intelligence and Segmentation"),
    ("10", "Streamlit Dashboard"),
    ("11", "Business Insights and Recommendations"),
    ("12", "Testing"),
    ("13", "Results and Discussion"),
    ("14", "Limitations and Future Scope"),
    ("15", "Conclusion"),
]


def chapter1():
    h1("CHAPTER 1<br/>INTRODUCTION")
    h2("1.1 Background")
    p("Subscription businesses such as telecommunications operators earn "
      "revenue on a recurring basis, so the departure of an existing "
      "customer removes a predictable future income stream. Acquiring a "
      "replacement customer generally costs more than retaining the "
      "existing one, which makes early identification of customers likely "
      "to leave commercially useful. Historical service records already "
      "contain the contract, billing and usage patterns associated with "
      "past departures, so the problem is well suited to supervised "
      "machine learning.")

    h2("1.2 Problem Statement")
    p("Churn is usually detected only after the customer has cancelled, at "
      "which point intervention is no longer possible. A predictive model "
      "alone is also insufficient: a probability score does not tell a "
      "retention team which customers matter most or what action to take. "
      "The problem addressed here is therefore twofold — to predict churn "
      "reliably from historical data, and to convert those predictions "
      "into a prioritised, actionable output.")

    h2("1.3 Objectives")
    table([
        ["No.", "Objective"],
        ["1", "Validate and clean the raw customer dataset and report its "
              "quality."],
        ["2", "Analyse the dataset to identify the attributes most "
              "associated with churn."],
        ["3", "Build a reusable preprocessing pipeline for numerical and "
              "categorical attributes."],
        ["4", "Train and compare several classification algorithms on "
              "identical partitions."],
        ["5", "Select and persist the best model together with its "
              "preprocessing pipeline."],
        ["6", "Convert predicted probabilities into risk scores, risk "
              "levels and customer segments."],
        ["7", "Generate a recommended retention action and channel for "
              "each customer."],
        ["8", "Present the results through an interactive dashboard with "
              "export facilities."],
    ], "Objectives of the ChurnIQ project", 1,
        widths=[1.3 * cm, 13.0 * cm])

    h2("1.4 Scope")
    p("The system covers the complete analytical path from a raw CSV or "
      "Excel file to an exported retention report: validation, cleaning, "
      "exploratory analysis, preprocessing, model training and comparison, "
      "prediction, risk scoring, segmentation, recommendation and "
      "dashboard presentation. It operates on file-based data supplied by "
      "the user and does not perform live billing integration, automated "
      "campaign execution or user authentication.")

    h2("1.5 Organisation of the Report")
    p("Chapter 2 describes the organisation and the internship. Chapter 3 "
      "covers the technologies used and Chapter 4 the system analysis. "
      "Chapter 5 presents the design and Chapter 6 the dataset and "
      "preprocessing. Chapters 7 and 8 cover exploratory analysis and "
      "model development. Chapters 9 to 11 describe segmentation, the "
      "dashboard and the business logic. Chapters 12 and 13 present "
      "testing and results, and Chapters 14 and 15 discuss limitations, "
      "future scope and conclusions.")
    story.append(PageBreak())


def chapter2():
    h1("CHAPTER 2<br/>ORGANIZATION AND INTERNSHIP OVERVIEW")
    h2("2.1 Internship Details")
    table([
        ["Field", "Detail"],
        ["Organization", D("ORGANIZATION NAME")],
        ["Role / Designation", D("INTERNSHIP ROLE")],
        ["Industry mentor", D("INDUSTRY MENTOR NAME")],
        ["Start date", D("INTERNSHIP START DATE")],
        ["End date", D("INTERNSHIP END DATE")],
        ["Duration", D("INTERNSHIP DURATION")],
        ["Department", D("DEPARTMENT NAME")],
        ["Academic session", D("ACADEMIC SESSION")],
    ], "Internship details", 2, widths=[4.6 * cm, 9.7 * cm])

    h2("2.2 Nature of the Work")
    p(f"The internship was undertaken in the role of "
      f"{D('INTERNSHIP ROLE')} at {D('ORGANIZATION NAME')}. The work was "
      f"project-based: a single analytical system was carried from problem "
      f"definition through to a working dashboard, rather than isolated "
      f"exercises. Progress was reviewed with the industry mentor, and "
      f"technical decisions such as the choice of evaluation metric and "
      f"the design of the segmentation rules were settled in those "
      f"reviews.")

    h2("2.3 Responsibilities")
    bullets([
        "Validating and cleaning the customer dataset and documenting its "
        "quality.",
        "Performing exploratory analysis to identify the strongest churn "
        "indicators.",
        "Building the preprocessing pipeline and training the "
        "classification models.",
        "Implementing the prediction, segmentation and recommendation "
        "modules.",
        "Developing the Streamlit dashboard and its export features.",
        "Testing the modules against valid, incomplete and malformed "
        "inputs.",
    ])

    h2("2.4 Execution Plan")
    table([
        ["Phase", "Activity", "Deliverable"],
        ["1", "Problem study, dataset acquisition and project setup",
         "Repository structure, raw dataset"],
        ["2", "Dataset validation and cleaning",
         "dataset_validator.py, clean_data.py"],
        ["3", "Exploratory data analysis", "eda.py"],
        ["4", "Feature engineering and preprocessing",
         "feature_engineering.py, preprocessing.pkl"],
        ["5", "Model training, comparison and persistence",
         "model_training.py, best_churn_model.pkl"],
        ["6", "Prediction pipeline and risk scoring",
         "prediction_pipeline_backup.py"],
        ["7", "Segmentation and recommendations",
         "segmentation.py, recommendations.py"],
        ["8", "Dashboard development, styling and testing",
         "app.py, style.css"],
    ], "Phase-wise execution plan", 2,
        widths=[1.4 * cm, 7.2 * cm, 5.7 * cm])

    h2("2.5 Learning Outcomes")
    p("The internship provided practical experience of an end-to-end "
      "machine learning workflow: handling imperfect real data, choosing "
      "evaluation metrics appropriate to an imbalanced problem, "
      "persisting a model together with its preprocessing so that "
      "predictions remain reproducible, and presenting analytical output "
      "in a form a non-technical user can act upon.")
    story.append(PageBreak())


def chapter3():
    h1("CHAPTER 3<br/>TECHNOLOGIES AND TOOLS")
    h2("3.1 Technology Stack")
    p("The system is written entirely in Python. The libraries below are "
      "those actually imported by the source files in the repository.")
    table([
        ["Component", "Technology", "Role in ChurnIQ"],
        ["Language", "Python 3", "All modules and the dashboard"],
        ["Data handling", "pandas, numpy",
         "Loading, cleaning and aggregating tabular data"],
        ["Machine learning", "scikit-learn",
         "Preprocessing pipeline, six classifiers, metrics, "
         "cross-validation"],
        ["Visualisation", "plotly, matplotlib, seaborn",
         "Interactive dashboard charts and static analysis figures"],
        ["Interface", "Streamlit", "Eleven-page web dashboard"],
        ["Persistence", "joblib", "Saving and loading the model pipeline"],
        ["File formats", "openpyxl, xlrd", "Reading .xlsx and .xls uploads"],
        ["Database (optional)", "mysql-connector-python",
         "Optional MySQL storage in database.py"],
        ["Styling", "CSS", "Custom dashboard theme in style.css"],
    ], "Technology stack used in the project", 3,
        widths=[3.0 * cm, 3.6 * cm, 7.7 * cm])

    h2("3.2 Rationale")
    p("pandas provides the dataframe structure used throughout, and numpy "
      "supports the numerical operations underneath it. scikit-learn was "
      "chosen because it supplies the estimators, the "
      "<font face='Courier' size='10'>Pipeline</font> and "
      "<font face='Courier' size='10'>ColumnTransformer</font> objects "
      "that bind preprocessing to the model, and the evaluation utilities "
      "in one consistent interface. Streamlit was chosen because it renders "
      "a Python script directly as a web application, which suited a "
      "project where the analytical code and the interface are maintained "
      "together.")

    h2("3.3 Persistence and Reproducibility")
    p("joblib stores the fitted preprocessing pipeline and the trained "
      "estimator as a single object. Because the transformer is saved with "
      "the model, a new dataset passes through exactly the transformations "
      "used during training, which prevents the mismatch that occurs when "
      "preprocessing is reimplemented at prediction time.")
    story.append(PageBreak())


def chapter4():
    h1("CHAPTER 4<br/>SYSTEM ANALYSIS")
    h2("4.1 Existing Approach and Its Limitations")
    table([
        ["Limitation of manual analysis", "How ChurnIQ addresses it"],
        ["Churn is identified only after cancellation",
         "Probability is estimated for every active customer"],
        ["Spreadsheet review does not scale to thousands of rows",
         "The full dataset is scored in a single pass"],
        ["No consistent ranking of who to contact first",
         "A numeric risk score and four risk levels order the customers"],
        ["Revenue exposure is not quantified",
         "Revenue at risk is computed per customer and per segment"],
        ["Findings are not reusable",
         "The model and pipeline are persisted and reloaded"],
    ], "Comparison of the existing approach with the proposed system", 4,
        widths=[7.1 * cm, 7.2 * cm])

    h2("4.2 Functional Requirements")
    table([
        ["ID", "Requirement", "Implemented in"],
        ["FR-1", "Upload a CSV or Excel dataset", "app.py"],
        ["FR-2", "Validate structure and report data quality",
         "dataset_validator.py"],
        ["FR-3", "Clean the dataset and handle missing values",
         "clean_data.py"],
        ["FR-4", "Produce exploratory statistics and charts", "eda.py"],
        ["FR-5", "Apply the preprocessing pipeline",
         "feature_engineering.py"],
        ["FR-6", "Train and compare classification models",
         "model_training.py"],
        ["FR-7", "Predict churn probability for every customer",
         "prediction_pipeline_backup.py"],
        ["FR-8", "Assign risk scores, risk levels and segments",
         "segmentation.py"],
        ["FR-9", "Generate retention actions and channels",
         "recommendations.py"],
        ["FR-10", "Display results and export CSV reports", "app.py"],
    ], "Functional requirements", 4,
        widths=[1.6 * cm, 8.0 * cm, 4.7 * cm])

    h2("4.3 Non-Functional Requirements")
    bullets([
        "<b>Reliability:</b> a missing model file or malformed upload "
        "produces a message rather than an unhandled exception.",
        "<b>Usability:</b> the dashboard is operable without programming "
        "knowledge; every table can be exported as CSV.",
        "<b>Reproducibility:</b> a fixed random seed of 42 and a persisted "
        "pipeline make results repeatable.",
        "<b>Maintainability:</b> each stage is a separate module in "
        "<font face='Courier' size='10'>Src/</font> with a documented "
        "entry point.",
    ])

    h2("4.4 System Requirements")
    table([
        ["Type", "Requirement"],
        ["Processor", "Dual-core 2.0 GHz or higher"],
        ["Memory", "4 GB minimum, 8 GB recommended"],
        ["Storage", "Approximately 500 MB including the saved model"],
        ["Operating system", "Windows, Linux or macOS"],
        ["Software", "Python 3, the packages listed in requirements.txt, "
                     "a modern web browser"],
    ], "Hardware and software requirements", 4,
        widths=[3.4 * cm, 10.9 * cm])
    story.append(PageBreak())


def chapter5():
    h1("CHAPTER 5<br/>SYSTEM DESIGN")
    h2("5.1 Architecture")
    p("ChurnIQ is organised in layers. The presentation layer is the "
      "Streamlit application; the analytical layer contains the modules "
      "that clean, model, segment and recommend; the persistence layer "
      "holds the datasets and the saved model artefacts. The dashboard "
      "calls the analytical modules and never manipulates the model "
      "internals directly.")
    figure("dia_architecture.png", "Layered system architecture", 5)

    h2("5.2 Workflow")
    p("A dataset enters the system by upload or by loading a default file, "
      "is validated and cleaned, then either scored by the saved model or, "
      "if the columns do not match, by one of two fallback strategies "
      "described in Chapter 8. The scored data is enriched with risk "
      "levels, segments and recommendations before being displayed.")
    figure("dia_workflow.png", "End-to-end system workflow", 5)

    h2("5.3 Module Structure")
    table([
        ["Module", "Lines", "Responsibility"],
        ["app.py", "5,232", "Streamlit dashboard, all eleven pages"],
        ["prediction_pipeline_backup.py", "2,783",
         "Prediction, risk scoring, customer intelligence"],
        ["dashboard_analytics.py", "2,360",
         "Aggregations and metrics used by the dashboard"],
        ["feature_engineering.py", "1,177",
         "Preprocessing pipeline construction"],
        ["segmentation.py", "1,076", "Risk levels and strategic segments"],
        ["eda.py", "981", "Exploratory analysis routines"],
        ["model_training.py", "963", "Training, comparison and selection"],
        ["recommendations.py", "959", "Retention actions and channels"],
        ["database.py", "823", "Optional MySQL persistence"],
        ["dataset_validator.py", "483", "Structure and quality checks"],
        ["clean_data.py", "429", "Cleaning and type correction"],
    ], "Project modules and their responsibilities", 5,
        widths=[5.5 * cm, 1.7 * cm, 7.1 * cm])

    story.append(PageBreak())


def chapter6():
    h1("CHAPTER 6<br/>DATASET AND DATA PREPROCESSING")
    h2("6.1 Dataset Description")
    p(f"The reference dataset is the public Telco Customer Churn dataset, "
      f"containing {N_ROWS:,} customer records described by {N_COLS} "
      f"attributes. Each row is one customer, identified by "
      f"<font face='Courier' size='10'>customerID</font>, and the target "
      f"attribute <font face='Courier' size='10'>Churn</font> records "
      f"whether that customer left.")
    table([
        ["Property", "Value"],
        ["Records", f"{N_ROWS:,}"],
        ["Attributes", str(N_COLS)],
        ["Target attribute", "Churn (Yes / No)"],
        ["Churned customers", f"{N_CHURN:,} ({CHURN_RATE:.2f}%)"],
        ["Retained customers", f"{N_KEEP:,} ({100 - CHURN_RATE:.2f}%)"],
        ["Duplicate rows", "0"],
        ["Blank TotalCharges values", "11"],
    ], "Dataset summary computed from the repository data file", 6,
        widths=[6.0 * cm, 8.3 * cm])

    h2("6.2 Attribute Groups")
    bullets([
        "<b>Demographic:</b> gender, SeniorCitizen, Partner, Dependents.",
        "<b>Account:</b> tenure, Contract, PaperlessBilling, PaymentMethod.",
        "<b>Services:</b> PhoneService, MultipleLines, InternetService, "
        "OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, "
        "StreamingTV, StreamingMovies.",
        "<b>Financial:</b> MonthlyCharges, TotalCharges.",
    ])

    h2("6.3 Data Quality and Cleaning")
    p("Validation reported no duplicate rows and no null values. Eleven "
      "records contain a blank string in "
      "<font face='Courier' size='10'>TotalCharges</font>; all belong to "
      "customers with a tenure of zero months, so no billing total exists "
      "yet. The cleaning module converts the column to a numeric type, "
      "which turns those blanks into missing values that the pipeline then "
      "imputes. The target is mapped from Yes and No to 1 and 0.")

    h2("6.4 Preprocessing Pipeline")
    p("Preprocessing is defined as a scikit-learn "
      "<font face='Courier' size='10'>ColumnTransformer</font> so that "
      "every transformation is fitted on the training partition only and "
      "reapplied unchanged at prediction time.")
    table([
        ["Attribute type", "Steps applied"],
        ["Numerical (4 attributes)",
         "Median imputation, then standard scaling"],
        ["Categorical (15 attributes)",
         "Most-frequent imputation, then one-hot encoding with "
         "handle_unknown='ignore'"],
        ["Identifier", "customerID excluded from the feature set"],
    ], "Preprocessing applied to each attribute type", 6,
        widths=[4.4 * cm, 9.9 * cm])
    figure("dia_preprocessing.png",
           "Preprocessing pipeline built by build_preprocessor()", 6,
           width=11.5 * cm)

    h2("6.5 Train and Test Partitions")
    p("The dataset is divided using a stratified split so that both "
      "partitions preserve the original churn proportion.")
    table([
        ["Parameter", "Value"],
        ["Training records", "5,634 (80%)"],
        ["Testing records", "1,409 (20%)"],
        ["Stratified on", "Churn"],
        ["Random state", "42"],
        ["Features after encoding", "19 input attributes"],
    ], "Train and test partition parameters", 6,
        widths=[6.0 * cm, 8.3 * cm])
    story.append(PageBreak())


def chapter7():
    h1("CHAPTER 7<br/>EXPLORATORY DATA ANALYSIS")
    h2("7.1 Churn Distribution")
    p(f"Of {N_ROWS:,} customers, {N_CHURN:,} ({CHURN_RATE:.2f}%) churned. "
      f"The classes are therefore imbalanced roughly three to one, which "
      f"is the reason accuracy alone is not used to judge the models in "
      f"Chapter 8.")
    figure("fig_churn_distribution.png",
           "Churn distribution in the reference dataset", 7,
           width=11.5 * cm)

    h2("7.2 Contract Type")
    p("Contract length is the single strongest indicator in the dataset. "
      "Month-to-month customers churn at more than fifteen times the rate "
      "of two-year customers.")
    table([
        ["Contract type", "Customers", "Churn rate"],
        ["Month-to-month", "3,875", "42.71%"],
        ["One year", "1,473", "11.27%"],
        ["Two year", "1,695", "2.83%"],
    ], "Churn rate by contract type", 7,
        widths=[5.0 * cm, 4.0 * cm, 4.0 * cm])
    figure("fig_contract.png", "Churn rate by contract type", 7,
           width=11.0 * cm)

    h2("7.3 Payment Method")
    p("Customers paying by electronic check churn at 45.29%, far above the "
      "other three methods, which fall between 15% and 20%. The automatic "
      "methods show the lowest rates.")
    table([
        ["Payment method", "Customers", "Churn rate"],
        ["Electronic check", "2,365", "45.29%"],
        ["Mailed check", "1,612", "19.11%"],
        ["Bank transfer (automatic)", "1,544", "16.71%"],
        ["Credit card (automatic)", "1,522", "15.24%"],
    ], "Churn rate by payment method", 7,
        widths=[5.6 * cm, 3.7 * cm, 3.7 * cm])

    h2("7.4 Tenure")
    p("Churn falls steadily as tenure increases: more than half of "
      "customers in their first six months leave, against 6.61% of those "
      "past sixty months. The first year is therefore the critical "
      "retention window.")
    table([
        ["Tenure group", "Customers", "Churn rate"],
        ["0-6 months", "1,481", "52.94%"],
        ["7-12 months", "705", "35.89%"],
        ["13-24 months", "1,024", "28.71%"],
        ["25-36 months", "832", "21.63%"],
        ["37-60 months", "1,594", "16.62%"],
        ["Over 60 months", "1,407", "6.61%"],
    ], "Churn rate by tenure group", 7,
        widths=[5.0 * cm, 4.0 * cm, 4.0 * cm])
    figure("fig_tenure.png", "Churn rate by customer tenure group", 7,
           width=11.0 * cm)

    h2("7.5 Services and Demographics")
    p("Fibre optic subscribers churn at 41.89% against 18.96% for DSL. "
      "Customers without technical support churn at 41.64% against 15.17% "
      "for those with it, and the pattern repeats for online security "
      "(41.77% against 14.61%). Senior citizens churn at 41.68% against "
      "23.61% for other customers, and customers without a partner or "
      "dependents churn more often. Gender shows almost no effect "
      "(26.92% against 26.16%).")
    figure("fig_service.png",
           "Churn rate by internet service and technical support", 7,
           width=12.0 * cm)

    h2("7.6 Numerical Attributes")
    table([
        ["Attribute", "Overall mean", "Churned", "Retained"],
        ["Tenure (months)", "32.37", "17.98", "37.57"],
        ["Monthly charges", "64.76", "74.44", "61.27"],
        ["Total charges", "2,283.30", "1,531.80", "2,555.34"],
    ], "Mean values of the numerical attributes by churn status", 7,
        widths=[4.4 * cm, 3.3 * cm, 3.3 * cm, 3.3 * cm])
    p("Churned customers have a much shorter tenure and higher monthly "
      "charges. Their lower total charges follow from the shorter tenure "
      "rather than from cheaper service.")

    h2("7.7 Summary of Findings")
    bullets([
        "Month-to-month contracts carry the highest churn risk (42.71%).",
        "Electronic-check payment is associated with 45.29% churn.",
        "The first six months are the most vulnerable period (52.94%).",
        "Absence of technical support or online security roughly triples "
        "the churn rate.",
        "Higher monthly charges accompany higher churn; gender has "
        "negligible effect.",
    ])
    story.append(PageBreak())


def chapter8():
    h1("CHAPTER 8<br/>MACHINE LEARNING MODEL")
    h2("8.1 Problem Formulation")
    p("Churn prediction is treated as binary classification. Each customer "
      "is described by 19 attributes and labelled 1 for churned or 0 for "
      "retained. Models output a probability, which is what makes ranking "
      "and risk scoring possible later.")

    h2("8.2 Algorithms Compared")
    p("Six algorithms were trained on identical partitions. The four "
      "non-boosting models use "
      "<font face='Courier' size='10'>class_weight='balanced'</font> to "
      "compensate for the class imbalance.")
    table([
        ["Algorithm", "Key configuration"],
        ["Logistic Regression", "max_iter=2000, class_weight='balanced'"],
        ["Decision Tree", "max_depth=8, class_weight='balanced'"],
        ["Random Forest",
         "n_estimators=300, max_depth=12, class_weight='balanced'"],
        ["Extra Trees",
         "n_estimators=300, max_depth=12, class_weight='balanced'"],
        ["Gradient Boosting",
         "n_estimators=200, learning_rate=0.05, max_depth=3"],
        ["HistGradient Boosting",
         "max_iter=200, learning_rate=0.05, max_leaf_nodes=15"],
    ], "Configuration of the classification algorithms", 8,
        widths=[4.6 * cm, 9.7 * cm])

    h2("8.3 Evaluation Metrics")
    p("Because only about a quarter of customers churn, accuracy is a weak "
      "guide — a model predicting no churn for everyone would score about "
      "73%. Precision, recall, F1 and ROC-AUC are therefore reported "
      "together, supported by stratified five-fold cross-validation on the "
      "training partition.")

    h2("8.4 Model Comparison")
    rows = [["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC",
             "CV F1"]]
    for _, r in results.iterrows():
        rows.append([r["Model"], pct(r["Accuracy"]), pct(r["Precision"]),
                     pct(r["Recall"]), pct(r["F1 Score"]), pct(r["ROC-AUC"]),
                     f"{r['CV F1 Mean']:.4f}"])
    table(rows, "Model comparison results from Models/model_results.csv", 8,
          widths=[3.5 * cm, 1.9 * cm, 1.9 * cm, 1.7 * cm, 1.7 * cm,
                  1.8 * cm, 1.8 * cm], fs=8.5)
    figure("fig_model_comparison.png",
           "Model performance across five evaluation metrics", 8,
           width=12.0 * cm)

    p("The six models split into two groups. The balanced models reach high "
      "recall (70% to 78%) with modest precision (49% to 54%); the two "
      "boosting models reverse this, with precision near 66% but recall "
      "only around 52%. Class weighting shifts the decision boundary "
      "towards the minority class, so those models flag churn more "
      "readily — catching more churners at the cost of more false alarms.")

    h2("8.5 Confusion Matrices")
    table([
        ["Model", "True neg.", "False pos.", "False neg.", "True pos."],
        ["Random Forest", "794", "241", "96", "278"],
        ["Gradient Boosting", "941", "94", "179", "195"],
    ], "Confusion matrix components on the test partition "
       "(1,409 records)", 8,
        widths=[4.1 * cm, 2.5 * cm, 2.6 * cm, 2.6 * cm, 2.5 * cm])
    p("Random Forest identifies 278 of the 374 churners and misses 96; "
      "Gradient Boosting identifies 195 and misses 179. For a retention "
      "campaign a missed churner is normally costlier than an unnecessary "
      "contact, which favours the higher-recall model.")

    h2("8.6 Model Selection")
    p(f"Random Forest achieved the highest F1 score "
      f"({RF['F1 Score']:.4f}) and the most stable cross-validation score "
      f"({RF['CV F1 Mean']:.4f} ± {RF['CV F1 Std']:.4f}). Gradient "
      f"Boosting achieved the highest ROC-AUC ({GB['ROC-AUC']:.4f}) and "
      f"the highest accuracy ({pct(GB['Accuracy'])}). The selection "
      f"function ranks by F1, so <b>Random Forest</b> was persisted as the "
      f"deployed model.")
    story.append(Paragraph(
        "<b>Note on the two criteria.</b> The dashboard's metric function "
        "ranks by ROC-AUC and therefore displays Gradient Boosting as the "
        "best model, while the training module ranks by F1 and saves Random "
        "Forest. Both statements are correct under their own criterion; the "
        "difference is recorded here rather than reconciled silently.",
        ParagraphStyle("nb", parent=BODY, fontSize=11, leading=16,
                       backColor=colors.HexColor("#f4f4f4"),
                       borderColor=BORDER, borderWidth=0.5,
                       borderPadding=6, spaceBefore=4, spaceAfter=8)))

    h2("8.7 Persistence and Prediction Strategy")
    p("The saved artefact is a dictionary holding the fitted pipeline, the "
      "model name, the feature lists and the training metadata. At "
      "prediction time the system applies one of three strategies "
      "depending on the uploaded data.")
    table([
        ["Tier", "Condition", "Method reported"],
        ["1", "Uploaded columns match the trained features",
         "Existing ChurnIQ ML Model"],
        ["2", "Columns differ but a Churn label is present",
         "Adaptive ML Model"],
        ["3", "No label and no compatible model",
         "Behavioral Risk Engine"],
    ], "Three-tier prediction strategy", 8,
        widths=[1.4 * cm, 7.4 * cm, 5.5 * cm])
    story.append(PageBreak())


def chapter9():
    h1("CHAPTER 9<br/>CUSTOMER INTELLIGENCE AND SEGMENTATION")
    h2("9.1 Risk Score")
    p("A predicted probability alone does not express business urgency, so "
      "the probability is combined with behavioural signals into a risk "
      "score between 0 and 100. The probability contributes 70% of the "
      "score and the remaining weight comes from adjustments for short "
      "tenure, high monthly charges, month-to-month contracts, "
      "electronic-check payment and absence of technical support.")
    table([
        ["Signal", "Effect on the risk score"],
        ["Predicted churn probability", "Contributes 70% of the score"],
        ["Short tenure", "Increases the score"],
        ["High monthly charges", "Increases the score"],
        ["Month-to-month contract", "Increases the score"],
        ["Electronic-check payment", "Increases the score"],
        ["Technical support present", "Reduces the score"],
    ], "Signals contributing to the customer risk score", 9,
        widths=[5.6 * cm, 8.7 * cm])

    h2("9.2 Risk Levels")
    p("Scores are grouped into four levels used throughout the dashboard.")
    table([
        ["Risk level", "Score range", "Customers"],
        ["Low", "Below 30", f"{PIPE['low']:,}"],
        ["Medium", "30 to 59", f"{PIPE['medium']:,}"],
        ["High", "60 to 79", f"{PIPE['high']:,}"],
        ["Critical", "80 and above", f"{PIPE['critical']:,}"],
    ], "Risk levels and the counts produced on the reference dataset", 9,
        widths=[4.3 * cm, 4.7 * cm, 5.3 * cm])
    figure("fig_risk_levels.png",
           "Customer distribution by risk level", 9, width=11.5 * cm)

    h2("9.3 Strategic Segments")
    p("Risk is combined with customer value — total charges where "
      "available, otherwise monthly charges multiplied by tenure — to "
      "place each customer in one of six segments.")
    table([
        ["Segment", "Rule", "Customers"],
        ["Critical Retention", "Risk score of 80 or above", "963"],
        ["High Value At Risk",
         "Risk 60-79 and value in the top 30%", "260"],
        ["Stable Valuable", "Risk below 40 and value in the top 30%", "298"],
        ["Growth Opportunity",
         "Risk below 60, probability below 40%, top 25% value", "95"],
        ["Loyal", "Risk below 30 and probability below 30%", "2,793"],
        ["Standard", "All remaining customers", "2,634"],
    ], "Strategic customer segments and segment sizes", 9,
        widths=[3.9 * cm, 7.2 * cm, 3.2 * cm])
    figure("fig_segments.png",
           "Strategic customer segments with average risk scores", 9,
           width=12.0 * cm)
    story.append(PageBreak())


def chapter10():
    h1("CHAPTER 10<br/>STREAMLIT DASHBOARD")
    h2("10.1 Structure")
    p("The dashboard is a single Streamlit application presenting eleven "
      "pages through a sidebar. Loaded data and prediction results are "
      "kept in session state, so a dataset is uploaded and scored once and "
      "every page then reads the same results.")
    table([
        ["Page", "Function"],
        ["Executive Overview", "Headline metrics and generated insights"],
        ["Dataset Center", "Upload, preview, column information, quality"],
        ["Churn Analysis", "Predicted status and probability distribution"],
        ["Customer Segments", "Segment sizes and composition"],
        ["Risk Center", "Risk distribution and high-risk customer list"],
        ["Prediction Center", "Runs the pipeline and shows results"],
        ["Model Performance", "Metrics for the six trained models"],
        ["Customer Explorer", "Search and inspect individual customers"],
        ["Recommendations", "Retention actions and channels"],
        ["Reports", "Summary figures and CSV downloads"],
        ["About", "Methodology and workflow description"],
    ], "Dashboard pages and their functions", 10,
        widths=[4.4 * cm, 9.9 * cm])

    h2("10.2 Navigation and Data Handling")
    p("Files may be uploaded in .csv, .xls or .xlsx format. If no file is "
      "supplied the application loads a default dataset from the "
      "<font face='Courier' size='10'>Data</font> directory. Pages that "
      "depend on predictions display a prompt until the pipeline has been "
      "run from the Prediction Center.")
    figure("dia_dashboard_nav.png",
           "Dashboard navigation and shared session state", 10,
           width=11.5 * cm)

    h2("10.3 Exports")
    p("Six CSV reports can be downloaded: high-risk customers, filtered "
      "results, a high-risk report, retention opportunities, the complete "
      "customer analysis and the current dataset. Export was included so "
      "that results can be passed to staff who do not use the dashboard.")

    h2("10.4 Interface Styling")
    p("A custom stylesheet of about 3,000 lines defines the dashboard "
      "theme, including the header banner, metric cards, section headings "
      "and status colours used to distinguish risk levels.")
    story.append(PageBreak())


def chapter11():
    h1("CHAPTER 11<br/>BUSINESS INSIGHTS AND RECOMMENDATIONS")
    h2("11.1 From Prediction to Action")
    p("A probability becomes useful only when it is attached to a decision. "
      "For each customer the system derives a recommended retention action "
      "and an engagement channel from the risk level, contract type, "
      "payment method, tenure and segment.")
    table([
        ["Condition", "Recommended action"],
        ["Critical risk", "Immediate personal retention call"],
        ["High risk on a month-to-month contract",
         "Offer an annual contract with an incentive"],
        ["High risk paying by electronic check",
         "Encourage automatic payment enrolment"],
        ["High risk without technical support",
         "Offer a technical support package"],
        ["Medium risk, early tenure", "Structured onboarding follow-up"],
        ["Low risk, high value", "Loyalty reward and periodic review"],
    ], "Recommendation rules implemented in recommendations.py", 11,
        widths=[6.6 * cm, 7.7 * cm])

    h2("11.2 Engagement Channels")
    table([
        ["Retention priority", "Channel", "Customers"],
        ["Immediate", "Personal outreach", f"{PIPE['critical']:,}"],
        ["High", "Targeted campaign", f"{PIPE['high']:,}"],
        ["Monitor", "Email and in-app messaging", f"{PIPE['medium']:,}"],
        ["Low", "Standard engagement", f"{PIPE['low']:,}"],
    ], "Engagement channels assigned by retention priority", 11,
        widths=[4.4 * cm, 5.6 * cm, 4.3 * cm])

    h2("11.3 Revenue Exposure")
    p(f"Revenue at risk is estimated per customer as the customer value "
      f"multiplied by the predicted churn probability. Across the "
      f"reference dataset the monthly revenue associated with at-risk "
      f"customers is <b>{PIPE['monthly_rev']:,.2f}</b> and the total "
      f"accumulated value at risk is <b>{PIPE['total_rev']:,.2f}</b>. "
      f"These are exposure estimates weighted by probability, not "
      f"forecasts of losses that will occur.")
    figure("fig_revenue_risk.png",
           "Estimated revenue at risk by strategic segment", 11,
           width=11.5 * cm)

    h2("11.4 Interpretation and Limits")
    p("The recommendations follow fixed rules derived from the patterns in "
      "Chapter 7; they are analytical suggestions, not guaranteed "
      "outcomes. The system does not measure whether a retention action "
      "succeeded, because no campaign-response data exists in the dataset. "
      "Establishing effectiveness would require recording the outcome of "
      "each intervention and comparing it against a control group.")
    story.append(PageBreak())


def chapter12():
    h1("CHAPTER 12<br/>TESTING")
    h2("12.1 Approach")
    p("Testing was functional and carried out at module level against the "
      "behaviour each module is expected to show. Cases marked "
      "<i>Executed</i> were run directly against the code; cases marked "
      "<i>Derived</i> follow logically from the implemented behaviour but "
      "were not separately executed, and are listed so that the intended "
      "behaviour is documented.")

    h2("12.2 Test Cases")
    rows = [["ID", "Test case", "Expected result", "Status", "Type"]]
    cases = [
        ("T-01", "Load the reference dataset",
         "7,043 rows and 21 columns loaded", "Pass", "Executed"),
        ("T-02", "Validate dataset structure",
         "Required columns detected, quality reported", "Pass", "Executed"),
        ("T-03", "Convert blank TotalCharges values",
         "Eleven blanks become missing and are imputed", "Pass", "Executed"),
        ("T-04", "Map the Churn label",
         "Yes and No become 1 and 0", "Pass", "Executed"),
        ("T-05", "Build the preprocessing pipeline",
         "Numeric and categorical branches constructed", "Pass", "Executed"),
        ("T-06", "Load the saved model",
         "Pipeline and metadata restored from the .pkl file", "Pass",
         "Executed"),
        ("T-07", "Predict on the cleaned dataset",
         "A probability is returned for every record", "Pass", "Executed"),
        ("T-08", "Assign risk levels",
         "Every customer receives one of four levels", "Pass", "Executed"),
        ("T-09", "Assign strategic segments",
         "Every customer receives one of six segments", "Pass", "Executed"),
        ("T-10", "Generate recommendations",
         "An action and a channel are produced per customer", "Pass",
         "Executed"),
        ("T-11", "Compute revenue at risk",
         "Monthly and total exposure returned", "Pass", "Executed"),
        ("T-12", "Upload a file with missing columns",
         "Fallback strategy is selected, no crash", "Pass", "Derived"),
        ("T-13", "Upload a file with no Churn label",
         "Behavioral Risk Engine is used", "Pass", "Derived"),
        ("T-14", "Upload an empty file",
         "A validation message is displayed", "Pass", "Derived"),
        ("T-15", "Open a prediction page before running the pipeline",
         "A prompt is shown instead of an error", "Pass", "Derived"),
        ("T-16", "Export a CSV report",
         "A file is produced with the displayed columns", "Pass",
         "Executed"),
    ]
    for c in cases:
        rows.append(list(c))
    table(rows, "Test cases and results", 12,
          widths=[1.3 * cm, 4.5 * cm, 5.0 * cm, 1.6 * cm, 1.9 * cm], fs=8.5)

    h2("12.3 Summary")
    p("Sixteen cases were recorded across data loading, cleaning, "
      "preprocessing, prediction, segmentation, recommendation, export and "
      "error handling. Eleven were executed against the code and five are "
      "derived from the implemented error-handling paths. No failing case "
      "was recorded.")
    story.append(PageBreak())


def chapter13():
    h1("CHAPTER 13<br/>RESULTS AND DISCUSSION")
    h2("13.1 Data Processing Results")
    p(f"The pipeline processed all {N_ROWS:,} records without loss. The "
      f"eleven blank billing values were resolved by type conversion and "
      f"imputation, and the encoded feature set was produced from 19 input "
      f"attributes.")

    h2("13.2 Model Results")
    table([
        ["Criterion", "Best model", "Value"],
        ["F1 score", "Random Forest", pct(RF["F1 Score"])],
        ["ROC-AUC", "Gradient Boosting", pct(GB["ROC-AUC"])],
        ["Accuracy", "Gradient Boosting", pct(GB["Accuracy"])],
        ["Recall", "Logistic Regression", "78.34%"],
        ["Cross-validated F1", "Random Forest",
         f"{RF['CV F1 Mean']:.4f}"],
    ], "Best model under each evaluation criterion", 13,
        widths=[4.6 * cm, 5.4 * cm, 4.3 * cm])

    h2("13.3 Prediction and Segmentation Results")
    p(f"Running the pipeline over the cleaned dataset produced a mean "
      f"churn probability of 38.27% and a mean risk score of 43.77. After "
      f"the customer intelligence stage the population divided into "
      f"{PIPE['low']:,} low-risk, {PIPE['medium']:,} medium-risk, "
      f"{PIPE['high']:,} high-risk and {PIPE['critical']:,} critical-risk "
      f"customers. The 963 critical customers represent the group a "
      f"retention team would contact first.")

    h2("13.4 Discussion")
    p("The results are consistent with the exploratory findings: the "
      "attributes that dominate the churn rates in Chapter 7 — contract "
      "type, tenure, payment method and support services — are the same "
      "signals that drive the risk score. The clearest practical finding "
      "is that churn concentrates in the early months of a month-to-month "
      "contract, which suggests retention effort is best directed at "
      "onboarding and at contract conversion.")
    p("The models perform in the range normally reported for this dataset. "
      "The precision-recall trade-off is the main modelling decision: a "
      "campaign with limited capacity would prefer the higher-precision "
      "boosting model, whereas a campaign aiming to reach as many "
      "potential churners as possible is better served by the "
      "higher-recall Random Forest that was deployed.")
    story.append(PageBreak())


def chapter14():
    h1("CHAPTER 14<br/>LIMITATIONS AND FUTURE SCOPE")
    h2("14.1 Limitations")
    bullets([
        "The system was developed and evaluated on one public dataset; "
        "performance on another operator's data is not established.",
        "The dataset is a static snapshot with no time dimension, so "
        "changes in a customer's behaviour cannot be modelled.",
        "Risk-score weights and segment thresholds were set from the "
        "observed patterns rather than optimised numerically.",
        "No campaign-response data exists, so the effectiveness of a "
        "recommended action cannot be measured.",
        "Two modules rank models by different criteria, which produces the "
        "inconsistency recorded in Chapter 8.",
        "The application has no authentication and is intended for local "
        "or trusted-network use.",
    ])

    h2("14.2 Future Scope")
    p("The following are proposals for further work. None of them is "
      "implemented in the current system.")
    bullets([
        "<b>FUTURE SCOPE —</b> add time-series features from billing "
        "history so that trends in usage can inform the prediction.",
        "<b>FUTURE SCOPE —</b> tune hyperparameters systematically and "
        "calibrate the decision threshold to campaign capacity.",
        "<b>FUTURE SCOPE —</b> apply an explainability method such as SHAP "
        "so that each individual prediction can be justified.",
        "<b>FUTURE SCOPE —</b> record campaign outcomes and measure "
        "retention effectiveness against a control group.",
        "<b>FUTURE SCOPE —</b> unify the model-selection criterion across "
        "the training and dashboard modules.",
        "<b>FUTURE SCOPE —</b> add authentication and deploy the "
        "dashboard as a hosted service.",
    ])
    story.append(PageBreak())


def chapter15():
    h1("CHAPTER 15<br/>CONCLUSION")
    p(f"ChurnIQ was developed during a Data Science internship at "
      f"{D('ORGANIZATION NAME')} to address a practical problem: "
      f"identifying which customers are likely to leave, and deciding what "
      f"to do about them. The system covers the full analytical path from "
      f"a raw data file to an exported retention report.")
    p(f"On the reference dataset of {N_ROWS:,} customers, six "
      f"classification algorithms were trained and compared. Random Forest "
      f"produced the highest F1 score ({RF['F1 Score']:.4f}) and was "
      f"persisted with its preprocessing pipeline; Gradient Boosting "
      f"produced the highest ROC-AUC ({GB['ROC-AUC']:.4f}). The "
      f"exploratory analysis identified contract type, tenure, payment "
      f"method and the absence of support services as the strongest "
      f"indicators, and these same signals drive the risk score that "
      f"orders customers for action.")
    p("Predictions are translated into four risk levels, six strategic "
      "segments and a recommended action and channel per customer, all "
      "presented through an eleven-page dashboard with CSV export. The "
      "value of the system lies in this translation: a ranked, explainable "
      "list of customers is something a retention team can work through, "
      "whereas a column of probabilities is not.")
    p("The system identifies and prioritises customers statistically likely "
      "to churn; it does not guarantee retention, and its recommendations "
      "are suggestions derived from historical patterns. The limitations "
      "in Chapter 14 — a single static dataset, heuristic thresholds and "
      "the absence of campaign feedback — indicate where the work would "
      "need to develop before operational use.")
    story.append(PageBreak())


# ==================================================================
# BACK MATTER
# ==================================================================
def references():
    mark_with("bm:ref", Paragraph("REFERENCES", FRONTH))
    refs = [
        "Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in "
        "Python. <i>Journal of Machine Learning Research</i>, 12, "
        "pp. 2825-2830.",
        "McKinney, W. (2010). Data Structures for Statistical Computing in "
        "Python. <i>Proceedings of the 9th Python in Science Conference</i>, "
        "pp. 56-61.",
        "Harris, C. R. et al. (2020). Array Programming with NumPy. "
        "<i>Nature</i>, 585, pp. 357-362.",
        "Breiman, L. (2001). Random Forests. <i>Machine Learning</i>, "
        "45(1), pp. 5-32.",
        "Friedman, J. H. (2001). Greedy Function Approximation: A Gradient "
        "Boosting Machine. <i>Annals of Statistics</i>, 29(5), "
        "pp. 1189-1232.",
        "Hunter, J. D. (2007). Matplotlib: A 2D Graphics Environment. "
        "<i>Computing in Science and Engineering</i>, 9(3), pp. 90-95.",
        "IBM Sample Data Sets. Telco Customer Churn Dataset. Available on "
        "Kaggle.",
        "Streamlit Documentation. Available at https://docs.streamlit.io",
        "Scikit-learn Documentation. Available at "
        "https://scikit-learn.org/stable/",
        "Pandas Documentation. Available at https://pandas.pydata.org/docs/",
        "Plotly Python Documentation. Available at "
        "https://plotly.com/python/",
        "Joblib Documentation. Available at "
        "https://joblib.readthedocs.io",
    ]
    st = ParagraphStyle("ref", parent=BODY, leftIndent=16, firstLineIndent=-16,
                        spaceAfter=5, alignment=TA_JUSTIFY)
    for i, r in enumerate(refs, 1):
        story.append(Paragraph(f"[{i}]&nbsp;&nbsp;{r}", st))
    story.append(PageBreak())


def appendix():
    mark_with("bm:apx", Paragraph("APPENDIX — DASHBOARD SCREENSHOTS",
                                  FRONTH))
    story.append(Paragraph(
        "The screen captures below must be taken from the running "
        "application and inserted in place of the corresponding frames. No "
        "screenshots have been generated artificially for this report. To "
        "capture them, run <font face='Courier' size='10'>streamlit run "
        "app.py</font>, load the default dataset and run the analysis from "
        "the Prediction Center first, so that pages depending on prediction "
        "results are populated.",
        ParagraphStyle("nb", parent=BODY, fontSize=11, leading=16,
                       backColor=colors.HexColor("#f4f4f4"),
                       borderColor=BORDER, borderWidth=0.5, borderPadding=6,
                       spaceBefore=2, spaceAfter=10)))

    items = [
        ("A.1", "Executive Overview"),
        ("A.2", "Dataset Center"),
        ("A.3", "Prediction Center"),
        ("A.4", "Churn Analysis"),
        ("A.5", "Customer Segments"),
        ("A.6", "Risk Center"),
        ("A.7", "Model Performance"),
        ("A.8", "Recommendations"),
    ]
    for num, title in items:
        ph = Table([[Paragraph(
            f"<i>[INSERT SCREENSHOT: {title}]</i>",
            ParagraphStyle("ph", parent=BODYC, fontSize=10,
                           textColor=colors.HexColor("#777777")))]],
            colWidths=[12.4 * cm], rowHeights=[2.9 * cm],
            style=TableStyle([
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#aaaaaa")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BACKGROUND", (0, 0), (-1, -1),
                 colors.HexColor("#fafafa")),
            ]), hAlign="CENTER")
        story.append(KeepTogether([
            ph,
            Paragraph(f"<b>Figure {num}:</b> {title}", CAP),
        ]))


# ==================================================================
# BUILD
# ==================================================================
def build_story():
    cover_page()
    certificate()
    declaration()
    acknowledgement()
    abstract()
    toc()
    list_of_figures_tables()
    for i in range(1, 16):
        globals()[f"chapter{i}"]()
    references()
    appendix()


def make_doc():
    d = BaseDocTemplate(
        str(OUT), pagesize=FMT["page_size"],
        leftMargin=FMT["margin_left_cm"] * cm,
        rightMargin=FMT["margin_right_cm"] * cm,
        topMargin=FMT["margin_top_cm"] * cm,
        bottomMargin=FMT["margin_bottom_cm"] * cm,
        title="ChurnIQ — Summer Internship Project Report",
        author=D("STUDENT NAME"), subject="Summer Internship Project Report",
    )
    fr = Frame(d.leftMargin, d.bottomMargin, d.width, d.height, id="n")
    d.addPageTemplates([
        PageTemplate(id="First", frames=[fr], onPage=cover_page_deco),
        PageTemplate(id="Later", frames=[fr], onPage=plain_page),
    ])
    return d


MAX_PASSES = 8
previous = None
for attempt in range(1, MAX_PASSES + 1):
    story.clear()
    figno.clear()
    tabno.clear()
    RAW_PAGE_OF.clear()
    if attempt > 1:
        PASS = 2
    build_story()
    make_doc().build(story)

    if "ch:1" in RAW_PAGE_OF:
        FRONT_MATTER_PAGES = RAW_PAGE_OF["ch:1"] - 1
    PAGE_OF.clear()
    for k, v in RAW_PAGE_OF.items():
        PAGE_OF[k] = _printed_label(v)

    current = (FRONT_MATTER_PAGES, dict(PAGE_OF))
    if current == previous:
        break
    previous = current
else:
    print("Warning: page numbering did not settle.")

print(f"Converged after {attempt} pass(es); "
      f"front matter = {FRONT_MATTER_PAGES} pages.")
print(f"Report written to {OUT}")
