# ChurnIQ — Summer Internship Project Report

Generates `ChurnIQ_Project_Report.pdf` (40 pages) in the format
prescribed by Amrapali University.

All technical content is read from the repository at build time — the
datasets in `Data/`, the metrics in `Models/model_results.csv` and the
metadata inside `Models/best_churn_model.pkl` — so the report cannot
drift away from the code.

## Rebuilding

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install reportlab pandas numpy scikit-learn joblib pillow

python report/build_report.py
```

The builder runs several passes and prints when it settles:

```
Converged after 3 pass(es); front matter = 8 pages.
```

It repeats because the table of contents carries real page numbers, and
filling it in shifts every page after it. If it ever reports that
numbering did not settle, the page numbers should not be trusted.

## Format applied

| Item | Setting |
| --- | --- |
| Paper | A4 |
| Font | Times New Roman 12 pt |
| Headings | 14 pt bold |
| Line spacing | 1.5 |
| Margins | Left 1.5 in, others 1 in |
| Printing | Single side |
| Length | 40 pages |

These live in the `FMT` dictionary near the top of `build_report.py`.

## Two things still to add

### 1. University logo

Save the logo as **`report/figures/university_logo.png`** and rebuild.
The builder also accepts `.jpg`, `.jpeg`, `.webp`, `.gif` and `.bmp`,
and will find it under a few common names (`logo`, `amrapali`, and so
on) in `report/figures/`, `report/` or the repository root.

The surrounding white border is cropped automatically before the logo is
placed, so a small mark on a large white canvas still prints at a
sensible size. Until a file is present the cover shows a labelled empty
frame rather than a substituted graphic.

### 2. Faculty mentor name

`[FACULTY MENTOR NAME]` appears on the certificate and in the
acknowledgement. This is the internal college guide, not the industry
mentor. Set it in `report/details.py` and rebuild.

### 3. Dashboard screenshots

The appendix holds eight labelled frames. To capture them:

```bash
streamlit run app.py
```

Load the default dataset and run the analysis from the Prediction Center
**first** — several pages stay empty until predictions exist.

## Details

Personal, college and internship details are in `report/details.py`.
Any field left in `[BRACKETS]` renders as a visible placeholder rather
than being invented, so nothing unverified can slip into the report
unnoticed.

The internship duration ("Approximately 6 weeks") is derived from the
supplied start and end dates, not assumed.

## A note on the two "best" models

The report records an inconsistency that exists in the code rather than
hiding it:

- `Src/model_training.py` ranks by **F1**, so the saved model is
  **Random Forest** (F1 0.6226).
- `Src/dashboard_analytics.py` ranks by **ROC-AUC**, so the dashboard
  displays **Gradient Boosting** (ROC-AUC 0.8437).

Both are correct under their own criterion. This is documented in
section 8.6. To present a single best model, change the ranking
criterion in the code and rebuild.

## Scope of claims

The report describes only functionality present in this repository.
Test cases in Chapter 12 are marked **Executed** or **Derived**, and
unimplemented improvements are labelled **FUTURE SCOPE**. The system is
described as prioritising customers likely to churn, not as guaranteeing
retention.
