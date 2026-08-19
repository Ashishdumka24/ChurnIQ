# ChurnIQ — Project Report

This directory generates `ChurnIQ_Project_Report.pdf`, a 106-page academic
report describing the ChurnIQ system.

Every number, table and figure in the report is derived from this repository
at build time — the dataset files in `Data/`, the metrics in
`Models/model_results.csv`, and the metadata inside
`Models/best_churn_model.pkl`. Nothing is transcribed by hand, so the report
cannot drift out of step with the code.

## Contents

| File | Purpose |
| --- | --- |
| `build_report.py` | Builds the PDF. Reads the repository's data and models directly. |
| `make_figures.py` | Regenerates the 15 data figures in `figures/`. |
| `make_diagrams.py` | Regenerates the 7 design diagrams in `figures/`. |
| `figures/` | 22 PNGs used by the report. |
| `pipeline_output_sample.csv` | Sample of real pipeline output, used in Appendix C. |
| `ChurnIQ_Project_Report.pdf` | The generated report. |

## Regenerating

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install reportlab pandas numpy scikit-learn joblib pillow matplotlib seaborn

python report/make_figures.py      # only if the data changed
python report/make_diagrams.py     # only if the design changed
python report/build_report.py
```

`build_report.py` runs from the repository root and writes the PDF next to
itself. The build is deterministic: the same inputs produce the same text.

### Why the build runs several passes

The table of contents, list of figures and list of tables carry real page
numbers. Filling those lists in changes their length, which shifts every page
after them, which changes the numbers again. The builder therefore records
where each heading, figure and table actually landed and rebuilds until the
page assignments stop changing — normally three passes. It prints:

```
Converged after 3 pass(es); front matter = 15 pages.
```

If it ever reports that numbering did not settle, the page numbers in the
front matter should not be trusted.

## Before submitting

Two things still need your input.

### 1. Fill in the bracketed placeholders

No personal, college or internship detail was invented. The report contains
**70 bracketed placeholders across 28 distinct fields**, all of which must be
replaced with real values. They are written as `[LIKE THIS]` so they are
impossible to miss on a read-through.

They are concentrated in a handful of places, so editing is quick:

| Where | Report pages | Fields |
| --- | --- | --- |
| Title, Certificate, Declaration, Acknowledgement | i–iv | name, roll number, course, college, university, department, mentors, degree, session, dates, place |
| Chapter 2 (Organisation, Duration) | 6–7 | organisation name and address, nature, industry mentor, mode, start/end dates, duration, working hours |
| Table 2.3 (Phase-wise plan) | 8 | `[WEEK RANGE]` for each of the 10 phases |

To find them all in the source:

```bash
grep -n '\[[A-Z][A-Z0-9 /-]*\]' report/build_report.py
```

Edit `build_report.py`, then rebuild. Do not edit the PDF directly, or your
changes will be lost the next time the report is generated.

### 2. Insert the dashboard screenshots

Appendix D contains **12 labelled placeholder boxes** rather than screenshots.
None were fabricated. To capture them:

```bash
streamlit run app.py
```

Open the URL shown, load the default dataset, then run the analysis from the
Prediction Center **first** — several pages stay empty until predictions
exist. Capture each of the 12 pages listed in Appendix D and insert them in
place of the corresponding boxes.

## A note on the two "best" models

The report does not paper over an inconsistency that exists in the code:

- `Src/model_training.py` → `select_best_model()` ranks by **F1**, so the
  model saved to `Models/best_churn_model.pkl` is **Random Forest**
  (F1 0.6226).
- `Src/dashboard_analytics.py` → `get_best_model_metrics()` ranks by
  **ROC-AUC**, so the dashboard displays **Gradient Boosting** as best
  (ROC-AUC 0.8437).

Both statements are true under their own criterion. This is documented as an
observed discrepancy in sections 8.8, 13.7 and 14.1 rather than resolved
silently. If you would rather the report present a single best model, change
the ranking criterion in the code first and rebuild — the report will follow.

## Scope of claims

The report describes only functionality present in this repository. Test cases
in Chapter 12 are marked either **Executed** (run against the code) or
**Derived** (logically implied by the implementation but not run), and
improvements that are not implemented are labelled **FUTURE SCOPE**. The
recommendation engine is described as producing analytical suggestions from
historical patterns, not guaranteed retention outcomes.
