# Adding the logo and the screenshots

Two image sets are still needed. Save them with the names below and run
one command — the builder finds them automatically and fits them to the
page, keeping the report at 40 pages.

```bash
python report/build_report.py
```

## 1. University logo

Save as:

```
report/figures/university_logo.png
```

`.jpg`, `.jpeg`, `.webp`, `.gif` and `.bmp` also work. The white border
around the artwork is cropped automatically before placing, so the logo
supplied as a small mark on a wide white canvas still prints at a proper
size on the cover.

## 2. Dashboard screenshots

Create the folder `report/figures/screenshots/` and save the eight
captures using these names:

| File name | Appendix figure |
| --- | --- |
| `executive_overview.png`  | Figure A.1 |
| `dataset_center.png`      | Figure A.2 |
| `prediction_center.png`   | Figure A.3 |
| `churn_analysis.png`      | Figure A.4 |
| `customer_segments.png`   | Figure A.5 |
| `risk_center.png`         | Figure A.6 |
| `model_performance.png`   | Figure A.7 |
| `recommendations.png`     | Figure A.8 |

Matching is on the page name, not the exact filename, so
`01 Executive Overview.png` or `Executive-Overview.jpg` work too. Any
capture that is missing keeps its labelled empty frame; the rest are
still placed.

Wide screenshots (around 1568x743, as captured from a maximised browser)
are scaled to fit four per page. The appendix then occupies two pages
and the report stays at 40, within the 30-40 limit.

## Checking the result

The builder prints its page count on every run. After adding the images,
confirm:

- the cover shows the logo instead of blank space;
- the appendix shows eight captures and no `[INSERT SCREENSHOT]` frames;
- the total is still 40 pages.

If adding the screenshots pushes the report past 40, reduce the height
limit in `build_report.py` — search for `four captures per page` and
lower the `4.55 * cm` value.
