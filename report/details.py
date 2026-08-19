"""
ChurnIQ report — personal, college and internship details.
===========================================================

FILL THIS FILE IN, then rebuild:

    python report/build_report.py

Every value below appears somewhere in the report. Anything left as-is
keeps its [BRACKETED PLACEHOLDER] form in the PDF, so nothing is ever
silently invented — an unfilled field is impossible to miss on a
read-through.

Only edit the text between the quotes.
"""

DETAILS = {

    # ------------------------------------------------------------------
    # 1. STUDENT  — title page, certificate, declaration, acknowledgement
    # ------------------------------------------------------------------
    "STUDENT NAME":              "[STUDENT NAME]",
    "ROLL NUMBER":               "[ROLL NUMBER]",

    # The programme you are enrolled in, e.g. "Computer Science and Engineering"
    "COURSE / BRANCH":           "[COURSE / BRANCH]",

    # The award, e.g. "Bachelor of Technology"
    "DEGREE":                    "[DEGREE]",

    # ------------------------------------------------------------------
    # 2. INSTITUTION
    # ------------------------------------------------------------------
    "COLLEGE NAME":              "[COLLEGE NAME]",
    "UNIVERSITY NAME":           "[UNIVERSITY NAME]",

    # e.g. "Department of Computer Science and Engineering"
    "DEPARTMENT NAME":           "[DEPARTMENT NAME]",

    # e.g. "2024-25"
    "ACADEMIC SESSION / YEAR":   "[ACADEMIC SESSION / YEAR]",

    # ------------------------------------------------------------------
    # 3. SUPERVISORS
    # ------------------------------------------------------------------
    "FACULTY MENTOR NAME":       "[FACULTY MENTOR NAME]",

    # Faculty mentor's rank, e.g. "Assistant Professor"
    "DESIGNATION":               "[DESIGNATION]",

    "HEAD OF DEPARTMENT NAME":   "[HEAD OF DEPARTMENT NAME]",

    # Leave as-is if the examiner signs an unnamed line
    "EXTERNAL EXAMINER":         "[EXTERNAL EXAMINER]",

    # ------------------------------------------------------------------
    # 4. SUBMISSION
    # ------------------------------------------------------------------
    "DATE OF SUBMISSION":        "[DATE OF SUBMISSION]",
    "PLACE":                     "[PLACE]",

    # ------------------------------------------------------------------
    # 5. INTERNSHIP / ORGANISATION  — Chapter 2, Tables 2.1 and 2.2
    #
    # If this was a self-driven academic project rather than an
    # internship at a company, see the note at the bottom of this file.
    # ------------------------------------------------------------------
    "ORGANIZATION NAME":         "[ORGANIZATION NAME]",
    "ORGANIZATION ADDRESS":      "[ORGANIZATION ADDRESS]",
    "DEPARTMENT / DIVISION":     "[DEPARTMENT / DIVISION]",

    # e.g. "Private limited company", "Academic institution"
    "NATURE OF ORGANIZATION":    "[NATURE OF ORGANIZATION]",

    "INDUSTRY MENTOR NAME":        "[INDUSTRY MENTOR NAME]",
    "INDUSTRY MENTOR DESIGNATION": "[INDUSTRY MENTOR DESIGNATION]",

    # "Onsite", "Remote" or "Hybrid"
    "ONSITE / REMOTE / HYBRID":  "[ONSITE / REMOTE / HYBRID]",

    "INTERNSHIP START DATE":     "[INTERNSHIP START DATE]",
    "INTERNSHIP END DATE":       "[INTERNSHIP END DATE]",

    # e.g. "8 weeks"
    "TOTAL DURATION IN WEEKS":   "[TOTAL DURATION IN WEEKS]",
    "WORKING DAYS PER WEEK":     "[WORKING DAYS PER WEEK]",
    "DAILY WORKING HOURS":       "[DAILY WORKING HOURS]",

    # e.g. "Weekly review meeting"
    "REVIEW / REPORTING FREQUENCY": "[REVIEW / REPORTING FREQUENCY]",
}


# ----------------------------------------------------------------------
# 6. PHASE SCHEDULE — Table 2.3, "Phase-wise execution plan"
#
# Ten phases, in order. Give the week span for each, e.g. "Week 1-2".
# The activities and deliverables are already filled in from the
# repository; only the durations are yours to supply.
#
# If the ten phases do not match how your time was actually spent, edit
# the wording in build_report.py rather than forcing a false schedule.
# ----------------------------------------------------------------------
PHASE_WEEKS = [
    "[WEEK RANGE]",   # 1  Problem study, dataset acquisition, project setup
    "[WEEK RANGE]",   # 2  Dataset validation and cleaning
    "[WEEK RANGE]",   # 3  Exploratory data analysis
    "[WEEK RANGE]",   # 4  Feature engineering and preprocessing pipeline
    "[WEEK RANGE]",   # 5  Model training and comparison
    "[WEEK RANGE]",   # 6  Model evaluation and persistence
    "[WEEK RANGE]",   # 7  Prediction pipeline development
    "[WEEK RANGE]",   # 8  Customer intelligence and segmentation
    "[WEEK RANGE]",   # 9  Recommendation engine and analytics
    "[WEEK RANGE]",   # 10 Dashboard development, styling and testing
]


# ----------------------------------------------------------------------
# NOTE — if this was NOT an internship
#
# Chapter 2 is written as "Internship / Project Overview" and Tables 2.1
# and 2.2 assume a host organisation. If the project was carried out
# independently or as coursework, do not invent an employer. Either:
#
#   * put your college in ORGANIZATION NAME and your faculty mentor in
#     INDUSTRY MENTOR NAME, or
#   * ask for Chapter 2 to be rewritten as a project-only chapter, and
#     the organisation tables removed.
# ----------------------------------------------------------------------
