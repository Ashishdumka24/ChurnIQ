# 📊 ChurnIQ

## Customer Churn Prediction & Business Intelligence System

ChurnIQ is an end-to-end **Customer Churn Prediction and Business Intelligence System** designed to transform customer data into meaningful predictive insights, customer-risk intelligence, segmentation results and actionable retention strategies.

The project combines **Data Science, Machine Learning, Data Visualization and Business Intelligence** into an interactive **Streamlit dashboard**.

---

## 📌 Table of Contents

* [Project Overview](#-project-overview)
* [Objectives](#-objectives)
* [Key Features](#-key-features)
* [System Workflow](#-system-workflow)
* [Technology Stack](#-technology-stack)
* [Project Structure](#-project-structure)
* [Machine Learning](#-machine-learning)
* [Dataset](#-dataset)
* [Dashboard Modules](#-dashboard-modules)
* [Installation](#-installation)
* [Running the Application](#-running-the-application)
* [Using ChurnIQ](#-using-churniq)
* [Model Evaluation](#-model-evaluation)
* [Business Intelligence](#-business-intelligence)
* [Learning Outcomes](#-learning-outcomes)
* [Future Improvements](#-future-improvements)
* [Project Significance](#-project-significance)
* [Developer](#-developer)
* [Conclusion](#-conclusion)

---

# 🔎 Project Overview

Customer churn is an important business problem where customers discontinue a company's products or services. Customer loss can affect revenue, customer lifetime value and long-term business growth.

ChurnIQ addresses this problem using a complete **Data Science and Machine Learning workflow**.

The system accepts customer information, performs data validation and processing, analyzes customer behavior, prepares predictive features and uses machine-learning models to generate churn predictions.

The predictions are then connected with:

* Customer risk analysis
* Customer segmentation
* Churn analytics
* Model evaluation
* Business insights
* Retention recommendations
* Interactive reporting

The main concept of ChurnIQ is:

> **Transform customer data into predictive intelligence that can support data-driven business decisions.**

---

# 🎯 Objectives

The primary objectives of ChurnIQ are:

1. Predict customers who are likely to churn.
2. Analyze customer behavior and churn patterns.
3. Perform data validation and cleaning.
4. Prepare meaningful predictive features.
5. Train classification-based machine-learning models.
6. Evaluate and compare model performance.
7. Identify customers with higher churn risk.
8. Segment customers for deeper analysis.
9. Generate actionable retention recommendations.
10. Present results through an interactive dashboard.
11. Demonstrate the practical application of Data Science to a real-world business problem.

---

# 🚀 Key Features

## 📂 Dataset Center

The Dataset Center provides a centralized location for managing customer datasets.

Features include:

* CSV upload
* XLS upload
* XLSX upload
* Dataset preview
* Row and column information
* Missing-cell analysis
* Data-type information
* Unique-value analysis
* Data-quality indicators
* Default dataset support

---

## 📊 Churn Analytics

The analytics module provides an overview of customer behavior and churn patterns.

It helps analyze:

* Customer distribution
* Churn distribution
* Customer characteristics
* Service usage
* Account information
* Billing patterns
* Customer behavior

---

## 🤖 Churn Prediction

The prediction module uses the trained machine-learning pipeline to generate customer-level churn predictions.

It provides predictive information such as:

* Churn prediction
* Prediction probability
* Customer-level results
* Risk-oriented information

---

## ⚠️ Risk Center

The Risk Center focuses on identifying customers who may require greater attention.

It helps users analyze customer risk and prioritize customers for further investigation or potential retention activities.

---

## 👥 Customer Segmentation

Customer segmentation groups customers according to relevant characteristics and analytical patterns.

This provides a broader understanding of customer populations rather than relying only on a single churn prediction.

---

## 🔎 Customer Explorer

The Customer Explorer allows users to search and inspect individual customer records.

This provides a detailed customer-level view alongside the overall dashboard analytics.

---

## 🧪 Model Performance

ChurnIQ evaluates classification models using multiple performance metrics, including:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC

Model evaluation helps determine the effectiveness of different machine-learning approaches.

---

## 💡 Business Recommendations

ChurnIQ converts analytical findings into practical business-oriented insights.

The recommendation system is intended to help users understand:

* Which customers may require attention
* Which customer groups have higher risk
* What customer characteristics may be associated with churn
* Where retention efforts can potentially be prioritized

---

## 📥 Reports

The reporting functionality provides customer and analytical information that can be used for further analysis and business reporting.

---

## 👨‍💻 About

The About section provides detailed information about:

* ChurnIQ
* Project objectives
* Project workflow
* Technology stack
* Machine-learning concepts
* Business value
* Learning outcomes
* Developer information

---

# 🔄 System Workflow

ChurnIQ follows a structured Data Science workflow:

```text
Customer Dataset
       ↓
Data Validation
       ↓
Data Cleaning
       ↓
Exploratory Data Analysis
       ↓
Feature Engineering
       ↓
Model Training
       ↓
Model Evaluation
       ↓
Churn Prediction
       ↓
Risk Intelligence
       ↓
Customer Segmentation
       ↓
Business Recommendations
       ↓
Interactive Dashboard
```

---

# 🛠️ Technology Stack

| Technology   | Purpose                        |
| ------------ | ------------------------------ |
| Python       | Core programming language      |
| Pandas       | Data manipulation and analysis |
| NumPy        | Numerical computation          |
| Scikit-learn | Machine Learning               |
| Matplotlib   | Data visualization             |
| Seaborn      | Statistical visualization      |
| Plotly       | Interactive visualization      |
| Streamlit    | Interactive web application    |
| MySQL        | Database support               |

---

# 📁 Project Structure

```text
CUSTOMER-CHURN-PREDICTION/
│
├── app.py
├── requirements.txt
├── style.css
│
├── Data/
│   ├── Telco-Customer-Churn.csv
│   ├── telco_churn_clean.csv
│   └── telco_churn_processed.csv
│
├── Models/
│   ├── best_churn_model.pkl
│   ├── preprocessing.pkl
│   └── model_results.csv
│
└── Src/
    ├── clean_data.py
    ├── dashboard_analytics.py
    ├── eda_analysis.py
    ├── eda.py
    ├── feature_engineering.py
    ├── model_evaluation.py
    ├── model_training.py
    ├── prediction_pipeline.py
    └── segmentation.py
```

---

# 🤖 Machine Learning

Machine Learning is the predictive foundation of ChurnIQ.

The general workflow consists of:

### 1. Data Preparation

Customer information is loaded and prepared for analysis.

### 2. Data Cleaning

Data-quality issues are inspected and handled.

### 3. Feature Engineering

Relevant customer attributes are transformed into useful predictive features.

### 4. Model Training

Classification models are trained using historical customer information.

### 5. Model Evaluation

Models are evaluated using appropriate classification metrics.

### 6. Model Selection

The best-performing model can be selected according to the evaluation results.

### 7. Prediction Pipeline

The preprocessing and trained model are connected through a reusable prediction pipeline.

### 8. Customer Prediction

The pipeline generates churn predictions and probability information for customers.

---

# 📊 Dataset

The primary dataset used by the project is the **IBM Telco Customer Churn dataset**.

The dataset contains customer information related to areas such as:

* Customer demographics
* Customer services
* Contract information
* Payment methods
* Tenure
* Monthly charges
* Total charges
* Churn status

The `Churn` variable represents whether the customer has discontinued the service.

---

# 🌐 Dashboard Modules

ChurnIQ provides an interactive dashboard containing multiple analytical areas.

### 📊 Dashboard

Provides an overall summary of customer and churn information.

### 📂 Dataset Center

Allows users to upload, inspect and manage datasets.

### 📈 Churn Analytics

Provides analytical views of customer churn patterns.

### 🤖 Churn Prediction

Generates customer-level churn predictions.

### ⚠️ Risk Center

Highlights customers based on predicted risk.

### 👥 Customer Segmentation

Provides customer-group analysis.

### 🔎 Customer Explorer

Allows individual customer records to be searched and inspected.

### 🧪 Model Performance

Provides model comparison and evaluation metrics.

### 💡 Recommendations

Provides retention-oriented insights.

### 📥 Reports

Provides reporting functionality.

### 👨‍💻 About

Explains the project, methodology, technology and developer.

---

# 💻 Installation

## 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/ChurnIQ.git
```

Move into the project directory:

```bash
cd ChurnIQ
```

---

## 2. Create a virtual environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
source venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

After running the command, Streamlit will provide a local URL.

Open the displayed URL in your browser to access ChurnIQ.

---

# 📤 Using ChurnIQ

### Step 1 — Open the Dashboard

Launch the application using Streamlit.

### Step 2 — Select Dataset Center

Navigate to:

```text
📂 Dataset Center
```

### Step 3 — Upload Data

Upload a supported:

```text
CSV
XLS
XLSX
```

file.

### Step 4 — Inspect the Dataset

Review:

* Rows
* Columns
* Missing values
* Data types
* Unique values
* Data quality

### Step 5 — Analyze

Explore churn patterns through the analytics modules.

### Step 6 — Predict

Run the churn prediction workflow.

### Step 7 — Analyze Risk

Open the Risk Center to examine customer-risk information.

### Step 8 — Explore Customers

Use Customer Explorer for individual customer analysis.

### Step 9 — Review Recommendations

Use the recommendation module to interpret customer-risk information.

### Step 10 — Generate Reports

Use the reporting functionality for further analysis.

---

# 🧪 Model Evaluation

ChurnIQ considers multiple classification metrics when evaluating predictive models.

### Accuracy

Measures the percentage of overall predictions that are correct.

### Precision

Measures how reliable positive churn predictions are.

### Recall

Measures how effectively churn customers are identified.

### F1-Score

Provides a balance between Precision and Recall.

### ROC-AUC

Measures the model's ability to distinguish between classes across different thresholds where applicable.

Using multiple metrics provides a more complete understanding of model performance.

---

# 💼 Business Intelligence

The key objective of ChurnIQ is to connect technical Machine Learning with practical Business Intelligence.

The platform follows the principle:

```text
DATA
  ↓
INFORMATION
  ↓
ANALYSIS
  ↓
PREDICTION
  ↓
RISK
  ↓
INSIGHT
  ↓
ACTION
```

Instead of simply answering:

> What happened?

ChurnIQ attempts to provide information that helps users investigate:

> Which customers may be at risk?

and:

> How can customer information support better retention decisions?

---

# 🎓 Learning Outcomes

This project demonstrates practical knowledge of:

* Python programming
* Data preprocessing
* Data cleaning
* Exploratory Data Analysis
* Data visualization
* Feature engineering
* Feature selection
* Classification algorithms
* Machine-learning model training
* Model evaluation
* Predictive analytics
* Customer segmentation
* Risk analysis
* Business intelligence
* Dashboard development
* Streamlit application development

---

# 🔮 Future Improvements

Possible future improvements for ChurnIQ include:

* Advanced model optimization
* Automated model retraining
* More machine-learning algorithms
* Explainable AI integration
* Advanced customer segmentation
* Real-time prediction
* Cloud database integration
* Automated email-based retention alerts
* Advanced report generation
* More interactive visualizations
* Deployment with scalable cloud infrastructure

---

# 🌟 Project Significance

ChurnIQ demonstrates how a complete Data Science project can progress from raw customer data to business-oriented intelligence.

The project brings together:

```text
Data Science
+
Machine Learning
+
Data Visualization
+
Business Intelligence
+
Interactive Web Development
```

The result is an integrated platform capable of analyzing customer information and presenting predictive insights through an accessible dashboard.

---

# 🏆 Conclusion

ChurnIQ demonstrates the practical application of Data Science and Machine Learning to the customer-retention problem.

By combining **data processing, exploratory analysis, feature engineering, machine learning, model evaluation, churn prediction, risk analysis, customer segmentation, visualization and business recommendations**, the system provides a complete analytical workflow.

The project shows how machine-learning predictions can be transformed into understandable business intelligence rather than remaining isolated technical results.

---

# 👨‍💻 Developer

## **Ashish Dumka**

**Project:** ChurnIQ — Customer Churn Prediction & Business Intelligence System

**Project Type:** College Data Science Project

**Focus:** Data Science, Machine Learning & Business Intelligence

---

# 📜 License

This project was developed for **educational and academic purposes**.

---

# ⭐ ChurnIQ

> **From Customer Data to Business Decisions.** 📊🤖💡

**Developed by Ashish Dumka**
