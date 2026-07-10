# 💼 LoanIQ – AI Loan Rejection Risk Predictor

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Logistic%20Regression-orange?style=for-the-badge&logo=scikitlearn)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red?style=for-the-badge&logo=streamlit)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Dashboard-3F4F75?style=for-the-badge&logo=plotly)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

LoanIQ is an AI-powered credit risk analysis platform that predicts the likelihood of a household experiencing a credit turn-down or avoiding a loan application due to fear of rejection. Built using **Logistic Regression**, the application provides transparent risk predictions, feature contribution analysis, and interactive visualizations through a modern Streamlit dashboard. :contentReference[oaicite:0]{index=0}

---

# 🌐 Live Demo

## 🚀 Try the Application

👉 **Add your Streamlit link here**

---

# 📌 Project Overview

Financial institutions rely on credit risk assessment to make informed lending decisions. LoanIQ demonstrates how Machine Learning can estimate loan rejection risk using demographic and financial indicators.

The project analyzes applicant information such as income category, debt ratio, credit history, education, employment, and assets to generate a comprehensive risk assessment. Every prediction is accompanied by feature-level explanations, making the model more transparent and interpretable. :contentReference[oaicite:1]{index=1}

---

# ✨ Features

✅ AI-powered Loan Rejection Risk Prediction

✅ Logistic Regression Model

✅ Interactive Streamlit Dashboard

✅ Risk Probability Gauge

✅ Feature Contribution Analysis

✅ Explainable AI Predictions

✅ Financial Risk Visualization

✅ Applicant Profile Analysis

✅ Modern Responsive UI

✅ Real-time Prediction

---

# 📊 Dataset

**Dataset Used**

Federal Reserve Survey of Consumer Finances (SCF 2019)

The dataset contains demographic and financial information collected from thousands of U.S. households.

### Features Include

- Age
- Gender
- Marital Status
- Number of Dependents
- Education Level
- Employment Status
- Home Ownership
- Income Category
- Net Worth Category
- Asset Category
- Debt-to-Income Ratio
- Leverage Ratio
- Credit Card Balance
- Late Payment History
- 60+ Day Late Payments
- Bankruptcy History

### Target Variable

- Credit Turn-Down Risk (TURNFEAR)

---

# 🧹 Data Preprocessing

The dataset undergoes several preprocessing steps before model training.

### Steps

- Data Cleaning
- Missing Value Handling
- Feature Selection
- Standard Scaling
- Feature Engineering
- Train-Test Split

---

# 🤖 Machine Learning Model

The project uses

## Logistic Regression

combined with

## StandardScaler

to estimate the probability of loan rejection risk while maintaining model interpretability through feature coefficients. :contentReference[oaicite:2]{index=2}

---

# 📈 Project Workflow

```text
Applicant Information
        │
        ▼
Data Cleaning
        │
        ▼
Feature Engineering
        │
        ▼
Feature Scaling
        │
        ▼
Logistic Regression
        │
        ▼
Risk Probability
        │
        ▼
Feature Contribution Analysis
        │
        ▼
Interactive Dashboard
```

---

# 🛠️ Tech Stack

## Programming Language

- Python

## Machine Learning

- Scikit-Learn
- Logistic Regression

## Data Processing

- Pandas
- NumPy

## Visualization

- Plotly

## Deployment

- Streamlit

---

# 📂 Project Structure

```text
LoanIQ/

│── app.py
│── Loan Rejection Predictor.ipynb
│── loan_dataset.csv
│── logistic_model.pkl
│── scaler.pkl
│── requirements.txt
│── README.md
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/shreya975/LoanIQ.git
```

## Navigate into Project

```bash
cd LoanIQ
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run the Streamlit Application

```bash
streamlit run app.py
```

The application will automatically open in your browser.

If it doesn't open automatically, visit:

```
http://localhost:8501
```

---

# 📊 Input Parameters

The prediction model considers:

- Applicant Age
- Gender
- Marital Status
- Employment Status
- Education
- Number of Dependents
- Income Category
- Asset Category
- Net Worth
- Debt-to-Income Ratio
- Leverage Ratio
- Credit Card Balance
- Payment History
- Bankruptcy History

---

# 📈 Dashboard Features

The application provides:

- 💼 Loan Rejection Risk Prediction
- 📊 Risk Probability Gauge
- 📉 Feature Contribution Analysis
- 📈 Applicant Risk Insights
- 📋 Explainable AI Results
- 📊 Model Analytics
- 📄 Detailed Risk Report
- 🎯 Interactive Visualizations

---

# 💻 Application Screens

## Home Dashboard

(Add Screenshot Here)

---

## Risk Prediction

(Add Screenshot Here)

---

## Model Analytics

(Add Screenshot Here)

---

# 🎯 Future Improvements

- XGBoost Classifier
- Random Forest Comparison
- SHAP Explainable AI
- Credit Score Integration
- PDF Report Generation
- REST API Deployment
- Cloud Database
- Multi-language Support

---

# 🚀 Deployment

The application is deployed using **Streamlit Cloud**.

### Live Application

👉 **Add your Streamlit deployment link here**

---

# 📈 Business Impact

LoanIQ helps financial institutions to:

- Assess applicant credit risk
- Improve lending decisions
- Increase model transparency
- Identify key financial risk factors
- Support data-driven credit analysis

---

# 👩‍💻 Author

## Shreya Mahajan

### GitHub

https://github.com/shreya975

### LinkedIn

https://www.linkedin.com/in/shreya-mahajan-b38b28385/

---

# 🤝 Contributing

Contributions are welcome!

Feel free to fork this repository and submit a Pull Request.

---
