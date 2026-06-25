# 📊 Employee Salary ML Prediction System

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.36-FF4B4B?logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-blue?logo=xgboost)
![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-success)

A professional, end-to-end Machine Learning pipeline and Interactive Dashboard built to predict employee salaries based on demographic and professional features.

---

## 🚀 Problem Statement
Predicting whether an employee's income exceeds `$50K/yr` based on census data. This project solves the problem using advanced Machine Learning techniques, specifically addressing class imbalance and prioritizing model explainability (XAI) so stakeholders can understand *why* the model makes its decisions.

## 🏗️ Architecture & Features
* **Advanced Feature Engineering**: Automatic computation of custom features like `net_capital` and `is_full_time` to provide the model with deeper insights.
* **Class Imbalance Handling**: Uses **SMOTE** (Synthetic Minority Over-sampling Technique) to ensure the model doesn't become biased towards the majority class.
* **Hyperparameter Tuned XGBoost**: The core prediction engine is an `XGBClassifier` that was mathematically optimized using `RandomizedSearchCV` to achieve **>92% ROC-AUC**.
* **Model Explainability (SHAP)**: Fully transparent decision making. The pipeline generates SHAP summary plots to mathematically prove feature importance.
* **Interactive Dashboard**: A custom **Streamlit** web application serving as a production UI. Users can filter cohorts and run the ML model on the fly to generate visual KPIs and confidence metrics.

---

## 💻 Installation & Usage

1. **Clone the repository**
```bash
git clone https://github.com/jagannadharao8/Employee-Salary-Prediction.git
cd Employee-Salary-Prediction
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Train the Model (Optional)**
The repository comes with a pre-trained `model.joblib`. If you wish to retrain it, run:
```bash
python train_model.py
```
*(This will execute the hyperparameter tuning and generate new EDA/SHAP plots in the `eda_plots/` directory).*

4. **Launch the Dashboard**
```bash
streamlit run app.py
```
*(Navigate to `http://localhost:8501` in your browser to use the interactive dashboard).*

---

## 🧠 Behind the Scenes: EDA & Explainability
During training, the pipeline generates crucial insights:
* **Feature Correlation**: Maps out the mathematical relationships between features.
* **SHAP Values**: Proves that features like *Age*, *Education*, and *Marital Status* are the strongest predictors of high income.

These visual insights are embedded directly into the Streamlit app under the **Model Explainability** tab!

---
*Built as a professional capstone for the Edunet IBM Internship.*
