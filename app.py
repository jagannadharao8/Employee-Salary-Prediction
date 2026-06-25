import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# Set page config
st.set_page_config(page_title="Employee ML Prediction & Search", page_icon="🔍", layout="wide")

st.title("🔍 Targeted ML Salary Predictions")
st.write("Filter your actual employee data and run the Machine Learning model on specific groups (e.g., all Single Males)!")

# Check if model exists
model_path = 'model.joblib'
csv_path = 'employeedata.csv'

if not os.path.exists(model_path):
    st.error("Model file 'model.joblib' not found. Please run 'train_model.py' first.")
    st.stop()

if not os.path.exists(csv_path):
    st.error(f"Could not find `{csv_path}` in the project folder!")
    st.stop()

# Load model & data
@st.cache_resource
def load_model():
    return joblib.load(model_path)

@st.cache_data
def load_data():
    df = pd.read_csv(csv_path)
    df = df.replace('?', np.nan)
    df['net_capital'] = df['capital-gain'] - df['capital-loss']
    df['is_full_time'] = (df['hours-per-week'] >= 40).astype(int)
    return df

model = load_model()
raw_data = load_data()

st.sidebar.header("1. Filter Your Data")
st.sidebar.write("Leave blank to include everyone.")

# Get unique values for dropdowns
genders = raw_data['gender'].dropna().unique().tolist()
marital_statuses = raw_data['marital-status'].dropna().unique().tolist()
educations = raw_data['education'].dropna().unique().tolist()
workclasses = raw_data['workclass'].dropna().unique().tolist()
occupations = raw_data['occupation'].dropna().unique().tolist()
races = raw_data['race'].dropna().unique().tolist()
incomes = raw_data['income'].dropna().unique().tolist()

# Define min/max for sliders
min_age, max_age = int(raw_data['age'].min()), int(raw_data['age'].max())
min_hpw, max_hpw = int(raw_data['hours-per-week'].min()), int(raw_data['hours-per-week'].max())

# Filters UI
selected_income = st.sidebar.multiselect("Actual Income", incomes)
selected_age_range = st.sidebar.slider("Age Range", min_age, max_age, (min_age, max_age))
selected_hpw_range = st.sidebar.slider("Hours per Week", min_hpw, max_hpw, (min_hpw, max_hpw))
selected_gender = st.sidebar.multiselect("Gender", genders)
selected_marital = st.sidebar.multiselect("Marital Status", marital_statuses)
selected_education = st.sidebar.multiselect("Education", educations)
selected_workclass = st.sidebar.multiselect("Workclass", workclasses)
selected_occupation = st.sidebar.multiselect("Occupation", occupations)
selected_race = st.sidebar.multiselect("Race", races)

# Apply filters
filtered_data = raw_data.copy()

# Filter by age and hours first
filtered_data = filtered_data[
    (filtered_data['age'] >= selected_age_range[0]) & (filtered_data['age'] <= selected_age_range[1])
]
filtered_data = filtered_data[
    (filtered_data['hours-per-week'] >= selected_hpw_range[0]) & (filtered_data['hours-per-week'] <= selected_hpw_range[1])
]

if selected_income:
    filtered_data = filtered_data[filtered_data['income'].isin(selected_income)]
if selected_gender:
    filtered_data = filtered_data[filtered_data['gender'].isin(selected_gender)]
if selected_marital:
    filtered_data = filtered_data[filtered_data['marital-status'].isin(selected_marital)]
if selected_education:
    filtered_data = filtered_data[filtered_data['education'].isin(selected_education)]
if selected_workclass:
    filtered_data = filtered_data[filtered_data['workclass'].isin(selected_workclass)]
if selected_occupation:
    filtered_data = filtered_data[filtered_data['occupation'].isin(selected_occupation)]
if selected_race:
    filtered_data = filtered_data[filtered_data['race'].isin(selected_race)]

st.write(f"### Found **{len(filtered_data)}** employees matching your filters.")
st.dataframe(filtered_data.head(10)) # show preview

st.write("---")
st.subheader("2. Run Machine Learning Predictions")
st.write("Press the button below to use the ML model to predict the salaries of these specific employees and compare them against their actual income!")

if st.button("Run ML Prediction on Filtered Data", type="primary"):
    if len(filtered_data) == 0:
        st.warning("No data matches your filters. Please adjust them and try again.")
    else:
        st.write("Predicting...")
        
        try:
            # We need to make sure the required features are present
            features_to_keep = [
                'age', 'workclass', 'education', 'educational-num',
                'marital-status', 'occupation', 'relationship',
                'race', 'gender', 'capital-gain', 'capital-loss',
                'hours-per-week', 'net_capital', 'is_full_time'
            ]
            
            # Filter data to just the required features for ML
            input_df = filtered_data[features_to_keep].copy()
            
            # Predict
            predictions = model.predict(input_df)
            probabilities = model.predict_proba(input_df)[:, 1]
            
            # Create a results dataframe
            results_df = filtered_data.copy()
            results_df['ML Predicted Income'] = np.where(predictions == 1, '>50K', '<=50K')
            results_df['Confidence (>50K)'] = (probabilities * 100).round(2).astype(str) + '%'
            
            # Move the new columns to the front for easier viewing
            cols = results_df.columns.tolist()
            cols = ['ML Predicted Income', 'Confidence (>50K)', 'income'] + [c for c in cols if c not in ['ML Predicted Income', 'Confidence (>50K)', 'income']]
            results_df = results_df[cols]
            
            st.success(f"✅ Successfully ran ML predictions for {len(results_df)} employees!")
            
            # Calculate metrics
            import matplotlib.pyplot as plt
            actual_y = filtered_data['income'].apply(lambda x: 1 if str(x).strip() == '>50K' else 0)
            correct_predictions = (predictions == actual_y).sum()
            accuracy = (correct_predictions / len(actual_y)) * 100
            
            st.markdown("### 📊 Analytical Summary")
            col1, col2, col3 = st.columns(3)
            predicted_high = (predictions == 1).sum()
            with col1:
                st.metric("Total Employees Analyzed", f"{len(results_df):,}")
            with col2:
                st.metric("Predicted >$50K", f"{predicted_high:,}")
            with col3:
                st.metric("Model Accuracy", f"{accuracy:.1f}%")
                
            st.markdown("### 📈 Visual Insights")
            
            # Bar chart comparing Actual vs Predicted
            actual_high = (actual_y == 1).sum()
            actual_low = (actual_y == 0).sum()
            pred_low = (predictions == 0).sum()
            
            chart_data = pd.DataFrame({
                "Actual Income": [actual_high, actual_low],
                "ML Predicted Income": [predicted_high, pred_low]
            }, index=[">50K", "<=50K"])
            
            st.write("**Actual vs Predicted Totals**")
            st.bar_chart(chart_data)
            
            # Histogram for confidence
            st.write("**Model Confidence Spread**")
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.hist(probabilities * 100, bins=20, color='teal', edgecolor='black', alpha=0.7)
            ax.set_title("How Confident is the Model? (Probability Spread)")
            ax.set_xlabel("Confidence % that income is >$50K")
            ax.set_ylabel("Number of Employees")
            st.pyplot(fig)
            
            st.write("---")
            with st.expander("🔍 Expand to see Detailed Raw Data"):
                st.write("### Full Prediction Results:")
                st.dataframe(results_df)
                
                # Provide a download button
                csv_export = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Results as CSV",
                    data=csv_export,
                    file_name="filtered_salary_predictions.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
