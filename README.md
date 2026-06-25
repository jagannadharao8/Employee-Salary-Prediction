# Employee Salary Prediction

## Project Overview

This project builds a **Machine Learning classification model** to predict whether an employee earns more than $50K or not. The solution helps HR teams automate and streamline high-income employee identification based on demographic and employment features.

### Key Objective
Predicts if an employee's income exceeds $50,000 annually using features such as:
- **Age** - Employee's age in years
- **Education** - Highest level of education attained
- **Occupation** - Type of job/profession
- **Hours per week** - Average working hours per week
- **Educational number** - Numeric encoding of education level

---

## Dataset

**Dataset Size:** 48,842 employee records (45,222 after preprocessing)

**Features:** 15 attributes including demographics, work characteristics, and income

### Feature Details
- `age`: Integer age of the employee
- `workclass`: Employment type (Private, Self-employed, Government, etc.)
- `education`: Education level (Bachelors, Masters, High School, etc.)
- `educational-num`: Numeric representation of education level
- `marital-status`: Marital status category
- `occupation`: Job occupation category
- `relationship`: Family relationship status
- `race`: Race/ethnicity
- `gender`: Gender (Male/Female)
- `capital-gain`: Capital gains in dollars
- `capital-loss`: Capital losses in dollars
- `hours-per-week`: Hours worked per week
- `native-country`: Country of origin

### Target Variable
- `income`: Binary classification (<=50K or >50K)
  - Class 0: Income <= $50,000
  - Class 1: Income > $50,000
  - Class distribution: 81.8% (<=50K) vs 18.2% (>50K)

---

## Data Preprocessing

The following preprocessing steps were applied:

1. **Handling Missing Values**
   - Removed rows with missing values (represented as '?')
   - Reduced dataset from 48,842 to 45,222 records

2. **Feature Selection**
   - Selected 5 key features: age, education, occupation, hours-per-week, educational-num
   - Dropped irrelevant features for model simplicity

3. **Encoding Categorical Variables**
   - Applied LabelEncoder to 'education' and 'occupation' columns
   - Converted income target variable to binary (0 and 1)

4. **Feature Scaling**
   - Used StandardScaler for normalizing numeric features
   - Essential for tree-based models to improve training efficiency

5. **Train-Test Split**
   - 80% training data (36,177 samples)
   - 20% testing data (9,045 samples)
   - Random state = 42 for reproducibility

---

## Model Architecture

### Algorithm: Random Forest Classifier

**Why Random Forest?**
- Handles categorical and numeric data well
- Robust to outliers and missing values
- Provides feature importance rankings
- No assumptions about data distribution
- Good balance between bias and variance

### Model Configuration
```python
RandomForestClassifier(
    n_estimators=100,      # 100 decision trees
    random_state=42        # For reproducibility
)
```

---

## Model Performance

### Accuracy Metrics
- **Overall Accuracy:** 77.21%
- **Precision (>50K):** 54%
- **Recall (>50K):** 41%
- **F1-Score (>50K):** 0.47

### Confusion Matrix
```
                Predicted
             <=50K    >50K
Actual <=50K  6086     756   (True Negative: 89%, False Positive: 11%)
Actual >50K   1305     898   (False Negative: 59%, True Positive: 41%)
```

### Classification Report
```
              precision    recall  f1-score   support

    <=50K         0.82      0.89      0.86      6842
    >50K          0.54      0.41      0.47      2203

  accuracy                           0.77      9045
  macro avg      0.68      0.65      0.66      9045
weighted avg    0.76      0.77      0.76      9045
```

### Feature Importance
The Random Forest model identifies the following features as most important for predictions:
1. **Age** - Strongest predictor of high income
2. **Educational Number** - Education level significantly influences income
3. **Hours Per Week** - Working more hours correlates with higher income
4. **Education** - Type of education matters
5. **Occupation** - Job category influences income levels

---

## Project Files

```
.
├── Employee salary prediction.ipynb     # Main Jupyter notebook with complete implementation
├── employeedata.csv                      # Dataset file (48,842 records)
├── Edunet ibm internship/                # IBM Edunet internship related files
├── project code images/                  # Visualization images and charts
└── README.md                             # This file
```

---

## Installation & Dependencies

### Required Libraries
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Versions Used
- Python 3.x
- pandas: For data manipulation and analysis
- numpy: For numerical computations
- matplotlib: For static visualizations
- seaborn: For statistical data visualization
- scikit-learn: For machine learning algorithms

---

## Usage

### Running the Project

1. **Clone the repository**
   ```bash
   git clone https://github.com/jagannadharao8/Employee-Salary-Prediction.git
   cd Employee-Salary-Prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Open the Jupyter notebook**
   ```bash
   jupyter notebook "Employee salary prediction.ipynb"
   ```

4. **Run all cells** to execute the complete pipeline

### Making Predictions

Example prediction for a 35-year-old employee:
```python
sample_employee = pd.DataFrame({
    'age': [35],
    'education': [1],              # Bachelors degree (encoded)
    'occupation': [6],             # Exec-managerial (encoded)
    'hours-per-week': [45],
    'educational-num': [13]
})

sample_scaled = scaler.transform(sample_employee)
prediction = model.predict(sample_scaled)
result = '>50K' if prediction[0] == 1 else '<=50K'
print(f"Predicted income: {result}")
```

---

## Model Insights

### Strengths
✅ High overall accuracy (77.21%)
✅ Excellent at identifying low-income earners (89% recall for <=50K)
✅ Robust handling of mixed data types
✅ Feature importance clearly indicates influential factors
✅ Reproducible results with fixed random state

### Limitations
⚠️ Class imbalance (4:1 ratio) affects >50K prediction accuracy
⚠️ Lower recall for >50K class (41%) - misses some high earners
⚠️ Precision for >50K is moderate (54%)
⚠️ Model may be biased toward the majority class

### Improvements for Future Work
1. **Address Class Imbalance**
   - Apply SMOTE or class weight balancing
   - Use stratified k-fold cross-validation

2. **Hyperparameter Tuning**
   - Grid search for optimal RandomForest parameters
   - Experiment with max_depth, min_samples_split

3. **Feature Engineering**
   - Create interaction features
   - Add polynomial features
   - Feature selection using mutual information

4. **Ensemble Methods**
   - Try XGBoost, LightGBM, or Gradient Boosting
   - Stack multiple models

5. **Model Interpretation**
   - Use SHAP or LIME for explainability
   - Analyze decision boundaries

---

## Results Visualization

The project includes the following visualizations:
- **Feature Importance Plot** - Bar chart showing feature relevance
- **Confusion Matrix** - Heatmap of prediction outcomes
- **Classification Report** - Detailed metrics for each class
- **ROC-AUC Curve** - Model performance across thresholds

---

## Business Applications

### HR Use Cases
1. **Salary Benchmarking** - Identify employees earning above market standards
2. **Compensation Analysis** - Find patterns in high earner demographics
3. **Retention Strategy** - Focus on high-value employees
4. **Recruitment** - Predict salary ranges for new positions
5. **Career Planning** - Guide employees toward high-earning paths

### Decision Support
- Automated flagging of potentially underpaid employees
- Compensation equity audits
- Workforce segmentation by income level
- Strategic HR resource allocation

---

## Author

**JALLA JAGANNADHARAO**

This project was developed as part of the IBM Edunet internship program, focusing on practical machine learning applications in HR analytics.

---

## License

This project is open source and available under the MIT License.

---

## Contact & Support

For questions or suggestions about this project:
- GitHub: [@jagannadharao8](https://github.com/jagannadharao8)
- Project Repository: [Employee-Salary-Prediction](https://github.com/jagannadharao8/Employee-Salary-Prediction)

---

## References

- Scikit-learn Documentation: https://scikit-learn.org
- Random Forest Classifier: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- Pandas Documentation: https://pandas.pydata.org
- Census Income Dataset: https://archive.ics.uci.edu/ml/datasets/adult

---

**Last Updated:** February 2026
**Status:** Project Complete
