"""
Machine Learning Training Pipeline for Employee Salary Prediction.

This script performs data loading, custom feature engineering, 
handles class imbalance using SMOTE, trains multiple models, 
performs hyperparameter tuning on an XGBoost classifier, 
and generates EDA/SHAP plots. The final pipeline is saved as model.joblib.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import shap

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def load_and_preprocess_data():
    print("Loading data...")
    # Load dataset
    data = pd.read_csv('employeedata.csv')
    
    # Clean data (replace '?' with NaN)
    data = data.replace('?', np.nan)
    
    # -----------------------------------------
    # FEATURE ENGINEERING
    # -----------------------------------------
    print("Engineering features...")
    data['net_capital'] = data['capital-gain'] - data['capital-loss']
    data['is_full_time'] = (data['hours-per-week'] >= 40).astype(int)
    
    # Features to keep
    features_to_keep = [
        'age', 'workclass', 'education', 'educational-num',
        'marital-status', 'occupation', 'relationship',
        'race', 'gender', 'capital-gain', 'capital-loss',
        'hours-per-week', 'net_capital', 'is_full_time', 'income'
    ]
    data = data[features_to_keep]
    
    # Drop rows where target is missing just in case
    data = data.dropna(subset=['income'])
    
    # Create EDA directory
    os.makedirs('eda_plots', exist_ok=True)
    
    # -----------------------------------------
    # ADVANCED EDA: Correlation Heatmap
    # -----------------------------------------
    print("Generating Advanced EDA plots...")
    plt.figure(figsize=(10, 8))
    # Select only numeric columns for correlation matrix
    numeric_df = data.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('eda_plots/correlation_heatmap.png')
    plt.close()
    
    # Target encoding
    X = data.drop('income', axis=1)
    y = data['income'].apply(lambda x: 1 if str(x).strip() == '>50K' else 0)
    
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def build_pipeline():
    # Identify column types
    numeric_features = ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'net_capital', 'is_full_time']
    categorical_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender']

    # Preprocessing for numerical data
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def train_and_evaluate():
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    preprocessor = build_pipeline()
    
    # -----------------------------------------
    # MODELS TO COMPARE (Using SMOTE for imbalance)
    # -----------------------------------------
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    }
    
    best_auc = 0
    best_model_name = ""
    best_pipeline = None
    
    print("\nTraining models and evaluating...\n")
    
    for name, model in models.items():
        print(f"--- {name} ---")
        
        pipeline = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', model)
        ])
        
        # -----------------------------------------
        # HYPERPARAMETER TUNING (Only for XGBoost)
        # -----------------------------------------
        if name == 'XGBoost':
            print("Running RandomizedSearchCV for XGBoost (This may take a few minutes)...")
            param_distributions = {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [3, 5, 7],
                'classifier__learning_rate': [0.01, 0.1, 0.2],
                'classifier__subsample': [0.8, 1.0]
            }
            search = RandomizedSearchCV(
                pipeline, 
                param_distributions=param_distributions, 
                n_iter=5, # Reduced for speed, typically 20-50
                cv=3, 
                scoring='roc_auc', 
                random_state=42,
                n_jobs=-1
            )
            search.fit(X_train, y_train)
            pipeline = search.best_estimator_
            print(f"Best parameters found: {search.best_params_}")
        else:
            pipeline.fit(X_train, y_train)
            
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        
        print(classification_report(y_test, y_pred))
        auc = roc_auc_score(y_test, y_pred_proba)
        print(f"ROC-AUC: {auc:.4f}\n")
        
        if auc > best_auc:
            best_auc = auc
            best_model_name = name
            best_pipeline = pipeline

    print(f"Best Model: {best_model_name} with ROC-AUC: {best_auc:.4f}")
    
    # -----------------------------------------
    # SHAP EXPLAINABILITY
    # -----------------------------------------
    print("Generating SHAP Explainability Plots...")
    # Extract the trained XGBoost model and preprocessor
    best_classifier = best_pipeline.named_steps['classifier']
    fitted_preprocessor = best_pipeline.named_steps['preprocessor']
    
    # Transform a sample of background data for SHAP
    X_train_sample = X_train.sample(500, random_state=42)
    X_train_transformed = fitted_preprocessor.transform(X_train_sample)
    
    # Get feature names after one-hot encoding
    cat_encoder = fitted_preprocessor.named_transformers_['cat'].named_steps['onehot']
    numeric_features = ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'net_capital', 'is_full_time']
    categorical_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender']
    cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
    all_feature_names = numeric_features + list(cat_feature_names)
    
    # Generate SHAP values
    explainer = shap.TreeExplainer(best_classifier)
    shap_values = explainer.shap_values(X_train_transformed)
    
    # Plot SHAP Summary
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_train_transformed, feature_names=all_feature_names, show=False)
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig('eda_plots/shap_summary.png')
    plt.close()
    print("SHAP plots saved to eda_plots/shap_summary.png")

    print(f"Saving the best model to 'model.joblib'...")
    joblib.dump(best_pipeline, 'model.joblib')
    print("Training complete! Model is ready for deployment.")

if __name__ == "__main__":
    train_and_evaluate()
