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
import json
from datetime import datetime, timezone

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
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
    data = data.replace('?', np.nan).drop_duplicates()
    
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
    y = data['income'].apply(lambda x: 1 if str(x).strip().rstrip('.') == '>50K' else 0)
    
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
            ('preprocessor', build_pipeline()),
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
            cv_auc = float(cross_val_score(pipeline, X_train, y_train, cv=3, scoring="roc_auc", n_jobs=-1).mean())
            
        auc = float(search.best_score_) if name == "XGBoost" else cv_auc
        print(f"Training-only cross-validation ROC-AUC: {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_model_name = name
            best_pipeline = pipeline

    print(f"Selected by training CV: {best_model_name}, CV ROC-AUC: {best_auc:.4f}")
    best_pipeline.fit(X_train, y_train)
    y_pred = best_pipeline.predict(X_test)
    test_auc = float(roc_auc_score(y_test, best_pipeline.predict_proba(X_test)[:, 1]))
    report = classification_report(y_test, y_pred, output_dict=True)
    print(f"Untouched test ROC-AUC: {test_auc:.4f}")
    joblib.dump(best_pipeline, 'model.joblib')
    metadata = {"model": best_model_name, "cv_roc_auc": best_auc, "test_roc_auc": test_auc,
                "test_report": report, "train_rows": len(X_train), "test_rows": len(X_test),
                "random_state": 42, "created_at": datetime.now(timezone.utc).isoformat(),
                "protocol": "duplicate removal, stratified 80/20 split; training-only 3-fold CV selection"}
    with open("evaluation.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    X_test.assign(income=y_test).to_csv("heldout.csv", index=False)
    
    # -----------------------------------------
    # SHAP EXPLAINABILITY
    # -----------------------------------------
    try:
        print("Generating SHAP Explainability Plots...")
        # Extract the trained XGBoost model and preprocessor
        best_classifier = best_pipeline.named_steps['classifier']
        fitted_preprocessor = best_pipeline.named_steps['preprocessor']
        
        # Transform a sample of background data for SHAP
        X_train_sample = X_train.sample(min(500, len(X_train)), random_state=42)
        X_train_transformed = fitted_preprocessor.transform(X_train_sample)
        
        # Get feature names after one-hot encoding
        cat_encoder = fitted_preprocessor.named_transformers_['cat'].named_steps['onehot']
        numeric_features = ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'net_capital', 'is_full_time']
        categorical_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender']
        cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
        all_feature_names = numeric_features + list(cat_feature_names)
        
        # Generate SHAP values
        if isinstance(best_classifier, LogisticRegression):
            explainer = shap.LinearExplainer(best_classifier, X_train_transformed)
        else:
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


    except Exception as exc:
        print(f"Optional SHAP plot unavailable: {type(exc).__name__}. Saved model and evaluation remain available.")

    print("Model and held-out evaluation saved. Inspect evaluation.json before quoting results.")

if __name__ == "__main__":
    train_and_evaluate()
