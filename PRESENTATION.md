# Income Classification Explorer — interview presentation outline

This outline describes the current implementation. Historical slides under docs/learning-archive contain earlier experiments and metrics; they are not the current benchmark.

## 1. Problem and scope
Classify a census record into income above or at most $50K. This is an educational classification demonstration, not a prediction of a person's numerical salary and not a tool for recruitment or compensation decisions.

## 2. Data and split
The supplied repository CSV is deduplicated before a stratified 80/20 split. The verified run used 39,032 training rows and 9,758 held-out test rows. Dataset source and licensing provenance still need independent documentation.

## 3. Training pipeline
Preprocessing and SMOTE are inside the cross-validation pipeline. Compare Logistic Regression and XGBoost with training-only three-fold CV; choose the model without consulting held-out labels.

## 4. Selection and results
XGBoost was selected with CV ROC-AUC 0.9228. The single held-out evaluation produced ROC-AUC 0.9207 and accuracy 85.29%. For the higher-income class, precision was 0.6680, recall 0.7667 and F1 0.7140 on 2,336 rows. These figures apply only to this dataset split.

## 5. Explainability
Choose LinearExplainer or TreeExplainer according to the fitted model. The verified XGBoost run generated a SHAP summary. Explanations describe fitted associations, not causality, fairness or guaranteed correctness.

## 6. Application and artifacts
The Streamlit application loads the saved pipeline and evaluation metadata. The default branch contains model.joblib, heldout.csv, evaluation.json, model.sha256 and generated plots. Displayed-row agreement includes training rows and is labelled separately from held-out accuracy.

## 7. Reproducibility evidence
[Training and model reload verification](https://github.com/jagannadharao8/Employee-Salary-Prediction/actions/runs/40697955712).
The saved pipeline reproduced the held-out ROC-AUC. Use requirements.txt and train_model.py to reproduce the protocol; inspect evaluation.json for exact results and artifact checksum.

## 8. Limitations and next experiment
SMOTE can interpolate one-hot categorical indicators. Compare class weighting and mixed-type sampling on training folds. Investigate provenance, class-wise performance and fairness before considering any real-world use. Model outputs are not calibrated probabilities.

Updated 28 October 2026. This is a speaking outline; it does not claim that a new PowerPoint deck was generated.
