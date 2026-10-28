# Income Classification Explorer
Educational portfolio project by Jalla Jagannadharao.

Predicts census income category (>50K or <=50K), not a numerical salary. The current training script compares Logistic Regression and XGBoost using training-only cross-validation, with preprocessing and SMOTE inside each pipeline. A stratified 20% test set is evaluated only after model selection.

## Reproduce
Use the pinned requirements and Python 3.10/3.11:
```sh
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```
Generated outputs: model.joblib, evaluation.json, heldout.csv, and optional SHAP plots in eda_plots. A SHAP failure does not prevent saving the model or evaluation. Linear/tree explanations are selected according to the winning model.

## Results status
A corrected GitHub Actions run completed on 28 October 2026: XGBoost selected by training-only CV (ROC-AUC 0.9228), with held-out ROC-AUC 0.9207 and accuracy 0.8529 on 9,758 test rows. These are results for this census-data split. See docs/evaluation-2026-10-28.json and https://github.com/jagannadharao8/Employee-Salary-Prediction/actions/runs/40697955712 . Reloading the generated model reproduced held-out ROC-AUC; SHAP generation succeeded. Run artifacts have seven-day retention. The existing committed model.joblib may be from an older training protocol and must be regenerated before deployment. Previous Random Forest and XGBoost descriptions referred to different project versions and are not comparable benchmark evidence.

The dashboard's agreement on displayed rows includes training records. It is explicitly not held-out test accuracy. Use evaluation.json for the saved run's test results. Model scores are not guaranteed calibrated probabilities.

## Limitations
SMOTE after one-hot encoding can interpolate categorical indicator values; compare class-weighted baselines or an appropriate mixed-type sampler in future evaluation. Census attributes can encode social bias; SHAP describes model associations, not causal explanations. This demonstration must not be used to make hiring or salary decisions.

## Repository organisation
Application entry points are app.py and train_model.py. Historical coursework, slides and certificates are learning materials rather than model evaluation evidence; they remain preserved pending organisation. Dataset provenance, licence and exact training environment should accompany published results.
