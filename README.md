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
A corrected GitHub Actions run completed on 28 October 2026: XGBoost selected by training-only CV (ROC-AUC 0.9228), with held-out ROC-AUC 0.9207 and accuracy 0.8529 on 9,758 test rows. These are results for this census-data split. See docs/evaluation-2026-10-28.json and https://github.com/jagannadharao8/Employee-Salary-Prediction/actions/runs/40697955712 . Reloading the generated model reproduced held-out ROC-AUC; SHAP generation succeeded. The matching model.joblib, heldout.csv, evaluation.json and SHAP plot are committed in the default branch; model.sha256 identifies the verified artifact. Downloadable Actions artifacts have seven-day retention, but the committed files remain available. Previous Random Forest and XGBoost descriptions referred to different project versions and are not comparable benchmark evidence.

The dashboard's agreement on displayed rows includes training records. It is explicitly not held-out test accuracy. Use evaluation.json for the saved run's test results. Model scores are not guaranteed calibrated probabilities.

## Limitations
SMOTE after one-hot encoding can interpolate categorical indicator values; compare class-weighted baselines or an appropriate mixed-type sampler in future evaluation. Census attributes can encode social bias; SHAP describes model associations, not causal explanations. This demonstration must not be used to make hiring or salary decisions.

## Repository organisation
Application entry points are app.py and train_model.py. Historical root-level coursework PDFs, slides and certificates are preserved under docs/learning-archive; code screenshots are under docs/legacy-screenshots. Application code and dataset paths remain at the root. The dataset is the repository's supplied census-income CSV; source and licensing provenance have not been independently established. The environment is pinned in requirements.txt. Historical notebooks/slides are learning records and do not supersede evaluation.json.
