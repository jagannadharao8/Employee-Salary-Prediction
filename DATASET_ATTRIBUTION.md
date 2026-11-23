# Dataset attribution and verification

The supplied employeedata.csv contains the UCI Adult (Census Income) record set. UCI describes it as an extraction by Barry Becker from the 1994 US Census database, used to classify annual income above or at most $50,000.

## Citation and licence

Becker, B. & Kohavi, R. (1996). Adult [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20.

- Official source: https://archive.ics.uci.edu/dataset/2/adult
- Official archive checked: https://archive.ics.uci.edu/static/public/2/adult.zip
- Licence stated by UCI: Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/
- Verification date: 23 November 2026.

## How the record set was verified

The checker downloaded adult.data and adult.test from the official archive and compared every supplied CSV record with their combined contents. It trimmed surrounding field whitespace and the terminal period on UCI test income labels, mapped education-num to educational-num and sex to gender, and ignored row ordering while preserving duplicate multiplicity.

All 48,842 row instances matched. Both representations contain 48,790 unique rows, and their normalized sorted-record SHA-256 is:

`6a07b01226845a861c786d71e2733b3bd3eef20789276e32bd5e347d84aea053`

[Recorded hashes, mapping and result](docs/data-provenance-verification.json) · [Successful verification run](https://github.com/jagannadharao8/Employee-Salary-Prediction/actions/runs/45197016596).

This identifies the data corpus; the original intermediate download history is unknown. The renamed gender column preserves UCI's sex variable values.

## Project processing and evaluation

The project pools the original UCI training/test records, removes duplicates, engineers net-capital and full-time indicators, selects its documented features, and creates its own stratified 80/20 split. Its reported ROC-AUC and accuracy therefore describe that project split. The CSV and fitted model were unchanged by this attribution update.

These are historical census records, with missing values and demographic attributes. The task is an educational classification example; the $50K category is defined by this historical dataset.

## Recheck

```sh
python scripts/verify_dataset_provenance.py
```

The standard-library checker downloads the official archive and writes docs/data-provenance-verification.json. The verification workflow can also be run from GitHub Actions. It emits aggregate comparisons and hashes.
