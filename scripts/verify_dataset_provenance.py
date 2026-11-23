"""Verify the supplied CSV against UCI Adult without changing training data or row order."""
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

UCI_URL = "https://archive.ics.uci.edu/static/public/2/adult.zip"
COLUMNS = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
           "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
           "hours-per-week", "native-country", "income"]
ALIASES = {"education-num": "educational-num", "sex": "gender"}

def normalise(row):
    result = [value.strip() for value in row]
    result[-1] = result[-1].rstrip(".")
    return tuple(result)

def fingerprint(rows):
    encoded = json.dumps(sorted(rows), separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()

def verify():
    raw = Path("employeedata.csv").read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    mapping = {column: column if column in reader.fieldnames else ALIASES.get(column, column) for column in COLUMNS}
    missing = [name for name in mapping.values() if name not in reader.fieldnames]
    if missing:
        raise ValueError("Required census columns are missing: " + ", ".join(missing))
    supplied = [normalise([record[mapping[column]] for column in COLUMNS]) for record in reader]
    request = urllib.request.Request(UCI_URL, headers={"User-Agent": "Portfolio-dataset-provenance/1.0"})
    with urllib.request.urlopen(request, timeout=45) as response:
        upstream_raw = response.read(4 * 1024 * 1024)
    upstream = []
    with zipfile.ZipFile(io.BytesIO(upstream_raw)) as archive:
        for name in ("adult.data", "adult.test"):
            match = next(path for path in archive.namelist() if Path(path).name == name)
            content = archive.read(match).decode("utf-8")
            for record in csv.reader(io.StringIO(content), skipinitialspace=True):
                if not record or record[0].lstrip().startswith("|"):
                    continue
                if len(record) != len(COLUMNS):
                    raise ValueError("Unexpected UCI field count")
                upstream.append(normalise(record))
    supplied_counts, upstream_counts = Counter(supplied), Counter(upstream)
    result = {
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "upstream_url": UCI_URL,
        "upstream_dataset": "UCI Adult",
        "doi": "10.24432/C5XW20",
        "repository_csv_sha256": hashlib.sha256(raw).hexdigest(),
        "upstream_zip_sha256": hashlib.sha256(upstream_raw).hexdigest(),
        "repository_rows": len(supplied),
        "upstream_rows": len(upstream),
        "repository_unique_rows": len(supplied_counts),
        "upstream_unique_rows": len(upstream_counts),
        "column_mapping": mapping,
        "normalisation": "strip surrounding field whitespace; strip trailing period from income label; compare row multisets ignoring row order",
        "repository_normalised_sha256": fingerprint(supplied),
        "upstream_normalised_sha256": fingerprint(upstream),
        "exact_normalised_multiset_match": supplied_counts == upstream_counts,
        "shared_row_instances": sum((supplied_counts & upstream_counts).values())
    }
    Path("docs").mkdir(exist_ok=True)
    Path("docs/data-provenance-verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["exact_normalised_multiset_match"]:
        raise SystemExit("Supplied CSV does not exactly match the normalised UCI Adult records.")
    return result

if __name__ == "__main__":
    verify()
