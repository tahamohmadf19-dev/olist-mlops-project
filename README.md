# Experimental Historical-Snapshot Late-Delivery Inference Demo

An educational Olist MLOps project for predicting whether an e-commerce order may arrive late from a historical order snapshot. It packages the fitted Task 2 scikit-learn classifier into an inference-only Python service, with tests, Docker, monitoring, and local model/data management.

**Status:** Local experimental demo. **Not production-ready.** The GitHub repository contains source code and DVC pointer files, **not** the trained model, data splits, or the local MLflow database. A fresh clone can run the model-independent tests and build the Docker image, but **cannot serve predictions until compatible model artifacts are supplied**.

## Historical-snapshot inference scope

Some engineered features depend on historical order-item/payment aggregates, including total item price, item count, average seller coordinates, total payment value and their derived log/distance features. The current bundle scores those historical snapshots; the availability of every input at the intended prediction time has **not been verified**.

Aggregates can incorporate records or updates that arrive after purchase. Using their final historical values as if they were available when the purchase occurred creates a **potential point-in-time leakage risk**. Blocking explicit future/target fields and matching Notebook 5 features do not establish prediction-time availability. This model is **not verified for real-time purchase-time inference**. That use requires a separate availability audit and evaluation using snapshots reconstructed strictly from information available at the scoring timestamp. The current model, feature transformations and training pipeline are unchanged.

## Architecture

```text
Input order (JSON)
        |
        v
FastAPI: /predict, /predict/batch
        |
        v
Schema + semantic input checks
        |
        v
Notebook-5-equivalent feature engineering
        |
        v
Previously fitted ColumnTransformer (no fit at inference)
        |
        v
Previously fitted LogisticRegression + saved decision threshold
        |
        v
Prediction, late probability, model SHA-256 fingerprint
        |
        +--> Structured JSON summaries to Docker logs (local rotation)
        +--> Prometheus metrics /metrics --> Prometheus TSDB --> local alert rules
                                                |
                                                +--> offline score-distribution drift check
```

Training, tuning, and evaluation remain in the six original notebooks. **Inference never retrains** the model.

## Technology

- Python 3.12, NumPy, pandas, SciPy, scikit-learn, joblib
- FastAPI, Uvicorn, Pydantic, pytest, Ruff, pre-commit
- DVC pointers and local cache for datasets and fitted artifacts
- Great Expectations for **offline validation** of the validation split
- MLflow local SQLite tracking and a registered PythonModel bundle (v2)
- Docker Compose, Prometheus metrics and local alert rules

## Project map

| Location | Purpose |
| --- | --- |
| `01_*.ipynb` through `06_*.ipynb` | Data preparation, feature engineering, model training/evaluation |
| `src/olist_inference/` | Config, feature engineering, validation, model loading and inference, drift calculations |
| `app/` | FastAPI adapter, monitoring, privacy-conscious prediction summaries |
| `config/inference.json` | Paths to fitted artifacts, maximum batch size |
| `scripts/` | Feature parity, Docker API parity, data validation, baseline and drift scripts |
| `tests/` | Synthetic unit and API integration tests |
| `requirements/` | Runtime, API, development and data-quality dependencies |
| `artifacts/**/*.dvc` | Pointers for tracked datasets, feature matrices and fitted artifacts |
| `compose.yaml` | Inference service with read-only artifact mounts, local port 8001 |
| `compose.monitoring.yaml` | Prometheus TSDB, local port 9090 |
| `compose.logging.yaml` | Docker API log rotation |
| `monitoring/` | Prometheus scrape config and two alert rules with promtool tests |
| `.github/workflows/ci.yml` | Lint, formatting, pytest and Docker image build |

## Setup: source-only clone

This section works **without the model files** for code quality and model-independent unit tests. Python 3.12 and Git are required.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements/dev.txt -r requirements/api.txt
.\.venv\Scripts\python.exe -m pip install ruff==0.17.3 pre-commit==4.7.0
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m ruff check --select E4,E7,E9,F app src scripts tests
.\.venv\Scripts\python.exe -m ruff format --check app src scripts tests
.\.venv\Scripts\python.exe -m pytest -q tests
```

Set up optional Git hooks. Ruff runs in an isolated pre-commit environment; pytest runs separately and in CI:

```powershell
.\.venv\Scripts\python.exe -m pre_commit install
.\.venv\Scripts\python.exe -m pre_commit run --all-files
```

The repository CI uses Python 3.12 on Ubuntu. The Docker job builds an image but does **not** run it, publish it, or deploy it. It needs no model files for the build.

## Run inference locally (artifacts required)

The following trusted Task 2 artifacts must exist at the exact paths configured in `config/inference.json`:

```text
artifacts/05_feature_engineering/05_preprocessor.joblib
artifacts/06_model/06_best_model.joblib
artifacts/06_model/06_results_summary.json
```

> A Git clone alone does not fetch these files. DVC tracks 10 files locally, but no shared DVC Remote is configured in this demo. DVC pointer files contain metadata, not the artifact contents; a fresh clone without the original local cache cannot restore them using `dvc checkout`, and `dvc pull` cannot fetch them without a configured, populated and accessible remote. Obtain the exact matching artifacts from a trusted copy, or separately configure and populate a DVC remote before restoring them. Reproducing artifacts with the Task 2 notebooks and verified dataset is a separate training task. Do not run `joblib.load` on untrusted files.

For the existing Windows development environment with those files already in place:

```powershell
# API plus Prometheus plus Docker log rotation
docker compose -f compose.yaml -f compose.monitoring.yaml -f compose.logging.yaml up -d --build --wait

# Compare actual Docker predictions against the trusted local inference bundle
.\.venv\Scripts\python.exe -m scripts.check_docker_api

# Compare Python feature engineering to the saved Notebook 5 validation matrix
# Requires artifacts/03_validation.csv and 05_validation_features.npz
.\.venv\Scripts\python.exe -m scripts.check_feature_parity
```

Endpoints (local only):

| URL | Description |
| --- | --- |
| `http://127.0.0.1:8001/docs` | Interactive OpenAPI docs |
| `http://127.0.0.1:8001/health` | API health |
| `http://127.0.0.1:8001/model` | Model fingerprint, threshold and feature count |
| `POST http://127.0.0.1:8001/predict` | One order: `{"order": {...}}` |
| `POST http://127.0.0.1:8001/predict/batch` | Many orders: `{"orders": [{...}]}` |
| `http://127.0.0.1:8001/metrics` | Prometheus exposition |
| `http://127.0.0.1:9090/targets` | Prometheus scrape health |
| `http://127.0.0.1:9090/alerts` | Prometheus alert rules |

The placeholder `{...}` is **not** a complete valid order. The model needs its original schema and dates; see `src/olist_inference/service.py`. `/health` passing alone does not prove a real prediction. The response contains `prediction`, `late_probability`, and `model_version`.

Stop without deleting the data volume:

```powershell
docker compose -f compose.yaml -f compose.monitoring.yaml -f compose.logging.yaml down
```

**Do not use `down -v`** if you want to retain Prometheus's local history.

## Data quality and tracking

Offline Great Expectations checks (requires validation data and its separate dependencies):

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements/data_quality.txt
.\.venv\Scripts\python.exe -m scripts.validate_dataset_gx
```

DVC pointer files reference the original splits, feature matrices, preprocessor, model and metadata. `dvc status` succeeds in the original local project. No DVC Remote is configured, so `dvc pull` after a fresh clone **will not work** until a remote is added and populated.

MLflow experiment tracking and Registry v2 are maintained in an **unpublished local SQLite store**. The deployed FastAPI service currently loads the saved local artifacts specified by the config, **not** the MLflow Registry. Registry registration scripts are one-time operations and should not be rerun blindly.

## Monitoring and drift

- Request count, status, latency, batch size and aggregate prediction-score distribution are exposed as Prometheus metrics.
- Prometheus scrapes the local API every 15 seconds and retains its local TSDB for seven days (configured retention; restart persistence not independently verified).
- Offline drift checker compares the fitted model's **training-set score histogram** with recent aggregate Prometheus buckets using Population Stability Index (PSI).
- The checker requires at least 200 observed scores; current local result: **`INSUFFICIENT_DATA`**, not proof of no drift. The numeric threshold is heuristic.
- Alert rules: `OlistAPIDown` and `OlistAPIHigh5xxRate`; `promtool` rule checks and unit tests passed locally. There is **no Alertmanager notification channel**.
- JSON prediction summaries do not include customer identifiers or raw input payloads. API logs use local Docker rotation and are **not** a durable, row-level outcome store for later label evaluation.

Offline drift commands (require the fitted artifacts and running Prometheus):

```powershell
.\.venv\Scripts\python.exe -m scripts.build_drift_baseline
.\.venv\Scripts\python.exe -m scripts.check_prediction_drift
```

## Verified local results

Results from the original Windows workstation, not from a public deployment:

- Verified locally: **172 passed, 1 warning** (non-blocking Starlette/AnyIO deprecation warning).
- Feature parity: **14,470 validation rows x 821 transformed features; maximum absolute difference = 0**.
- Real Docker inference parity succeeded for **5 orders**, including validation of forbidden target fields.
- Original fitted model content fingerprint: `sha256:a2485e81e9f4`.
- Drift training reference: **67,529 rows**, 13 histogram bins.
- Docker API and Prometheus containers were healthy; Prometheus scraping, two alert rules, and structured prediction logging were verified locally.
- **GitHub Actions: verified successful.** [Run ID: `42859752354`](https://github.com/tahamohmadf19-dev/olist-mlops-project/actions/runs/42859752354) confirmed **172 passed, 1 warning**; Ruff lint and format passed; Docker image build succeeded.

## Limitations and security

1. **Experimental only**. Do not expose API or Prometheus to the public Internet without authentication, TLS, request limits, and infrastructure review.
2. **Historical-snapshot limitation and potential point-in-time leakage**: availability of order-item/payment aggregates and delivery-estimate fields at scoring time remains unverified. Feature parity does **not** prove online feature availability; the model is not verified for real-time purchase-time inference.
3. **Reproducibility gap**: model files, DVC cache, and MLflow store stay local; therefore, a clean clone cannot run inference until trusted compatible assets are supplied. This does not yet meet the original clean-machine, one-command startup requirement.
4. **DVC local only**: no external recovery copy. Avoid treating GitHub as a backup of data/model files.
5. **Monitoring gap**: predictions and eventual ground-truth delivery labels are not joined and stored in a durable evaluation database; local aggregate logs cannot support that requirement alone.
6. **CI delivery gap**: the workflow builds, but does not push, the Docker image; no GitHub run has been verified.
7. **Privacy**: `database_check.ipynb` contains sensitive connection details and must remain outside Git. Six public-facing notebooks had saved outputs removed, but all code and text must still be manually reviewed before publishing.

## Learning references

- [Task 3: From Notebooks to Production](https://madewithml.com/)
- [DVC: Remote Storage](https://doc.dvc.org/user-guide/data-management/remote-storage)
- [FastAPI documentation](https://fastapi.tiangolo.com/)
- [MLflow documentation](https://mlflow.org/docs/latest/)
- [Prometheus documentation](https://prometheus.io/docs/)
- [GitHub Actions: Python](https://docs.github.com/en/actions/tutorials/build-and-test-code/python)

**Project type:** Experimental Historical-Snapshot Late-Delivery Inference Demo; an educational local project, not a deployed product.
