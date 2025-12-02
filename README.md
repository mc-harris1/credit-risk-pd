# Credit Risk Scoring — Probability of Default (PD) Prediction

This project builds an **end-to-end credit risk scoring system** that predicts the **probability that a loan will default**, using real-world consumer lending datasets.

It includes a full ML pipeline (data → features → modeling → evaluation → serving), a **FastAPI scoring service**, and an optional **Streamlit dashboard** for model exploration and applicant-level explanations.

The goal is to demonstrate production-quality ML engineering, risk modeling, explainability, and deployment patterns commonly used in **FinTech** environments.

---

## 🧰 Tooling: `uv`, `ruff`, & `pre-commit`

This project uses:

- **`uv`** for Python dependency management, virtual environments, and running commands.
- **`ruff`** for linting and formatting.
- **`pre-commit`** to enforce code quality (including `ruff`) on every commit.
- **GitHub Actions** for CI (runs `uv sync`, `pre-commit`, `ruff`, `pytest` with coverage, and builds a Docker image).

All commands assume:

- You have `uv` installed.
- You use `uv run ...` or `uvx ...` instead of raw `python` / `pip`.

---

## 📌 Features

- **Probability of Default (PD)** prediction for consumer loans  
- Time-based train/validation/test split  
- LightGBM / XGBoost and logistic regression baseline  
- SHAP-based explainability (global + local)  
- FastAPI real-time scoring service  
- Dockerized deployment with a consistent image name/tag strategy  
- Linting + formatting via `ruff` (locally and in CI)  
- Tests via `pytest` with coverage reports (XML + HTML)  
- Optional Streamlit UI for interactive demo  

---

## 📁 Repository Structure

```text
credit-risk-pd/
├── README.md
├── pyproject.toml
├── Makefile
├── .pre-commit-config.yaml
├── docker/
│   └── Dockerfile
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── models/
│   ├── artifacts/
│   └── metadata/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_prototyping.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load_data.py
│   │   └── preprocess.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── predict.py
│   │   └── metrics.py
│   ├── serving/
│   │   ├── __init__.py
│   │   └── api.py
│   └── utils/
│       ├── __init__.py
│       └── io.py
├── streamlit_app/
│   └── app.py
└── tests/
    ├── __init__.py
    ├── test_api.py
    ├── test_data.py
    ├── test_features.py
    └── test_models.py
```

---

## 📊 Data

### Primary Dataset

This project uses **LendingClub-style consumer loan data**, featuring:

- Loan amount, term, interest rate  
- Income, employment length, home ownership  
- Debt-to-income (DTI), revolving utilization  
- Grade & subgrade  
- Loan status (fully paid, default, charged-off, etc.)

### Target Definition

```text
default = 1 → charged-off, default, or severe delinquency  
default = 0 → fully paid
```

Ambiguous statuses (“Current”, “In Grace Period”, etc.) are excluded.

---

## 🛠️ ML Pipeline Overview

### 1. Preprocessing

- Clean categories, handle missing data  
- Filter relevant loan statuses  
- Remove leakage fields  

### 2. Feature Engineering

- Loan-to-income ratio  
- DTI normalization  
- Interest rate buckets  
- Grade & subgrade encoding  
- One-hot or target encoding  

### 3. Modeling

Baseline:
- Logistic Regression

Advanced:
- LightGBM / XGBoost  
- Optional probability calibration  

Metrics:
- ROC-AUC  
- PR-AUC  
- Brier Score  
- Calibration curves  
- Confusion matrices  

### 4. Explainability (SHAP)

- Global feature importance  
- Local explanations per applicant  

---

## 🚀 Model Serving (FastAPI)

### Endpoints

**GET /health**  
Health check.

**POST /score**  
Returns PD, risk band, and (optionally) SHAP-based explanations.

### Run API (local)

```bash
uv run uvicorn src.serving.api:app --reload --host 0.0.0.0 --port 8000
```

Swagger UI:  
`http://localhost:8000/docs`

---

## 🧪 Tests, Linting, & Pre-commit

### Install dependencies (including dev)

```bash
uv sync --all-groups
```

### Run tests

```bash
uv run pytest -q
```

### Run tests with coverage

```bash
uv run pytest --cov=src --cov-report=xml --cov-report=html
```

This will generate:

- `coverage.xml` – for tools like Codecov / Sonar / CI
- `htmlcov/` – browsable HTML coverage report

### Lint & format with `ruff`

```bash
uvx ruff check src tests
uvx ruff format src tests
```

### Pre-commit hooks

Install pre-commit locally (once):

```bash
uv run pre-commit install
```

Run against all files:

```bash
uv run pre-commit run --all-files
```

The `.pre-commit-config.yaml` is configured to:

- Run `ruff` (`check` + `format`)
- Enforce basic hygiene (trailing whitespace, EOF newline, large-file checks, etc.)

---

## 🐳 Docker Image Strategy

The **Dockerfile** lives at `docker/Dockerfile` and is built around:

- Python 3.11 slim base image  
- `uv` installed in the container  
- `uv sync --no-dev` for production-only dependencies  
- Default command: run the FastAPI app via `uvicorn`

Example local build & run:

```bash
# Local image name (matches Makefile default)
docker build -t credit-risk-api -f docker/Dockerfile .
docker run -p 8000:8000 credit-risk-api
```

### CI Image Tagging

The GitHub Actions workflow uses a consistent image name/tag strategy:

- Base name:  
  `ghcr.io/<owner>/credit-risk-pd`
- Tags:
  - `ghcr.io/<owner>/credit-risk-pd:<GITHUB_SHA>`
  - `ghcr.io/<owner>/credit-risk-pd:latest`

The CI job builds the image with both tags so you get:

- A unique image per commit (SHA tag)
- A rolling `latest` tag for quick testing

> Note: pushing to GHCR requires configuring `docker login` with a GitHub token. The provided workflow only **builds** the image; you can add a push step if desired.

---

## 🤖 Continuous Integration (GitHub Actions)

The CI workflow (`.github/workflows/ci.yml`) runs on pushes and PRs and:

1. Checks out the repo  
2. Sets up Python 3.11  
3. Installs `uv`  
4. Runs `uv sync --all-groups` (including dev tools)  
5. Runs `pre-commit` on all files (includes `ruff`)  
6. Runs `ruff` explicitly on `src` and `tests`  
7. Runs `pytest` with coverage reports  
8. Builds a Docker image with consistent tags  
9. Uploads coverage artifacts (HTML + XML) for inspection

So every PR gets:

- Style & lint checks
- Automated tests
- Coverage reports (as downloadable artifacts)
- A Docker image build using the same Dockerfile you use locally

---

## 📊 Optional Streamlit Demo

```bash
uv run streamlit run streamlit_app/app.py
```

This launches a simple UI that:

- Collects loan application parameters
- Calls the FastAPI `/score` endpoint
- Displays PD and risk band (and, later, explanations)

---

## 🧭 Full Pipeline (with `uv`)

Install dependencies:

```bash
uv sync --all-groups
```

Preprocess:

```bash
uv run python -m src.data.preprocess
```

Feature build:

```bash
uv run python -m src.features.build_features
```

Train:

```bash
uv run python -m src.models.train
```

Serve:

```bash
uv run uvicorn src.serving.api:app --reload --host 0.0.0.0 --port 8000
```

---

## 📦 Future Work

- Add LGD/EAD for full expected loss modeling  
- Survival analysis for time-to-default  
- Reject inference to correct approval bias  
- Data & score drift monitoring (PSI, feature drift)  
- Push Docker images to GHCR or another registry from CI  
- Deploy to AWS (Lambda/ECS), GCP (Cloud Run), or Azure  

---

<<<<<<< HEAD
## 📄 License

MIT or your preferred license.
=======
## **📄 License**
No license.
>>>>>>> b97d1e2a5d6b8e04e6673e9b30f39182f1034b6a
