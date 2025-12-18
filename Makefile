# ---- Project configuration ----
APP_MODULE = src.serving.api:app
UVICORN_HOST = 0.0.0.0
UVICORN_PORT = 8000

# ---- General ----

.PHONY: help
help:
	@echo "Common commands:"
	@echo	"  make sync        - Install dependencies with uv"
	@echo	"  make lint        - Run ruff lint"
	@echo	"  make fmt         - Run ruff formatter"
	@echo	"  make test        - Run pytest"
	@echo	"  make pre-commit  - Run all pre-commit checks"
	@echo	"  make api         - Run FastAPI app with uvicorn"
	@echo	"  make streamlit   - Run Streamlit demo app"
	@echo   "  make data-kaggle - Download dataset from Kaggle"
	@echo	"  make preprocess  - Run data preprocessing"
	@echo	"  make features    - Run feature engineering"
	@echo 	"  make tune	    - Run hyperparameter tuning"
	@echo	"  make train       - Run model training"
	@echo	"  make evaluate    - Run model evaluation"
	@echo	"  make monitor     - Run model monitoring"
	@echo	"  make pipeline    - Run full ML pipeline (data-kaggle, preprocess, features, tune, train, evaluate, monitor)"
	@echo	"  make explain     - Run model explanation"
	@echo	"  make docker-build- Build Docker image"
	@echo	"  make docker-run  - Run Docker container"

# ---- Env / deps ----

.PHONY: sync
sync:
	uv sync

# ---- Code quality ----

.PHONY: lint
lint:
	uvx ruff check src tests

.PHONY: fmt
fmt:
	uvx ruff format src tests

.PHONY: test
test:
	uv run pytest -q

.PHONY: pre-commit
pre-commit:
	uvx pre-commit run --all-files

# ---- Pipeline steps ----

.PHONY: data-kaggle
data-kaggle:
	uv run python -m src.data.download_kaggle

.PHONY: preprocess
preprocess:
	uv run python -m src.data.preprocess

.PHONY: features
features:
	uv run python -m src.features.build_features

.PHONY: tune
tune:
	uv run python -m src.models.tune

.PHONY: train
train:
	uv run python -m src.models.train

.PHONY: evaluate
evaluate:
	uv run python -m src.models.evaluate

.PHONY: monitor
monitor:
	uv run python -m src.models.monitor

.PHONY: pipeline
pipeline: data-kaggle preprocess features tune train evaluate monitor

.PHONY: explain
explain:
	uv run python -m src.models.explain

# ---- App servers ----

.PHONY: api
api:
	uv run uvicorn $(APP_MODULE) --reload --host $(UVICORN_HOST) --port $(UVICORN_PORT)

.PHONY: streamlit
streamlit:
	uv run streamlit run streamlit_app/app.py

# ---- Docker ----

IMAGE_NAME = credit-risk-api

.PHONY: docker-build
docker-build:
	docker build -t $(IMAGE_NAME) -f docker/Dockerfile .

.PHONY: docker-run
docker-run:
	docker run -p 8000:8000 $(IMAGE_NAME)
