from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse

from .registry import ModelRegistry, ModelRegistryError, get_registry
from .schemas import (
    ExplainRequest,
    ExplainResponse,
    FeatureAttribution,
    HealthResponse,
    PredictionResponse,
)
from .shap_utils import ShapError, compute_shap_for_single_row, format_top_attributions

app = FastAPI(title="Credit Risk PD API", version="1.0.0")


@app.exception_handler(ModelRegistryError)
def _registry_error_handler(_, exc: ModelRegistryError):
    return JSONResponse(
        status_code=500,
        content={"error": "model_registry_error", "detail": str(exc)},
    )


@app.exception_handler(ShapError)
def _shap_error_handler(_, exc: ShapError):
    return JSONResponse(
        status_code=500,
        content={"error": "shap_error", "detail": str(exc)},
    )


@app.get("/health", response_model=HealthResponse)
def health(registry: ModelRegistry = Depends(get_registry)) -> HealthResponse:
    try:
        loaded = registry.is_loaded()
        if not loaded:
            # Try lazy load to give a meaningful health signal
            registry.load()
            loaded = True
    except Exception:
        loaded = False

    return HealthResponse(model_loaded=loaded, model_version=registry.version())


@app.post("/predict", response_model=PredictionResponse)
def predict(
    payload: dict,  # keep flexible; validate in schema layer below
    registry: ModelRegistry = Depends(get_registry),
) -> PredictionResponse:
    """
    Backwards/forwards compatible:
    - Accept either {"loan_amnt": ...} or {"application": {...}}
    """
    application = payload.get("application", payload)

    try:
        # Pydantic schema validation
        from .schemas import LoanApplication

        app_model = LoanApplication.model_validate(application)
        pd_val, _df = registry.predict_from_payload(app_model.model_dump())
        return PredictionResponse(pd=pd_val, model_version=registry.version())
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


@app.post("/explain", response_model=ExplainResponse)
def explain(
    req: ExplainRequest,
    registry: ModelRegistry = Depends(get_registry),
) -> ExplainResponse:
    pd_val, df = registry.predict_from_payload(req.application.model_dump())

    meta = {
        "top_k": req.top_k,
        "note": "Attributions sorted by |shap_value| desc (class=1 probability).",
    }

    base_value = None
    attributions: list[FeatureAttribution] = []

    try:
        base_value_raw, shap_vec = compute_shap_for_single_row(registry._model, df)  # noqa: SLF001
        attr_dicts = format_top_attributions(df, shap_vec, top_k=req.top_k)
        attributions = [FeatureAttribution(**a) for a in attr_dicts]
        base_value = base_value_raw if req.include_base_value else None
    except Exception as e:
        # best-effort response
        meta["explain_error"] = str(e)
        base_value = None
        attributions = []

    return ExplainResponse(
        pd=pd_val,
        model_version=registry.version(),
        base_value=base_value,
        attributions=attributions,
        meta=meta,
    )
