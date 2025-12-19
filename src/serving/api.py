from __future__ import annotations

from typing import Any, Optional

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


def _risk_band_from_pd(pd_val: float) -> str:
    """
    Simple banding. Tune thresholds to match your business policy.
    """
    if pd_val < 0.10:
        return "Low"
    if pd_val < 0.25:
        return "Medium"
    return "High"


@app.get("/health", response_model=HealthResponse)
def health(registry: ModelRegistry = Depends(get_registry)) -> HealthResponse:
    try:
        loaded = registry.is_loaded()
        if not loaded:
            registry.load()
            loaded = True
    except Exception:
        loaded = False

    return HealthResponse(model_loaded=loaded, model_version=registry.version())


@app.post("/predict", response_model=PredictionResponse)
def predict(
    payload: dict,
    registry: ModelRegistry = Depends(get_registry),
) -> PredictionResponse:
    application = payload.get("application", payload)

    try:
        from .schemas import LoanApplication

        app_model = LoanApplication.model_validate(application)

        pd_val, df = registry.predict_from_payload(app_model.model_dump())
        pd_val_f = float(pd_val)
        risk_band = _risk_band_from_pd(pd_val_f)

        top_factors: list[FeatureAttribution] = []
        shap_error: Optional[str] = None

        # ✅ Attempt SHAP and keep the error if it fails
        try:
            base_value, shap_vec = compute_shap_for_single_row(registry._model, df)  # noqa: SLF001
            attr_dicts = format_top_attributions(df, shap_vec, top_k=10)
            top_factors = [FeatureAttribution(**a) for a in attr_dicts]
        except Exception as e:
            shap_error = str(e)
            top_factors = []

        return PredictionResponse(
            pd=pd_val_f,
            model_version=registry.version(),
            risk_band=risk_band,
            top_factors=top_factors,
            shap_error=shap_error,
        )

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

    meta: dict[str, Any] = {
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
        meta["explain_error"] = str(e)
        base_value = None
        attributions = []

    return ExplainResponse(
        pd=float(pd_val),
        model_version=registry.version(),
        base_value=base_value,
        attributions=attributions,
        meta=meta,
    )
