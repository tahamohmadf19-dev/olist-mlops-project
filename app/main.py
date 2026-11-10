"""Local-only FastAPI HTTP adapter. Model training is never performed here."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Protocol

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.monitoring import new_monitoring
from app.prediction_audit import emit_prediction_event
from src.olist_inference.config import InferenceConfig
from src.olist_inference.service import (
    ArtifactValidationError,
    InferenceService,
    InputValidationError,
)

logger = logging.getLogger("olist.api")
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "inference.json"


class PredictionService(Protocol):
    """The narrow contract the HTTP adapter depends on."""

    version: str
    threshold: float
    expected_features: Sequence[str]

    def predict(self, orders: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]: ...


class OrderRequest(BaseModel):
    """Fields are checked against the trained model's schema by InferenceService."""

    model_config = ConfigDict(extra="forbid")
    order: dict[str, Any]


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    orders: list[dict[str, Any]] = Field(min_length=1, max_length=1000)


class PredictionResponse(BaseModel):
    prediction: int = Field(ge=0, le=1)
    late_probability: float = Field(ge=0, le=1)
    model_version: str


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: str


class ModelInfoResponse(BaseModel):
    model_version: str
    decision_threshold: float
    input_feature_count: int


def create_app(
    *,
    service: PredictionService | None = None,
    config_path: str | Path | None = None,
) -> FastAPI:
    """Create an HTTP adapter; inject a fake service during tests.

    Production artifacts are loaded once, on startup. An invalid bundle stops
    startup rather than allowing an API to serve incorrect predictions.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if service is not None:
            app.state.inference = service
        else:
            path = (
                config_path
                or os.environ.get("OLIST_INFERENCE_CONFIG")
                or DEFAULT_CONFIG
            )
            app.state.inference = InferenceService(InferenceConfig.from_json(path))
        logger.info(
            "Inference service ready model_version=%s", app.state.inference.version
        )
        try:
            yield
        finally:
            app.state.inference = None

    metrics = new_monitoring()

    api = FastAPI(
        title="Olist Late Delivery Prediction API",
        version="0.2.0",
        description="Educational, local-only inference API using Task 2 fitted artifacts.",
        lifespan=lifespan,
    )

    @api.middleware("http")
    async def collect_metrics(request: Request, call_next: Any) -> Any:
        route_names = {
            ("GET", "/health"): "health",
            ("GET", "/model"): "model",
            ("POST", "/predict"): "predict",
            ("POST", "/predict/batch"): "predict_batch",
        }
        route = route_names.get((request.method, request.url.path))
        if route is None:
            return await call_next(request)
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            metrics.requests.labels(
                route=route, method=request.method, status=str(status)
            ).inc()
            metrics.latency.labels(route=route, method=request.method).observe(
                time.perf_counter() - started
            )

    @api.middleware("http")
    async def request_timing(request: Request, call_next: Any) -> Any:
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("Unhandled server error path=%s", request.url.path)
            raise
        finally:
            duration_ms = (time.perf_counter() - start) * 1000.0
        logger.info(
            "HTTP path=%s method=%s status=%s latency_ms=%.2f",
            request.url.path,
            request.method,
            response.status_code,
            duration_ms,
        )
        return response

    @api.exception_handler(InputValidationError)
    async def input_validation_handler(
        request: Request, exc: InputValidationError
    ) -> JSONResponse:
        logger.warning("Input rejected route=%s reason=%s", request.url.path, str(exc))
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @api.exception_handler(ArtifactValidationError)
    async def artifact_failure_handler(
        request: Request, exc: ArtifactValidationError
    ) -> JSONResponse:
        logger.error(
            "Prediction backend unavailable path=%s reason=%s",
            request.url.path,
            str(exc),
        )
        return JSONResponse(
            status_code=503, content={"detail": "Prediction service unavailable"}
        )

    @api.get("/metrics", include_in_schema=False)
    def get_metrics() -> Response:
        # Local-only endpoint; protect with network controls before public deployment.
        return Response(
            content=generate_latest(metrics.registry),
            media_type=CONTENT_TYPE_LATEST,
        )

    @api.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @api.get("/model", response_model=ModelInfoResponse)
    def model_info(request: Request) -> ModelInfoResponse:
        current: PredictionService = request.app.state.inference
        return ModelInfoResponse(
            model_version=current.version,
            decision_threshold=current.threshold,
            input_feature_count=len(current.expected_features),
        )

    @api.post("/predict", response_model=PredictionResponse)
    def predict_single(payload: OrderRequest, request: Request) -> PredictionResponse:
        current: PredictionService = request.app.state.inference
        started = time.perf_counter()
        result = current.predict([payload.order])[0]
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        metrics.observe_predictions([result])
        emit_prediction_event(
            [result], model_version=current.version, inference_latency_ms=elapsed_ms
        )
        logger.info(
            "Prediction completed count=1 positives=%d model_version=%s",
            result["prediction"],
            current.version,
        )
        return PredictionResponse.model_validate(result)

    @api.post("/predict/batch", response_model=BatchResponse)
    def predict_batch(payload: BatchRequest, request: Request) -> BatchResponse:
        current: PredictionService = request.app.state.inference
        started = time.perf_counter()
        results = current.predict(payload.orders)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        metrics.observe_predictions(results)
        emit_prediction_event(
            results, model_version=current.version, inference_latency_ms=elapsed_ms
        )
        logger.info(
            "Prediction completed count=%d positives=%d model_version=%s",
            len(results),
            sum(int(item["prediction"]) for item in results),
            current.version,
        )
        return BatchResponse(
            predictions=[PredictionResponse.model_validate(item) for item in results]
        )

    return api


app = create_app()
