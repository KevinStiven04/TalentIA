
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import (
    predictions,
    gemini,
    health,
    analytics,
    employees
)


app = FastAPI(
    title="TalentIA API",
    description="API para predicción de rotación laboral",
    version="1.0.0"
)


# Configuración CORS para Kobweb.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# Router de predicciones y SHAP.
app.include_router(
    predictions.router,
    prefix="/api/v1"
)

# Router de Gemini.
app.include_router(
    gemini.router,
    prefix="/api/v1"
)

# Router de salud.
app.include_router(
    health.router,
    prefix="/api/v1"
)

# Router de Analytics.
app.include_router(
    analytics.router,
    prefix="/api/v1"
)

app.include_router(
    employees.router,
    prefix="/api/v1"
)
