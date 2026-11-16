from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class PredictionHistoryItem(BaseModel):
    """Registro individual del historial de predicciones."""

    id: int
    employee_id: int | None = None
    employee_number: int | None = None
    display_name: str | None = None
    department: str
    job_role: str
    predicted_attrition: Literal["Yes", "No"]
    probability_yes: float
    threshold: float
    created_at: datetime

    model_config = {
        "from_attributes": True
    }


class PredictionHistoryResponse(BaseModel):
    """Respuesta paginada del historial de predicciones."""

    items: list[PredictionHistoryItem]
    total: int
    page: int
    page_size: int
    total_pages: int
