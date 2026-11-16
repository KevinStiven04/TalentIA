
from pydantic import BaseModel, Field
from typing import Literal


class PredictionResponse(BaseModel):
    """
    Resultado de la predicción de rotación laboral.
    """

    predicted_attrition: Literal["Yes", "No"] = Field(
        ...,
        description="Clase predicha por el modelo"
    )

    probability_yes: float = Field(
        ...,
        ge=0,
        le=1,
        description="Probabilidad estimada de Attrition Yes"
    )

    threshold: float = Field(
        ...,
        ge=0,
        le=1,
        description="Umbral utilizado para clasificar"
    )
