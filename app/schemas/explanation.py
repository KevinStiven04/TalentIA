from typing import Literal
from pydantic import BaseModel


class FeatureExplanation(BaseModel):
    """Explicación de la contribución de una característica."""

    feature_key: str
    feature: str
    value: str | int | float

    shap_value: float
    absolute_impact: float

    direction: Literal[
        "toward_yes",
        "toward_no",
        "neutral"
    ]

class SHAPExplanationResponse(BaseModel):
    """Respuesta de la explicación SHAP individual."""

    explained_class: Literal["Yes"]
    base_value: float
    features: list[FeatureExplanation]