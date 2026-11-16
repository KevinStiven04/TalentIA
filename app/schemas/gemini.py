
from pydantic import BaseModel


class GeminiAnalysisResponse(BaseModel):
    """Respuesta del análisis generado por Gemini."""

    interpretation: str

    factors_toward_yes: list[str]

    factors_toward_no: list[str]

    recommendations: list[str]

    disclaimer: str
