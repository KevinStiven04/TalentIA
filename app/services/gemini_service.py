
import json
import logging

from functools import lru_cache

from google import genai
from google.genai import types

from app.core.config import settings
from app.schemas.prediction import PredictionResponse
from app.schemas.explanation import SHAPExplanationResponse


logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_gemini_client():
    """
    Crea y reutiliza el cliente de Gemini.
    """

    settings.validate_gemini_config()

    return genai.Client(
        api_key=settings.GEMINI_API_KEY,
        http_options=types.HttpOptions(
            timeout=60000,
            retry_options=types.HttpRetryOptions(
                attempts=3,
                initial_delay=1.0,
                max_delay=5.0,
                exp_base=2.0,
                jitter=1.0
            )
        )
    )


# Estructura JSON esperada de Gemini.
GEMINI_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "interpretation": {
            "type": "string"
        },
        "factors_toward_yes": {
            "type": "array",
            "items": {"type": "string"}
        },
        "factors_toward_no": {
            "type": "array",
            "items": {"type": "string"}
        },
        "recommendations": {
            "type": "array",
            "items": {"type": "string"}
        },
        "disclaimer": {
            "type": "string"
        }
    },
    "required": [
        "interpretation",
        "factors_toward_yes",
        "factors_toward_no",
        "recommendations",
        "disclaimer"
    ]
}


def generate_gemini_analysis(
    prediction: PredictionResponse,
    explanation: SHAPExplanationResponse
) -> dict:
    """
    Genera una interpretación en lenguaje natural
    de la predicción y las contribuciones SHAP.
    """

    try:
        # 1. Obtener el cliente de Gemini.
        client = get_gemini_client()

        # 2. Preparar los datos que recibirá Gemini.
        analysis_data = {
            "predicted_attrition": prediction.predicted_attrition,
            "probability_yes": prediction.probability_yes,
            "threshold": prediction.threshold,
            "shap_factors": [
                {
                    "feature": feature.feature,
                    "shap_value": feature.shap_value,
                    "direction": feature.direction
                }
                for feature in explanation.features
            ]
        }

        # 3. Convertir los datos a JSON.
        input_json = json.dumps(
            analysis_data,
            ensure_ascii=False
        )

        # 4. Definir las instrucciones para Gemini.
        system_instruction = """
        Eres un asistente especializado en interpretar
        resultados de modelos de Machine Learning aplicados
        al análisis de rotación laboral.

        Tu tarea es interpretar los resultados del modelo
        y las contribuciones SHAP proporcionadas.

        REGLAS IMPORTANTES:

        1. No debes modificar ni recalcular la predicción.
        2. No debes cambiar la probabilidad del modelo.
        3. Utiliza exclusivamente la información recibida.
        4. Explica qué características contribuyeron hacia
           Attrition Yes y cuáles hacia Attrition No.
        5. No afirmes que una característica causa la rotación.
        6. No inventes información sobre el empleado.
        7. Genera recomendaciones generales para la organización.
        8. No recomiendes despedir, contratar o sancionar
           empleados basándote en esta predicción.
        9. Responde siempre en español.
        10. Aclara que las predicciones son estimaciones
            y no certezas sobre el comportamiento futuro.
        """

        # 5. Enviar la solicitud a Gemini.
        response = client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=input_json,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_json_schema=GEMINI_RESPONSE_SCHEMA,
                thinking_config=types.ThinkingConfig(
                    thinking_level="low"
                )
            )
        )

        # 6. Comprobar que Gemini haya generado una respuesta.
        if not response.text:
            raise ValueError(
                "Gemini no devolvió contenido."
            )

        # 7. Convertir la respuesta JSON a diccionario.
        analysis = json.loads(response.text)

        logger.info(
            "Interpretación de TalentIA generada correctamente."
        )

        # 8. Devolver el resultado.
        return analysis

    except Exception:
        logger.exception(
            "Error al generar la interpretación con Gemini."
        )
        raise
