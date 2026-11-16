import logging

from fastapi import APIRouter, HTTPException, status

from app.schemas.employee import EmployeeFeatures
from app.schemas.prediction import PredictionResponse
from app.schemas.explanation import SHAPExplanationResponse
from app.schemas.gemini import GeminiAnalysisResponse

from app.services.prediction_service import calculate_employee_prediction
from app.services.shap_service import explain_employee
from app.services.gemini_service import generate_gemini_analysis

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/gemini",
    tags=["Gemini"]
)


@router.post(
    "/analyze",
    response_model=GeminiAnalysisResponse,
    status_code=status.HTTP_200_OK
)
def analyze_employee(data: EmployeeFeatures):
    """
    Genera una interpretación de la predicción
    y las explicaciones SHAP utilizando Gemini
    sin duplicar registros en prediction_history.
    """

    try:
        # 1. Calcular la predicción del empleado sin persistir en BD.
        logger.info("Iniciando cálculo de predicción para Gemini...")
        prediction = calculate_employee_prediction(data)
        logger.info("Predicción calculada.")

        # 2. Obtener las explicaciones SHAP.
        logger.info("Iniciando explicación SHAP...")
        explanation_data = explain_employee(data)
        logger.info("Explicación SHAP completada.")

        # 3. Validar la estructura de la explicación SHAP.
        logger.info("Validando resultados SHAP...")
        explanation = SHAPExplanationResponse.model_validate(
            explanation_data
        )

        # 4. Enviar los resultados a Gemini.
        logger.info("Enviando información a Gemini...")
        analysis = generate_gemini_analysis(
            prediction,
            explanation
        )
        logger.info("Gemini respondió correctamente.")

        # 5. Devolver el análisis generado.
        return analysis

    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(error)
        )

    except Exception:
        logger.exception(
            "Error al generar el análisis de Gemini."
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No fue posible generar el análisis con Gemini."
        )
