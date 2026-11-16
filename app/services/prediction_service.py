
import logging

import pandas as pd
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.ml.model_loader import get_model
from app.models.employee import Employee
from app.models.prediction_history import PredictionHistory
from app.schemas.employee import EmployeeFeatures
from app.schemas.prediction import PredictionResponse


logger = logging.getLogger(__name__)

# Umbral de clasificación
THRESHOLD = 0.5

# Correspondencia entre los atributos recibidos por la API
# y los nombres originales del dataset IBM HR Analytics.
FEATURE_MAPPING = {
    "department": "Department",
    "job_role": "JobRole",
    "monthly_income": "MonthlyIncome",
    "job_satisfaction": "JobSatisfaction",
    "environment_satisfaction": "EnvironmentSatisfaction",
    "over_time": "OverTime",
    "work_life_balance": "WorkLifeBalance",
    "years_at_company": "YearsAtCompany",
}


class EmployeeNotFoundError(LookupError):
    """Error lanzado cuando el employee_number indicado no existe en Neon."""


def calculate_employee_prediction(
    data: EmployeeFeatures
) -> PredictionResponse:
    """
    Calcula la predicción de rotación laboral sin persistir en base de datos.

    Excluye employee_number y conserva únicamente las 8 características
    originales del modelo.
    """

    try:
        # 1. Obtener el modelo entrenado.
        model = get_model()

        # 2. Excluir employee_number y tomar únicamente las 8 variables del modelo.
        raw_features = data.model_dump(exclude={"employee_number"})

        # 3. Adaptar los nombres de los atributos al dataset original.
        employee_data = {
            dataset_column: raw_features[api_field]
            for api_field, dataset_column in FEATURE_MAPPING.items()
        }

        # 4. Crear un DataFrame con las columnas esperadas en su orden exacto.
        employee_df = pd.DataFrame(
            [employee_data],
            columns=list(FEATURE_MAPPING.values())
        )

        # 5. Obtener las probabilidades de predicción.
        probabilities = model.predict_proba(employee_df)[0]

        # 6. Identificar la probabilidad correspondiente a Attrition = Yes.
        classes = list(model.classes_)
        positive_class_index = classes.index(1)

        probability_yes = float(probabilities[positive_class_index])

        # 7. Clasificar según el umbral establecido.
        predicted_attrition = (
            "Yes" if probability_yes >= THRESHOLD else "No"
        )

        # 8. Construir la respuesta.
        return PredictionResponse(
            predicted_attrition=predicted_attrition,
            probability_yes=probability_yes,
            threshold=THRESHOLD
        )

    except Exception:
        logger.exception("Error al calcular la predicción de rotación.")
        raise


def predict_employee(
    data: EmployeeFeatures,
    db: Session
) -> PredictionResponse:
    """
    Calcula la predicción de rotación laboral y la almacena en Neon PostgreSQL.
    """

    # 1. Calcular la predicción del modelo sin modificar la base de datos.
    prediction = calculate_employee_prediction(data)

    try:
        # 2. Resolver la relación con Employee si se proporcionó employee_number.
        employee_id: int | None = None

        if data.employee_number is not None:
            employee = db.scalar(
                select(Employee).where(
                    Employee.employee_number == data.employee_number
                )
            )

            if employee is None:
                raise EmployeeNotFoundError(
                    f"No existe un empleado con employee_number={data.employee_number}."
                )

            employee_id = employee.id

        # 3. Crear el registro histórico de predicción.
        prediction_record = PredictionHistory(
            employee_id=employee_id,
            department=data.department,
            job_role=data.job_role,
            predicted_attrition=prediction.predicted_attrition,
            probability_yes=prediction.probability_yes,
            threshold=prediction.threshold
        )

        db.add(prediction_record)
        db.commit()
        db.refresh(prediction_record)

        return prediction

    except EmployeeNotFoundError:
        db.rollback()
        raise

    except SQLAlchemyError:
        db.rollback()
        logger.exception(
            "Error de base de datos al guardar la predicción en Neon."
        )
        raise

    except Exception:
        db.rollback()
        logger.exception(
            "Error inesperado al guardar la predicción en Neon."
        )
        raise

