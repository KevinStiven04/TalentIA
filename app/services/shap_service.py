
import logging

import numpy as np
import pandas as pd
import shap

from app.ml.model_loader import get_model
from app.schemas.employee import EmployeeFeatures


logger = logging.getLogger(__name__)


# Correspondencia entre los atributos de la API
# y las columnas originales del dataset.
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


# Nombres que mostraremos en la interfaz.
FEATURE_LABELS = {
    "Department": "Departamento",
    "JobRole": "Cargo",
    "MonthlyIncome": "Ingreso mensual",
    "JobSatisfaction": "Satisfacción laboral",
    "EnvironmentSatisfaction": "Satisfacción con el entorno",
    "OverTime": "Horas extras",
    "WorkLifeBalance": "Equilibrio vida-trabajo",
    "YearsAtCompany": "Antigüedad en la empresa",
}


def _extract_positive_class_values(
    shap_values,
    positive_class_index: int,
    number_of_features: int
) -> np.ndarray:
    """
    Extrae los valores SHAP correspondientes a la clase positiva (Yes).

    Contempla los formatos habituales devueltos por diferentes
    versiones de SHAP.
    """

    if isinstance(shap_values, list):
        values = np.asarray(shap_values[positive_class_index])

    else:
        values_array = np.asarray(shap_values)

        if values_array.ndim == 3:
            # Formato: muestras, características, clases
            if (
                values_array.shape[1] == number_of_features
                and values_array.shape[2] > positive_class_index
            ):
                values = values_array[0, :, positive_class_index]

            # Formato alternativo: clases, muestras, características
            elif (
                values_array.shape[0] > positive_class_index
                and values_array.shape[2] == number_of_features
            ):
                values = values_array[positive_class_index, 0, :]

            else:
                raise ValueError(
                    "No se reconoce el formato de los valores SHAP."
                )

        elif values_array.ndim == 2:
            values = values_array[0]

        elif values_array.ndim == 1:
            values = values_array

        else:
            raise ValueError(
                "La dimensión de los valores SHAP no es compatible."
            )

    values = np.asarray(values).reshape(-1)

    if len(values) != number_of_features:
        raise ValueError(
            "La cantidad de valores SHAP no coincide con "
            "las características transformadas."
        )

    return values


def _extract_base_value(
    expected_value,
    positive_class_index: int
) -> float:
    """
    Obtiene el valor base correspondiente a la clase Yes.
    """

    values = np.asarray(expected_value)

    if values.ndim == 0:
        return float(values)

    if values.ndim == 1:
        return float(values[positive_class_index])

    values = values.reshape(-1, values.shape[-1])

    return float(values[0, positive_class_index])


def _get_original_feature(transformed_name: str) -> str | None:
    """
    Identifica a qué variable original pertenece una característica
    después de aplicar el preprocesamiento.
    """

    # Elimina el prefijo del transformador, por ejemplo:
    # categoricas__Department_Sales -> Department_Sales
    clean_name = transformed_name.split("__", 1)[-1]

    # Se ordenan por longitud para comprobar primero los nombres
    # más específicos.
    original_features = sorted(
        FEATURE_MAPPING.values(),
        key=len,
        reverse=True
    )

    for feature in original_features:
        if (
            clean_name == feature
            or clean_name.startswith(f"{feature}_")
        ):
            return feature

    return None


def explain_employee(data: EmployeeFeatures) -> dict:
    """
    Genera una explicación SHAP individual para un empleado.

    Devuelve las características que más influyeron en la
    clasificación de Attrition = Yes.
    """

    try:
        # 1. Obtener el Pipeline entrenado.
        model = get_model()

        # 2. Comprobar que el modelo sea un Pipeline.
        if not hasattr(model, "steps") or len(model.steps) < 2:
            raise ValueError(
                "El modelo debe ser un Pipeline con preprocesador "
                "y clasificador."
            )

        # 3. Separar el preprocesamiento y el clasificador.
        preprocessing_pipeline = model[:-1]
        classifier = model.steps[-1][1]

        # 4. Obtener los datos del empleado excluyendo employee_number.
        employee_data = data.model_dump(exclude={"employee_number"})

        # 5. Cambiar los nombres para que coincidan con el dataset.
        dataset_data = {
            FEATURE_MAPPING[key]: employee_data[key]
            for key in FEATURE_MAPPING
        }

        # 6. Crear el DataFrame con las columnas originales.
        employee_df = pd.DataFrame(
            [dataset_data],
            columns=list(FEATURE_MAPPING.values())
        )

        # 7. Aplicar exactamente el preprocesamiento del entrenamiento.
        transformed_data = preprocessing_pipeline.transform(
            employee_df
        )

        # SHAP necesita trabajar con datos numéricos densos
        # para este flujo de explicación.
        if hasattr(transformed_data, "toarray"):
            transformed_data = transformed_data.toarray()

        # 8. Obtener los nombres de las características transformadas.
        feature_names = (
            preprocessing_pipeline.get_feature_names_out()
        )

        # 9. Identificar la posición de la clase positiva (Yes = 1).
        classes = list(classifier.classes_)
        positive_class_index = classes.index(1)

        # 10. Crear el explicador SHAP utilizando el clasificador.
        explainer = shap.TreeExplainer(classifier)

        # 11. Calcular los valores SHAP del empleado.
        raw_shap_values = explainer.shap_values(
            transformed_data
        )

        shap_values = _extract_positive_class_values(
            raw_shap_values,
            positive_class_index,
            len(feature_names)
        )

        # 12. Obtener el valor base de la clase Yes.
        base_value = _extract_base_value(
            explainer.expected_value,
            positive_class_index
        )

        # 13. Agrupar las contribuciones de las variables
        # transformadas en las ocho variables originales.
        grouped_values = {
            feature: 0.0
            for feature in FEATURE_MAPPING.values()
        }

        for transformed_name, shap_value in zip(
            feature_names,
            shap_values
        ):
            original_feature = _get_original_feature(
                str(transformed_name)
            )

            if original_feature is None:
                logger.warning(
                    "No se pudo identificar la variable SHAP: %s",
                    transformed_name
                )
                continue

            grouped_values[original_feature] += float(shap_value)

        # 14. Preparar los resultados para el frontend.
        feature_explanations = []

        for original_feature, shap_value in grouped_values.items():

            # Buscar el nombre utilizado por la API.
            api_feature = next(
                key
                for key, value in FEATURE_MAPPING.items()
                if value == original_feature
            )

            feature_explanations.append({
                "feature_key": api_feature,
                "feature": FEATURE_LABELS[original_feature],
                "value": employee_data[api_feature],
                "shap_value": float(shap_value),
                "absolute_impact": abs(float(shap_value)),
                "direction": (
                    "toward_yes"
                    if shap_value > 0
                    else "toward_no"
                    if shap_value < 0
                    else "neutral"
                )
            })

        # 15. Ordenar de mayor a menor influencia absoluta.
        feature_explanations.sort(
            key=lambda item: item["absolute_impact"],
            reverse=True
        )

        # 16. Devolver la explicación.
        return {
            "explained_class": "Yes",
            "base_value": base_value,
            "features": feature_explanations
        }

    except Exception:
        logger.exception(
            "Error al generar la explicación SHAP del empleado."
        )
        raise
