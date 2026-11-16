
import logging

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import case, desc, func, select, union_all
from sqlalchemy.exc import InterfaceError, OperationalError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.employee import Employee
from app.models.prediction_history import PredictionHistory


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/analytics",
    tags=["Analytics"]
)


# Nombres de los meses para el gráfico del dashboard.
MONTH_LABELS = [
    "ene", "feb", "mar", "abr",
    "may", "jun", "jul", "ago",
    "sep", "oct", "nov", "dic"
]


def get_latest_predictions():
    """
    Obtiene la última predicción de cada empleado registrado.

    Si un empleado tiene varias predicciones,
    solamente se considera la más reciente.
    """

    ranked_predictions = (
        select(
            PredictionHistory.id.label("prediction_id"),
            PredictionHistory.employee_id.label("employee_id"),
            PredictionHistory.predicted_attrition.label(
                "predicted_attrition"
            ),
            PredictionHistory.probability_yes.label(
                "probability_yes"
            ),
            PredictionHistory.created_at.label("created_at"),

            func.row_number().over(
                partition_by=PredictionHistory.employee_id,
                order_by=[
                    PredictionHistory.created_at.desc(),
                    PredictionHistory.id.desc()
                ]
            ).label("position")
        )
        .where(PredictionHistory.employee_id.is_not(None))
        .subquery()
    )

    latest_predictions = (
        select(ranked_predictions)
        .where(ranked_predictions.c.position == 1)
        .subquery()
    )

    return latest_predictions


def get_current_analyses(latest_predictions):
    """
    Combina la última predicción de cada empleado registrado
    con las predicciones manuales (employee_id IS NULL)
    para los indicadores de estado actual.
    """

    employee_analyses = select(
        latest_predictions.c.prediction_id,
        latest_predictions.c.employee_id,
        latest_predictions.c.predicted_attrition,
        latest_predictions.c.probability_yes,
        latest_predictions.c.created_at
    )

    manual_analyses = (
        select(
            PredictionHistory.id.label("prediction_id"),
            PredictionHistory.employee_id.label("employee_id"),
            PredictionHistory.predicted_attrition.label(
                "predicted_attrition"
            ),
            PredictionHistory.probability_yes.label(
                "probability_yes"
            ),
            PredictionHistory.created_at.label("created_at")
        )
        .where(PredictionHistory.employee_id.is_(None))
    )

    return union_all(employee_analyses, manual_analyses).subquery()


def get_last_six_months():
    """
    Genera los últimos seis meses incluyendo el actual.
    """

    now = datetime.now(timezone.utc)

    months = []

    current_index = now.year * 12 + now.month - 1

    for offset in range(5, -1, -1):

        month_index = current_index - offset

        year = month_index // 12
        month = month_index % 12 + 1

        months.append({
            "month": f"{year}-{month:02d}",
            "label": MONTH_LABELS[month - 1]
        })

    first_month = months[0]["month"]

    start_date = datetime.strptime(
        first_month,
        "%Y-%m"
    ).replace(tzinfo=timezone.utc)

    return months, start_date


@router.get("/dashboard")
def get_dashboard_analytics(
    db: Session = Depends(get_db)
):
    """
    Obtiene las estadísticas necesarias para el dashboard
    de TalentIA.
    """

    try:

        # --------------------------------------------------
        # 1. Obtener última predicción por empleado y análisis manuales.
        # --------------------------------------------------

        latest_predictions = get_latest_predictions()
        current_analyses = get_current_analyses(latest_predictions)

        # --------------------------------------------------
        # 2. Obtener estadísticas de predicciones vigentes.
        # --------------------------------------------------

        prediction_stats = db.execute(
            select(
                func.count(
                    current_analyses.c.prediction_id
                ).label("employees_analyzed"),

                func.sum(
                    case(
                        (
                            current_analyses.c.predicted_attrition
                            == "Yes",
                            1
                        ),
                        else_=0
                    )
                ).label("high_risk"),

                func.avg(
                    current_analyses.c.probability_yes
                ).label("average_probability")
            )
        ).one()

        employees_analyzed = int(
            prediction_stats.employees_analyzed or 0
        )

        high_risk = int(
            prediction_stats.high_risk or 0
        )

        average_probability = (
            float(prediction_stats.average_probability)
            if prediction_stats.average_probability is not None
            else None
        )

        high_risk_percentage = (
            (high_risk / employees_analyzed) * 100
            if employees_analyzed > 0
            else None
        )

        # --------------------------------------------------
        # 3. Obtener la cantidad total de predicciones registradas.
        # --------------------------------------------------

        prediction_records = db.scalar(
            select(
                func.count(PredictionHistory.id)
            )
        ) or 0

        # --------------------------------------------------
        # 4. Obtener la rotación histórica del dataset.
        # --------------------------------------------------

        observed_stats = db.execute(
            select(
                func.count(Employee.id).label("total"),

                func.sum(
                    case(
                        (
                            Employee.observed_attrition == "Yes",
                            1
                        ),
                        else_=0
                    )
                ).label("yes_count")
            )
            .where(
                Employee.observed_attrition.is_not(None)
            )
        ).one()

        dataset_employees = int(
            observed_stats.total or 0
        )

        observed_yes = int(
            observed_stats.yes_count or 0
        )

        observed_attrition_rate = (
            (observed_yes / dataset_employees) * 100
            if dataset_employees > 0
            else None
        )

        # --------------------------------------------------
        # 5. Obtener empleados por departamento.
        # --------------------------------------------------

        department_rows = db.execute(
            select(
                Employee.department,
                func.count(Employee.id).label("total")
            )
            .group_by(Employee.department)
            .order_by(desc("total"))
        ).all()

        departments = [
            {
                "department": row.department,
                "employees": int(row.total)
            }
            for row in department_rows
        ]

        # --------------------------------------------------
        # 6. Obtener todas las predicciones de los últimos 6 meses.
        # --------------------------------------------------

        months, start_date = get_last_six_months()

        month_expression = func.date_trunc(
            "month",
            func.timezone("UTC", PredictionHistory.created_at)
        )

        monthly_rows = db.execute(
            select(
                month_expression.label("month"),

                func.count(
                    PredictionHistory.id
                ).label("total"),

                func.sum(
                    case(
                        (
                            PredictionHistory.predicted_attrition
                            == "Yes",
                            1
                        ),
                        else_=0
                    )
                ).label("yes_count")
            )
            .where(
                PredictionHistory.created_at >= start_date
            )
            .group_by(month_expression)
            .order_by(month_expression)
        ).all()

        monthly_data = {
            row.month.strftime("%Y-%m"): row
            for row in monthly_rows
        }

        monthly_trend = []

        for month in months:

            row = monthly_data.get(month["month"])

            total = int(row.total or 0) if row else 0
            yes_count = int(row.yes_count or 0) if row else 0

            yes_rate = (
                (yes_count / total) * 100
                if total > 0
                else None
            )

            monthly_trend.append({
                "month": month["month"],
                "label": month["label"],
                "analyses": total,
                "yes_count": yes_count,
                "yes_rate_percentage": (
                    round(yes_rate, 2)
                    if yes_rate is not None
                    else None
                )
            })

        # --------------------------------------------------
        # 7. Calcular cambios frente al mes anterior.
        # --------------------------------------------------

        current_month = monthly_trend[-1]
        previous_month = monthly_trend[-2]

        monthly_change_analyses = None
        monthly_change_percentage_points = None

        if (
            current_month["analyses"] > 0
            and previous_month["analyses"] > 0
        ):
            monthly_change_analyses = (
                current_month["analyses"]
                - previous_month["analyses"]
            )

            monthly_change_percentage_points = round(
                current_month["yes_rate_percentage"]
                - previous_month["yes_rate_percentage"],
                2
            )

        # --------------------------------------------------
        # 8. Obtener los cinco empleados registrados con mayor riesgo.
        # --------------------------------------------------

        risk_rows = db.execute(
            select(
                Employee.id.label("employee_id"),
                Employee.employee_number,
                Employee.display_name,
                Employee.department,
                latest_predictions.c.probability_yes,
                latest_predictions.c.created_at
            )
            .join(
                latest_predictions,
                Employee.id == latest_predictions.c.employee_id
            )
            .where(
                latest_predictions.c.predicted_attrition == "Yes"
            )
            .order_by(
                latest_predictions.c.probability_yes.desc(),
                latest_predictions.c.created_at.desc()
            )
            .limit(5)
        ).all()

        high_risk_employees = [
            {
                "employee_id": row.employee_id,
                "employee_number": row.employee_number,
                "display_name": row.display_name,
                "department": row.department,
                "probability_yes": round(
                    float(row.probability_yes) * 100,
                    2
                ),
                "predicted_at": row.created_at.isoformat()
            }
            for row in risk_rows
        ]

        # --------------------------------------------------
        # 9. Construir respuesta para el dashboard.
        # --------------------------------------------------

        return {
            "summary": {
                "employees_analyzed": employees_analyzed,
                "dataset_employees": dataset_employees,
                "observed_attrition_rate": (
                    round(observed_attrition_rate, 2)
                    if observed_attrition_rate is not None
                    else None
                ),
                "high_risk": high_risk,
                "high_risk_percentage": (
                    round(high_risk_percentage, 2)
                    if high_risk_percentage is not None
                    else None
                ),
                "average_probability_yes": (
                    round(average_probability * 100, 2)
                    if average_probability is not None
                    else None
                ),
                "prediction_records": int(prediction_records),
                "monthly_change_analyses": monthly_change_analyses,
                "monthly_change_percentage_points": (
                    monthly_change_percentage_points
                )
            },

            "monthly_trend": monthly_trend,

            "departments": departments,

            "high_risk_employees": high_risk_employees,

            "definitions": {
                "observed_attrition_rate": (
                    "Proporción histórica de Attrition=Yes "
                    "en el dataset IBM HR Analytics."
                ),
                "monthly_trend": (
                    "Porcentaje mensual de predicciones "
                    "clasificadas como Yes en TalentIA."
                ),
                "average_probability_yes": (
                    "Promedio de probability_yes de la "
                    "última predicción de cada empleado y análisis manuales."
                )
            }
        }

    except (OperationalError, InterfaceError):
        logger.exception(
            "Error de conexión con Neon al consultar el dashboard."
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No fue posible conectar con la base de datos."
        )

    except SQLAlchemyError:
        logger.exception(
            "Error de base de datos al obtener las estadísticas del dashboard."
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error al consultar las estadísticas del dashboard."
        )

    except Exception:
        logger.exception(
            "Error al obtener las estadísticas del dashboard."
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno al obtener las estadísticas del dashboard."
        )
