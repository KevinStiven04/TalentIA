
import logging
import math
from datetime import date, datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import String, and_, cast, func, literal, or_, select
from sqlalchemy.exc import InterfaceError, OperationalError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.employee import Employee
from app.models.prediction_history import PredictionHistory
from app.schemas.employee import EmployeeFeatures
from app.schemas.explanation import SHAPExplanationResponse
from app.schemas.prediction import PredictionResponse
from app.schemas.prediction_history import (
    PredictionHistoryItem,
    PredictionHistoryResponse,
)
from app.services.prediction_service import (
    EmployeeNotFoundError,
    predict_employee,
)
from app.services.shap_service import explain_employee


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/predictions",
    tags=["Predictions"]
)


def _parse_history_date(
    raw_value: str | None,
    *,
    is_end: bool
) -> tuple[datetime | None, bool]:
    """
    Convierte una fecha YYYY-MM-DD o ISO-8601 a datetime con zona horaria.

    Cuando se recibe una fecha sin hora para el límite final (is_end=True),
    devuelve el inicio del día siguiente para usar comparación estricta (<)
    e incluir todo el día final.
    """
    if raw_value is None or not raw_value.strip():
        return None, False

    cleaned = raw_value.strip()

    if len(cleaned) == 10:
        try:
            parsed_date = date.fromisoformat(cleaned)
        except ValueError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Formato de fecha inválido: '{cleaned}'. Usa YYYY-MM-DD o ISO 8601."
            ) from error

        dt = datetime(
            parsed_date.year,
            parsed_date.month,
            parsed_date.day,
            tzinfo=timezone.utc
        )
        if is_end:
            return dt + timedelta(days=1), True
        return dt, True

    try:
        parsed_dt = datetime.fromisoformat(cleaned)
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Formato de fecha y hora inválido: '{cleaned}'."
        ) from error

    if parsed_dt.tzinfo is None:
        parsed_dt = parsed_dt.replace(tzinfo=timezone.utc)

    return parsed_dt, False


@router.get(
    "/history",
    response_model=PredictionHistoryResponse,
    status_code=status.HTTP_200_OK
)
def get_prediction_history(
    search: str | None = Query(
        default=None,
        description="Buscar por número de empleado, nombre, departamento o cargo"
    ),
    from_date: str | None = Query(
        default=None,
        alias="from",
        description="Fecha inicial del historial (YYYY-MM-DD o ISO 8601)"
    ),
    to_date: str | None = Query(
        default=None,
        alias="to",
        description="Fecha final del historial (YYYY-MM-DD o ISO 8601)"
    ),
    attrition: str | None = Query(
        default=None,
        description="Filtrar por clasificación Yes o No"
    ),
    page: int = Query(
        default=1,
        ge=1,
        description="Número de página"
    ),
    page_size: int = Query(
        default=10,
        ge=1,
        le=100,
        description="Cantidad de registros por página"
    ),
    db: Session = Depends(get_db)
):
    """
    Consulta el historial de predicciones almacenado en Neon PostgreSQL
    con búsqueda, rango de fechas, filtro de clasificación y paginación.
    """

    try:
        filters = []

        # 1. Búsqueda general por número de empleado, nombre, departamento o cargo.
        if search and search.strip():
            search_term = f"%{search.strip()}%"
            filters.append(
                or_(
                    Employee.display_name.ilike(search_term),
                    cast(Employee.employee_number, String).ilike(search_term),
                    PredictionHistory.department.ilike(search_term),
                    PredictionHistory.job_role.ilike(search_term),
                    and_(
                        PredictionHistory.employee_id.is_(None),
                        literal("Predicción manual").ilike(search_term)
                    )
                )
            )

        # 2. Filtro por clasificación Yes / No.
        if attrition and attrition.strip():
            normalized_attrition = attrition.strip().lower()
            if normalized_attrition == "yes":
                filters.append(PredictionHistory.predicted_attrition == "Yes")
            elif normalized_attrition == "no":
                filters.append(PredictionHistory.predicted_attrition == "No")
            else:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="El parámetro attrition debe ser 'Yes' o 'No'."
                )

        # 3. Filtros por fecha inicial y final.
        start_dt, _ = _parse_history_date(from_date, is_end=False)
        end_dt, end_is_exclusive = _parse_history_date(to_date, is_end=True)

        if start_dt is not None and end_dt is not None:
            invalid_range = (
                start_dt >= end_dt if end_is_exclusive else start_dt > end_dt
            )
            if invalid_range:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="La fecha inicial no puede ser posterior a la fecha final."
                )

        if start_dt is not None:
            filters.append(PredictionHistory.created_at >= start_dt)

        if end_dt is not None:
            if end_is_exclusive:
                filters.append(PredictionHistory.created_at < end_dt)
            else:
                filters.append(PredictionHistory.created_at <= end_dt)

        # 4. Consultar el total de registros que cumplen los filtros.
        total = db.scalar(
            select(func.count(PredictionHistory.id))
            .select_from(PredictionHistory)
            .outerjoin(
                Employee,
                PredictionHistory.employee_id == Employee.id
            )
            .where(*filters)
        ) or 0

        total_pages = math.ceil(total / page_size) if total > 0 else 0

        # 5. Consultar la página solicitada ordenada de más reciente a más antigua.
        rows = db.execute(
            select(
                PredictionHistory.id,
                PredictionHistory.employee_id,
                Employee.employee_number,
                Employee.display_name,
                PredictionHistory.department,
                PredictionHistory.job_role,
                PredictionHistory.predicted_attrition,
                PredictionHistory.probability_yes,
                PredictionHistory.threshold,
                PredictionHistory.created_at,
            )
            .select_from(PredictionHistory)
            .outerjoin(
                Employee,
                PredictionHistory.employee_id == Employee.id
            )
            .where(*filters)
            .order_by(
                PredictionHistory.created_at.desc(),
                PredictionHistory.id.desc()
            )
            .offset((page - 1) * page_size)
            .limit(page_size)
        ).all()

        items = [
            PredictionHistoryItem(
                id=row.id,
                employee_id=row.employee_id,
                employee_number=row.employee_number,
                display_name=row.display_name,
                department=row.department,
                job_role=row.job_role,
                predicted_attrition=row.predicted_attrition,
                probability_yes=float(row.probability_yes),
                threshold=float(row.threshold),
                created_at=row.created_at,
            )
            for row in rows
        ]

        return PredictionHistoryResponse(
            items=items,
            total=int(total),
            page=page,
            page_size=page_size,
            total_pages=total_pages
        )

    except HTTPException:
        raise

    except (OperationalError, InterfaceError):
        logger.exception("Error de conexión con Neon al consultar el historial.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No fue posible conectar con la base de datos."
        )

    except SQLAlchemyError:
        logger.exception("Error de base de datos al consultar el historial.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error al consultar el historial de predicciones."
        )

    except Exception:
        logger.exception("Error inesperado al consultar el historial.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno al consultar el historial de predicciones."
        )


@router.post(
    "",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK
)
def create_prediction(
    data: EmployeeFeatures,
    db: Session = Depends(get_db)
):
    """
    Realiza una predicción individual de rotación laboral y la persiste en Neon.
    """

    try:
        return predict_employee(data, db)

    except EmployeeNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error)
        )

    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(error)
        )

    except (OperationalError, InterfaceError):
        logger.exception("Error de conexión con la base de datos Neon.")

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No fue posible conectar con la base de datos."
        )

    except SQLAlchemyError:
        logger.exception("Error al guardar la predicción en la base de datos.")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error al guardar la predicción en la base de datos."
        )

    except Exception:
        logger.exception("Error en el endpoint de predicciones.")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno al realizar la predicción."
        )

@router.post(
    "/explain",
    response_model=SHAPExplanationResponse
)
def explain_prediction(data: EmployeeFeatures):
    """
    Genera una explicación SHAP individual
    de la clasificación del empleado.
    """

    try:
        return explain_employee(data)

    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(error)
        )

    except Exception:
        logger.exception(
            "Error al generar la explicación SHAP."
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno al generar la explicación."
        )