
import logging
import math

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, func, or_, cast, String
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.employee import Employee
from app.schemas.employee_directory import EmployeeDirectoryResponse


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/employees",
    tags=["Employees"]
)


@router.get(
    "",
    response_model=EmployeeDirectoryResponse
)
def get_employees(
    search: str | None = Query(
        default=None,
        description="Buscar por número, nombre, departamento o cargo"
    ),
    department: str | None = Query(
        default=None,
        description="Filtrar por departamento"
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
        description="Cantidad de empleados por página"
    ),
    db: Session = Depends(get_db)
):
    """
    Obtiene el directorio de empleados con búsqueda,
    filtros y paginación.
    """

    try:
        # 1. Construir los filtros.
        filters = []

        # 2. Aplicar búsqueda general.
        if search and search.strip():

            search_term = f"%{search.strip()}%"

            filters.append(
                or_(
                    Employee.display_name.ilike(search_term),
                    cast(
                        Employee.employee_number,
                        String
                    ).ilike(search_term),
                    Employee.department.ilike(search_term),
                    Employee.job_role.ilike(search_term)
                )
            )

        # 3. Filtrar por departamento.
        if department and department.strip():

            filters.append(
                Employee.department.ilike(
                    department.strip()
                )
            )

        # 4. Consultar el total de empleados encontrados.
        total = db.scalar(
            select(func.count(Employee.id)).where(*filters)
        ) or 0

        # 5. Calcular el total de páginas.
        total_pages = math.ceil(total / page_size)

        # 6. Consultar los empleados de la página solicitada.
        statement = (
            select(Employee)
            .where(*filters)
            .order_by(Employee.employee_number.asc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )

        employees = db.scalars(statement).all()

        # 7. Devolver la respuesta paginada.
        return EmployeeDirectoryResponse(
            items=employees,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages
        )

    except Exception:
        logger.exception(
            "Error al consultar el directorio de empleados."
        )
        raise
