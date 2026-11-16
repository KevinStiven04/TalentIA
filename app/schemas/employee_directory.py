
from pydantic import BaseModel


class EmployeeDirectoryItem(BaseModel):
    """Información de un empleado para el directorio."""

    id: int
    employee_number: int
    display_name: str
    department: str
    job_role: str
    monthly_income: float
    years_at_company: int
    observed_attrition: str | None = None

    model_config = {
        "from_attributes": True
    }


class EmployeeDirectoryResponse(BaseModel):
    """Respuesta paginada del directorio de empleados."""

    items: list[EmployeeDirectoryItem]
    total: int
    page: int
    page_size: int
    total_pages: int
