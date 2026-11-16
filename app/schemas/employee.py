
from pydantic import BaseModel, Field
from typing import Literal


class EmployeeFeatures(BaseModel):
    """
    Características de un empleado utilizadas
    para predecir la rotación laboral.
    """

    employee_number: int | None = Field(
        default=None,
        description=(
            "Número identificador del empleado en la base de datos "
            "(opcional para predicciones manuales)"
        )
    )

    department: str = Field(
        ...,
        min_length=1,
        description="Departamento del empleado"
    )

    job_role: str = Field(
        ...,
        min_length=1,
        description="Cargo del empleado"
    )

    monthly_income: float = Field(
        ...,
        gt=0,
        description="Ingreso mensual del empleado"
    )

    job_satisfaction: int = Field(
        ...,
        ge=1,
        le=4,
        description="Satisfacción laboral (1 a 4)"
    )

    environment_satisfaction: int = Field(
        ...,
        ge=1,
        le=4,
        description="Satisfacción con el entorno laboral (1 a 4)"
    )

    over_time: Literal["Yes", "No"] = Field(
        ...,
        description="Indica si el empleado realiza horas extras"
    )

    work_life_balance: int = Field(
        ...,
        ge=1,
        le=4,
        description="Equilibrio entre vida laboral y personal (1 a 4)"
    )

    years_at_company: int = Field(
        ...,
        ge=0,
        description="Años de antigüedad en la empresa"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "employee_number": 1001,
                    "department": "Research & Development",
                    "job_role": "Research Scientist",
                    "monthly_income": 4500,
                    "job_satisfaction": 2,
                    "environment_satisfaction": 2,
                    "over_time": "Yes",
                    "work_life_balance": 2,
                    "years_at_company": 3
                }
            ]
        }
    }
