
from sqlalchemy import (
    Integer,
    String,
    Float,
    DateTime,
    func
)

from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class Employee(Base):

    __tablename__ = "employees"



    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True
    )

    employee_number: Mapped[int] = mapped_column(
        Integer,
        unique=True,
        nullable=False,
        index=True
    )

    display_name: Mapped[str] = mapped_column(
        String(80),
        nullable=False
    )

    department: Mapped[str] = mapped_column(
        String(100),
        nullable=False
    )

    job_role: Mapped[str] = mapped_column(
        String(100),
        nullable=False
    )

    monthly_income: Mapped[float] = mapped_column(
        Float,
        nullable=False
    )

    job_satisfaction: Mapped[int] = mapped_column(
        Integer,
        nullable=False
    )

    environment_satisfaction: Mapped[int] = mapped_column(
        Integer,
        nullable=False
    )

    over_time: Mapped[str] = mapped_column(
        String(3),
        nullable=False
    )

    work_life_balance: Mapped[int] = mapped_column(
        Integer,
        nullable=False
    )

    years_at_company: Mapped[int] = mapped_column(
        Integer,
        nullable=False
    )

    observed_attrition: Mapped[str | None] = mapped_column(
        String(3),
        nullable=True
    )

    source: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="ibm_dataset"
    )

    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

    # Relación con el historial de predicciones.
    predictions: Mapped[list["PredictionHistory"]] = relationship(
        back_populates="employee"
    )
