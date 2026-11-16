
from sqlalchemy import (
    Integer,
    String,
    Float,
    DateTime,
    ForeignKey,
    func
)

from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class PredictionHistory(Base):

    __tablename__ = "prediction_history"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True
    )

    employee_id: Mapped[int | None] = mapped_column(
        ForeignKey("employees.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )

    # Guardamos el departamento y el cargo como fotografía
    # de los datos usados en esa predicción.
    department: Mapped[str] = mapped_column(
        String(100),
        nullable=False
    )

    job_role: Mapped[str] = mapped_column(
        String(100),
        nullable=False
    )

    predicted_attrition: Mapped[str] = mapped_column(
        String(3),
        nullable=False
    )

    probability_yes: Mapped[float] = mapped_column(
        Float,
        nullable=False
    )

    threshold: Mapped[float] = mapped_column(
        Float,
        nullable=False
    )

    created_at: Mapped[DateTime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

    employee: Mapped["Employee | None"] = relationship(
        back_populates="predictions"
    )
