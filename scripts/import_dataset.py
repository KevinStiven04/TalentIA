
from pathlib import Path

import pandas as pd

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert

from app.core.database import Base, engine, SessionLocal
from app.models import Employee, PredictionHistory


# Ruta principal del backend.
BASE_DIR = Path(__file__).resolve().parents[1]

# Ruta del CSV.
DATASET_PATH = (
    BASE_DIR
    / "data"
    / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
)


# Columnas que necesitamos del dataset.
REQUIRED_COLUMNS = [
    "EmployeeNumber",
    "Department",
    "JobRole",
    "MonthlyIncome",
    "JobSatisfaction",
    "EnvironmentSatisfaction",
    "OverTime",
    "WorkLifeBalance",
    "YearsAtCompany",
    "Attrition"
]


def import_dataset():
    """
    Importa los empleados del dataset IBM HR Analytics
    a la tabla employees de Neon PostgreSQL.
    """

    # 1. Comprobar que exista el archivo.
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset en: {DATASET_PATH}"
        )

    print("Leyendo dataset...")

    # 2. Leer el CSV.
    df = pd.read_csv(DATASET_PATH)

    # 3. Comprobar que existan las columnas necesarias.
    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Faltan columnas en el CSV: {missing_columns}"
        )

    # 4. Seleccionar únicamente las columnas necesarias.
    df = df[REQUIRED_COLUMNS].copy()

    # 5. Eliminar registros duplicados por identificador.
    df = df.drop_duplicates(
        subset=["EmployeeNumber"],
        keep="last"
    )

    # 6. Comprobar valores faltantes.
    if df[REQUIRED_COLUMNS].isnull().any().any():
        raise ValueError(
            "El dataset contiene valores vacíos en las "
            "columnas seleccionadas."
        )

    # 7. Preparar los registros para PostgreSQL.
    records = []

    for _, row in df.iterrows():

        employee_number = int(row["EmployeeNumber"])

        records.append({
            "employee_number": employee_number,

            # Nombre identificativo generado.
            # No representa el nombre real de una persona.
            "display_name": f"Empleado #{employee_number}",

            "department": str(row["Department"]),
            "job_role": str(row["JobRole"]),

            "monthly_income": float(row["MonthlyIncome"]),

            "job_satisfaction": int(
                row["JobSatisfaction"]
            ),

            "environment_satisfaction": int(
                row["EnvironmentSatisfaction"]
            ),

            "over_time": str(row["OverTime"]),

            "work_life_balance": int(
                row["WorkLifeBalance"]
            ),

            "years_at_company": int(
                row["YearsAtCompany"]
            ),

            "observed_attrition": str(row["Attrition"]),

            "source": "ibm_dataset"
        })

    # 8. Crear las tablas si todavía no existen.
    Base.metadata.create_all(bind=engine)

    # 9. Insertar o actualizar empleados.
    # Si el EmployeeNumber ya existe, no se duplica.
    with SessionLocal() as session:

        statement = insert(Employee).values(records)

        update_fields = [
            "display_name",
            "department",
            "job_role",
            "monthly_income",
            "job_satisfaction",
            "environment_satisfaction",
            "over_time",
            "work_life_balance",
            "years_at_company",
            "observed_attrition",
            "source"
        ]

        statement = statement.on_conflict_do_update(
            index_elements=[Employee.employee_number],
            set_={
                field: getattr(statement.excluded, field)
                for field in update_fields
            }
        )

        try:
            session.execute(statement)
            session.commit()

        except Exception:
            session.rollback()
            raise

        # 10. Consultar el total de empleados almacenados.
        total_employees = session.scalar(
            select(func.count()).select_from(Employee)
        )

    print("\nImportación completada correctamente.")
    print(f"Registros procesados: {len(records)}")
    print(f"Empleados almacenados: {total_employees}")


if __name__ == "__main__":
    import_dataset()
