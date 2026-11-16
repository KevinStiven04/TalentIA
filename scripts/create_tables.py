
from app.core.database import Base, engine

# Importar los modelos para registrarlos en Base.metadata.
from app.models import Employee, PredictionHistory


def create_tables():
    """
    Crea las tablas de TalentIA en Neon PostgreSQL.
    """

    Base.metadata.create_all(bind=engine)

    print("Tablas creadas correctamente en Neon.")
    print(" - employees")
    print(" - prediction_history")


if __name__ == "__main__":
    create_tables()
