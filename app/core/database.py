
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.core.config import settings


# Comprobar que exista la URL de conexión.
if not settings.DATABASE_URL:
    raise ValueError(
        "No se encontró DATABASE_URL en el archivo .env"
    )


# Adaptar la URL para utilizar Psycopg 3.
database_url = settings.DATABASE_URL

if database_url.startswith("postgresql://"):
    database_url = database_url.replace(
        "postgresql://",
        "postgresql+psycopg://",
        1
    )

elif database_url.startswith("postgres://"):
    database_url = database_url.replace(
        "postgres://",
        "postgresql+psycopg://",
        1
    )


# Crear el motor de conexión.
engine = create_engine(
    database_url,
    pool_pre_ping=True,
    pool_recycle=300
)


# Crear la fábrica de sesiones.
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False
)


# Clase base de los modelos de SQLAlchemy.
class Base(DeclarativeBase):
    pass


# Dependencia para obtener sesiones de base de datos.
def get_db():
    db = SessionLocal()

    try:
        yield db

    finally:
        db.close()
