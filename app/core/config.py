
import os
from pathlib import Path

from dotenv import load_dotenv


# Obtener la ruta principal del backend.
BASE_DIR = Path(__file__).resolve().parents[2]

# Ubicación del archivo .env.
ENV_FILE = BASE_DIR / ".env"

# Cargar las variables de entorno.
load_dotenv(dotenv_path=ENV_FILE)


class Settings:
    """
    Configuración general de TalentIA.
    """

    # Información de la aplicación.
    APP_NAME = "TalentIA API"
    APP_VERSION = "1.0.0"
    API_V1_PREFIX = "/api/v1"

    DATABASE_URL = os.getenv(
        "DATABASE_URL",
        ""
    ).strip()
    # Configuración de Gemini.
    GEMINI_API_KEY = os.getenv(
        "GEMINI_API_KEY", ""
    ).strip()

    GEMINI_MODEL = os.getenv(
        "GEMINI_MODEL",
        "gemini-3.5-flash-lite"
    ).strip()

    def validate_gemini_config(self):
        """
        Comprueba que la configuración de Gemini esté completa.
        """

        if not self.GEMINI_API_KEY:
            raise ValueError(
                "No se encontró GEMINI_API_KEY en el archivo .env"
            )

        if not self.GEMINI_MODEL:
            raise ValueError(
                "No se encontró GEMINI_MODEL en el archivo .env"
            )


# Instancia global de configuración.
settings = Settings()
