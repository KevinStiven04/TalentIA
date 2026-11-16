
from pathlib import Path
from functools import lru_cache
import logging
import joblib

logger = logging.getLogger(__name__)

# Ruta raíz del backend
BASE_DIR = Path(__file__).resolve().parents[2]

# Ubicación del modelo entrenado en Google Colab
MODEL_PATH = BASE_DIR / "artifacts" / "modelo_rotacion.joblib"


@lru_cache(maxsize=1)
def load_model():
    """
    Carga el modelo de Machine Learning una sola vez.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en: {MODEL_PATH}"
        )

    logger.info("Cargando modelo de TalentIA...")

    model = joblib.load(MODEL_PATH)

    # Comprobar que el modelo tenga los métodos necesarios
    if not hasattr(model, "predict"):
        raise ValueError(
            "El archivo no contiene un modelo válido."
        )

    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "El modelo no permite calcular probabilidades."
        )

    logger.info("Modelo de TalentIA cargado correctamente.")

    return model


def get_model():
    """
    Obtiene el modelo cargado.
    Si ya está cargado, reutiliza la misma instancia.
    """

    return load_model()
