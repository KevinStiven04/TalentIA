
from fastapi import APIRouter


router = APIRouter(
    prefix="/health",
    tags=["Health"]
)


@router.get("")
def health_check():
    """
    Comprueba que TalentIA API esté funcionando.
    """

    return {
        "status": "ok",
        "message": "TalentIA API funcionando"
    }
