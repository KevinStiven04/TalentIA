from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, func, select, text

from app.core.database import SessionLocal
from app.models.employee import Employee
from app.models.prediction_history import PredictionHistory
from main import app


client = TestClient(app)


@pytest.fixture
def isolated_history():
    """
    Mantiene la tabla limpia durante la prueba de historial y restaura
    eliminando únicamente los registros creados durante el test.
    """
    with SessionLocal() as db:
        initial_max_id = db.scalar(
            select(func.coalesce(func.max(PredictionHistory.id), 0))
        ) or 0

    yield initial_max_id

    with SessionLocal() as db:
        db.execute(
            delete(PredictionHistory).where(
                PredictionHistory.id > initial_max_id
            )
        )
        db.commit()


def _seed_predictions(initial_max_id: int) -> dict:
    """
    Inserta un conjunto controlado de registros de prueba con fechas distintas
    (empleados reales y predicciones manuales) y retorna sus metadatos.
    """
    now = datetime.now(timezone.utc)
    yesterday = now - timedelta(days=1)
    three_days_ago = now - timedelta(days=3)

    with SessionLocal() as db:
        emp4 = db.scalar(select(Employee).where(Employee.employee_number == 4))
        emp7 = db.scalar(select(Employee).where(Employee.employee_number == 7))
        emp10 = db.scalar(select(Employee).where(Employee.employee_number == 10))
        assert emp4 is not None and emp7 is not None and emp10 is not None

        records = [
            PredictionHistory(
                employee_id=emp4.id,
                department=emp4.department,
                job_role=emp4.job_role,
                predicted_attrition="Yes",
                probability_yes=0.662,
                threshold=0.5,
                created_at=now,
            ),
            PredictionHistory(
                employee_id=None,
                department="Sales",
                job_role="Sales Representative",
                predicted_attrition="Yes",
                probability_yes=0.715,
                threshold=0.5,
                created_at=yesterday,
            ),
            PredictionHistory(
                employee_id=emp7.id,
                department=emp7.department,
                job_role=emp7.job_role,
                predicted_attrition="No",
                probability_yes=0.145,
                threshold=0.5,
                created_at=yesterday - timedelta(hours=2),
            ),
            PredictionHistory(
                employee_id=emp10.id,
                department="Human Resources",
                job_role="Human Resources",
                predicted_attrition="No",
                probability_yes=0.225,
                threshold=0.5,
                created_at=three_days_ago,
            ),
        ]
        db.add_all(records)
        db.commit()
        for r in records:
            db.refresh(r)

        return {
            "now": now,
            "yesterday": yesterday,
            "three_days_ago": three_days_ago,
            "ids": [r.id for r in records],
            "emp4": emp4,
            "emp7": emp7,
            "emp10": emp10,
        }


def test_history_empty_and_no_matches(isolated_history):
    """Escenario 9: consultar historial vacío o sin coincidencias."""
    # Si la tabla está vacía, GET sin filtros devuelve 0 resultados.
    with SessionLocal() as db:
        existing_count = db.scalar(select(func.count(PredictionHistory.id))) or 0

    if existing_count == 0:
        resp_empty = client.get("/api/v1/predictions/history")
        assert resp_empty.status_code == 200
        body = resp_empty.json()
        assert body["items"] == []
        assert body["total"] == 0
        assert body["page"] == 1
        assert body["page_size"] == 10
        assert body["total_pages"] == 0

    # También comprobar búsqueda sin coincidencias.
    resp_no_match = client.get(
        "/api/v1/predictions/history",
        params={"search": "__SIN_COINCIDENCIAS_99999__"},
    )
    assert resp_no_match.status_code == 200
    assert resp_no_match.json()["items"] == []
    assert resp_no_match.json()["total"] == 0
    assert resp_no_match.json()["total_pages"] == 0


def test_history_unfiltered_search_filters_and_pagination(isolated_history):
    """
    Escenarios 1, 2, 3, 4, 5, 6, 7 y 8:
    - Sin filtros (ordenado por created_at DESC)
    - Búsqueda por employee_number, departamento y cargo
    - Filtro Yes / No
    - Filtro de fechas incluyendo día final sin hora
    - Paginación (page, page_size, total_pages)
    - Predicción manual con employee_id NULL
    """
    seeded = _seed_predictions(isolated_history)

    # 1. Consultar historial sin filtros (incluyendo parámetros vacíos como en el ejemplo).
    r1 = client.get(
        "/api/v1/predictions/history?search=&from=&to=&attrition=&page=1&page_size=10"
    )
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["total"] >= 4
    assert len(d1["items"]) >= 4
    # Verificar orden descendente por created_at y sin IDs duplicados.
    ids = [item["id"] for item in d1["items"]]
    assert len(ids) == len(set(ids))
    timestamps = [item["created_at"] for item in d1["items"]]
    assert timestamps == sorted(timestamps, reverse=True)

    # 2. Buscar por número de empleado ("4" / "10").
    r2 = client.get("/api/v1/predictions/history", params={"search": "10"})
    assert r2.status_code == 200
    assert any(item["employee_number"] == 10 for item in r2.json()["items"])

    # 3. Buscar por departamento y por cargo.
    r3_dept = client.get(
        "/api/v1/predictions/history", params={"search": "Human Resources"}
    )
    assert r3_dept.status_code == 200
    assert any(
        item["department"] == "Human Resources"
        for item in r3_dept.json()["items"]
    )

    r3_role = client.get(
        "/api/v1/predictions/history", params={"search": "Sales Representative"}
    )
    assert r3_role.status_code == 200
    assert any(
        item["job_role"] == "Sales Representative"
        for item in r3_role.json()["items"]
    )

    # 4. Filtrar por clasificación Yes.
    r4 = client.get("/api/v1/predictions/history", params={"attrition": "Yes"})
    assert r4.status_code == 200
    assert r4.json()["total"] >= 2
    assert all(
        item["predicted_attrition"] == "Yes" for item in r4.json()["items"]
    )

    # 5. Filtrar por clasificación No.
    r5 = client.get("/api/v1/predictions/history", params={"attrition": "No"})
    assert r5.status_code == 200
    assert r5.json()["total"] >= 2
    assert all(
        item["predicted_attrition"] == "No" for item in r5.json()["items"]
    )

    # 6. Filtrar por fechas sin hora (YYYY-MM-DD): debe incluir el día final completo.
    y_str = seeded["yesterday"].date().isoformat()
    n_str = seeded["now"].date().isoformat()
    r6 = client.get(
        "/api/v1/predictions/history",
        params={"from": y_str, "to": n_str},
    )
    assert r6.status_code == 200
    returned_ids = {item["id"] for item in r6.json()["items"]}
    # El registro de hace 3 días NO debe entrar; los de hoy y ayer SÍ deben entrar.
    assert seeded["ids"][0] in returned_ids
    assert seeded["ids"][1] in returned_ids
    assert seeded["ids"][2] in returned_ids
    assert seeded["ids"][3] not in returned_ids

    # 7. Cambiar de página (page=1 y page=2 con page_size=2).
    p1 = client.get(
        "/api/v1/predictions/history", params={"page": 1, "page_size": 2}
    )
    p2 = client.get(
        "/api/v1/predictions/history", params={"page": 2, "page_size": 2}
    )
    assert p1.status_code == 200 and p2.status_code == 200
    p1_ids = {item["id"] for item in p1.json()["items"]}
    p2_ids = {item["id"] for item in p2.json()["items"]}
    assert len(p1_ids) == 2
    assert len(p2_ids) == 2
    assert p1_ids.isdisjoint(p2_ids)
    assert p1.json()["total_pages"] >= 2

    # 8. Consultar predicción manual con employee_id NULL (sin inventar nombres).
    r8 = client.get("/api/v1/predictions/history", params={"search": "manual"})
    assert r8.status_code == 200
    manual_items = r8.json()["items"]
    assert len(manual_items) >= 1
    manual_row = next(
        item for item in manual_items if item["id"] == seeded["ids"][1]
    )
    assert manual_row["employee_id"] is None
    assert manual_row["employee_number"] is None
    assert manual_row["display_name"] is None
    assert manual_row["department"] == "Sales"
    assert manual_row["job_role"] == "Sales Representative"

    # Verificar contra SQL directo en Neon que el total coincida exactamente.
    with SessionLocal() as db:
        sql_total = db.execute(
            text("SELECT COUNT(*) FROM prediction_history")
        ).scalar()
        assert d1["total"] == sql_total


def test_new_prediction_appears_in_history_without_duplicates(isolated_history):
    """
    Escenarios 10 y 11:
    - Realizar una nueva predicción y comprobar que aparezca inmediatamente en el historial.
    - Ejecutar SHAP y Gemini sobre el mismo empleado sin crear duplicados.
    """
    payload = {
        "employee_number": 4,
        "department": "Research & Development",
        "job_role": "Laboratory Technician",
        "monthly_income": 2090,
        "job_satisfaction": 3,
        "environment_satisfaction": 4,
        "over_time": "Yes",
        "work_life_balance": 3,
        "years_at_company": 0,
    }

    before_hist = client.get("/api/v1/predictions/history")
    assert before_hist.status_code == 200
    before_total = before_hist.json()["total"]

    pred_resp = client.post("/api/v1/predictions", json=payload)
    assert pred_resp.status_code == 200

    # SHAP y Gemini no deben crear registros duplicados.
    explain_resp = client.post("/api/v1/predictions/explain", json=payload)
    assert explain_resp.status_code == 200
    gemini_resp = client.post("/api/v1/gemini/analyze", json=payload)
    assert gemini_resp.status_code == 200

    after_hist = client.get("/api/v1/predictions/history")
    assert after_hist.status_code == 200
    after_data = after_hist.json()
    assert after_data["total"] == before_total + 1

    newest = after_data["items"][0]
    assert newest["employee_number"] == 4
    assert newest["display_name"] == "Empleado #4"
    assert newest["department"] == "Research & Development"
    assert newest["job_role"] == "Laboratory Technician"
    assert newest["predicted_attrition"] == pred_resp.json()["predicted_attrition"]
    assert newest["probability_yes"] == pytest.approx(
        pred_resp.json()["probability_yes"]
    )
