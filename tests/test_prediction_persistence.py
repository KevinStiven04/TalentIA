from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, func, select

from app.core.database import SessionLocal
from app.models.employee import Employee
from app.models.prediction_history import PredictionHistory
from app.services.prediction_service import calculate_employee_prediction
from main import app


client = TestClient(app)


@pytest.fixture
def cleanup_test_predictions():
    """
    Registra el ID máximo actual de prediction_history antes de la prueba
    y elimina únicamente los registros creados durante el test al finalizar.
    """
    with SessionLocal() as db:
        initial_max_id = db.scalar(
            select(func.coalesce(func.max(PredictionHistory.id), 0))
        ) or 0

    created_ids: list[int] = []
    yield created_ids

    with SessionLocal() as db:
        db.execute(
            delete(PredictionHistory).where(
                PredictionHistory.id > initial_max_id
            )
        )
        db.commit()


def _get_employee_payload(employee: Employee, include_number: bool = True) -> dict:
    payload = {
        "department": employee.department,
        "job_role": employee.job_role,
        "monthly_income": employee.monthly_income,
        "job_satisfaction": employee.job_satisfaction,
        "environment_satisfaction": employee.environment_satisfaction,
        "over_time": employee.over_time,
        "work_life_balance": employee.work_life_balance,
        "years_at_company": employee.years_at_company,
    }
    if include_number:
        payload["employee_number"] = employee.employee_number
    return payload


def test_a_existing_employee_prediction_persists_with_internal_id(
    cleanup_test_predictions,
):
    """
    Prueba A: empleado existente.
    Verifica HTTP 200, inserción en prediction_history y uso de Employee.id interno.
    """
    with SessionLocal() as db:
        employee = db.scalar(
            select(Employee)
            .where(Employee.id != Employee.employee_number)
            .order_by(Employee.id.asc())
            .limit(1)
        )
        assert employee is not None
        before_count = db.scalar(select(func.count(PredictionHistory.id))) or 0

    payload = _get_employee_payload(employee, include_number=True)
    response = client.post("/api/v1/predictions", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["predicted_attrition"] in ("Yes", "No")
    assert 0.0 <= body["probability_yes"] <= 1.0
    assert body["threshold"] == 0.5

    with SessionLocal() as db:
        after_count = db.scalar(select(func.count(PredictionHistory.id))) or 0
        assert after_count == before_count + 1

        record = db.scalar(
            select(PredictionHistory)
            .order_by(PredictionHistory.id.desc())
            .limit(1)
        )
        assert record is not None
        assert record.employee_id == employee.id
        assert record.employee_id != employee.employee_number
        assert record.department == employee.department
        assert record.job_role == employee.job_role
        assert record.predicted_attrition == body["predicted_attrition"]
        assert record.probability_yes == pytest.approx(body["probability_yes"])
        assert record.threshold == pytest.approx(body["threshold"])


def test_b_manual_prediction_persists_with_null_employee_id(
    cleanup_test_predictions,
):
    """
    Prueba B: predicción manual sin employee_number.
    Verifica HTTP 200 e inserción con employee_id = NULL.
    """
    with SessionLocal() as db:
        before_count = db.scalar(select(func.count(PredictionHistory.id))) or 0

    payload = {
        "department": "Research & Development",
        "job_role": "Research Scientist",
        "monthly_income": 4500,
        "job_satisfaction": 2,
        "environment_satisfaction": 2,
        "over_time": "Yes",
        "work_life_balance": 2,
        "years_at_company": 3,
    }
    response = client.post("/api/v1/predictions", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["predicted_attrition"] in ("Yes", "No")
    assert 0.0 <= body["probability_yes"] <= 1.0

    with SessionLocal() as db:
        after_count = db.scalar(select(func.count(PredictionHistory.id))) or 0
        assert after_count == before_count + 1

        record = db.scalar(
            select(PredictionHistory)
            .order_by(PredictionHistory.id.desc())
            .limit(1)
        )
        assert record is not None
        assert record.employee_id is None
        assert record.department == payload["department"]
        assert record.job_role == payload["job_role"]
        assert record.predicted_attrition == body["predicted_attrition"]
        assert record.probability_yes == pytest.approx(body["probability_yes"])


def test_c_nonexistent_employee_returns_404_and_does_not_persist(
    cleanup_test_predictions,
):
    """
    Prueba C: empleado inexistente.
    Verifica error controlado (404) y cero inserciones en prediction_history.
    """
    nonexistent_number = 99999999

    with SessionLocal() as db:
        existing = db.scalar(
            select(Employee).where(
                Employee.employee_number == nonexistent_number
            )
        )
        assert existing is None
        before_count = db.scalar(select(func.count(PredictionHistory.id))) or 0

    payload = {
        "employee_number": nonexistent_number,
        "department": "Sales",
        "job_role": "Sales Executive",
        "monthly_income": 5200,
        "job_satisfaction": 3,
        "environment_satisfaction": 3,
        "over_time": "No",
        "work_life_balance": 3,
        "years_at_company": 4,
    }
    response = client.post("/api/v1/predictions", json=payload)

    assert response.status_code == 404
    assert "detail" in response.json()

    with SessionLocal() as db:
        after_count = db.scalar(select(func.count(PredictionHistory.id))) or 0
        assert after_count == before_count


def test_d_shap_and_gemini_flow_no_duplicate_records(
    cleanup_test_predictions,
):
    """
    Prueba D: flujo completo predicción -> SHAP -> Gemini sin duplicar registros.
    """
    with SessionLocal() as db:
        employee = db.scalar(
            select(Employee)
            .where(Employee.employee_number == 4)
            .limit(1)
        )
        assert employee is not None
        initial_count = db.scalar(select(func.count(PredictionHistory.id))) or 0

    payload = _get_employee_payload(employee, include_number=True)

    # 1. POST /api/v1/predictions -> crea exactamente 1 registro.
    pred_response = client.post("/api/v1/predictions", json=payload)
    assert pred_response.status_code == 200

    with SessionLocal() as db:
        after_pred_count = db.scalar(select(func.count(PredictionHistory.id))) or 0
        assert after_pred_count == initial_count + 1

    # 2. POST /api/v1/predictions/explain -> funciona con employee_number y no inserta.
    shap_response = client.post("/api/v1/predictions/explain", json=payload)
    assert shap_response.status_code == 200
    shap_body = shap_response.json()
    assert shap_body["explained_class"] == "Yes"
    assert len(shap_body["features"]) == 8
    assert all(
        f["feature_key"] != "employee_number" for f in shap_body["features"]
    )

    with SessionLocal() as db:
        after_shap_count = db.scalar(select(func.count(PredictionHistory.id))) or 0
        assert after_shap_count == initial_count + 1

    # 3. POST /api/v1/gemini/analyze -> funciona y NO duplica el registro.
    gemini_response = client.post("/api/v1/gemini/analyze", json=payload)
    assert gemini_response.status_code == 200
    gemini_body = gemini_response.json()
    assert gemini_body["interpretation"]
    assert isinstance(gemini_body["factors_toward_yes"], list)
    assert isinstance(gemini_body["factors_toward_no"], list)
    assert isinstance(gemini_body["recommendations"], list)
    assert gemini_body["disclaimer"]

    with SessionLocal() as db:
        after_gemini_count = db.scalar(select(func.count(PredictionHistory.id))) or 0
        assert after_gemini_count == initial_count + 1

    # Verificar que calculate_employee_prediction da el mismo resultado con y sin employee_number.
    from app.schemas.employee import EmployeeFeatures

    calc_with_id = calculate_employee_prediction(EmployeeFeatures(**payload))
    payload_without_id = _get_employee_payload(employee, include_number=False)
    calc_without_id = calculate_employee_prediction(
        EmployeeFeatures(**payload_without_id)
    )
    assert calc_with_id.probability_yes == pytest.approx(
        calc_without_id.probability_yes
    )
    assert calc_with_id.predicted_attrition == calc_without_id.predicted_attrition


def test_e_analytics_dashboard_updates_accurately(
    cleanup_test_predictions,
):
    """
    Prueba E: Analytics dashboard.
    Verifica:
    - Múltiples predicciones para el mismo empleado cuentan una sola vez en estado actual.
    - Predicciones manuales (employee_id NULL) se contabilizan como análisis y en tendencia mensual.
    - Promedio de probability_yes exacto.
    - Evolución mensual considera todos los registros históricos.
    - high_risk_employees muestra empleados registrados con riesgo Yes.
    - Rotación observada del dataset permanece independiente.
    """
    baseline = client.get("/api/v1/analytics/dashboard")
    assert baseline.status_code == 200
    baseline_data = baseline.json()

    with SessionLocal() as db:
        emp_high = db.scalar(
            select(Employee).where(Employee.employee_number == 4).limit(1)
        )
        emp_low = db.scalar(
            select(Employee).where(Employee.employee_number == 7).limit(1)
        )
        assert emp_high is not None and emp_low is not None

    # 1. Predecir emp_high dos veces (simular re-evaluación).
    p_high_1 = _get_employee_payload(emp_high, include_number=True)
    r1 = client.post("/api/v1/predictions", json=p_high_1)
    assert r1.status_code == 200

    p_high_2 = dict(p_high_1)
    p_high_2["job_satisfaction"] = 1
    p_high_2["environment_satisfaction"] = 1
    p_high_2["work_life_balance"] = 1
    r2 = client.post("/api/v1/predictions", json=p_high_2)
    assert r2.status_code == 200

    # 2. Predecir emp_low una vez.
    p_low = _get_employee_payload(emp_low, include_number=True)
    r3 = client.post("/api/v1/predictions", json=p_low)
    assert r3.status_code == 200

    # 3. Predicción manual (sin employee_number).
    p_manual = {
        "department": "Sales",
        "job_role": "Sales Representative",
        "monthly_income": 2100,
        "job_satisfaction": 1,
        "environment_satisfaction": 1,
        "over_time": "Yes",
        "work_life_balance": 1,
        "years_at_company": 1,
    }
    r4 = client.post("/api/v1/predictions", json=p_manual)
    assert r4.status_code == 200

    dashboard = client.get("/api/v1/analytics/dashboard")
    assert dashboard.status_code == 200
    data = dashboard.json()

    summary = data["summary"]
    # 4 predicciones nuevas, pero emp_high solo cuenta una vez en estado actual -> +3 análisis.
    assert (
        summary["employees_analyzed"]
        == baseline_data["summary"]["employees_analyzed"] + 3
    )
    assert (
        summary["prediction_records"]
        == baseline_data["summary"]["prediction_records"] + 4
    )
    assert (
        summary["observed_attrition_rate"]
        == baseline_data["summary"]["observed_attrition_rate"]
    )
    assert (
        summary["dataset_employees"]
        == baseline_data["summary"]["dataset_employees"]
    )

    # Verificar promedio exacto si la tabla empezó vacía en baseline.
    if baseline_data["summary"]["employees_analyzed"] == 0:
        probs = [
            r2.json()["probability_yes"],
            r3.json()["probability_yes"],
            r4.json()["probability_yes"],
        ]
        expected_avg = round((sum(probs) / len(probs)) * 100, 2)
        assert summary["average_probability_yes"] == pytest.approx(expected_avg)

        expected_high = sum(
            1
            for resp in (r2.json(), r3.json(), r4.json())
            if resp["predicted_attrition"] == "Yes"
        )
        assert summary["high_risk"] == expected_high
        assert summary["high_risk_percentage"] == pytest.approx(
            round((expected_high / 3) * 100, 2)
        )

    # Verificar tendencia mensual del mes UTC actual (debe incluir las 4 predicciones).
    current_ym = datetime.now(timezone.utc).strftime("%Y-%m")
    current_month_entry = next(
        m for m in data["monthly_trend"] if m["month"] == current_ym
    )
    baseline_month_entry = next(
        m for m in baseline_data["monthly_trend"] if m["month"] == current_ym
    )
    assert (
        current_month_entry["analyses"]
        == baseline_month_entry["analyses"] + 4
    )

    # Verificar que emp_high (si dio Yes) figura una sola vez en high_risk_employees.
    if r2.json()["predicted_attrition"] == "Yes":
        emp_numbers = [
            item["employee_number"] for item in data["high_risk_employees"]
        ]
        assert emp_high.employee_number in emp_numbers
        assert emp_numbers.count(emp_high.employee_number) == 1
        # Las predicciones manuales no deben aparecer en high_risk_employees.
        assert all(
            item["employee_number"] is not None
            for item in data["high_risk_employees"]
        )
