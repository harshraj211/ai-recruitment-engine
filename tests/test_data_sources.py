import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.candidate_store import load_candidates, use_local_dataset


@pytest.fixture(autouse=True)
def reset_candidate_source():
    use_local_dataset()
    yield
    use_local_dataset()


def test_data_source_status_defaults_to_local_dataset() -> None:
    client = TestClient(app)
    response = client.get("/api/v1/data-source")

    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "local"
    assert payload["candidate_count"] == 20


def test_mock_candidates_endpoint_returns_candidate_data() -> None:
    client = TestClient(app)
    response = client.get("/api/v1/mock-candidates")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 20
    assert {"id", "full_name", "skills", "expected_salary_usd"}.issubset(payload[0])


def test_upload_json_replaces_active_candidate_dataset() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/v1/data-source/upload",
        json={
            "candidates": [
                {
                    "name": "Neha Kapoor",
                    "role": "Backend Engineer",
                    "skills": ["Python", "FastAPI", "PostgreSQL"],
                    "experience": 4,
                    "salary": 70000,
                },
                {
                    "name": "Arjun Rao",
                    "role": "ML Engineer",
                    "skills": "Python, PyTorch, AWS",
                    "experience": "6",
                    "salary": "95k",
                },
            ]
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "upload"
    assert payload["candidate_count"] == 2
    assert [candidate.full_name for candidate in load_candidates()] == ["Neha Kapoor", "Arjun Rao"]
    assert load_candidates()[1].expected_salary_usd == 95000


def test_upload_json_validates_required_fields() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/v1/data-source/upload",
        json={"candidates": [{"name": "Missing Salary", "skills": ["Python"], "experience": 3}]},
    )

    assert response.status_code == 422
    assert "salary" in response.json()["detail"]
