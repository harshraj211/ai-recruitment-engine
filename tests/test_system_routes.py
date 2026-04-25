from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root_route() -> None:
    response = client.get("/")
    assert response.status_code == 200
    payload = response.json()
    assert payload["stage"] == "step_9_api_ready"
    assert payload["match_url"] == "/api/v1/match"


def test_health_route() -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["app_name"] == "AI-Powered Talent Scouting & Engagement Agent"
