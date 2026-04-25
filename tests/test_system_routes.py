from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root_route_serves_frontend() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Talent Scout" in response.text


def test_api_info_route_returns_service_metadata() -> None:
    response = client.get("/api")
    assert response.status_code == 200
    payload = response.json()
    assert payload["stage"] == "step_9_api_ready"
    assert payload["match_url"] == "/api/v1/match"
    assert payload["stream_match_url"] == "/api/v1/match/stream"


def test_health_route() -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["app_name"] == "AI-Powered Talent Scouting & Engagement Agent"
