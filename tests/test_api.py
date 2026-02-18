import pytest
from fastapi.testclient import TestClient
from ticket_ai.api.main import app

client = TestClient(app)

@pytest.fixture(autouse=True)
def mock_llm(monkeypatch):
    # Força o caminho "LLM habilitado" para o endpoint chamar gerar_resposta()
    # (que está mockado abaixo).
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "ticket_ai.api.main.gerar_resposta",
        lambda texto, categoria: "Resposta mockada",
    )

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "timestamp_utc" in data

def test_predict_endpoint_contract():
    payload = {"texto": "Quero cancelar minha assinatura"}
    response = client.post("/predict", json=payload)

    # Em ambiente de testes, o modelo pode não existir.
    assert response.status_code in (200, 503)

    if response.status_code == 200:
        data = response.json()
        assert "categoria" in data
        assert "resposta" in data
        assert data["resposta"] == "Resposta mockada"

def test_predict_rejects_too_short_text():
    payload = {"texto": "  a "}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # validação Pydantic