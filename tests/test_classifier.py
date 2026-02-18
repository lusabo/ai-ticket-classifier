from pathlib import Path
import pytest
from ticket_ai.services.classifier import TicketClassifier

MODEL_PATH = Path("models/ticket_clf.joblib")

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Modelo não encontrado em models/ticket_clf.joblib (rode o treino antes).",
)

def test_model_loads():
    # Verifica se o modelo é carregado corretamente.
    clf = TicketClassifier()
    assert clf is not None

def test_prediction_returns_string():
    # Verifica se a predição retorna uma categoria válida (string não vazia).
    clf = TicketClassifier()
    categoria = clf.predict("Quero cancelar minha assinatura")
    assert isinstance(categoria, str)
    assert len(categoria) > 0

def test_prediction_probabilities_sum_to_one():
    # Verifica se probabilidades somam aproximadamente 1 (quando disponível).
    clf = TicketClassifier()
    probas = clf.predict_proba("Produto chegou com defeito")
    total = sum(probas.values())
    assert pytest.approx(total, 0.01) == 1.0