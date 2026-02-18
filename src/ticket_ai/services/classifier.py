import os
from joblib import load
from pathlib import Path

DEFAULT_MODEL_PATH = Path(os.getenv("TICKET_AI_MODEL_PATH", "models/ticket_clf.joblib"))


class TicketClassifier:
    """Serviço de classificação de tickets."""

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Modelo não encontrado em '{model_path}'. "
                "Execute o pipeline de treinamento."
            )
        self.model = load(model_path)

    def predict(self, texto: str) -> str:
        """Classifica ticket e retorna categoria."""
        return str(self.model.predict([texto])[0])

    def predict_proba(self, texto: str) -> dict[str, float]:
        """Retorna probabilidades por categoria."""
        probas = self.model.predict_proba([texto])[0]
        classes = self.model.classes_
        return {str(cls): float(p) for cls, p in zip(classes, probas)}