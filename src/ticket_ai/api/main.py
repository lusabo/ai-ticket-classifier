from fastapi import FastAPI, HTTPException
from datetime import datetime, timezone
import os
import logging
from contextlib import asynccontextmanager

from ticket_ai.schemas import PredictRequest, PredictResponse
from ticket_ai.services.classifier import TicketClassifier
from ticket_ai.services.llm import gerar_resposta, resposta_fallback

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("ticket_ai_api")

START_TIME = datetime.now(timezone.utc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicialização e teardown do app."""
    try:
        app.state.clf = TicketClassifier()
        logger.info("✅ Modelo carregado com sucesso.")
    except FileNotFoundError as e:
        app.state.clf = None
        logger.error(f"❌ Modelo não encontrado: {e}")
    except Exception as e:
        app.state.clf = None
        logger.exception(f"❌ Erro inesperado ao carregar modelo: {e}")
    yield

app = FastAPI(
    title="Ticket AI",
    description="API de classificação automática de tickets com geração de resposta via LLM.",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/")
def root():
    return {
        "service": "Ticket AI",
        "version": app.version,
        "docs": "/docs",
        "health": "/health",
    }

@app.get("/health")
def health_check():
    now = datetime.now(timezone.utc)
    uptime = (now - START_TIME).total_seconds()
    llm_configured = bool(os.getenv("OPENAI_API_KEY"))

    clf = getattr(app.state, "clf", None)
    if clf is None:
        status = "model_not_loaded"
    elif not llm_configured:
        status = "llm_not_configured"
    else:
        status = "healthy"

    return {
        "status": status,
        "uptime_seconds": uptime,
        "model_loaded": clf is not None,
        "llm_configured": llm_configured,
        "timestamp_utc": now.isoformat(),
    }

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    clf = getattr(app.state, "clf", None)
    if clf is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo não disponível. Execute o pipeline de treinamento.",
        )

    llm_configured = bool(os.getenv("OPENAI_API_KEY"))

    try:
        # 1) Classificação
        categoria = clf.predict(req.texto)

        # 2) Probabilidades (se disponível)
        probas = {}
        confidence = None
        try:
            probas = clf.predict_proba(req.texto)
            confidence = max(probas.values()) if probas else None
        except Exception:
            # Não derruba request se probas falhar (modelo pode mudar no futuro)
            probas = {}
            confidence = None


        # 3) Resposta (LLM ou fallback)
        if not llm_configured:
            resposta = resposta_fallback(categoria)
        else:
            try:
                resposta = gerar_resposta(req.texto, categoria)
            except RuntimeError:
                # Degradação graciosa (produção): não derruba a API por falha no provedor LLM
                logger.warning(
                    "Falha no LLM; retornando fallback.",
                    extra={"categoria": categoria, "texto_length": len(req.texto)},
                )
                resposta = resposta_fallback(categoria)

        # 4) Log do request (metadados seguros)
        logger.info(
            "Predição realizada",
            extra={
                "categoria": categoria,
                "texto_length": len(req.texto),
                "confidence": confidence,
            },
        )

        return PredictResponse(categoria=categoria, resposta=resposta)

    except Exception:
        # ✅ Importante para debug: stacktrace completo (sem PII)
        logger.exception(
            "Erro inesperado no /predict",
            extra={
                "texto_length": len(req.texto) if getattr(req, "texto", None) else None,
            },
        )
        raise HTTPException(status_code=500, detail="Erro interno no servidor.")