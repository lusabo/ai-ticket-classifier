import os
import time
import random
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_client: OpenAI | None = None

def _get_client() -> OpenAI:
    global _client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY não configurada.")
    if _client is None:
        _client = OpenAI(api_key=api_key)
    return _client

def resposta_fallback(categoria: str) -> str:
    """
    Resposta padrão para quando o LLM estiver desabilitado ou falhar.
    Mantém o serviço funcional sem depender do provedor externo.
    """
    return (
        "Obrigado por entrar em contato! "
        f"Identificamos seu ticket como '{categoria}'. "
        "Nossa equipe vai analisar e retornar com orientações em breve."
    )

def gerar_resposta(texto: str, categoria: str) -> str:
    """
    Gera resposta profissional usando LLM.
    - Reutiliza client global
    - Timeout para evitar pendurar request
    - Retries com backoff/jitter para reduzir falhas transitórias
    """
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    client = _get_client()

    system_msg = "Você é um atendente profissional e empático."
    user_msg = (
        f"Categoria do ticket: {categoria}\n"
        f"Mensagem do cliente: {texto}\n\n"
        "Gere uma resposta educada, objetiva e profissional.\n"
        "Use português do Brasil.\n"
        "Limite-se a 3-4 frases."
    )

    max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "2"))
    backoff_base = float(os.getenv("OPENAI_RETRY_BACKOFF_BASE_SECONDS", "0.5"))
    backoff_max = float(os.getenv("OPENAI_RETRY_BACKOFF_MAX_SECONDS", "4.0"))

    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.7,
                max_tokens=300,
                timeout=30,
            )
            content = response.choices[0].message.content
            return content.strip() if content else ""
        except Exception as e:
            last_err = e
            if attempt >= max_retries:
                break
            sleep_s = min(backoff_max, backoff_base * (2 ** attempt))
            sleep_s = sleep_s * (0.8 + 0.4 * random.random())  # jitter ~ [0.8x .. 1.2x]
            time.sleep(sleep_s)

    raise RuntimeError("Falha ao gerar resposta via LLM (após retries).") from last_err