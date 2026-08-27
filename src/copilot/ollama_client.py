from ollama import Client

from src.config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)


# =========================================================
# 1. Ollama client
# =========================================================

OLLAMA_CLIENT = Client(
    host=OLLAMA_BASE_URL
)


# =========================================================
# 2. Chat helper
# =========================================================

def chat_with_ollama(
    messages,
    tools=None,
):
    """
    Send a non-streaming chat request to Ollama.

    The Ollama runtime may run locally on the host
    while FastAPI runs either locally or inside
    a Docker container.

    Tool schemas may be supplied to allow the model
    to request explicitly allowlisted backend tools.
    """

    kwargs = {
        "model": OLLAMA_MODEL,

        "messages": messages,

        "stream": False,

        # Keep context controlled on a
        # 24GB Apple Silicon development machine.
        "options": {
            "num_ctx": 8192,
            "temperature": 0.1,
        },
    }

    if tools:

        kwargs[
            "tools"
        ] = tools

    response = (
        OLLAMA_CLIENT.chat(
            **kwargs
        )
    )

    return response