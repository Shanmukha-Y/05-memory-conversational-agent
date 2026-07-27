"""Thin wrapper around the Ollama client: chat, embeddings, and JSON-mode
structured generation with Pydantic validation + one retry.

All live-LLM calls in the codebase go through this module, which keeps it
the single place that needs to handle Ollama's failure modes: the Ollama
python client raises `ollama.ResponseError` for HTTP-level errors, but a
slow/overloaded server (this one is shared with other builders) can also
surface as a bare socket `TimeoutError` that never gets wrapped -- that has
crashed at least one other agent's CLI, so we catch it explicitly here
alongside `ollama.RequestError`/`ollama.ResponseError` and re-raise as
`LLMError`, a single exception type callers can handle.
"""

from __future__ import annotations

import json

import ollama
from pydantic import BaseModel, ValidationError

from memagent.config import CONFIG

Message = dict[str, str]


class LLMError(RuntimeError):
    """Raised for any failure talking to the local Ollama server."""


def _client() -> ollama.Client:
    return ollama.Client(host=CONFIG.ollama_host, timeout=CONFIG.llm_timeout_s)


def chat(messages: list[Message], temperature: float = CONFIG.chat_temperature) -> str:
    """Single chat completion. Returns the assistant's text content."""
    try:
        response = _client().chat(
            model=CONFIG.generation_model,
            messages=messages,
            options={"temperature": temperature},
        )
    except (ollama.RequestError, ollama.ResponseError) as exc:
        raise LLMError(f"Ollama chat call failed: {exc}") from exc
    except TimeoutError as exc:
        raise LLMError(f"Ollama chat call timed out after {CONFIG.llm_timeout_s}s: {exc}") from exc
    return response["message"]["content"]


def embed(text: str) -> list[float]:
    """Embed a single string with the configured embedding model."""
    try:
        response = _client().embed(model=CONFIG.embedding_model, input=text)
    except (ollama.RequestError, ollama.ResponseError) as exc:
        raise LLMError(f"Ollama embed call failed: {exc}") from exc
    except TimeoutError as exc:
        raise LLMError(f"Ollama embed call timed out after {CONFIG.llm_timeout_s}s: {exc}") from exc
    embeddings = response["embeddings"]
    return list(embeddings[0])


def generate_json(
    prompt: str,
    schema: type[BaseModel],
    system: str | None = None,
    temperature: float = 0.0,
    max_retries: int = CONFIG.llm_max_retries,
) -> BaseModel:
    """Call the generation model in JSON mode and validate the result
    against `schema`, retrying once (by default) with the validation error
    fed back to the model if parsing/validation fails."""
    messages: list[Message] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = _client().chat(
                model=CONFIG.generation_model,
                messages=messages,
                format="json",
                options={"temperature": temperature},
            )
            raw = response["message"]["content"]
            data = json.loads(raw)
            return schema.model_validate(data)
        except (ollama.RequestError, ollama.ResponseError) as exc:
            raise LLMError(f"Ollama JSON call failed: {exc}") from exc
        except TimeoutError as exc:
            raise LLMError(f"Ollama JSON call timed out after {CONFIG.llm_timeout_s}s: {exc}") from exc
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
            if attempt < max_retries:
                messages.append({"role": "assistant", "content": raw})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "That response was not valid JSON matching the required "
                            f"schema. Error: {exc}. Reply again with ONLY valid JSON."
                        ),
                    }
                )
    raise LLMError(f"Model did not return schema-valid JSON after {max_retries + 1} attempt(s): {last_error}")
