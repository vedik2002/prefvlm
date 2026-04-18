"""OpenAI client with retry, structured output helpers, and vision support."""

import base64
import json
from pathlib import Path
from typing import Any, Optional, Type, TypeVar

from loguru import logger
from openai import OpenAI, RateLimitError, APIError, APIConnectionError
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)
import logging

from prefvlm.config import cfg

T = TypeVar("T", bound=BaseModel)

_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        cfg.validate()
        _client = OpenAI(api_key=cfg.openai_api_key)
    return _client


def _retry_decorator():
    return retry(
        retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(6),
        before_sleep=before_sleep_log(logging.getLogger("tenacity"), logging.WARNING),
        reraise=True,
    )


def encode_image(image_path: Path) -> str:
    """Base64-encode a local image for inclusion in API messages."""
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def image_message_part(image_path: Path, detail: str = "high") -> dict:
    """Build the image content part for a chat message."""
    b64 = encode_image(image_path)
    suffix = image_path.suffix.lstrip(".").lower()
    mime = "image/jpeg" if suffix in ("jpg", "jpeg") else f"image/{suffix}"
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{b64}", "detail": detail},
    }


@_retry_decorator()
def chat_completion(
    *,
    model: str,
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 1024,
    json_mode: bool = False,
    extra_kwargs: Optional[dict] = None,
) -> str:
    """Raw chat completion returning the assistant message text."""
    client = get_client()
    kwargs: dict[str, Any] = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    logger.debug(f"chat_completion model={model} msgs={len(messages)} temp={temperature}")
    response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content
    logger.debug(f"chat_completion -> {len(content or '')} chars")
    return content or ""


@_retry_decorator()
def structured_completion(
    *,
    model: str,
    messages: list[dict],
    response_model: Type[T],
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> T:
    """Chat completion parsed into a Pydantic model via JSON mode."""
    # Add JSON instruction to the last user message if not already present
    system_instruction = (
        f"You must respond with a valid JSON object matching this schema:\n"
        f"{json.dumps(response_model.model_json_schema(), indent=2)}\n"
        f"Return ONLY the JSON object, no markdown fencing."
    )
    augmented = [{"role": "system", "content": system_instruction}] + messages
    raw = chat_completion(
        model=model,
        messages=augmented,
        temperature=temperature,
        max_tokens=max_tokens,
        json_mode=True,
    )
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model returned non-JSON: {raw[:200]}") from e
    return response_model.model_validate(data)


@_retry_decorator()
def get_embedding(text: str, model: Optional[str] = None) -> list[float]:
    """Return a text embedding vector."""
    model = model or cfg.models.embeddings
    client = get_client()
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding


def load_prompt(name: str) -> str:
    """Load a prompt template from prefvlm/prompts/{name}.txt"""
    path = cfg.paths.prompts_dir / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text().strip()
