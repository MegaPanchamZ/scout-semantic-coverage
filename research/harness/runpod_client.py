import os
from pathlib import Path

import runpod
from dotenv import load_dotenv
from typing import Any

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _get_api_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        raise RuntimeError(
            "RUNPOD_API_KEY not set. Set it in research/.env or export RUNPOD_API_KEY=..."
        )
    return key


def _client(endpoint_id: str) -> runpod.Endpoint:
    runpod.api_key = _get_api_key()
    return runpod.Endpoint(endpoint_id)


def run_vllm_endpoint(
    endpoint_id: str,
    vllm_endpoint: str,
    payload: dict[str, Any],
    timeout: int = 300,
) -> dict[str, Any] | None:
    ep = _client(endpoint_id)
    result = ep.run_sync({
        "input": {
            "endpoint": vllm_endpoint,
            "payload": payload,
        }
    }, timeout=timeout)
    if isinstance(result, dict) and "error" in result:
        raise RuntimeError(f"RunPod worker error: {result['error']}")
    return result


def check_models(endpoint_id: str, timeout: int = 600) -> dict[str, Any] | None:
    return run_vllm_endpoint(endpoint_id, "models", {}, timeout=timeout)


def chat_completion(
    endpoint_id: str,
    messages: list[dict[str, str]],
    model: str = "nvidia/Cosmos-Transfer1-7B-Sample-AV",
    max_tokens: int = 1024,
    temperature: float = 0.7,
    timeout: int = 300,
) -> dict[str, Any] | None:
    return run_vllm_endpoint(
        endpoint_id,
        "chat/completions",
        {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=timeout,
    )


def batch_run(
    endpoint_id: str,
    requests: list[dict[str, Any]],
    timeout: int = 600,
) -> list[dict[str, Any] | None]:
    ep = _client(endpoint_id)
    payloads = [
        {
            "input": {
                "endpoint": r.get("vllm_endpoint", "chat/completions"),
                "payload": r.get("payload", {}),
            }
        }
        for r in requests
    ]
    results = ep.run_bulk_sync(payloads, timeout=timeout)
    for i, r in enumerate(results):
        if isinstance(r, dict) and "error" in r:
            results[i] = None
    return results
