# Model serving backends

The harness can drive AutoVLA through several serving stacks without changing
the experiment. Sensor capture, routing, pure-pursuit tracking and all coverage
observers stay in `research/harness/autovla_bridge.py`; only the model call is
swapped, so every backend sees the identical CARLA episode.

## Selection

`run_shakedown.py` (and `policy_search.py`, which passes through) accepts:

| flag | values | meaning |
| --- | --- | --- |
| `--autovla-backend` | `torch` (default), `http`, `openai` | serving stack |
| `--autovla-endpoint` | URL | remote server root |
| `--autovla-timeout` | seconds | request timeout |
| `--autovla-model` | name | served model name (openai protocol) |

### `torch` (default)

In-process Hugging Face AutoVLA. Historical path, unchanged behaviour. This is
the **only** backend that exposes a torch model, so it is required for
activation-based coverage (KMNC / LSA) and the coverage observer's activation
hooks. Signature:

```
research/.venv/bin/python research/harness/run_shakedown.py \
  --agent-kind autovla \
  --agent-repo-path research/models/AutoVLA \
  --agent-config research/models/AutoVLA/checkpoints/AutoVLA-hf ...
```

### `http` — batched planning server (recommended for throughput)

Talks to a server exposing `POST /plan` with AutoVLA's **native video** contract
(3 cameras x 4 frames). Reference implementation:
`/root/work/autovla_serve/plan_server.py` (vLLM offline engine, continuous
batching; measured ~45 planning steps/s at batch 128 on an H100).

```
--autovla-backend http --autovla-endpoint http://127.0.0.1:8101
```

This backend needs only numpy + the codebook, so the harness environment does
not need torch. Activation coverage is unavailable (the coverage observer
records `unsupported-agent`); engine/semantic coverage is unaffected.

### `openai` — OpenAI-compatible chat-completions server

Works with stock servers hosting the converted checkpoint (`vllm serve`,
SGLang, `llama-server`). Uses the **image** modality (12 frames), which costs
roughly twice the vision tokens of the native video prompt but needs no custom
endpoint.

```
--autovla-backend openai --autovla-endpoint http://127.0.0.1:8100 --autovla-model autovla
```

## Starting a server

```bash
# vLLM /plan server (video contract, batched)
VLLM_USE_FLASHINFER_SAMPLER=0 /root/work/vllm_env/bin/python \
  /root/work/autovla_serve/plan_server.py

# or stock OpenAI server (image contract)
VLLM_USE_FLASHINFER_SAMPLER=0 /root/work/vllm_env/bin/vllm serve \
  research/models/AutoVLA/checkpoints/AutoVLA-hf \
  --served-model-name autovla --port 8100 \
  --limit-mm-per-prompt '{"image":12,"video":3}' \
  --mm-processor-kwargs '{"min_pixels":109760,"max_pixels":109760}'
```

## Campaign use

`run_fse_multimap_autovla.sh` forwards two optional environment variables:

```bash
SCOUT_AUTOVLA_BACKEND=http \
SCOUT_AUTOVLA_ENDPOINT=http://127.0.0.1:8101 \
bash research/experiments/EXP-020-policy-comparison/proof-of-concept/run_fse_multimap_autovla.sh A
```

## Caveats

- A remote backend changes execution but **not** the coverage-engine semantics
  (semantic/engine metrics are computed by the harness from the semantic stream,
  independently of how the model was served).
- `policy_search.py`'s checkpoint fingerprint does not currently encode the
  backend, so do **not** resume a campaign across a backend change — use a fresh
  `--output-dir` (mix only same-backend rows).
- Backends are validated by `research/tests/test_model_backends.py` (payload
  construction and trajectory decode; no network or torch required).
