from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any
from urllib import error, request


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from research.harness.dota_symbolic import (  # noqa: E402
    DEFAULT_METADATA_PATH,
    DotaClip,
    build_archetype_seed,
    build_prompt_context,
    load_dota_metadata,
    normalize_dota_class,
    resolve_openai_api_key,
    select_representative_clip,
    slugify_label,
)


DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "research/logs/synthesis"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Use an LLM to convert a DoTA anomaly class or clip into formal symbolic and scenario-generation artefacts."
    )
    parser.add_argument("--dota-class", default=None)
    parser.add_argument("--clip-id", default=None)
    parser.add_argument("--metadata-json", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--description", default=None, help="Optional manual natural-language description override.")
    parser.add_argument("--model", default="gpt-4.1")
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument("--disable-openai", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("LLM response was empty.")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Expected the LLM to return a JSON object.")
    return payload


def _chat_completion(api_key: str, model: str, system_prompt: str, user_prompt: str) -> dict[str, Any]:
    payload = {
        "model": model,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    raw_request = request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with request.urlopen(raw_request, timeout=90) as response:
        response_payload = json.loads(response.read().decode("utf-8"))
    choice = (((response_payload.get("choices") or [None])[0]) or {}).get("message") or {}
    content = str(choice.get("content") or "")
    return _extract_json_object(content)


def _build_system_prompt() -> str:
    return (
        "You translate traffic anomaly descriptions into structured neuro-symbolic ADS-testing artefacts. "
        "Return only JSON. Preserve field names exactly. "
        "Use executable, concrete, minimal outputs that align with STSL, Scallop, and CARLA scenario generation."
    )


def _build_user_prompt(canonical_class: str, prompt_context: str, description: str | None, seed: dict[str, Any]) -> str:
    manual_description = description or seed["description"]
    return json.dumps(
        {
            "task": "Convert the supplied DoTA anomaly context into a formal symbolic specification and a CARLA scenario template.",
            "constraints": [
                "Keep the STSL formula concise and executable-looking.",
                "Use predicate names already compatible with LASER/Scallop-style traffic reasoning.",
                "Keep verification_predicates short and tied to semantic validation in simulation.",
                "scenario_template must be JSON-friendly and not depend on live CARLA coordinates.",
                "scallop_rules must be a list of concise rule strings.",
            ],
            "input": {
                "canonical_class": canonical_class,
                "prompt_context": prompt_context,
                "manual_description": manual_description,
                "seed_formula": seed["stsl_formula"],
                "seed_intent": seed["scenario_intent"],
                "seed_verification_predicates": seed["verification_predicates"],
            },
            "output_schema": {
                "natural_language_description": "string",
                "stsl_formula": "string",
                "verification_predicates": ["string"],
                "scenario_intent": "string",
                "scenario_template": {
                    "controller": "string",
                    "controller_params": {},
                    "map_preferences": ["string"],
                    "weather_variants": ["string"],
                },
                "scallop_rules": ["string"],
                "assumptions": ["string"],
            },
        },
        indent=2,
    )


def _heuristic_synthesis(canonical_class: str, prompt_context: str, description: str | None, clip: DotaClip | None) -> dict[str, Any]:
    seed = build_archetype_seed(canonical_class)
    map_preferences = ["Town01", "Town03"]
    if "turn" in canonical_class or "lateral" in canonical_class:
        map_preferences = ["Town03", "Town01"]
    output = {
        "natural_language_description": description or seed["description"],
        "stsl_formula": seed["stsl_formula"],
        "verification_predicates": list(seed["verification_predicates"]),
        "scenario_intent": seed["scenario_intent"],
        "scenario_template": {
            "controller": seed["controller"],
            "controller_params": dict(seed["controller_params"]),
            "map_preferences": map_preferences,
            "weather_variants": ["ClearNoon", "WetCloudyNoon", "HardRainNoon", "SoftRainSunset"],
        },
        "scallop_rules": list(seed["scallop_rules"]),
        "assumptions": [
            "This artefact was generated from class metadata plus deterministic seed templates.",
            "The scenario template is a map-agnostic seed and still requires route anchoring for executable ScenarioSpec emission.",
        ],
    }
    if clip is not None:
        output["assumptions"].append(
            f"Representative clip {clip.clip_id} was used as timing context with anomaly window {clip.anomaly_start}-{clip.anomaly_end}."
        )
    output["prompt_context"] = prompt_context
    return output


def synthesize_archetype(
    *,
    canonical_class: str,
    clip: DotaClip | None,
    description: str | None,
    model: str,
    env_file: Path | None,
    disable_openai: bool,
) -> dict[str, Any]:
    prompt_context = build_prompt_context(canonical_class, clip=clip, clip_count=1 if clip is not None else None)
    seed = build_archetype_seed(canonical_class)
    heuristic = _heuristic_synthesis(canonical_class, prompt_context, description, clip)
    if disable_openai:
        return {
            "source": "heuristic-fallback",
            **heuristic,
        }

    api_key = resolve_openai_api_key(env_file)
    if not api_key:
        return {
            "source": "heuristic-fallback",
            "warning": "OPENAI_API_KEY was not found in the environment or any discovered .env file.",
            **heuristic,
        }

    try:
        llm_payload = _chat_completion(
            api_key,
            model,
            _build_system_prompt(),
            _build_user_prompt(canonical_class, prompt_context, description, seed),
        )
        return {
            "source": "openai-chat-completions",
            "model": model,
            "prompt_context": prompt_context,
            "natural_language_description": str(
                llm_payload.get("natural_language_description") or heuristic["natural_language_description"]
            ),
            "stsl_formula": str(llm_payload.get("stsl_formula") or heuristic["stsl_formula"]),
            "verification_predicates": list(
                llm_payload.get("verification_predicates") or heuristic["verification_predicates"]
            ),
            "scenario_intent": str(llm_payload.get("scenario_intent") or heuristic["scenario_intent"]),
            "scenario_template": llm_payload.get("scenario_template") or heuristic["scenario_template"],
            "scallop_rules": list(llm_payload.get("scallop_rules") or heuristic["scallop_rules"]),
            "assumptions": list(llm_payload.get("assumptions") or heuristic["assumptions"]),
        }
    except (error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
        return {
            "source": "heuristic-fallback",
            "warning": f"OpenAI synthesis failed and the deterministic fallback was used: {exc}",
            **heuristic,
        }


def main() -> None:
    args = build_parser().parse_args()
    metadata = load_dota_metadata(args.metadata_json)

    clip: DotaClip | None = None
    canonical_class = normalize_dota_class(args.dota_class) if args.dota_class else None
    if args.clip_id:
        clip = metadata.get(args.clip_id)
        if clip is None:
            raise SystemExit(f"DoTA clip '{args.clip_id}' was not found in {args.metadata_json}.")
        canonical_class = clip.anomaly_class

    if canonical_class is None:
        raise SystemExit("Either --dota-class or --clip-id is required.")

    if clip is None:
        class_clips = [item for item in metadata.values() if item.anomaly_class == canonical_class]
        clip = select_representative_clip(class_clips)

    synthesis = synthesize_archetype(
        canonical_class=canonical_class,
        clip=clip,
        description=args.description,
        model=args.model,
        env_file=args.env_file,
        disable_openai=args.disable_openai,
    )
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "canonical_class": canonical_class,
        "clip_id": clip.clip_id if clip is not None else None,
        "clip_metadata": (
            {
                "clip_id": clip.clip_id,
                "subset": clip.subset,
                "video_start": clip.video_start,
                "video_end": clip.video_end,
                "anomaly_start": clip.anomaly_start,
                "anomaly_end": clip.anomaly_end,
                "num_frames": clip.num_frames,
                "anomaly_progress_ratio": clip.anomaly_progress_ratio,
                "anomaly_duration_ratio": clip.anomaly_duration_ratio,
            }
            if clip is not None
            else None
        ),
        "synthesis": synthesis,
    }

    output_path = args.output
    if output_path is None:
        output_path = DEFAULT_OUTPUT_DIR / f"{slugify_label(canonical_class)}-{clip.clip_id if clip else 'archetype'}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()