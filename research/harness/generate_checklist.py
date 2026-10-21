from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from research.harness.dota_symbolic import (  # noqa: E402
    DEFAULT_CHECKLIST_PATH,
    DEFAULT_METADATA_PATH,
    DotaClip,
    build_prompt_context,
    load_dota_metadata,
    select_representative_clip,
    slugify_label,
)
from research.harness.llm_synthesizer import synthesize_archetype  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a Symbolic Checklist of DoTA anomaly archetypes for CARLA replay and symbolic coverage accounting."
    )
    parser.add_argument("--metadata-json", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--model", default="gpt-4.1")
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument("--disable-openai", action="store_true")
    parser.add_argument("--sample-limit", type=int, default=5)
    parser.add_argument("--output", type=Path, default=DEFAULT_CHECKLIST_PATH)
    return parser


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.fmean(values))


def _build_archetype_entry(
    canonical_class: str,
    clips: list[DotaClip],
    *,
    sample_limit: int,
    model: str,
    env_file: Path | None,
    disable_openai: bool,
) -> dict[str, Any]:
    representative_clip = select_representative_clip(clips)
    synthesis = synthesize_archetype(
        canonical_class=canonical_class,
        clip=representative_clip,
        description=None,
        model=model,
        env_file=env_file,
        disable_openai=disable_openai,
    )
    subset_counts = Counter(clip.subset for clip in clips)
    sorted_clip_ids = sorted(clip.clip_id for clip in clips)
    return {
        "archetype_id": slugify_label(canonical_class),
        "canonical_class": canonical_class,
        "clip_count": len(clips),
        "sample_clip_ids": sorted_clip_ids[:sample_limit],
        "representative_clip_id": representative_clip.clip_id if representative_clip is not None else None,
        "subset_breakdown": dict(sorted(subset_counts.items())),
        "anomaly_progress_ratio_mean": _mean([clip.anomaly_progress_ratio for clip in clips]),
        "anomaly_duration_ratio_mean": _mean([clip.anomaly_duration_ratio for clip in clips]),
        "prompt_context": build_prompt_context(canonical_class, representative_clip, clip_count=len(clips)),
        "verification_predicates": list(synthesis.get("verification_predicates") or []),
        "scenario_intent": synthesis.get("scenario_intent"),
        "scenario_template": synthesis.get("scenario_template"),
        "stsl_formula": synthesis.get("stsl_formula"),
        "natural_language_description": synthesis.get("natural_language_description"),
        "scallop_rules": synthesis.get("scallop_rules"),
        "synthesis": synthesis,
    }


def main() -> None:
    args = build_parser().parse_args()
    metadata = load_dota_metadata(args.metadata_json)
    grouped: dict[str, list[DotaClip]] = defaultdict(list)
    for clip in metadata.values():
        grouped[clip.anomaly_class].append(clip)

    archetypes = [
        _build_archetype_entry(
            canonical_class,
            sorted(clips, key=lambda clip: clip.clip_id),
            sample_limit=args.sample_limit,
            model=args.model,
            env_file=args.env_file,
            disable_openai=args.disable_openai,
        )
        for canonical_class, clips in sorted(grouped.items())
    ]

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "DoTA",
        "metadata_json": str(args.metadata_json),
        "total_clip_count": len(metadata),
        "archetype_count": len(archetypes),
        "generation_mode": "class-grouped",
        "archetypes": archetypes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()