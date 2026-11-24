"""Strict, versioned evidence files. No simulator imports or implicit path guessing."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import os
import tempfile

from research.harness.outcome_models import (
    CaptureManifest, ChannelCoverage, FrameEvidence, SensorEvidence, strict_json,
)


class EvidenceReadError(ValueError):
    pass


def _reject_constant(value):
    raise ValueError(f"Non-finite JSON constant: {value}")


def _parse_finite_float(token: str) -> float:
    val = float(token)
    if not math.isfinite(val):
        raise ValueError(f"Non-finite JSON float literal: {token}")
    return val


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    obj: dict[str, object] = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"Duplicate JSON object key: {key!r}")
        obj[key] = value
    return obj


def loads_strict_json(text: str):
    try:
        return json.loads(
            text,
            parse_constant=_reject_constant,
            parse_float=_parse_finite_float,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except RecursionError as exc:
        raise ValueError("JSON nesting depth exceeded") from exc


def read_strict_json(path: Path):
    return loads_strict_json(Path(path).read_bytes().decode("utf-8"))


def atomic_json(path: Path, value) -> None:
    text = strict_json(value, indent=2) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class EvidenceWriter:
    """Append-only trace; an interrupted writer leaves an inspectable partial file."""

    def __init__(self, path: Path, run_id: str):
        self.path = Path(path)
        self.run_id = run_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("x", encoding="utf-8")
        self._count = 0

    def append(self, record: FrameEvidence | SensorEvidence):
        if not isinstance(record, (FrameEvidence, SensorEvidence)) or record.run_id != self.run_id:
            raise ValueError("Evidence record type/run identity mismatch")
        envelope = {
            "schema_version": 2,
            "kind": "frame" if isinstance(record, FrameEvidence) else "sensor",
            "data": record,
        }
        line = strict_json(envelope) + "\n"
        self._stream.write(line)
        self._stream.flush()
        self._count += 1

    def close(self):
        if not self._stream.closed:
            self._stream.close()

    def finish(self, manifest: CaptureManifest) -> Path:
        if manifest.run_id != self.run_id:
            raise ValueError("Manifest run identity mismatch")
        self.close()
        target = self.path.with_suffix(self.path.suffix + ".manifest.json")
        atomic_json(target, {"capture_manifest": manifest, "evidence_file": self.path.name,
                             "evidence_sha256": file_digest(self.path), "record_count": self._count})
        return target


def iter_evidence(path: Path):
    """Stream normalized evidence records, catching torn UTF-8 bytes and malformed JSON."""
    offset = 0
    with Path(path).open("rb") as stream:
        for number, raw_line in enumerate(stream, 1):
            line_offset = offset
            offset += len(raw_line)
            try:
                line = raw_line.decode("utf-8")
                envelope = loads_strict_json(line)
                if not isinstance(envelope, dict):
                    raise ValueError("Evidence record must be a JSON object")
                if type(envelope.get("schema_version")) is not int or envelope["schema_version"] != 2:
                    raise ValueError("Unsupported evidence version")
                if envelope["kind"] == "frame":
                    yield FrameEvidence.from_dict(envelope["data"])
                elif envelope["kind"] == "sensor":
                    raw_sensor = envelope["data"]
                    if not isinstance(raw_sensor, dict):
                        raise ValueError("Sensor evidence data must be a JSON object")
                    data = dict(raw_sensor)
                    if data.get("impulse_ns") is not None:
                        if not isinstance(data["impulse_ns"], (list, tuple)):
                            raise ValueError("impulse_ns must be a list of three finite numbers")
                        data["impulse_ns"] = tuple(data["impulse_ns"])
                    if data.get("lane_markings") is not None:
                        if not isinstance(data["lane_markings"], (list, tuple)):
                            raise ValueError("lane_markings must be a list of strings")
                        data["lane_markings"] = tuple(data["lane_markings"])
                    yield SensorEvidence(**data)
                else:
                    raise ValueError("Unknown evidence kind")
            except (UnicodeDecodeError, ValueError, TypeError, KeyError, OverflowError, RecursionError) as exc:
                raise EvidenceReadError(f"{path}:{number}: byte {line_offset}: {exc}") from exc


def load_capture_manifest(path: Path) -> tuple[CaptureManifest, Path]:
    path = Path(path)
    try:
        payload = read_strict_json(path)
        if not isinstance(payload, dict):
            raise EvidenceReadError(f"Capture manifest must be a JSON object: {path}")
        evidence_file = payload["evidence_file"]
        expected_sha = payload["evidence_sha256"]
        raw_manifest = payload["capture_manifest"]
        if not isinstance(evidence_file, str) or not evidence_file:
            raise ValueError("evidence_file must be a non-empty string")
        if Path(evidence_file).name != evidence_file or evidence_file in (".", ".."):
            raise ValueError("evidence_file must be a plain relative filename inside the manifest directory")
        if not isinstance(expected_sha, str) or not expected_sha:
            raise ValueError("evidence_sha256 must be a non-empty string")
        if not isinstance(raw_manifest, dict):
            raise ValueError("capture_manifest must be an object")
        evidence = path.parent / evidence_file
        if evidence.is_symlink() or evidence.resolve().parent != path.parent.resolve():
            raise ValueError("evidence_file must not be a symlink or escape the manifest directory")
        if file_digest(evidence) != expected_sha:
            raise EvidenceReadError("Evidence checksum mismatch")
        return CaptureManifest.from_dict(raw_manifest), evidence
    except EvidenceReadError:
        raise
    except (KeyError, TypeError, ValueError, OSError, OverflowError, UnicodeDecodeError, RecursionError) as exc:
        raise EvidenceReadError(f"Invalid capture manifest {path}: {exc}") from exc


def write_outcome(path: Path, outcome):
    atomic_json(path, outcome.to_dict() if hasattr(outcome, "to_dict") else outcome)


def recover_partial_trace(
    corrupt_trace_path: Path,
    output_dir: Path,
    *,
    run_id: str,
    termination_reason: str = "truncated_trace_recovered",
) -> tuple[Path, int, str | None]:
    """Copy the valid prefix of an interrupted/corrupt JSONL trace into a fresh labelled partial capture.

    Never mutates the original file and never fabricates complete channel
    coverage: every recovered channel is marked ``partial`` with explicit
    truncation provenance.
    """
    corrupt_trace_path = Path(corrupt_trace_path)
    output_dir = Path(output_dir)
    if (
        not isinstance(run_id, str)
        or not run_id.strip()
        or "\x00" in run_id
        or "/" in run_id
        or "\\" in run_id
        or run_id in (".", "..")
        or Path(f"{run_id}.recovered.jsonl").name != f"{run_id}.recovered.jsonl"
    ):
        bad_id = repr(run_id) if isinstance(run_id, str) else type(run_id).__name__
        raise ValueError(f"Invalid run_id for trace recovery path: {bad_id}")
    recovered_trace = output_dir / f"{run_id}.recovered.jsonl"
    recovered_manifest = recovered_trace.with_suffix(recovered_trace.suffix + ".manifest.json")
    corrupt_resolved = corrupt_trace_path.resolve()
    if corrupt_resolved in (recovered_trace.resolve(), recovered_manifest.resolve()):
        raise ValueError("Recovered output path must not overwrite corrupt_trace_path")
    output_dir.mkdir(parents=True, exist_ok=True)
    if recovered_trace.resolve().parent != output_dir.resolve():
        raise ValueError(f"Recovered trace path escapes output_dir for run_id {run_id!r}")
    writer = EvidenceWriter(recovered_trace, run_id)
    finished = False
    try:
        records: list[FrameEvidence | SensorEvidence] = []
        corruption_error: str | None = None
        try:
            for rec in iter_evidence(corrupt_trace_path):
                if rec.run_id != run_id:
                    raise EvidenceReadError(f"Run identity mismatch: expected {run_id}, got {rec.run_id}")
                writer.append(rec)
                records.append(rec)
        except EvidenceReadError as exc:
            corruption_error = str(exc)

        eval_times = [r.sim_time_s for r in records if r.phase == "evaluation"]
        start_s = min(eval_times) if eval_times else 0.0
        end_s = max(eval_times) if eval_times else start_s

        channels: dict[str, ChannelCoverage] = {}
        for rec in records:
            if rec.phase != "evaluation":
                continue
            if isinstance(rec, FrameEvidence):
                for ch in rec.available_channels:
                    prev = channels.get(ch)
                    lo = min(prev.start_s, rec.sim_time_s) if (prev and prev.start_s is not None) else rec.sim_time_s
                    hi = max(prev.end_s, rec.sim_time_s) if (prev and prev.end_s is not None) else rec.sim_time_s
                    channels[ch] = ChannelCoverage("partial", lo, hi, reasons=("truncated_trace_recovery",))
            elif isinstance(rec, SensorEvidence):
                prev = channels.get(rec.channel)
                lo = min(prev.start_s, rec.sim_time_s) if (prev and prev.start_s is not None) else rec.sim_time_s
                hi = max(prev.end_s, rec.sim_time_s) if (prev and prev.end_s is not None) else rec.sim_time_s
                channels[rec.channel] = ChannelCoverage("partial", lo, hi, reasons=("truncated_trace_recovery",))

        manifest = CaptureManifest(
            run_id=run_id,
            start_s=start_s,
            end_s=end_s,
            execution_status="partial",
            termination_reason=termination_reason,
            channels=channels,
            provenance={
                "recovered_from": str(corrupt_trace_path),
                "recovered_record_count": len(records),
                "corruption_error": corruption_error,
            },
        )
        manifest_path = writer.finish(manifest)
        finished = True
        return manifest_path, len(records), corruption_error
    finally:
        try:
            writer.close()
        finally:
            if not finished:
                recovered_trace.unlink(missing_ok=True)
