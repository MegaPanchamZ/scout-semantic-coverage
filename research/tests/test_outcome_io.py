from dataclasses import asdict, replace
import json

import pytest

from research.harness.outcome_io import EvidenceWriter, EvidenceReadError, iter_evidence, load_capture_manifest, atomic_json
from research.harness.outcome_models import SensorEvidence
from research.harness.safety_outcomes import classify_run
from research.scripts.reclassify_outcomes import audit_archive, summarize_records, replay_capture, main
from research.tests.outcome_helpers import frame, manifest


def test_capture_roundtrip_and_replay_identical(tmp_path):
    evidence = [frame(i) for i in range(11)]
    evidence.append(SensorEvidence("run", 4, .4, "collision", "car", impulse_ns=(1, 2, 3)))
    path = tmp_path / "trace.jsonl"
    writer = EvidenceWriter(path, "run")
    for item in evidence:
        writer.append(item)
    manifest_path = writer.finish(manifest())
    assert list(iter_evidence(path)) == evidence
    capture, resolved = load_capture_manifest(manifest_path)
    assert capture == manifest()
    assert resolved == path
    direct = classify_run({"capture_manifest": asdict(capture)}, evidence=evidence)
    assert replay_capture(manifest_path).to_dict() == direct
    assert main(["--capture-manifest", str(manifest_path), "--output-dir", str(tmp_path / "result")]) == 0
    assert json.loads((tmp_path / "result/outcome.json").read_text()) == direct


def test_corrupt_trace_rejected_and_partial_prefix_remains_readable(tmp_path):
    path = tmp_path / "trace.jsonl"
    writer = EvidenceWriter(path, "run")
    writer.append(frame(0))
    manifest_path = writer.finish(manifest())
    with path.open("a") as stream:
        stream.write('{"schema_version":')
    reader = iter_evidence(path)
    assert next(reader) == frame(0)
    with pytest.raises(EvidenceReadError, match=":2:"):
        next(reader)
    with pytest.raises(EvidenceReadError, match="checksum"):
        load_capture_manifest(manifest_path)


def test_writer_checks_identity_and_cannot_clobber(tmp_path):
    writer = EvidenceWriter(tmp_path / "trace", "run")
    with pytest.raises(ValueError):
        writer.append(replace(frame(0), run_id="wrong"))
    with pytest.raises(FileExistsError):
        EvidenceWriter(tmp_path / "trace", "run")
    with pytest.raises(ValueError):
        writer.finish(replace(manifest(), run_id="wrong"))
    writer.close()


def test_atomic_strict_write_preserves_existing_file_on_serialization_error(tmp_path):
    path = tmp_path / "test.json"
    atomic_json(path, {"ok": 1})
    with pytest.raises(ValueError):
        atomic_json(path, {"bad": float("nan")})
    assert json.loads(path.read_text()) == {"ok": 1}


def test_archive_retains_missing_unreadable_corrupt_and_valid_rows(tmp_path):
    root = tmp_path / "campaign" / "random"
    root.mkdir(parents=True)
    run = tmp_path / "run.json"
    run.write_text(json.dumps({"ticks_executed": 10, "reached_goal": True, "safety_metrics": {"min_ttc": float("inf")}}))
    broken = tmp_path / "bad.json"
    broken.write_text("[")
    rows = [json.dumps({"eval_index": 0, "run_json_path": str(run)}),
            json.dumps({"eval_index": 1}), json.dumps({"eval_index": 2, "run_json_path": str(broken)}), "broken"]
    (root / "rows.jsonl").write_text("\n".join(rows))
    records = audit_archive(root.parent)
    assert len(records) == 4
    assert [r["status"] for r in records] == ["classified_legacy", "missing_run", "unreadable_run", "corrupt_row"]
    report = summarize_records(records)
    metrics = report["policies"]["random"]["metrics"]
    assert metrics["collision"]["unknown"] == 3
    assert metrics["collision"]["assessable_rate"] is None
    assert metrics["route_incomplete"]["absent"] == 1
    assert metrics["route_incomplete"]["unknown"] == 2
    before = run.read_bytes()
    output = tmp_path / "audit"
    assert main(["--search-root", str(root.parent), "--output-dir", str(output)]) == 0
    assert run.read_bytes() == before
    with pytest.raises(SystemExit):
        main(["--search-root", str(root.parent), "--output-dir", str(output)])


def test_empty_legacy_run_and_false_counter_are_unassessable(tmp_path):
    root = tmp_path / "policy"
    root.mkdir()
    (tmp_path / "run.json").write_text('{"collision_count":0}')
    (root / "rows.jsonl").write_text('{"run_json_path":"run.json"}')
    record = audit_archive(root, workspace_root=tmp_path)[0]
    assert record["outcome"]["categories"]["safety"]["collision"]["status"] == "unknown"


def test_evidence_with_nonfinite_json_rejected(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"schema_version":2,"kind":"frame","data":{"sim_time_s":NaN}}')
    with pytest.raises(EvidenceReadError, match="Non-finite"):
        list(iter_evidence(path))


def test_context_and_validity_survive_persisted_replay(tmp_path):
    from research.tests.test_task_outcomes import trace, TASK_CHANNELS
    from research.tests.test_scenario_quality import VALID
    from research.harness.outcome_models import ChannelCoverage
    evidence = trace()
    writer = EvidenceWriter(tmp_path / "context.jsonl", "run")
    for record in evidence:
        writer.append(record)
    capture = manifest(10, channels={c: ChannelCoverage("complete", 0, 10) for c in TASK_CHANNELS}, scenario_checks=VALID)
    path = writer.finish(capture)
    assert list(iter_evidence(tmp_path / "context.jsonl")) == evidence
    result = replay_capture(path).to_dict()
    assert result["categories"]["task"]["stuck"]["status"] == "present"
    assert result["categories"]["task"]["no_progress"]["status"] == "present"
    assert result["categories"]["quality"]["low_interaction"]["status"] == "present"


def test_recover_partial_trace_writes_labelled_partial_manifest_without_mutating_source(tmp_path):
    from research.harness.outcome_io import recover_partial_trace
    path = tmp_path / "corrupt.jsonl"
    writer = EvidenceWriter(path, "run")
    writer.append(frame(0))
    writer.append(frame(1))
    writer.close()
    with path.open("a", encoding="utf-8") as stream:
        stream.write('{"schema_version":2,"kind":"frame","data":')
    before = path.read_bytes()

    manifest_path, count, err = recover_partial_trace(path, tmp_path / "recovered", run_id="run")
    assert path.read_bytes() == before
    assert count == 2
    assert err is not None and ":3:" in err
    cap, rec_trace = load_capture_manifest(manifest_path)
    assert cap.execution_status == "partial"
    assert all(ch.status == "partial" for ch in cap.channels.values())
    assert len(list(iter_evidence(rec_trace))) == 2


def test_iter_evidence_and_recover_handle_torn_multibyte_utf8(tmp_path):
    from research.harness.outcome_io import recover_partial_trace

    path = tmp_path / "torn_utf8.jsonl"
    writer = EvidenceWriter(path, "run")
    writer.append(frame(0))
    writer.close()
    with path.open("ab") as stream:
        stream.write(b'{"schema_version":2,"kind":"frame","data":\xc3')  # torn 2-byte UTF-8 sequence

    reader = iter_evidence(path)
    assert next(reader) == frame(0)
    with pytest.raises(EvidenceReadError, match=":2:"):
        next(reader)

    manifest_path, count, err = recover_partial_trace(path, tmp_path / "rec_utf8", run_id="run")
    assert count == 1
    assert err is not None and ":2:" in err and "byte" in err
    cap, rec_trace = load_capture_manifest(manifest_path)
    assert cap.execution_status == "partial"
    assert len(list(iter_evidence(rec_trace))) == 1


def test_iter_evidence_and_recover_reject_overflow_int_and_list_of_pairs_data(tmp_path):
    from dataclasses import asdict
    from research.harness.outcome_io import recover_partial_trace
    from research.harness.outcome_models import CaptureManifest, ChannelCoverage, RoadRuleContext, RouteProgress, SensorEvidence

    # Overflowing int sim_time_s (10**400) on an otherwise valid frame must raise EvidenceReadError (not OverflowError)!
    huge_path = tmp_path / "huge_time.jsonl"
    writer = EvidenceWriter(huge_path, "run")
    writer.append(frame(0))
    writer.close()
    valid_f1 = asdict(frame(1))
    valid_f1["sim_time_s"] = 10**400
    with huge_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"schema_version": 2, "kind": "frame", "data": valid_f1}) + "\n")
    manifest_path, count, err = recover_partial_trace(huge_path, tmp_path / "recovered_huge", run_id="run")
    assert count == 1 and err is not None and ":2:" in err
    assert manifest_path.is_file()

    # Deeply nested torn row (1,100 unclosed arrays) must raise EvidenceReadError (not RecursionError) so recovery works!
    deep_path = tmp_path / "deep_nest.jsonl"
    writer_d = EvidenceWriter(deep_path, "run")
    writer_d.append(frame(0))
    writer_d.close()
    with deep_path.open("a", encoding="utf-8") as stream:
        stream.write("[" * 1100 + "\n")
    rec_deep_manifest, deep_count, deep_err = recover_partial_trace(deep_path, tmp_path / "recovered_deep", run_id="run")
    assert deep_count == 1 and deep_err is not None and ":2:" in deep_err
    assert rec_deep_manifest.is_file()

    # Array-of-pairs data payload, float schema_version=2.0 on an otherwise valid frame,
    # non-string road_context.evidence_refs=[7], and non-string lane_markings=[1] must all reject as EvidenceReadError!
    valid_f1_ok = asdict(frame(1))
    bad_rc_frame = {**valid_f1_ok, "road_context": {"stop_sign_applicable": True, "evidence_refs": [7]}}
    for bad_line in (
        '{"schema_version":2,"kind":"frame","data":[["sim_time_s",1.0],["sim_time_s",2.0]]}\n',
        '{"schema_version":2,"kind":"sensor","data":[["sim_time_s",1.0],["sim_time_s",2.0]]}\n',
        json.dumps({"schema_version": 2.0, "kind": "frame", "data": valid_f1_ok}) + "\n",
        json.dumps({"schema_version": 2, "kind": "frame", "data": bad_rc_frame}) + "\n",
        json.dumps({"schema_version": 2, "kind": "sensor", "data": {
            "run_id": "run", "sim_frame": 1, "sim_time_s": 0.1, "channel": "lane_invasion",
            "actor_id": "ego", "lane_markings": [1],
        }}) + "\n",
    ):
        bad_path = tmp_path / "bad_shape.jsonl"
        bad_path.write_text(bad_line, encoding="utf-8")
        with pytest.raises(EvidenceReadError):
            list(iter_evidence(bad_path))

    # CaptureManifest and load_capture_manifest strict schema & exception wrapping checks!
    with pytest.raises(EvidenceReadError):
        load_capture_manifest(tmp_path / "does_not_exist.manifest.json")
    broken_m = tmp_path / "broken.manifest.json"
    broken_m.write_text("{not_json", encoding="utf-8")
    with pytest.raises(EvidenceReadError):
        load_capture_manifest(broken_m)
    m_base = asdict(manifest())
    for bad_m in (
        {**m_base, "schema_version": 2.0},
        {**m_base, "run_id": 123},
        {**m_base, "termination_reason": 123},
        {**m_base, "provenance": []},
        {**m_base, "channels": {"ego_pose": {"status": "unavailable", "reasons": "abc"}}},
    ):
        with pytest.raises(ValueError):
            CaptureManifest.from_dict(bad_m)
    with pytest.raises(ValueError):
        RoadRuleContext(evidence_refs="xyz")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        RouteProgress(route_id=123, status="unknown", total_m=10.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ChannelCoverage(status="unavailable", reasons="abc")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        SensorEvidence(run_id="r", sim_frame=0, sim_time_s=0.0, channel="c", actor_id="a", lane_markings=[1])  # type: ignore[arg-type]
    # Unhashable list values passed to enum-like fields raise ValueError (not TypeError)!
    with pytest.raises(ValueError):
        RouteProgress(route_id="r", status=[], total_m=10.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        RoadRuleContext(lane_type=[])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ChannelCoverage(status=[])  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        CaptureManifest.from_dict({**m_base, "execution_status": []})
    # Invalid UTF-8, deeply nested recursive manifest, and path-traversal evidence_file raise EvidenceReadError!
    bad_utf8_m = tmp_path / "bad_utf8.manifest.json"
    bad_utf8_m.write_bytes(b"\xff\xfe")
    with pytest.raises(EvidenceReadError):
        load_capture_manifest(bad_utf8_m)
    deep_m = tmp_path / "deep.manifest.json"
    deep_m.write_text("[" * 1100, encoding="utf-8")
    with pytest.raises(EvidenceReadError):
        load_capture_manifest(deep_m)
    for bad_ev_file in ("../escape.jsonl", "/tmp/abs.jsonl"):
        trav_m = tmp_path / "trav.manifest.json"
        trav_m.write_text(json.dumps({
            "capture_manifest": m_base,
            "evidence_file": bad_ev_file,
            "evidence_sha256": "0" * 64,
        }), encoding="utf-8")
        with pytest.raises(EvidenceReadError, match="plain relative filename"):
            load_capture_manifest(trav_m)
    # Symlink evidence_file inside the manifest directory pointing outside must be rejected!
    outside_trace = tmp_path / "outside.jsonl"
    outside_w = EvidenceWriter(outside_trace, "run")
    outside_w.append(frame(0))
    outside_w.close()
    sym_dir = tmp_path / "sym_manifest_dir"
    sym_dir.mkdir()
    sym_trace = sym_dir / "linked.jsonl"
    sym_trace.symlink_to(outside_trace)
    sym_m = sym_dir / "linked.manifest.json"
    from research.harness.outcome_io import file_digest, loads_strict_json
    sym_m.write_text(json.dumps({
        "capture_manifest": m_base,
        "evidence_file": "linked.jsonl",
        "evidence_sha256": file_digest(outside_trace),
    }), encoding="utf-8")
    with pytest.raises(EvidenceReadError, match="symlink"):
        load_capture_manifest(sym_m)
    # loads_strict_json converts RecursionError on deeply nested JSON into ValueError!
    with pytest.raises(ValueError, match="nesting depth"):
        loads_strict_json("[" * 1100)
    # Valid sensor row AND CaptureManifest provenance with 600 nested arrays must finish & recover cleanly!
    nested_meta: object = 1
    for _ in range(600):
        nested_meta = [nested_meta]
    sensor_deep_path = tmp_path / "sensor_deep_meta.jsonl"
    sw = EvidenceWriter(sensor_deep_path, "run")
    sw.append(SensorEvidence(run_id="run", sim_frame=0, sim_time_s=0.0, channel="collision", actor_id="ego", metadata={"a": nested_meta}))
    deep_cap = manifest(provenance={"a": nested_meta})
    deep_m_path = sw.finish(deep_cap)
    assert deep_m_path.is_file()
    with sensor_deep_path.open("a", encoding="utf-8") as stream:
        stream.write('{"schema_version":2,"kind":"frame","data":')
    s_rec_m, s_rec_cnt, s_rec_err = recover_partial_trace(sensor_deep_path, tmp_path / "rec_sensor_deep", run_id="run")
    assert s_rec_cnt == 1 and s_rec_err is not None and s_rec_m.is_file()
    # Even if writer.close() raises during failed recovery, the partial .recovered.jsonl must be unlinked so retry works!
    orig_close = EvidenceWriter.close
    orig_finish = EvidenceWriter.finish
    try:
        def _boom_finish(self, m):
            raise RuntimeError("boom_finish")
        def _boom_close(self):
            orig_close(self)
            raise OSError("boom_close")
        EvidenceWriter.finish = _boom_finish  # type: ignore[method-assign]
        EvidenceWriter.close = _boom_close  # type: ignore[method-assign]
        with pytest.raises(OSError, match="boom_close"):
            recover_partial_trace(sensor_deep_path, tmp_path / "rec_cleanup_dir", run_id="run")
    finally:
        EvidenceWriter.finish = orig_finish  # type: ignore[method-assign]
        EvidenceWriter.close = orig_close  # type: ignore[method-assign]
    assert not (tmp_path / "rec_cleanup_dir" / "run.recovered.jsonl").exists()
    retry_m, retry_cnt, _ = recover_partial_trace(sensor_deep_path, tmp_path / "rec_cleanup_dir", run_id="run")
    assert retry_cnt == 1 and retry_m.is_file()
    # recover_partial_trace must reject path-escaping, whitespace, deeply nested, or self-overwriting paths!
    deep_id: object = "x"
    for _ in range(1100):
        deep_id = [deep_id]
    for bad_run_id in ("../escape", "sub/escape", "..", ".", "   ", 10**5000, deep_id):
        with pytest.raises(ValueError, match="Invalid run_id"):
            recover_partial_trace(sensor_deep_path, tmp_path / "rec_escape_dir", run_id=bad_run_id)  # type: ignore[arg-type]
    assert not (tmp_path / "escape.recovered.jsonl").exists()
    # Self-overwrite guard: corrupt_trace_path matching the output manifest path must raise ValueError and preserve bytes!
    self_dir = tmp_path / "rec_self_overwrite"
    self_dir.mkdir()
    self_src = self_dir / "run.recovered.jsonl.manifest.json"
    sw_self = EvidenceWriter(self_src, "run")
    sw_self.append(frame(0))
    sw_self.close()
    orig_self_bytes = self_src.read_bytes()
    with pytest.raises(ValueError, match="must not overwrite"):
        recover_partial_trace(self_src, self_dir, run_id="run")
    assert self_src.read_bytes() == orig_self_bytes
    assert not (self_dir / "run.recovered.jsonl").exists()
