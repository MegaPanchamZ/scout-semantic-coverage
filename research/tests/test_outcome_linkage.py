"""Tests for semantic-to-outcome linkage, intended interaction checks, and guidance diagnostics."""
from __future__ import annotations

from research.harness.outcome_linkage import (
    build_guidance_diagnostics,
    evaluate_intended_interaction,
    link_semantics_to_events,
)
from research.tests.outcome_helpers import actor, frame


def test_linkage_requires_same_actor_time_and_excludes_circular_and_post_event_facts():
    semantic_frames = [
        {
            "tick": 1,
            "sim_time_s": 1.0,
            "predicates": [
                {"predicate": "crossing_path", "signature": "crossing_path(pedestrian,ego)", "actor_ids": ["ped1", "ego"], "actor_roles": ["pedestrian_subject", "ego"]},
                {"predicate": "in_front_of", "signature": "in_front_of(vehicle,ego)", "actor_ids": ["other_veh", "ego"], "actor_roles": ["background", "ego"]},
                {"predicate": "jaywalking", "signature": "jaywalking(pedestrian)"},  # missing actor identity -> rejected
            ],
        },
        {
            "tick": 2,
            "sim_time_s": 2.0,
            "predicates": [
                {"predicate": "colliding", "signature": "colliding(ego,pedestrian)", "actor_ids": ["ego", "ped1"], "actor_roles": ["ego", "pedestrian_subject"], "outcome_derived": True},
            ],
        },
        {
            "tick": 3,
            "sim_time_s": 2.5,
            "predicates": [
                {"predicate": "obstructing", "signature": "obstructing(pedestrian,ego)", "actor_ids": ["ped1", "ego"], "actor_roles": ["pedestrian_subject", "ego"]},
            ],
        },
    ]
    events = [
        {"event_id": "ev1", "label": "collision", "actor_ids": ["ped1"], "start_s": 2.0, "end_s": 2.1},
    ]
    linked = link_semantics_to_events(
        semantic_frames, events, antecedent_window_s=2.0, actor_role_map={"ego": "ego", "ped1": "pedestrian_subject", "other_veh": "background"}
    )
    assert len(linked["antecedent_links"]) == 1
    assert linked["antecedent_links"][0]["predicate"] == "crossing_path"
    assert linked["antecedent_links"][0]["lag_s"] == 1.0
    # Circular colliding predicate separated out, never in antecedent_links
    assert len(linked["outcome_derived_links"]) == 1
    assert linked["outcome_derived_links"][0]["circular_outcome_derived"] is True
    assert linked["diagnostics"]["n_rejected_missing_identity"] == 1
    assert linked["diagnostics"]["n_rejected_post_event"] == 1


def test_evaluate_intended_interaction_checks_conjunction_order_and_physical_presence():
    sem = [
        {"sim_time_s": 0.5, "predicates": [{"predicate": "in_front_of", "signature": "in_front_of(pedestrian,ego)", "actor_roles": ["ped_subj", "ego"]}]},
        {"sim_time_s": 1.2, "predicates": [{"predicate": "crossing_path", "signature": "crossing_path(pedestrian,ego)", "actor_roles": ["ped_subj", "ego"]}]},
    ]
    phys = [frame(0, actors=(actor("p", actor_type="pedestrian", role="ped_subj"),))]

    ok = evaluate_intended_interaction(
        {"intended_role": "ped_subj", "required_predicates": ["in_front_of", "crossing_path"], "ordered_predicates": ["in_front_of", "crossing_path"]},
        sem, phys,
    )
    assert ok["witnessed"] is True and ok["stage"] == "witnessed"

    bad_order = evaluate_intended_interaction(
        {"intended_role": "ped_subj", "required_predicates": ["in_front_of", "crossing_path"], "ordered_predicates": ["crossing_path", "in_front_of"]},
        sem, phys,
    )
    assert bad_order["witnessed"] is False and bad_order["stage"] == "predicate_order_violated"

    missing_actor = evaluate_intended_interaction(
        {"intended_role": "ghost_actor", "required_predicates": ["in_front_of"]},
        sem, phys,
    )
    assert missing_actor["witnessed"] is False and missing_actor["stage"] == "actor_not_instantiated_physically"


def test_guidance_diagnostics_identifies_saturation_clipping_and_failure_stages():
    attempts = [
        {"hazard_target": "t1", "candidate_key": "c1", "trigger_activated": False, "fitness": 0.0, "mutation_clipped": True},
        {"hazard_target": "t1", "candidate_key": "c2", "trigger_activated": True, "engine_run_obligations": ["t1"], "engine_new_obligations": 0, "fitness": 0.0, "outcome_cluster_digest": "k1"},
        {"hazard_target": "t1", "candidate_key": "c2", "trigger_activated": True, "engine_run_obligations": ["t1"], "engine_new_obligations": 1, "fitness": 1.0, "terminated_by_collision": True, "outcome_cluster_digest": "k1"},
        {"hazard_target": "t2", "candidate_key": "c3", "run_error": "spawn failed", "fallback_selection": True},
    ]
    diag = build_guidance_diagnostics(attempts)
    assert diag["n_attempts"] == 4
    assert diag["repeated_candidates"] == 1
    assert diag["clipped_mutations"] == 1
    assert diag["fallback_selections"] == 1
    assert diag["repeat_outcome_clusters"] == 1
    assert diag["failure_stage_counts"]["controller_not_triggered"] == 1
    assert diag["failure_stage_counts"]["coverage_saturated_no_new_credit"] == 1
    assert diag["failure_stage_counts"]["safety_event_discovered"] == 1
    assert diag["failure_stage_counts"]["execution_or_setup_failure"] == 1


def test_linkage_rejects_cross_run_and_contradictory_actor_ids_with_same_role():
    semantic_frames = [
        {
            "run_id": "run_A",
            "sim_time_s": 1.0,
            "predicates": [
                # Same role 'pedestrian_subject' as ped1, but contradictory actor_id 'ped2' -> must NOT match!
                {"predicate": "crossing_path", "signature": "crossing_path(pedestrian,ego)", "actor_ids": ["ped2", "ego"], "actor_roles": ["pedestrian_subject", "ego"]},
                # Missing identity counted once even across multiple events
                {"predicate": "jaywalking", "signature": "jaywalking(pedestrian)"},
            ],
        },
        {
            "run_id": "run_B",  # Different run_id -> must NOT link to run_A events!
            "sim_time_s": 1.0,
            "predicates": [
                {"predicate": "crossing_path", "signature": "crossing_path(pedestrian,ego)", "actor_ids": ["ped1", "ego"], "actor_roles": ["pedestrian_subject", "ego"]},
            ],
        },
    ]
    events = [
        {"event_id": "ev1", "run_id": "run_A", "label": "collision", "actor_ids": ["ped1"], "start_s": 2.0, "end_s": 2.1},
        {"event_id": "ev2", "run_id": "run_A", "label": "near_miss", "actor_ids": ["ped1"], "start_s": 2.5, "end_s": 2.6},
    ]
    linked = link_semantics_to_events(
        semantic_frames,
        events,
        antecedent_window_s=3.0,
        actor_role_map={"ego": "ego", "ped1": "pedestrian_subject", "ped2": "pedestrian_subject"},
    )
    assert linked["antecedent_links"] == []
    assert linked["diagnostics"]["n_rejected_missing_identity"] == 1
    assert linked["diagnostics"]["n_rejected_cross_run"] == 2


def test_intended_interaction_does_not_splice_different_actors_or_coerce_missing_time():
    # Two different pedestrians sharing role 'ped_subj', each satisfies only one predicate -> must NOT pass!
    sem_split = [
        {"sim_time_s": 0.5, "predicates": [{"predicate": "in_front_of", "actor_ids": ["p1", "ego"], "actor_roles": ["ped_subj", "ego"]}]},
        {"sim_time_s": 1.0, "predicates": [{"predicate": "crossing_path", "actor_ids": ["p2", "ego"], "actor_roles": ["ped_subj", "ego"]}]},
    ]
    res_split = evaluate_intended_interaction(
        {"intended_role": "ped_subj", "required_predicates": ["in_front_of", "crossing_path"]},
        sem_split,
    )
    assert res_split["witnessed"] is False
    assert res_split["stage"] == "intended_predicate_not_witnessed"

    # Missing timestamp must fail ordered_predicates instead of being coerced to 0.0
    sem_notime = [
        {"predicates": [{"predicate": "in_front_of", "actor_ids": ["p1"], "actor_roles": ["ped_subj"]}]},
        {"sim_time_s": 1.0, "predicates": [{"predicate": "crossing_path", "actor_ids": ["p1"], "actor_roles": ["ped_subj"]}]},
    ]
    res_notime = evaluate_intended_interaction(
        {"intended_role": "ped_subj", "required_predicates": ["in_front_of", "crossing_path"], "ordered_predicates": ["in_front_of", "crossing_path"]},
        sem_notime,
    )
    assert res_notime["witnessed"] is False
    assert res_notime["stage"] == "predicate_order_violated"
    assert any("missing_timestamp" in f for f in res_notime["order_failures"])


def test_guidance_diagnostics_separates_witness_unavailable_and_outcome_unknown():
    attempts = [
        {"hazard_target": "t1", "trigger_activated": True},  # no semantic observation -> witness_unavailable
        {
            "hazard_target": "t1",
            "trigger_activated": True,
            "engine_run_obligations": ["t1"],
            "engine_new_obligations": 1,
            "outcome_v2": {"schema_version": 2, "aggregates": {"safety_event": {"status": "unknown"}}},
        },
    ]
    diag = build_guidance_diagnostics(attempts)
    assert diag["failure_stage_counts"]["witness_unavailable"] == 1
    assert diag["failure_stage_counts"]["outcome_unknown"] == 1
    assert diag["denominators"]["eligible_for_witness"] == 1
    assert diag["denominators"]["eligible_for_outcome"] == 0


def test_intended_interaction_numeric_ego_id_non_splicing_multi_occurrence_order_and_missing_run_id():
    # 1. Numeric CARLA ego ID '42' must NOT splice predicates from actors '101' and '102'
    sem_numeric = [
        {"sim_time_s": 0.5, "predicates": [{"predicate": "in_front_of", "actor_ids": ["101", "42"], "actor_roles": ["pedestrian_subject", "ego"]}]},
        {"sim_time_s": 1.0, "predicates": [{"predicate": "crossing_path", "actor_ids": ["102", "42"], "actor_roles": ["pedestrian_subject", "ego"]}]},
    ]
    phys = [frame(0, ego=actor("42", role="ego"), actors=(actor("101", actor_type="pedestrian", role="pedestrian_subject"), actor("102", actor_type="pedestrian", role="pedestrian_subject")))]
    res_num = evaluate_intended_interaction(
        {"intended_role": "pedestrian_subject", "required_predicates": ["in_front_of", "crossing_path"]},
        sem_numeric,
        phys,
    )
    assert res_num["witnessed"] is False

    # 2. Multi-occurrence sequence B@0.1, A@0.2, B@0.3 satisfies ordered A -> B
    sem_multi = [
        {"sim_time_s": 0.1, "predicates": [{"predicate": "B", "actor_ids": ["101", "42"], "actor_roles": ["pedestrian_subject", "ego"]}]},
        {"sim_time_s": 0.2, "predicates": [{"predicate": "A", "actor_ids": ["101", "42"], "actor_roles": ["pedestrian_subject", "ego"]}]},
        {"sim_time_s": 0.3, "predicates": [{"predicate": "B", "actor_ids": ["101", "42"], "actor_roles": ["pedestrian_subject", "ego"]}]},
    ]
    res_multi = evaluate_intended_interaction(
        {"intended_role": "pedestrian_subject", "required_predicates": ["A", "B"], "ordered_predicates": ["A", "B"]},
        sem_multi,
        phys,
    )
    assert res_multi["witnessed"] is True and res_multi["matched_actor_key"] == "101"

    # 3. Fact without run_id must NOT link to an event that carries run_id
    linked = link_semantics_to_events(
        [{"sim_time_s": 1.0, "predicates": [{"predicate": "crossing_path", "actor_ids": ["ped1", "42"], "actor_roles": ["pedestrian_subject", "ego"]}]}],
        [{"event_id": "e1", "run_id": "run_X", "label": "collision", "actor_ids": ["ped1"], "start_s": 2.0, "end_s": 2.1}],
        actor_role_map={"42": "ego", "ped1": "pedestrian_subject"},
    )
    assert linked["antecedent_links"] == []
    assert linked["diagnostics"]["n_rejected_cross_run"] == 1

    # 4. Without actor_role_map, numeric ego ID '42' in positional (actor_ids, actor_roles) must NOT bridge vehicle 102 -> 101
    linked_no_map = link_semantics_to_events(
        [{"run_id": "r1", "sim_time_s": 1.0, "predicates": [{"predicate": "in_front_of", "actor_ids": ["102", "42"], "actor_roles": ["other_vehicle", "ego"]}]}],
        [{"event_id": "e1", "run_id": "r1", "label": "collision", "actor_ids": ["42", "101"], "start_s": 2.0, "end_s": 2.1}],
    )
    assert linked_no_map["antecedent_links"] == []

    # 5. Empty semantic trace with single-item ordered_predicates must NOT return witnessed=True
    empty_ord = evaluate_intended_interaction({"intended_role": "ped", "required_predicates": [], "ordered_predicates": ["A"]}, [])
    assert empty_ord["witnessed"] is False
