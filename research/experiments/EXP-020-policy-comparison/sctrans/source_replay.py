"""Read SCTrans CommonRoad sources without changing coordinates or time."""
from __future__ import annotations

import bisect
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class State:
    t: float
    x: float
    y: float
    yaw: float
    speed: float


@dataclass
class Actor:
    actor_id: str
    kind: str
    length: float
    width: float
    states: list[State]

    def at(self, t: float) -> State | None:
        """Interpolate recorded samples only; never extrapolate or clamp time."""
        if t < self.states[0].t - 1e-8 or t > self.states[-1].t + 1e-8:
            return None
        i = bisect.bisect_right([s.t for s in self.states], t)
        if i == 0:
            return self.states[0]
        if i == len(self.states):
            return self.states[-1]
        a, b = self.states[i - 1:i + 1]
        u = (t - a.t) / (b.t - a.t)
        angle = math.atan2(math.sin(b.yaw - a.yaw), math.cos(b.yaw - a.yaw))
        return State(t, a.x + u * (b.x-a.x), a.y + u * (b.y-a.y),
                     a.yaw + u * angle, a.speed + u * (b.speed-a.speed))


@dataclass
class Source:
    path: Path
    dt: float
    ego: State
    actors: list[Actor]
    lanes: dict[str, dict]
    goals: list[tuple[float, float]]

    @property
    def end_time(self):
        return max((a.states[-1].t for a in self.actors), default=0.0)

    def manifest(self):
        return {
            "source": str(self.path), "sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
            "dt": self.dt, "ego": asdict(self.ego), "actor_count": len(self.actors),
            "actor_classes": dict(Counter(a.kind for a in self.actors)),
            "goals": self.goals, "end_time_s": self.end_time, "lanelet_count": len(self.lanes),
            "actors": [{"id": a.actor_id, "class": a.kind, "length": a.length, "width": a.width,
                        "samples": len(a.states), "start_s": a.states[0].t, "end_s": a.states[-1].t}
                       for a in self.actors],
        }


def _number(e, path, default=None):
    val = e.findtext(path)
    if val is None:
        if default is None:
            raise ValueError(f"Missing required {path}")
        return default
    value = float(val)
    if not math.isfinite(value):
        raise ValueError(f"Nonfinite {path}")
    return value


def _state(e, dt):
    return State(_number(e, "time/exact") * dt,
                 _number(e, "position/point/x"), _number(e, "position/point/y"),
                 _number(e, "orientation/exact"), _number(e, "velocity/exact", 0.0))


def load_source(path: Path) -> Source:
    root = ET.parse(path).getroot()
    dt = float(root.attrib["timeStepSize"])
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("Invalid source timestep")
    pp = root.find("planningProblem")
    if pp is None:
        raise ValueError("No planningProblem")
    ego = _state(pp.find("initialState"), dt)
    actors = []
    for e in root:
        if e.tag not in ("obstacle", "dynamicObstacle", "staticObstacle"):
            continue
        kind = e.findtext("type")
        states = [_state(e.find("initialState"), dt)]
        states += [_state(s, dt) for s in e.findall("trajectory/state")]
        if any(b.t <= a.t for a,b in zip(states,states[1:])):
            raise ValueError(f"Non-increasing timestamps: actor {e.attrib['id']}")
        shape = e.find("shape/rectangle")
        if shape is None:
            raise ValueError(f"Unsupported shape: actor {e.attrib['id']}")
        actors.append(Actor(e.attrib["id"], kind, _number(shape,"length"), _number(shape,"width"),states))
    if len({a.actor_id for a in actors}) != len(actors):
        raise ValueError("Duplicate actor IDs")
    lanes = {}
    for lane in root.findall("lanelet"):
        def points(side):
            return [(_number(p,"x"),_number(p,"y")) for p in lane.findall(side + "/point")]
        left,right=points("leftBound"),points("rightBound")
        lanes[lane.attrib["id"]] = {"left":left,"right":right,
             "successors":[s.attrib["ref"] for s in lane.findall("successor")]}
    goals=[]
    for goal in pp.findall("goalState/position"):
        rect=goal.find("rectangle/center")
        if rect is not None:
            goals.append((_number(rect,"x"),_number(rect,"y")))
        for ref in goal.findall("lanelet"):
            lane=lanes[ref.attrib["ref"]]
            if lane["left"] and lane["right"]:
                l,r=lane["left"],lane["right"]
                goals.append(((l[-1][0]+r[-1][0])/2,(l[-1][1]+r[-1][1])/2))
        point=goal.find("point")
        if point is not None:
            goals.append((_number(point,"x"),_number(point,"y")))
    return Source(path.resolve(),dt,ego,actors,lanes,goals)


def validation_verdict(row, *, min_displacement=2.0, max_replay_error=0.25):
    """A successful process or unrelated hazard cannot pass scene validation."""
    reasons=[]
    if row.get('validation_schema_version')!=1 or not row.get('code_sha256'):
        reasons.append('missing_validation_provenance')
    if row.get("run_error"):
        reasons.append("execution_error")
    if row.get("ads_control_overrides",0):
        reasons.append("ads_modified")
    if row.get('diagnostic_no_npcs'):
        reasons.append('diagnostic_only')
    if row.get('fidelity_class')=='source_derived_variant':
        reasons.append('perturbed_baseline')
    displacement=row.get('ego_displacement_m',0)
    if not math.isfinite(displacement) or displacement < min_displacement:
        reasons.append("ego_stalled")
    if not row.get("intended_hazard_detected",False):
        reasons.append("intended_hazard_not_detected")
    if not row.get("intended_predicates_detected",False):
        reasons.append("intended_predicates_not_detected")
    if row.get("spawn_failures"):
        reasons.append("actor_spawn_failed")
    expected_actors=row.get('source',{}).get('actor_count')
    if expected_actors is None or len(row.get('actor_mapping',{}))!=expected_actors:
        reasons.append('incomplete_actor_replay')
    error=row.get("max_replay_error_m",float("inf"))
    if not math.isfinite(error) or error > max_replay_error:
        reasons.append("trajectory_mismatch")
    if row.get('max_replay_yaw_error_deg',0)>5:
        reasons.append('trajectory_heading_mismatch')
    expected_ticks=row.get('ticks_expected',0)
    if expected_ticks<=0 or row.get('ticks_executed') != expected_ticks:
        reasons.append('incomplete_execution')
    if not row.get("map_validated",False):
        reasons.append("map_not_validated")
    if row.get("collision_count",0):
        reasons.append("collision_in_base_scenario")
    if not row.get('collision_sensor_attached') or 'collision_count' not in row:
        reasons.append('missing_collision_sensor_evidence')
    return {"passed":not reasons,"reasons":reasons}


if __name__ == "__main__":
    import argparse,json
    p=argparse.ArgumentParser();p.add_argument("source_dir",type=Path);p.add_argument("--output",type=Path)
    args=p.parse_args();rows=[]
    for path in sorted(args.source_dir.glob("*.xml")):
        try:
            row=load_source(path).manifest()
            row["source_issues"]=(["missing_goal"] if not row["goals"] else [])
            if row["end_time_s"] < 2:
                row["source_issues"].append("recorded_trajectory_shorter_than_2s")
            rows.append(row)
        except Exception as e:
            rows.append({"source":str(path),"source_issues":[str(e)]})
    result=json.dumps(rows,indent=2)
    if args.output:args.output.write_text(result+"\n")
    for r in rows:
        print(Path(r['source']).name,r.get('actor_count'),r.get('end_time_s'),r['source_issues'])
