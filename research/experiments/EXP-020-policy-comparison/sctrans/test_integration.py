import json
import math
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace as NS
import unittest

from gap import bounding_box_gap
from map_preflight import audit_map
from source_replay import Actor,State
from variants import sample_actor,DEFAULTS
from route_compatibility import bounded_topology


class GeometryTests(unittest.TestCase):
    def actor(self,x,y,yaw):
        vertices=[]
        for a,b in [(-2,-1),(-2,1),(2,-1),(2,1)]:
            vertices.append(NS(x=x+a*math.cos(yaw)-b*math.sin(yaw),y=y+a*math.sin(yaw)+b*math.cos(yaw)))
        return NS(bounding_box=NS(get_world_vertices=lambda tf:vertices),get_transform=lambda:None)

    def test_oriented_boxes_and_overlap_are_distances_only(self):
        ego=self.actor(0,0,0)
        self.assertAlmostEqual(bounding_box_gap(ego,self.actor(5,0,0)),1)
        self.assertAlmostEqual(bounding_box_gap(ego,self.actor(5,0,math.pi/2)),2)
        self.assertEqual(bounding_box_gap(ego,self.actor(1,0,0)),0)

    def test_invalid_map_is_rejected_before_simulator(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.xodr'
            p.write_text('<OpenDRIVE><road id="1"><planView><geometry length="0"><spiral curvEnd="nan"/></geometry></planView></road></OpenDRIVE>')
            result=audit_map(p)
            self.assertFalse(result['passed'])
            self.assertIn('road:1:nonpositive_geometry_length',result['issues'])
            self.assertIn('road:1:nonfinite_spiral.curvEnd',result['issues'])

    def test_no_successor_does_not_crash_topology_walk(self):
        class Location:
            def __init__(self,x):self.x=x;self.y=0;self.z=0
            def distance(self,b):return abs(self.x-b.x)
        def waypoint(x):return NS(transform=NS(location=Location(x)),next=lambda d:[],road_id=1,section_id=0,lane_id=-1,s=x)
        entry,exit=waypoint(0),waypoint(3)
        dao=NS(_sampling_resolution=1,_wmap=NS(get_topology=lambda:[(entry,exit)]))
        topology=bounded_topology(dao)
        self.assertEqual(topology[0]['path'],[])
        self.assertIs(topology[0]['exit'],exit)

    def test_identity_variant_preserves_recording_and_never_extrapolates(self):
        actor=Actor('7','car',4,2,[State(0,0,0,0,10),State(1,10,0,0,10)])
        identity=dict(DEFAULTS,actor_id='7')
        self.assertEqual(sample_actor(actor,.5,identity),actor.at(.5))
        delayed=dict(identity,time_offset_s=-.2)
        self.assertIsNone(sample_actor(actor,.1,delayed))
        self.assertEqual(sample_actor(actor,.5,delayed).x,3)
        self.assertIsNone(sample_actor(actor,1.1,identity))


class PolicyTests(unittest.TestCase):
    def test_failed_baseline_is_rejected_before_launch(self):
        from compare_policies import admitted,AdmissionError
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'result.json'
            p.write_text(json.dumps({'validation':{'passed':True},'ego_displacement_m':.02}))
            with self.assertRaises(AdmissionError):admitted(p)

    def test_imported_exp020_policies_match_direct_calls(self):
        from real_policies import ALIASES,RealPolicy,policy_search,baselines,make_space
        for alias,canonical in ALIASES.items():
            with self.subTest(policy=alias):
                integrated=RealPolicy(alias,42);space=make_space()
                direct=baselines.make_policy_state(canonical,space,policy_search.PolicyState)
                seen=set();rng=random.Random(42);values=dict(DEFAULTS)
                # Compare the actual feedback/candidate sequence, including novelty changes.
                for i in range(4):
                    row={'covered_obligations':['moving(vehicle)']+(['hazard(other: turning)'] if i>1 else []),
                         'min_gap_m':8-i,'collision_count':int(i==3)}
                    scored=integrated.observe(values,row)
                    witnessed=set(scored['engine_run_obligations'])
                    before=set() if i==0 else previous
                    direct.observe(values,policy_search._semantic_fitness(scored,canonical,engine_active=True),
                                   policy_search._semantic_tiebreak(scored,canonical,engine_active=True),
                                   run_obligations=witnessed,covered_before=before,row=scored)
                    seen.add(space.key(space.validate(values)));previous=before|witnessed
                    expected=direct.next_candidate(rng,seen);seen.add(space.key(expected))
                    actual=integrated.next()
                    self.assertEqual(actual,expected.to_dict())
                    values=actual


if __name__=='__main__':unittest.main()
