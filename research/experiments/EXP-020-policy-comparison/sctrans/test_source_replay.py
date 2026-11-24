import math
import unittest
from source_replay import Actor,State,validation_verdict


class ReplayTests(unittest.TestCase):
    def complete(self):
        return dict(validation_schema_version=1,code_sha256={'runner':'test'},
                    source={'actor_count':1},actor_mapping={'7':{'carla_id':42}},
                    ticks_expected=20,ticks_executed=20,collision_sensor_attached=True,collision_count=0,
                    ego_displacement_m=4,intended_hazard_detected=True,
                    intended_predicates_detected=True,max_replay_error_m=.01,map_validated=True)

    def test_timestamps_and_heading_wrap(self):
        actor=Actor('7','car',5,2,[State(0,0,0,math.radians(179),2),State(.2,4,2,math.radians(-179),4)])
        mid=actor.at(.1)
        self.assertAlmostEqual(mid.x,2)
        self.assertAlmostEqual(mid.y,1)
        self.assertAlmostEqual(mid.yaw,math.pi)
        self.assertIsNone(actor.at(.3))
        self.assertIsNone(actor.at(-.1))

    def test_unrelated_hazard_and_stalled_ego_fail(self):
        row=self.complete()
        row.update(ego_displacement_m=.02,intended_hazard_detected=False,intended_predicates_detected=False)
        self.assertEqual(validation_verdict(row)['reasons'],
             ['ego_stalled','intended_hazard_not_detected','intended_predicates_not_detected'])

    def test_success_requires_all_evidence(self):
        row=self.complete()
        self.assertTrue(validation_verdict(row)['passed'])
        row['ads_control_overrides']=1
        self.assertFalse(validation_verdict(row)['passed'])
        row['ads_control_overrides']=0
        row['spawn_failures']=['7']
        self.assertFalse(validation_verdict(row)['passed'])

    def test_nan_incomplete_and_diagnostics_cannot_pass(self):
        for update,reason in [({'max_replay_error_m':float('nan')},'trajectory_mismatch'),
                             ({'ego_displacement_m':float('nan')},'ego_stalled'),
                             ({'ticks_executed':19},'incomplete_execution'),
                             ({'actor_mapping':{}},'incomplete_actor_replay'),
                             ({'diagnostic_no_npcs':True},'diagnostic_only'),
                             ({'collision_sensor_attached':False},'missing_collision_sensor_evidence')]:
            with self.subTest(reason=reason):
                row=self.complete();row.update(update)
                self.assertIn(reason,validation_verdict(row)['reasons'])


if __name__=='__main__': unittest.main()
