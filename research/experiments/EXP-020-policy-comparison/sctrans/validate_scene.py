#!/usr/bin/env python3
"""Run one source replay with a fresh, unmodified InterFuser instance.

NPC poses are reset to recorded samples each tick; finite-difference velocities
advance them to the next sample through CARLA physics. Actual post-tick error
is measured, never assumed. Collisions come exclusively from CARLA's sensor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

from source_replay import load_source,validation_verdict
from paths import workspace_root
from variants import sample_actor,validate_candidate,DEFAULTS
from map_preflight import audit_map

ROOT=workspace_root()
sys.path.insert(0,str(ROOT))

TARGETS={
    'DEU_Flensburg-10_1_T-1.xml':['hazard(other: turning)'],
    'DEU_Ffb-1_3_T-1.xml':['hazard(other: turning)'],
    'USA_Lanker-2_8_T-1.xml':['hazard(other: oncoming)'],
    # Source audit found parallel, same-direction traffic; an oncoming target
    # would invent an interaction. Keep this as a map/ADS feasibility check.
    'CHN_Cho-1_2_T-1.xml':[],
    'CHN_Sha-14_1_T-1.xml':['hazard(other: turning)'],
    'USA_Peach-4_1_T-1.xml':['hazard(other: lateral)'],
}


def write_json(path,payload):
    path.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')


def evaluate(args):
    print('Loading CARLA and unmodified InterFuser dependencies',flush=True)
    import carla
    from gap import bounding_box_gap
    from research.harness.coverage_engine import compute_cov,load_oracle,load_semantic_trace
    from research.harness.models import RunResult,ScenarioSpec
    from research.harness.observers.semantic import SemanticObserver,SemanticObserverConfig
    from research.harness.pcla_bridge import PclaAdapter

    source=load_source(args.source)
    candidate=validate_candidate(json.loads(args.candidate.read_text()),source) if args.candidate else None
    args.output.mkdir(parents=True,exist_ok=True)
    record={'source':source.manifest(),'map':str(args.map),
            'map_sha256':hashlib.sha256(args.map.read_bytes()).hexdigest(),
            'replay_mode':'recorded_trajectories_with_physics_and_measured_error',
            'map_fidelity':'upstream_commonroad_to_opendrive_conversion',
            'ego_initial_speed_mps':source.ego.speed,'ads_control_overrides':0,
            'collision_source':'sensor.other.collision',
            'gap_metric':'horizontal_world_oriented_bounding_box_polygon_distance_m',
            'targets':TARGETS[args.source.name], 'spawn_failures':[],
            'actor_mapping':{},'collision_events':[], 'max_replay_error_m':0.0,
            'map_validated':False,'map_checks':{},'ego_displacement_m':0.0,
            'max_replay_yaw_error_deg':0.0,'ticks_expected':round(source.end_time/source.dt),
            'intended_hazard_detected':False,'intended_predicates_detected':False,
            'diagnostic_no_npcs':args.no_npcs}
    record['candidate']=candidate
    record['fidelity_class']='source_derived_variant' if candidate and any(candidate[k]!=v for k,v in DEFAULTS.items()) else 'recorded_actor_replay_on_converted_map'
    record['validation_schema_version']=1
    record['code_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                          [Path(__file__),Path(__file__).with_name('source_replay.py'),
                           Path(__file__).with_name('variants.py'),ROOT/'research/harness/pcla_bridge.py',
                           ROOT/'research/harness/coverage_engine.py',ROOT/'research/harness/observers/semantic.py',
                           ROOT/'research/models/PCLA/pcla_agents/interfuser/interfuser_agent.py']}
    record['oracle_sha256']=hashlib.sha256(args.oracle.read_bytes()).hexdigest()
    record['map_preflight']=audit_map(args.map)
    record['route_compatibility_fix']='bounded_topology_walk' if args.safe_route_topology else None
    write_json(args.output/'provenance.json',record)
    client=carla.Client('127.0.0.1',args.port);client.set_timeout(180)
    owned=[];adapter=None;world=None;original=None;obs=None;ctx=None
    start=time.monotonic();collisions=[];ticks=0
    try:
        if not source.goals or source.end_time<2:
            raise ValueError('Source has no spatial goal or insufficient recorded duration')
        if not record['map_preflight']['passed']:
            raise ValueError('Converted map failed preflight: '+str(record['map_preflight']['issues']))
        print('Generating source map',args.source.name,flush=True)
        world=client.generate_opendrive_world(args.map.read_text(),carla.OpendriveGenerationParameters(
            vertex_distance=1.0,max_road_length=50.0,wall_height=0.0,
            additional_width=0.0,smooth_junctions=True,enable_mesh_visibility=True))
        original=world.get_settings();settings=world.get_settings()
        settings.synchronous_mode=True;settings.fixed_delta_seconds=source.dt
        settings.substepping=True;settings.max_substep_delta_time=.01;settings.max_substeps=10
        world.apply_settings(settings);world.tick()
        world.set_weather(carla.WeatherParameters.ClearNoon)
        world_map=world.get_map();lib=world.get_blueprint_library()

        def pose(state,z=.3):
            return carla.Transform(carla.Location(x=state.x,y=-state.y,z=z),
                                   carla.Rotation(yaw=-math.degrees(state.yaw)))

        def road_error(state):
            loc=pose(state).location
            wp=world_map.get_waypoint(loc,project_to_road=True,lane_type=carla.LaneType.Driving)
            if wp is None:return 1e9
            # A point inside the converted lane need not lie on its centerline.
            return max(0.0,math.hypot(loc.x-wp.transform.location.x,loc.y-wp.transform.location.y)-wp.lane_width/2)

        checks=[road_error(source.ego)]
        for actor in source.actors:
            checks.extend(road_error(s) for s in actor.states[::10])
        goal=carla.Location(x=source.goals[0][0],y=-source.goals[0][1],z=.3)
        goal_wp=world_map.get_waypoint(goal,project_to_road=True,lane_type=carla.LaneType.Driving)
        goal_error=goal.distance(goal_wp.transform.location) if goal_wp else 1e9
        record['map_checks']={'max_recorded_point_outside_driving_lane_m':max(checks),
             'ego_outside_lane_m':checks[0], 'goal_centerline_distance_m':goal_error,
             'sample_count':len(checks),'tolerance_m':1.0}
        record['map_validated']=max(checks)<=1.0 and goal_error<=5.0
        print('MAP_CHECK',json.dumps(record['map_checks']),flush=True)

        ego_bp=lib.find('vehicle.lincoln.mkz_2017');ego_bp.set_attribute('role_name','hero')
        ego=world.try_spawn_actor(ego_bp,pose(source.ego))
        if ego is None:raise RuntimeError('Ego could not spawn at source initial position')
        owned.append(ego)
        sensor=world.spawn_actor(lib.find('sensor.other.collision'),carla.Transform(),attach_to=ego)
        owned.append(sensor)

        def on_collision(event):
            impulse=event.normal_impulse
            collisions.append({'frame':event.frame,'other_actor_id':event.other_actor.id,
               'other_actor_type':event.other_actor.type_id,
               'impulse':math.sqrt(impulse.x**2+impulse.y**2+impulse.z**2)})
        sensor.listen(on_collision)
        record['collision_sensor_attached']=True
        # Settle only before t=0; no controls or internal ADS state are overwritten.
        for _ in range(5):world.tick()
        adapter=PclaAdapter(world,ego,client,None,'if_if')
        if args.safe_route_topology:
            from route_compatibility import install
            install()
        adapter.set_destination(goal)
        for _ in range(3):world.tick()
        route=[{'x':wp.transform.location.x,'y':wp.transform.location.y}
               for wp in adapter._route_waypoints]
        if len(route)<2:raise RuntimeError('ADS route has fewer than two waypoints')
        record['route']=route

        ego.set_transform(pose(source.ego,.1))
        ego.set_target_velocity(carla.Vector3D(x=source.ego.speed*math.cos(source.ego.yaw),
                                              y=-source.ego.speed*math.sin(source.ego.yaw),z=0))
        spec=ScenarioSpec(args.source.stem,'OpenDriveMap','ClearNoon',0,0,
                          'SCTrans source replay; no ADS overrides',max_ticks=round(source.end_time/source.dt))
        obs=SemanticObserver(SemanticObserverConfig(trace_output_dir=args.output,
                       stream_output_dir=args.output,stream_every_ticks=1))
        ctx={'world':world,'carla':carla,'ego_vehicle':ego,'scenario':spec,'route_waypoints':route}
        obs.on_run_start(spec,ctx)
        initial=pose(source.ego).location;active={};spawned=set();min_gap=None;max_speed=0

        def spawn_actor(actor,state):
            bps={'car':'vehicle.tesla.model3','truck':'vehicle.carlamotors.carlacola',
                 'bus':'vehicle.mitsubishi.fusorosa','bicycle':'vehicle.bh.crossbike',
                 'motorcycle':'vehicle.harley-davidson.low_rider','pedestrian':'walker.pedestrian.0001'}
            bp_id=bps.get(actor.kind)
            if bp_id is None:
                record['spawn_failures'].append({'id':actor.actor_id,'reason':'unsupported_class','class':actor.kind});return
            bp=lib.find(bp_id)
            if bp.has_attribute('role_name'):bp.set_attribute('role_name','source_'+actor.actor_id)
            if bp.has_attribute('is_invincible'):bp.set_attribute('is_invincible','false')
            a=world.try_spawn_actor(bp,pose(state,.15 if actor.kind!='pedestrian' else .9))
            if a is None:
                record['spawn_failures'].append({'id':actor.actor_id,'reason':'spawn_collision_or_invalid_pose'});return
            owned.append(a)
            record['actor_mapping'][actor.actor_id]={'carla_id':a.id,'blueprint':a.type_id,
                'source_dimensions_m':[actor.length,actor.width],
                'carla_dimensions_m':[2*a.bounding_box.extent.x,2*a.bounding_box.extent.y]}
            active[actor.actor_id]=a

        # The first simulation step uses source initial state; callbacks from setup are excluded by frame.
        first_frame=world.get_snapshot().frame+1
        with (args.output/'telemetry.jsonl').open('w') as log:
            for i in range(spec.max_ticks):
                t=round(i*source.dt,8);next_t=round((i+1)*source.dt,8);expected={}
                if not args.no_npcs:
                    for actor in source.actors:
                        state=sample_actor(actor,t,candidate);nxt=sample_actor(actor,next_t,candidate)
                        # Do not leave a stationary actor beyond its recorded horizon.
                        if state is None or nxt is None:
                            old=active.pop(actor.actor_id,None)
                            if old is not None:old.destroy()
                            continue
                        if actor.actor_id not in spawned:
                            spawn_actor(actor,state);spawned.add(actor.actor_id)
                        a=active.get(actor.actor_id)
                        if a is None:continue
                        current_tf=a.get_transform()
                        a.set_transform(pose(state,max(.05,current_tf.location.z)))
                        if nxt is not None:
                            a.set_target_velocity(carla.Vector3D(x=(nxt.x-state.x)/source.dt,
                                                   y=-(nxt.y-state.y)/source.dt,z=0))
                            dyaw=math.atan2(math.sin(nxt.yaw-state.yaw),math.cos(nxt.yaw-state.yaw))
                            a.set_target_angular_velocity(carla.Vector3D(z=-math.degrees(dyaw)/source.dt))
                            expected[actor.actor_id]=nxt
                        else:a.set_target_velocity(carla.Vector3D())
                frame=world.tick();ticks=i+1
                control=adapter.run_step()
                ego.apply_control(control)
                loc=ego.get_location();vel=ego.get_velocity();speed=math.hypot(vel.x,vel.y)
                max_speed=max(max_speed,speed);tick_error=0;closest=None;gap=None
                for actor_id,a in active.items():
                    if actor_id in expected:
                        ex=expected[actor_id];actual=a.get_location()
                        err=math.hypot(actual.x-ex.x,actual.y+ex.y)
                        tick_error=max(tick_error,err)
                        yaw=math.radians(a.get_transform().rotation.yaw)+ex.yaw
                        yaw_error=abs(math.degrees(math.atan2(math.sin(yaw),math.cos(yaw))))
                        record['max_replay_yaw_error_deg']=max(record['max_replay_yaw_error_deg'],yaw_error)
                    d=bounding_box_gap(ego,a)
                    if gap is None or d<gap:gap=d;closest=actor_id
                record['max_replay_error_m']=max(record['max_replay_error_m'],tick_error)
                if gap is not None:min_gap=gap if min_gap is None else min(min_gap,gap)
                ctx['telemetry']={'speed_mps':speed,'speed_kph':3.6*speed,
                       'distance_to_goal_m':loc.distance(goal),
                       'control':{'throttle':float(control.throttle),'brake':float(control.brake),'steer':float(control.steer)},
                       'agent_step':adapter.last_step_info}
                obs.on_tick(ticks,ctx)
                log.write(json.dumps({'tick':ticks,'frame':frame,'t_s':next_t,'ego_x':loc.x,'ego_y':loc.y,
                     'speed_mps':speed,'control':ctx['telemetry']['control'],'bbox_gap_m':gap,
                     'closest_source_actor':closest,'max_replay_error_m':tick_error})+'\n');log.flush()
                if ticks%20==0:print('TICK',ticks,'speed',round(speed,2),'error',round(tick_error,3),flush=True)
        record.update(ego_displacement_m=ego.get_location().distance(initial),max_ego_speed_mps=max_speed,
                      min_gap_m=min_gap,ticks_executed=ticks)
        # Allow the asynchronous collision callback to deliver the final frame.
        time.sleep(.1)
        sensor.stop()
        record['collision_events']=[e for e in collisions if e['frame']>=first_frame]
        record['collision_count']=len(record['collision_events'])
        result=RunResult(spec.scenario_id,spec.town,'pcla',True,False,ticks,adapter.done(),
                         bool(record['collision_count']),record['collision_count'])
        obs.on_run_end(result,ctx)
        stream=Path(result.metadata['semantic']['stream_dump_path'])
        oracle=load_oracle(args.oracle)
        cov=compute_cov(oracle,load_semantic_trace(stream),ego_route=route)
        write_json(args.output/'coverage.json',cov)
        obligations=[o for axis in cov['dimensions'].values() for o in axis['obligations'] if o['covered']]
        matched=[o for o in obligations if o['signature'] in record['targets']]
        record['covered_obligations']=[o['signature'] for o in obligations]
        record['hazard_witnesses']=[o for o in obligations if o['signature'].startswith('hazard(')]
        record['intended_witnesses']=matched
        record['intended_hazard_detected']=bool(record['targets']) and set(record['targets'])<={o['signature'] for o in matched}
        mapped_ids={str(a['carla_id']) for a in record['actor_mapping'].values()}
        record['intended_predicates_detected']=bool(matched) and all(
            all(o.get('predicate_ticks',{}).get(p) for p in o['sim_predicate'].split(','))
            and str(o.get('witness_actor','')).split('#')[-1] in mapped_ids for o in matched)
        record['unintended_hazards']=[o['signature'] for o in record['hazard_witnesses'] if o['signature'] not in record['targets']]
        record['semantic_stream']=str(stream)
    except Exception as e:
        record['run_error']=f'{type(e).__name__}: {e}'
        (args.output/'error.txt').write_text(traceback.format_exc())
        print(traceback.format_exc(),flush=True)
    finally:
        if adapter is not None:
            try:adapter.destroy()
            except Exception as e:record['adapter_cleanup_error']=str(e)
        for a in reversed(owned):
            try:
                if a.is_alive:a.destroy()
            except Exception:pass
        if world is not None and original is not None:
            try:world.apply_settings(original)
            except Exception as e:record['settings_restore_error']=str(e)
        record['elapsed_s']=time.monotonic()-start
        record['validation']=validation_verdict(record)
        write_json(args.output/'result.json',record)
        print('VALIDATION',json.dumps(record['validation']),flush=True)
    return record


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--map',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--port',type=int,default=2510);p.add_argument('--no-npcs',action='store_true')
    p.add_argument('--safe-route-topology',action='store_true')
    p.add_argument('--candidate',type=Path,help='Explicit source-derived variant JSON; omit for original replay')
    p.add_argument('--oracle',type=Path,default=ROOT/'research/experiments/EXP-018-nuscenes-oracle-inventory/artifacts/oracle_inventory_v1.0-trainval.json')
    args=p.parse_args();result=evaluate(args)
    sys.stdout.flush();sys.stderr.flush()
    # PCLA/CARLA background sensor threads can block CPython shutdown after cleanup.
    os._exit(0 if result['validation']['passed'] else 1)


if __name__=='__main__':main()
