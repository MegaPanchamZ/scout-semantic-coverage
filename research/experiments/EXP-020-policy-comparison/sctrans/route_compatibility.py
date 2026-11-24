"""Bound PCLA's topology walk at CARLA dead ends on generated maps.

This repairs route input construction, not agent inference or returned controls.
The original PCLA DAO indexes next()[0] even at map boundaries. Keep advertised
endpoints and stop sampling when no successor exists. Never invent a connection.
"""
import math


def bounded_topology(dao):
    topology=[]
    for entry,exit in dao._wmap.get_topology():
        a,b=entry.transform.location,exit.transform.location
        segment={'entry':entry,'exit':exit,
                 'entryxyz':tuple(round(v) for v in (a.x,a.y,a.z)),
                 'exitxyz':tuple(round(v) for v in (b.x,b.y,b.z)), 'path':[]}
        current=entry;seen=set()
        for _ in range(10000):
            if current.transform.location.distance(b)<=dao._sampling_resolution:break
            successors=current.next(dao._sampling_resolution)
            if not successors:break
            # The advertised segment identifies its lane. Do not select an unrelated branch.
            same=[w for w in successors if (w.road_id,w.section_id,w.lane_id)==
                  (entry.road_id,entry.section_id,entry.lane_id)]
            if not same:break
            current=min(same,key=lambda w:w.transform.location.distance(b))
            key=(current.road_id,current.section_id,current.lane_id,round(current.s,3))
            if key in seen:raise RuntimeError('Cycle while sampling route topology')
            seen.add(key);segment['path'].append(current)
        else:raise RuntimeError('Unbounded route topology segment')
        topology.append(segment)
    return topology


def install():
    from leaderboard_codes.global_route_planner_dao import GlobalRoutePlannerDAO
    GlobalRoutePlannerDAO.get_topology=bounded_topology
