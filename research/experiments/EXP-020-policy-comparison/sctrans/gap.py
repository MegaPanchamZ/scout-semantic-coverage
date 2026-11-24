"""Horizontal distance between the actual world-space oriented CARLA boxes."""
from shapely.geometry import MultiPoint


def footprint(actor):
    vertices=actor.bounding_box.get_world_vertices(actor.get_transform())
    return MultiPoint([(v.x,v.y) for v in vertices]).convex_hull


def bounding_box_gap(ego,other):
    return float(footprint(ego).distance(footprint(other)))
