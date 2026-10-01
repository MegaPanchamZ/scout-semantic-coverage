from __future__ import annotations

import math
from typing import Any

from research.harness.models import CollisionEvent, SafetyMetrics


class SafetyOracle:
    def __init__(self) -> None:
        self.metrics = SafetyMetrics()
        self._stuck_ticks = 0
    
    def tick(self, world: Any, ego_vehicle: Any) -> None:
        try:
            ego_vel = ego_vehicle.get_velocity()
            speed = math.sqrt(ego_vel.x ** 2 + ego_vel.y ** 2 + ego_vel.z ** 2)
            
            if speed < 0.1:
                self._stuck_ticks += 1
                if self._stuck_ticks > 50:
                    self.metrics.stuck_frames += 1
            else:
                self._stuck_ticks = 0
                
            ego_loc = ego_vehicle.get_location()
            
            for actor in world.get_actors().filter('*vehicle*'):
                if actor.id == ego_vehicle.id:
                    continue
                
                other_loc = actor.get_location()
                other_vel = actor.get_velocity()
                
                dx = other_loc.x - ego_loc.x
                dy = other_loc.y - ego_loc.y
                dist = math.sqrt(dx**2 + dy**2)
                
                rel_vx = ego_vel.x - other_vel.x
                rel_vy = ego_vel.y - other_vel.y
                
                if dist > 0:
                    nx, ny = dx / dist, dy / dist
                    approach_speed = rel_vx * nx + rel_vy * ny
                    if approach_speed > 0:
                        ttc = dist / approach_speed
                        if ttc < self.metrics.min_ttc:
                            self.metrics.min_ttc = ttc
                            
                # Near collision definition: within 2.5 meters
                if 0 < dist < 2.5:
                    self.metrics.near_collisions += 1
                    
            if ego_vehicle.is_at_traffic_light():
                tl = ego_vehicle.get_traffic_light()
                if tl and str(tl.get_state()) == "Red" and speed > 2.0:
                    self.metrics.rule_violations += 1
        except Exception:
            pass


class CollisionOracle:
    def __init__(self) -> None:
        self.events: list[CollisionEvent] = []
        self.injected_actor_ids: set[int] = set()
        self._sensor: Any | None = None

    def attach(self, world: Any, ego_vehicle: Any) -> Any:
        blueprint = world.get_blueprint_library().find("sensor.other.collision")
        sensor = world.spawn_actor(blueprint, self._identity_transform(world), attach_to=ego_vehicle)
        weak_self = self

        def _on_collision(event: Any) -> None:
            impulse = event.normal_impulse
            intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
            other_actor = event.other_actor
            weak_self.events.append(
                CollisionEvent(
                    frame=event.frame,
                    actor_id=other_actor.id,
                    actor_type=other_actor.type_id,
                    intensity=float(intensity),
                    injected_actor=other_actor.id in weak_self.injected_actor_ids,
                )
            )

        sensor.listen(_on_collision)
        self._sensor = sensor
        return sensor

    def destroy(self) -> None:
        if self._sensor is not None:
            try:
                self._sensor.stop()
            except RuntimeError:
                pass
            try:
                self._sensor.destroy()
            except RuntimeError:
                pass
            self._sensor = None

    @staticmethod
    def _identity_transform(world: Any) -> Any:
        carla = type(world).__module__.split(".")[0]
        module = __import__(carla)
        return module.Transform()
