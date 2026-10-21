"""Follow the current ego in CARLA's visible spectator view without ticking it."""
import argparse
import time
import carla


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--hours", type=float, default=8)
    args = parser.parse_args()
    client = carla.Client("127.0.0.1", args.port)
    client.set_timeout(2)
    deadline = time.monotonic() + args.hours * 3600
    while time.monotonic() < deadline:
        try:
            world = client.get_world()
            vehicles = sorted(world.get_actors().filter("vehicle.*"), key=lambda actor: actor.id)
            ego = next((actor for actor in vehicles if actor.attributes.get("role_name") in ("hero", "ego")),
                       vehicles[0] if vehicles else None)
            if ego is not None:
                transform = ego.get_transform()
                forward = transform.get_forward_vector()
                position = transform.location - carla.Location(x=forward.x * 8, y=forward.y * 8) + carla.Location(z=4)
                world.get_spectator().set_transform(carla.Transform(position, carla.Rotation(pitch=-15, yaw=transform.rotation.yaw)))
        except RuntimeError:
            pass
        time.sleep(0.5)


if __name__ == "__main__":
    main()
