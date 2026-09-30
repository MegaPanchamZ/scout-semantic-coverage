#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

# Try importing tomli or tomllib (Python 3.11+)
try:
    import tomllib
except ImportError:
    import tomli as tomllib


class DSLParser:
    """Parses TOML-based DSL definitions for controllable world generation."""

    WEATHER_MAPPINGS = {
        "ClearNoon": "bright sunny daylight, clear skies, sharp shadows",
        "ClearSunset": "golden hour sunset lighting, warm orange glow, long shadows",
        "HeavyRain": "heavy downpour, wet roads with puddles and reflections, rainy windshield",
        "WetNoon": "overcast day, damp wet roads, soft diffuse shadows",
        "SoftRain": "light drizzle, foggy atmosphere, misty air",
    }

    TIME_MAPPINGS = {
        "Noon": "midday sun",
        "Sunset": "sunset golden hour",
        "Night": "nighttime streetlights, dark sky, headlights casting beams",
    }

    MANEUVER_MAPPINGS = {
        "highway_entry": "ego vehicle entering a highway on-ramp",
        "highway_exit": "ego vehicle taking a highway exit",
        "lane_change": "ego vehicle making a lane change",
        "overtaking": "ego vehicle overtaking another vehicle",
        "merging": "traffic merging into the ego vehicle's lane",
        "intersection": "approaching a busy intersection",
        "pedestrian_crossing": "approaching a pedestrian crossing",
    }

    def __init__(self, dsl_path: Path):
        self.dsl_path = dsl_path
        self.data = self._load(dsl_path)

    def _load(self, path: Path) -> Dict[str, Any]:
        with open(path, "rb") as f:
            return tomllib.load(f)

    def compile_cosmos_prompt(self) -> str:
        """Compiles DSL visual tags and base prompt into a rich Cosmos generation prompt."""
        dsl = self.data.get("dsl") or {}
        base_prompt = self.data.get("cosmos_prompt", {}).get("base_prompt") or "A front dashcam video of a car driving."
        style = self.data.get("cosmos_prompt", {}).get("style") or "photorealistic, 4k movie, highly detailed"

        modifiers = []

        weather = dsl.get("weather")
        if weather in self.WEATHER_MAPPINGS:
            modifiers.append(self.WEATHER_MAPPINGS[weather])
        elif weather:
            modifiers.append(f"weather style of {weather}")

        time_of_day = dsl.get("time_of_day")
        if time_of_day in self.TIME_MAPPINGS:
            modifiers.append(self.TIME_MAPPINGS[time_of_day])
        elif time_of_day:
            modifiers.append(f"during {time_of_day}")

        road = dsl.get("road_type")
        if road:
            modifiers.append(f"on an {road.lower()}")

        maneuver = dsl.get("maneuver_type")
        if maneuver in self.MANEUVER_MAPPINGS:
            modifiers.append(self.MANEUVER_MAPPINGS[maneuver])
        elif maneuver:
            modifiers.append(f"performing {maneuver.replace('_', ' ')}")

        full_prompt = base_prompt
        if modifiers:
            full_prompt += ", " + ", ".join(modifiers)
        if style:
            full_prompt += ", " + style

        return full_prompt

    def get_stsg_obligations(self) -> Dict[str, Any]:
        """Compiles the target semantic contracts to be verified by the STSG oracle."""
        ob = self.data.get("stsg_obligations") or {}
        nodes = list(ob.get("nodes") or [])
        attributes = list(ob.get("attributes") or [])
        relations = list(ob.get("relations") or [])
        
        # Automatically derive relations from maneuver_type if provided
        dsl = self.data.get("dsl") or {}
        maneuver = dsl.get("maneuver_type")
        if maneuver == "lane_change":
            if "ego" not in nodes:
                nodes.append("ego")
            relations.append("changes_lane(ego)")
        elif maneuver == "overtaking":
            if "ego" not in nodes:
                nodes.append("ego")
            if "npc1" not in nodes:
                nodes.append("npc1")
            relations.append("overtakes(ego, npc1)")
        elif maneuver == "merging":
            if "ego" not in nodes:
                nodes.append("ego")
            if "npc1" not in nodes:
                nodes.append("npc1")
            relations.append("merges_behind(npc1, ego)")
            
        return {
            "nodes": list(set(nodes)),
            "attributes": list(set(attributes)),
            "relations": list(set(relations)),
        }

    def validate(self) -> bool:
        """Validates that the required schema fields exist and are well-formed."""
        if "dsl" not in self.data:
            raise ValueError("Missing [dsl] section in DSL TOML")
        if "stsg_obligations" not in self.data:
            raise ValueError("Missing [stsg_obligations] section in DSL TOML")
        
        # Verify lists
        ob = self.data["stsg_obligations"]
        for key in ["nodes", "attributes", "relations"]:
            if key in ob and not isinstance(ob[key], list):
                raise TypeError(f"stsg_obligations.{key} must be a list of strings")
        
        return True


def main():
    parser = argparse.ArgumentParser(description="Parse and compile ADS generation DSL TOML.")
    parser.add_argument("dsl_toml", type=Path, help="Path to the DSL TOML file.")
    parser.add_argument("--validate-only", action="store_true", help="Only validate without printing.")

    args = parser.parse_args()

    if not args.dsl_toml.exists():
        print(f"Error: file not found '{args.dsl_toml}'", file=sys.stderr)
        sys.exit(1)

    try:
        dsl = DSLParser(args.dsl_toml)
        dsl.validate()
        if args.validate_only:
            print("DSL TOML structure is valid.")
            sys.exit(0)

        prompt = dsl.compile_cosmos_prompt()
        ob = dsl.get_stsg_obligations()

        print("--- Compiled Cosmos Prompt ---")
        print(prompt)
        print("\n--- Target STSG Obligations ---")
        print(f"Nodes: {ob['nodes']}")
        print(f"Attributes: {ob['attributes']}")
        print(f"Relations: {ob['relations']}")

    except Exception as e:
        print(f"Validation Error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
