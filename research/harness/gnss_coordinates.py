from __future__ import annotations

from typing import Any, Callable

import numpy as np


class GnssRouteNormalizer:
    """Convert measured GNSS into the legacy route planner's coordinate frame.

    CARLA map projections can reverse latitude relative to the legacy
    Mercator conversion. Calibrate once from map geometry, then transform
    the actual sensor samples (including their noise), never ego ground truth.
    The local affine calibration also accounts for projection scale/offset.
    """

    def __init__(self, carla_map: Any, reference: Any,
                 route_gps: Callable[[Any], dict[str, float]]) -> None:
        location_type = type(reference)
        anchors = [
            location_type(x=reference.x, y=reference.y, z=reference.z),
            location_type(x=reference.x + 100.0, y=reference.y, z=reference.z),
            location_type(x=reference.x, y=reference.y + 100.0, z=reference.z),
        ]
        observed, expected = [], []
        for location in anchors:
            geo = carla_map.transform_to_geolocation(location)
            target = route_gps(location)
            observed.append([geo.latitude, geo.longitude])
            expected.append([target["lat"], target["lon"]])
        self._observed_origin = np.asarray(observed[0])
        self._expected_origin = np.asarray(expected[0])
        source_basis = (np.asarray(observed[1:]) - observed[0]).T
        target_basis = (np.asarray(expected[1:]) - expected[0]).T
        self._matrix = np.linalg.solve(source_basis.T, target_basis.T).T
        if not np.isfinite(self._matrix).all():
            raise ValueError("Invalid CARLA GNSS coordinate calibration")

    def normalize(self, sample: Any) -> np.ndarray:
        corrected = np.asarray(sample, dtype=np.float64).copy()
        corrected[:2] = self._expected_origin + self._matrix @ (
            corrected[:2] - self._observed_origin
        )
        return corrected
