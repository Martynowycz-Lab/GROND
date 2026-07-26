from types import SimpleNamespace

import numpy as np

from grond.legacy.engine.build_dynrows_xds import (
    header_detector_geometry,
    xds_detector_origin_and_beam_center,
)


def test_xds_org_is_not_off_axis_direct_beam_center():
    origin, beam_center, normal = xds_detector_origin_and_beam_center(
        fast=[1.0, 0.0, 0.0],
        slow=[0.0, 1.0, 0.0],
        beam_direction=[1.140061, 0.050374, 39.857662],
        orgx=23.36,
        orgy=490.19,
        qx=0.056,
        qy=0.056,
        distance=961.06,
    )

    assert np.allclose(origin, [-1.30816, -27.45064, 961.06])
    assert np.allclose(normal, [0.0, 0.0, 1.0])
    assert np.allclose(beam_center, [514.244, 511.880], atol=0.002)


def test_header_detector_geometry_reports_physical_origin_and_beam_center():
    data = SimpleNamespace(
        header={
            "NX": "1024 NY= 1024 QX= 0.056 QY= 0.056",
            "ORGX": "23.36 ORGY= 490.19 DETECTOR_DISTANCE= 961.06",
            "DIRECTION_OF_DETECTOR_X-AXIS": "1 0 0",
            "DIRECTION_OF_DETECTOR_Y-AXIS": "0 1 0",
            "DETECTOR": "GENERIC",
        },
        s0=np.asarray([1.140061, 0.050374, 39.857662]),
        iseg=np.asarray([0]),
    )

    geometry = header_detector_geometry(data)

    assert geometry["xds_org_px"] == [23.36, 490.19]
    assert np.allclose(
        geometry["detector_origin_mm"], [-1.30816, -27.45064, 961.06]
    )
    assert np.allclose(
        geometry["beam_center_px"], [514.244, 511.880], atol=0.002
    )
