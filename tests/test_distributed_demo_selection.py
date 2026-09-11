import numpy as np
import pytest
from fiber_robotics_sim import experiments, demo_extensions


@pytest.mark.parametrize(
    "mode", ["Rayleigh/OFDR", "φ-OTDR / DAS", "Brillouin", "Raman"]
)
def test_exact_profile_selection_and_units(mode):
    p = dict(
        mode=mode,
        fiber_length_mm=300.0,
        event_position_mm=140.0,
        event_strength=600.0,
        spatial_spacing_mm=10,
        sample_rate_hz=50,
    )
    r = experiments.run_distributed_experiment(p)["results"]
    args = {**p, "inspection_index": 12}
    d = demo_extensions.distributed_demo(args)
    f = d["frames"][d["initial_index"]]
    assert f["sample_index"] == 12
    assert d["positions"][12] == r["profile_position_mm"][12]
    assert np.isclose(f["signals"][0], r["profile_value"][12])
    assert d["unit"] == r["unit"]
    assert args["inspection_index"] == 12
    if mode == "Brillouin":
        assert d["truth"] is None and d["estimate"] is None
    assert (
        demo_extensions.distributed_demo({**p, "inspection_index": 9999})[
            "initial_index"
        ]
        == len(d["positions"]) - 1
    )
