import numpy as np
from fiber_robotics_sim import demos, eskin


def test_selected_channel_current_frame_and_clamping():
    p = dict(
        sensor_count=8,
        touch_points=[(24, 30, 6)],
        skin_width_mm=80,
        skin_height_mm=60,
        receptive_width_mm=18,
        temperature_c=36,
        noise_nm=0.001,
        seed=7,
    )
    args = {**p, "inspection_channel": 99}
    demo = demos.skin_demo(args)
    assert demo["inspection_channel"] == 7
    assert args["inspection_channel"] == 99
    r = eskin.simulate_fbg_skin(**p)
    np.testing.assert_allclose(
        demo["frames"][demo["initial_index"]]["signals"], r["compensated_shift_nm"]
    )
    assert "SKIN-N08-08" in demo["subtitle"]
    assert "本动画帧温补响应" in demo["frames"][60]["caption"]
    assert demos.skin_demo(p)["inspection_channel"] is None
