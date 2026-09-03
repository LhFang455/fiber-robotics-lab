import numpy as np

from fiber_robotics_sim.eskin import (
    simulate_dynamic_skin_event,
    simulate_fbg_skin,
    simulate_pressure_reconstruction,
    simulate_triaxial_taxel,
)
from fiber_robotics_sim.eskin_experiments import (
    ESKIN_PRESETS,
    eskin_csv_bytes,
    eskin_report_bytes,
    eskin_result_records,
)
from fiber_robotics_sim.eskin_visuals import (
    dynamic_event_figure,
    fbg_skin_figure,
    pressure_reconstruction_figure,
    taxel_channel_figure,
)


def sample_results():
    return {
        "三轴单元": simulate_triaxial_taxel(
            fx_n=2, fy_n=1, fz_n=8, temperature_c=35, reference_match=0.98, seed=1
        ),
        "光学皮肤": simulate_fbg_skin(
            sensor_count=8, touch_points=[(30, 30, 8)], temperature_c=30, seed=1
        ),
        "压力重建": simulate_pressure_reconstruction("双点接触", 4, 16, seed=1),
        "动态判别": simulate_dynamic_skin_event("即将滑移", seed=1),
    }


def test_presets_cover_every_electronic_skin_experiment_mode():
    assert set(ESKIN_PRESETS) == {
        "三轴温漂校正", "双点光学触觉", "稀疏双点重建", "即将滑移", "热物体"
    }
    assert {preset["mode"] for preset in ESKIN_PRESETS.values()} == {
        "三轴单元", "光学皮肤", "压力重建", "动态判别"
    }
    for preset in ESKIN_PRESETS.values():
        assert preset["description"]
        assert preset["state"]
        assert all(key.startswith("eskin_") for key in preset["state"])


def test_result_records_and_csv_use_tidy_section_specific_columns():
    expected_columns = {
        "三轴单元": {"channel", "active_pf", "reference_pf", "corrected_pf"},
        "光学皮肤": {"sensor", "x_mm", "y_mm", "measured_shift_nm", "compensated_shift_nm"},
        "压力重建": {"row", "column", "truth_kpa", "reconstruction_kpa", "error_kpa"},
        "动态判别": {"time_s", "normal_force_n", "shear_force_n", "shear_ratio", "centroid_x_mm", "temperature_c"},
    }
    for section, result in sample_results().items():
        records = eskin_result_records(section, result)
        assert records
        assert expected_columns[section] <= set(records[0])
        csv_text = eskin_csv_bytes(section, result).decode("utf-8-sig")
        assert all(column in csv_text.splitlines()[0] for column in expected_columns[section])


def test_report_marks_simulation_boundary_and_keeps_parameters_and_metrics():
    report = eskin_report_bytes(
        "三轴触觉单元",
        {"温度": "35 °C", "参考匹配": "98%"},
        {"校正后 MAE": "0.12 N"},
        "灵敏度矩阵不是器件标定数据。",
    ).decode("utf-8-sig")
    assert "# 电子皮肤教学实验报告" in report
    assert "## 输入参数" in report
    assert "## 结果指标" in report
    assert "35 °C" in report and "0.12 N" in report
    assert "教学仿真" in report
    assert "不是器件标定数据" in report


def test_electronic_skin_figures_expose_the_expected_signals():
    results = sample_results()
    taxel = taxel_channel_figure(results["三轴单元"])
    optical = fbg_skin_figure(results["光学皮肤"], 80.0, 60.0)
    pressure = pressure_reconstruction_figure(results["压力重建"])
    dynamic = dynamic_event_figure(results["动态判别"])

    assert len(taxel.data) == 3
    assert {trace.name for trace in taxel.data} == {"主动信号", "参考信号", "校正信号"}
    assert len(optical.data) >= 2
    assert len(pressure.data) == 3
    assert all(np.asarray(trace.z).ndim == 2 for trace in pressure.data)
    assert len(dynamic.data) >= 4
