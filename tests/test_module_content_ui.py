from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def metric(app, label):
    return next(item.value for item in app.metric if item.label == label)


def test_tactile_scenario_exposes_metrics_baseline_and_exports():
    """捕获触觉场景不能载入、基线不能冻结或结果不可导出的问题。"""
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()

    app.selectbox(key="tactile_preset").set_value("高扰动薄板")
    app.button(key="load_tactile_preset").click().run()

    assert not app.exception
    assert app.selectbox(key="tactile_material").value == "薄板"
    assert app.slider(key="tactile_pattern_noise").value == 0.35
    assert app.slider(key="global_noise").value == 0.01
    assert metric(app, "类别间隔").endswith("百分点")
    assert metric(app, "重复一致率").endswith("%")

    app.button(key="save_tactile_baseline").click().run()
    baseline = app.session_state["tactile_baseline"]
    app.slider(key="tactile_pattern_noise").set_value(0.50).run()
    assert app.session_state["tactile_baseline"] == baseline
    tactile_table = next(table.value for table in app.dataframe if "基线 A" in table.value.columns)
    assert {"握持力", "接触面积", "模式扰动", "温度变化", "波长噪声", "随机种子"} <= set(
        tactile_table["指标"]
    )

    labels = {button.label for button in app.get("download_button")}
    assert "下载触觉实验 CSV" in labels
    assert "下载触觉实验报告" in labels


def test_tactile_baseline_table_marks_invalid_contact_metrics_not_applicable():
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()

    app.slider(key="tactile_grip_force").set_value(0.0).run()
    app.button(key="save_tactile_baseline").click().run()
    table = next(table.value for table in app.dataframe if "基线 A" in table.value.columns)
    values = dict(zip(table["指标"], table["基线 A"], strict=True))

    assert values["类别间隔"] == "不适用"
    assert values["模板偏差"] == "不适用"
    assert values["重复一致率"] == "不适用"


def test_foot_swing_scenario_marks_cop_limit_and_exports_comparison():
    """捕获摆动期仍显示可靠 CoP，或足底页缺少误差与导出的问题。"""
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()

    app.selectbox(key="global_failed_channel").set_value("足底区域 1")
    app.slider(key="global_drift").set_value(0.01)
    app.selectbox(key="foot_preset").set_value("摆动期低载荷")
    app.button(key="load_foot_preset").click().run()

    assert not app.exception
    assert app.selectbox(key="global_failed_channel").value == "无"
    assert app.slider(key="global_drift").value == 0.0
    assert app.selectbox(key="foot_support").value == "摆动期"
    assert metric(app, "区域 MAE").endswith("N")
    assert metric(app, "CoP 位置误差") != ""
    assert any("CoP 仅供参考" in item.value for item in app.info)

    app.button(key="save_foot_baseline").click().run()
    assert app.session_state["foot_baseline"] is not None
    foot_table = next(table.value for table in app.dataframe if "基线 A" in table.value.columns)
    assert {"输入载荷", "温度变化", "波长噪声", "模拟失效", "替代漂移"} <= set(
        foot_table["指标"]
    )

    labels = {button.label for button in app.get("download_button")}
    assert "下载足底实验 CSV" in labels
    assert "下载足底实验报告" in labels

    app.selectbox(key="foot_preset").set_value("平地中期")
    app.button(key="load_foot_preset").click().run()
    app.selectbox(key="global_failed_channel").set_value("足底区域 1").run()
    assert any("模拟通道失效" in item.value for item in app.info)
    assert not any("当前为低载荷状态" in item.value for item in app.info)
