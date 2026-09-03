from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def metric(app, label):
    return next(item.value for item in app.metric if item.label == label)


def test_shape_health_optical_and_chain_form_complete_experiment_flows():
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()

    app.selectbox(key="shape_preset").set_value("芯间温差对照")
    app.button(key="load_shape_preset").click().run()
    assert not app.exception
    assert app.slider(key="shape_temperature_gradient").value == 8.0
    assert metric(app, "中心线 RMSE").endswith("mm")
    assert metric(app, "末端误差").endswith("mm")
    app.button(key="save_shape_baseline").click().run()
    assert app.session_state["shape_baseline"] is not None

    app.selectbox(key="health_preset").set_value("高密度阵列对照")
    app.button(key="load_health_preset").click().run()
    assert not app.exception
    assert app.select_slider(key="arm_fbg_count").value == 8
    assert metric(app, "定位误差").endswith("mm")
    app.button(key="save_health_baseline").click().run()
    assert app.session_state["health_baseline"] is not None

    app.selectbox(key="optical_preset").set_value("温度交叉敏感")
    app.button(key="load_optical_preset").click().run()
    assert not app.exception
    assert app.slider(key="global_temperature").value == 20.0
    assert metric(app, "EFPI 腔长变化").endswith("nm")
    assert metric(app, "温度引起的椭圆率偏移").endswith("°")
    app.button(key="save_optical_baseline").click().run()
    assert app.session_state["optical_baseline"] is not None

    app.selectbox(key="chain_preset").set_value("强滤波对照")
    app.button(key="load_chain_preset").click().run()
    assert app.select_slider(key="chain_filter_window").value == 15
    assert metric(app, "滤波后噪声 RMS").endswith("nm")
    assert metric(app, "理论滤波延迟").endswith("ms")
    assert metric(app, "指令一致率").endswith("%")
    app.button(key="save_chain_baseline").click().run()
    assert app.session_state["chain_baseline"] is not None

    app.selectbox(key="chain_experiment").set_value("连续体形状重建").run()
    assert any("任务步骤" in item.value for item in app.markdown)
    labels = {button.label for button in app.get("download_button")}
    assert {
        "下载形状重建 CSV", "下载形状重建报告",
        "下载健康监测 CSV", "下载健康监测报告",
        "下载偏振与干涉 CSV", "下载偏振与干涉报告",
        "下载当前实验报告",
    } <= labels


def test_optical_chain_and_assembly_expose_learning_scaffolds_and_comparisons():
    source = APP_PATH.read_text(encoding="utf-8")
    optical = source[source.index("with polarization_tab:"):source.index("with chain_tab:")]
    chain = source[source.index("with chain_tab:"):source.index("with assembly_tab:")]
    assembly = source[source.index("with assembly_tab:"):]

    assert "module_learning_frame(" in optical
    assert "机制响应曲线" in optical
    assert "基线 A 与当前 B" in optical
    assert "module_learning_frame(" in chain
    assert "滤波窗口" in chain
    assert "基线 A 与当前 B" in chain
    assert "module_learning_frame(" in assembly


def test_invalid_direction_health_localisation_and_chain_calibration_show_not_applicable():
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()

    app.slider(key="shape_curvature").set_value(0.0).run()
    assert metric(app, "反演弯曲方向") == "不适用"

    app.selectbox(key="health_preset").set_value("健康基线")
    app.button(key="load_health_preset").click().run()
    assert metric(app, "可疑位置") == "未形成定位"
    assert any("点式 FBG 阵列" in item.value for item in app.subheader)

    app.slider(key="calibration_offset").set_value(0.0)
    app.selectbox(key="chain_experiment").set_value("弯曲标定与温补").run()
    assert metric(app, "弯曲标定与温补：角度反演误差") == "不可反演"


def test_shape_comparison_preserves_all_inputs_that_change_the_result():
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()

    app.button(key="save_shape_baseline").click().run()
    table = next(table.value for table in app.dataframe if "基线 A" in table.value.columns)

    assert {"光纤长度", "温度变化", "随机种子"} <= set(table["指标"])


def test_distributed_page_has_presets_baseline_and_reproducible_exports():
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()

    app.selectbox(key="distributed_preset").set_value("稀疏采样对照")
    app.button(key="load_distributed_preset").click().run()
    assert app.slider(key="distributed_spatial_spacing").value == 40
    app.button(key="save_distributed_baseline").click().run()
    assert app.session_state["distributed_baseline"] is not None

    labels = {button.label for button in app.get("download_button")}
    assert "下载分布式实验 CSV" in labels
    assert "下载分布式实验报告" in labels


def test_sidebar_quick_navigation_can_reach_the_last_domain_lab():
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()

    app.selectbox(key="sidebar_module_navigation").set_value("⑥ 分布式光学与数据").run()

    assert app.session_state["main_navigation"] == "⑥ 分布式光学与数据"
