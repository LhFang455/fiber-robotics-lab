from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_electronic_skin_is_the_seventh_domain_lab_with_five_learning_areas():
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()
    assert not app.exception
    assert tuple(app.selectbox(key="sidebar_module_navigation").options)[-1] == "⑦ 电子皮肤与多模态感知"

    headings = {item.value for item in app.subheader}
    assert {
        "电子皮肤系统总览与机制对照",
        "三轴触觉单元：主动/参考信号与力反演",
        "FBG 光学皮肤：感受野、温补与压力质心",
        "稀疏压力重建：采样、插值与误差",
        "动态滑移与多模态决策",
    } <= headings


def test_electronic_skin_exposes_core_controls_metrics_baselines_and_exports():
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()
    assert not app.exception
    assert app.slider(key="eskin_fx_n")
    assert app.slider(key="eskin_reference_match")
    assert app.select_slider(key="eskin_fbg_sensor_count")
    assert app.selectbox(key="eskin_contact_mode")
    assert app.selectbox(key="eskin_pressure_scenario")
    assert app.select_slider(key="eskin_sparse_size")
    assert app.selectbox(key="eskin_dynamic_event")
    assert app.slider(key="eskin_slip_threshold")
    assert app.slider(key="eskin_repeat_count")

    metric_labels = {item.label for item in app.metric}
    assert {
        "校正后力 MAE", "矩阵条件数", "载荷误差", "质心定位误差",
        "压力场 RMSE", "通道节省率", "滑移判定", "重复告警率",
    } <= metric_labels

    button_keys = {item.key for item in app.button}
    assert {
        "save_eskin_taxel_baseline", "save_eskin_fbg_baseline",
        "save_eskin_pressure_baseline", "save_eskin_dynamic_baseline",
    } <= button_keys

    downloads = {item.label for item in app.get("download_button")}
    assert {
        "下载三轴单元 CSV", "下载三轴单元报告",
        "下载光学皮肤 CSV", "下载光学皮肤报告",
        "下载压力重建 CSV", "下载压力重建报告",
        "下载动态事件 CSV", "下载动态事件报告",
    } <= downloads


def test_electronic_skin_preset_updates_the_matching_internal_experiment():
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()
    app.selectbox(key="eskin_preset").set_value("双点光学触觉")
    app.button(key="load_eskin_preset").click().run()

    assert not app.exception
    assert app.session_state["eskin_navigation"] == "FBG 光学皮肤"
    assert app.select_slider(key="eskin_fbg_sensor_count").value == 8
    assert app.selectbox(key="eskin_contact_mode").value == "双点"
    assert app.slider(key="eskin_touch2_force").value == 4.0


def test_electronic_skin_states_boundaries_and_research_links_are_explicit():
    source = APP_PATH.read_text(encoding="utf-8")
    assert "透明教学模型" in source
    assert "不能替代器件标定、实物测试或安全认证" in source
    assert "https://www.nature.com/articles/s42256-022-00487-3" in source
    assert "https://opg.optica.org/jlt/abstract.cfm?uri=jlt-42-8-3022" in source
    assert "https://pubmed.ncbi.nlm.nih.gov/42284403/" in source
