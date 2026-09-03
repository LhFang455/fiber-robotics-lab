import ast
from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def route_definition():
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    assignment = next(
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "GUIDED_ROUTES" for target in node.targets)
    )
    return ast.literal_eval(assignment.value)


def test_guided_routes_cover_every_non_overview_module_with_learning_metadata():
    routes = route_definition()
    covered_tabs = {
        step[0]
        for route in routes.values()
        for step in route["steps"]
    }

    assert list(routes) == [
        "基础标定与解调", "抓取与触觉", "足底与装配",
        "形状与结构健康", "光学机制与数据兼容", "电子皮肤与多模态",
    ]
    assert covered_tabs == set(range(1, 7))
    for route in routes.values():
        assert route["duration"].startswith("约 ")
        assert route["prerequisite"]
        assert route["deliverable"]


def test_route_selector_and_downloadable_record_support_all_routes():
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()

    selector = app.selectbox(key="guided_route")
    assert selector.options == [
        "基础标定与解调", "抓取与触觉", "足底与装配",
        "形状与结构健康", "光学机制与数据兼容", "电子皮肤与多模态",
    ]
    assert any(button.label == "下载当前路线学习记录" for button in app.get("download_button"))


def test_guided_route_shows_bounded_steps_without_changing_simulation_state():
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()
    assert not app.exception
    public_before = {
        key: app.session_state[key]
        for key in ("global_temperature", "global_noise", "global_sample_rate", "global_failed_channel")
    }

    assert app.selectbox(key="guided_route").value == "基础标定与解调"
    route_checks = [item for item in app.checkbox if item.key and item.key.startswith("guided_progress_")]
    assert [item.key for item in route_checks] == [
        "guided_progress_calibration_0",
        "guided_progress_calibration_1",
        "guided_progress_calibration_2",
    ]
    assert not any(item.value for item in route_checks)
    assert any("进度 0 / 3" in item.value for item in app.caption)
    assert any("完成勾选仅记录学习进度" in item.value for item in app.caption)
    assert {key: app.session_state[key] for key in public_before} == public_before


def test_route_progress_is_isolated_and_reset_only_affects_selected_route():
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()
    app.checkbox(key="guided_progress_calibration_0").check()
    app.checkbox(key="guided_progress_calibration_1").check().run()
    assert not app.exception
    assert app.session_state["guided_route_progress"]["calibration"][:2] == [True, True]

    app.selectbox(key="guided_route").set_value("抓取与触觉").run()
    assert not app.exception
    grasp_checks = [item for item in app.checkbox if item.key and item.key.startswith("guided_progress_grasp_")]
    assert len(grasp_checks) == 4
    grasp_checks[0].check().run()
    app.button(key="reset_guided_route").click().run()

    assert not app.exception
    assert app.session_state["guided_progress_grasp_0"] is False
    assert app.session_state["guided_route_progress"]["calibration"][:2] == [True, True]
    app.selectbox(key="guided_route").set_value("基础标定与解调").run()
    assert app.checkbox(key="guided_progress_calibration_0").value is True
    assert app.checkbox(key="guided_progress_calibration_1").value is True


def test_each_route_exposes_expected_number_of_steps_and_report_destination():
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()
    expected = {
        "基础标定与解调": 3,
        "抓取与触觉": 4,
        "足底与装配": 3,
        "形状与结构健康": 4,
        "光学机制与数据兼容": 4,
        "电子皮肤与多模态": 4,
    }
    for route_name, count in expected.items():
        app.selectbox(key="guided_route").set_value(route_name).run()
        assert not app.exception
        route_checks = [item for item in app.checkbox if item.key and item.key.startswith("guided_progress_")]
        assert len(route_checks) == count
    assert all(
        any(step[1:3] == ("foundation_navigation", "解调与实验任务") for step in route["steps"])
        for name, route in route_definition().items()
        if name != "电子皮肤与多模态"
    )


def test_route_module_and_home_navigation_use_tracked_native_tabs():
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()
    app.button(key="tab_jump_1_进入第 1 步").click().run()
    assert not app.exception
    assert app.session_state["main_navigation"] == "② FBG 基础与解调"
    assert app.session_state["foundation_navigation"] == "弯曲标定与诊断"

    app.button(key="tab_jump_0_← 返回主页").click().run()
    assert app.session_state["main_navigation"] == "① 系统总览"
    app.button(key="module_jump_5").click().run()
    assert not app.exception
    assert app.session_state["main_navigation"] == "⑥ 分布式光学与数据"

    app.button(key="tab_jump_0_← 返回主页").click().run()
    assert not app.exception
    assert app.session_state["main_navigation"] == "① 系统总览"


def test_existing_next_buttons_follow_the_documented_learning_order():
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()
    app.button(key="tab_jump_1_进入第 1 步").click().run()
    app.button(key="tab_jump_2_下一步 → 手部：三维抓取").click().run()
    assert app.session_state["main_navigation"] == "③ 手部抓取与触觉"
    assert app.session_state["hand_navigation"] == "三维抓取"
    app.button(key="tab_jump_4_下一步 → 结构健康").click().run()
    assert app.session_state["main_navigation"] == "⑤ 形状与结构监测"
    assert app.session_state["structure_navigation"] == "结构健康"
    app.button(key="tab_jump_5_下一步 → 分布式感知").click().run()
    assert not app.exception
    assert app.session_state["main_navigation"] == "⑥ 分布式光学与数据"
    assert app.session_state["optics_navigation"] == "分布式感知"


def test_top_tab_changes_keep_the_sidebar_module_selector_in_sync():
    source = APP_PATH.read_text(encoding="utf-8")

    assert "def sync_sidebar_navigation(widget_key: str)" in source
    assert 'navigation_key == "main_navigation"' in source
