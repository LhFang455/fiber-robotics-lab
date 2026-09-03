import ast
from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
EXPECTED_LABS = (
    "① 系统总览",
    "② FBG 基础与解调",
    "③ 手部抓取与触觉",
    "④ 足底感知与装配",
    "⑤ 形状与结构监测",
    "⑥ 分布式光学与数据",
    "⑦ 电子皮肤与多模态感知",
)
EXPECTED_SECTIONS = {
    "foundation": ("弯曲标定与诊断", "解调与实验任务"),
    "hand": ("二维抓取", "三维抓取", "材质识别"),
    "foot": ("平衡与步态", "装配校验"),
    "structure": ("连续体形状", "结构健康"),
    "optics": ("分布式感知", "偏振与干涉", "数据兼容"),
    "eskin": ("三轴触觉单元", "FBG 光学皮肤", "稀疏压力重建", "动态滑移与多模态"),
}


def literal_assignment(name):
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    assignment = next(
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    )
    return ast.literal_eval(assignment.value)


def test_top_level_navigation_is_six_domain_labs_with_stable_internal_sections():
    assert literal_assignment("LAB_LABELS") == EXPECTED_LABS
    assert literal_assignment("LAB_SECTIONS") == EXPECTED_SECTIONS

    source = APP_PATH.read_text(encoding="utf-8")
    assert "= tracked_tabs(\n    LAB_LABELS, \"main_navigation\"" in source
    for navigation_key in (
        "foundation_navigation", "hand_navigation", "foot_navigation",
        "structure_navigation", "optics_navigation",
    ):
        assert f'"{navigation_key}"' in source


def test_sidebar_only_lists_domain_labs_and_existing_content_remains_available():
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()
    assert not app.exception
    assert tuple(app.selectbox(key="sidebar_module_navigation").options) == EXPECTED_LABS

    headings = {item.value for item in app.subheader}
    assert {
        "单根手指 FBG 弯曲标定",
        "机器人手：FBG 弯曲、指尖触觉与关节状态",
        "三维抓取传感：独立接触与 FBG 读数",
        "多材质触觉识别：五指与掌心 FBG 接触分布",
        "机器人足：六区足底接触、地形与步态相位",
        "三芯光纤的连续体机器人 3D 形状重建",
        "机械臂结构健康监测：点式 FBG 阵列局部异常定位",
        "分布式光纤感知：连续空间上的应变、振动与温度",
        "偏振与干涉传感：偏振态、旋转与微腔光程差",
        "实验任务与报告",
        "可更换式足底组件：二维装配状态预测",
        "FBG-SimPlus 兼容：通用八列数据适配",
    } <= headings

    downloads = {item.label for item in app.get("download_button")}
    assert {
        "下载可恢复记录 JSON", "下载三维抓取实验报告", "下载触觉实验报告",
        "下载足底实验报告", "下载形状重建报告", "下载健康监测报告",
        "下载分布式实验报告", "下载偏振与干涉报告", "下载当前实验报告",
        "下载装配验证参数摘要", "下载通用八列文本模板",
    } <= downloads


def test_guided_routes_target_a_domain_lab_and_internal_section():
    routes = literal_assignment("GUIDED_ROUTES")
    valid_sections = {
        (lab_index, section_label)
        for lab_index, lab_id in enumerate((None, "foundation", "hand", "foot", "structure", "optics", "eskin"))
        if lab_id is not None
        for section_label in EXPECTED_SECTIONS[lab_id]
    }
    covered = set()
    for route in routes.values():
        for step in route["steps"]:
            lab_index, section_key, section_label, title, action, observation = step
            assert section_key.endswith("_navigation")
            assert title and action and observation
            covered.add((lab_index, section_label))
    assert covered == valid_sections


def test_programmatic_navigation_remounts_tabs_while_preserving_stable_state_keys():
    source = APP_PATH.read_text(encoding="utf-8")
    assert "def request_navigation(" in source
    assert "def tracked_tabs(" in source
    assert 'widget_key = f"{navigation_key}_widget_{revision}"' in source
    assert 'target_key = f"{navigation_key}_target"' in source
    assert "st.session_state.pop(widget_key, None)" in source
    assert "st.tabs(labels, default=default, key=widget_key" in source
    for key in (
        "main_navigation", "foundation_navigation", "hand_navigation",
        "foot_navigation", "structure_navigation", "optics_navigation",
        "eskin_navigation",
    ):
        assert f'"{key}"' in source
