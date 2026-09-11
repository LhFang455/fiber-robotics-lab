from pathlib import Path
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / 'app.py'
SECTIONS = {
    'foundation': ('弯曲标定与诊断', '解调与实验任务'),
    'hand': ('二维抓取', '三维抓取', '材质识别'),
    'foot': ('平衡与步态', '装配校验'),
    'structure': ('连续体形状', '结构健康'),
    'optics': ('分布式感知', '偏振与干涉', '数据兼容'),
    'eskin': ('三轴触觉单元', 'FBG 光学皮肤', '稀疏压力重建', '动态滑移与多模态'),
}


def test_all_experiments_open_directly_and_keep_shared_and_local_state():
    app = AppTest.from_file(APP_PATH, default_timeout=40).run()
    assert not app.exception
    assert len(app.selectbox(key='quick_experiment').options) == 16
    app.slider(key='global_temperature').set_value(12.0).run()
    app.slider(key='hand_bend_angle').set_value(35.0).run()
    app.button(key='save_calibration_baseline').click().run()
    baseline = app.session_state['calibration_baseline']
    for index, (domain, sections) in enumerate(SECTIONS.items(), start=1):
        for section in sections:
            app.selectbox(key='quick_experiment').set_value(section).run()
            assert not app.exception
            assert app.session_state[f'{domain}_navigation'] == section
            assert app.get('tab_container')[0].proto.tab_container.default_tab_index == index
            assert app.selectbox(key='quick_experiment').value is None
            assert app.slider(key='global_temperature').value == 12.0
            assert app.slider(key='hand_bend_angle').value == 35.0
            assert app.session_state['calibration_baseline'] == baseline
    # Cards also jump directly to a non-default experiment.
    app.button(key='experiment_jump_hand_三维抓取').click().run()
    assert not app.exception
    assert app.session_state['hand_navigation'] == '三维抓取'
    assert app.get('tab_container')[0].proto.tab_container.default_tab_index == 2
    ids = ('planar','three-d','tactile','foot','calibration','shape','health','distributed',
           'data','optical','chain','assembly','taxel','skin','pressure','dynamic')
    anchors = {heading.proto.anchor for heading in app.subheader}
    assert {f'exports-{ident}' for ident in ids} <= anchors


def test_visible_settings_remain_live_and_assembly_has_local_primary_control():
    app = AppTest.from_file(APP_PATH, default_timeout=40).run()
    assert not app.exception
    labels = {item.label: item for item in app.expander}
    assert not {'公共测量参数与动画', '按步骤学习：六条实验路线', '实验说明与观察方法', '保存与恢复实验记录', '查看抓稳条件与当前读数'} & labels.keys()
    assert not labels['系统原理、配置与使用边界'].proto.expanded
    app.selectbox(key='assembly_case_choice').set_value('单侧错位').run()
    assert not app.exception
    assert app.selectbox(key='sole_assembly_case').value == '单侧错位'
    assert any(m.label == '当前复装工况' and m.value == '单侧错位' for m in app.metric)
    app.button(key='demo_preset').click().run()
    assert not app.exception
    assert app.selectbox(key='assembly_case_choice').value == '正常装配'


def test_first_visit_entry_and_visible_guides_cover_every_experiment():
    import ast
    app = AppTest.from_file(APP_PATH, default_timeout=40).run()
    assert not app.exception
    assert any(h.value == '第一次使用，从这里开始' for h in app.subheader)
    for heading in ('**1 · 动手操作**', '**2 · 看懂结果**', '**3 · 留下记录**'):
        assert sum(m.value == heading for m in app.markdown) == 16
    app.slider(key='global_temperature').set_value(8.0).run()
    app.button(key='start_first_experiment').click().run()
    assert not app.exception
    assert app.session_state['foundation_navigation'] == '弯曲标定与诊断'
    assert app.get('tab_container')[0].proto.tab_container.default_tab_index == 1
    assert app.slider(key='global_temperature').value == 8.0
    # Guidance must precede each placeholder, rather than rendering below the model.
    tree = ast.parse(APP_PATH.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.With):
            continue
        name = getattr(node.items[0].context_expr, 'id', None)
        if name not in {'tactile_tab', 'foot_tab', 'shape_tab', 'health_tab', 'polarization_tab',
                        'taxel_tab', 'optical_skin_tab', 'pressure_tab', 'dynamic_tab'}:
            continue
        guide = next(n for n in ast.walk(node) if isinstance(n, ast.Call)
                     and getattr(n.func, 'id', None) == 'module_learning_frame')
        slot = next(n for n in ast.walk(node) if isinstance(n, ast.Assign)
                    and any(getattr(t, 'id', '').endswith('_demo_slot') for t in n.targets))
        assert guide.lineno < slot.lineno
