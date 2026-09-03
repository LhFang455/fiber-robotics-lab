from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / 'app.py'


def grasp_conditions(app):
    return next(item.value for item in app.dataframe if '判定条件' in item.value.columns)


def test_grasp_feedback_tracks_failed_then_successful_contact_conditions():
    app = AppTest.from_file(APP_PATH, default_timeout=20).run()
    assert not app.exception
    assert grasp_conditions(app)['状态'].tolist() == ['未满足'] * 3
    assert any('尚未满足' in item.value and '拇指' in item.value for item in app.warning)
    app.button(key='start_three_d_grasp_task').click().run()
    for _ in range(3):
        app.button(key='advance_three_d_grasp_task').click().run()
    assert not app.exception
    assert grasp_conditions(app)['状态'].tolist() == ['已满足'] * 3
    assert next(item.value for item in app.metric if item.label == '三维抓取状态') == 'FBG 已抓稳'
    app.slider(key='three_d_can_x').set_value(3.0).run()
    assert not app.exception
    assert '未满足' in grasp_conditions(app)['状态'].tolist()
    assert next(item.value for item in app.metric if item.label == '三维抓取状态') == 'FBG 未抓稳'


def test_grasp_channel_inspector_maps_palm_and_separates_display_noise():
    app = AppTest.from_file(APP_PATH, default_timeout=20).run()
    app.selectbox(key='three_d_inspect_channel').set_value('掌心').run()
    assert not app.exception
    assert any('总览第 6 路' in item.value and '细分 FBG 15' in item.value for item in app.caption)
    assert any('不参与当前抓稳判定' in item.value for item in app.caption)
    # 通道选择只是查看，不应改变姿态或抓取结果。
    assert app.slider(key='three_d_thumb_mcp').value == 0.0
    assert next(item.value for item in app.metric if item.label == '三维抓取状态') == 'FBG 未抓稳'


def test_planar_and_three_dimensional_grasp_pages_export_reports():
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()

    labels = {button.label for button in app.get('download_button')}

    assert "下载二维抓取 FBG 读数 CSV" in labels
    assert "下载二维抓取实验报告" in labels
    assert "下载三维抓取 FBG 读数 CSV" in labels
    assert "下载三维抓取实验报告" in labels


def test_planar_and_three_dimensional_pages_expose_quantitative_grasp_experiments():
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()

    slider_keys = {item.key for item in app.get("select_slider")}
    labels = {button.label for button in app.get("download_button")}

    assert {"planar_repeat_samples", "three_d_repeat_samples"} <= slider_keys
    assert "下载二维重复采样 CSV" in labels
    assert "下载二维抓取稳健性报告" in labels
    assert "下载三维重复采样 CSV" in labels
    assert "下载三维抓取稳健性报告" in labels
    assert any("可完整执行当前判定" in item.value.columns for item in app.dataframe)
