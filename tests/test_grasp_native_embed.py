import base64
from unittest.mock import patch
from fiber_robotics_sim import embedded_view


def test_only_grasp_views_use_native_opaque_iframe():
    for key in ('planar-hand','spatial-hand'):
        with patch.object(embedded_view.st,'iframe') as native, patch.object(embedded_view.st,'html') as custom:
            document='<svg>原姿态</svg><script>const duration=520;</script>'
            embedded_view.render_html(document,key=key,height=560)
            native.assert_called_once()
            assert not custom.called
            source=native.call_args.args[0]
            assert source.startswith('data:text/html;charset=utf-8;base64,')
            assert base64.b64decode(source.split(',')[1]).decode().endswith(document)
            assert native.call_args.kwargs['height']==560
    with patch.object(embedded_view.st,'iframe') as native, patch.object(embedded_view.st,'html') as custom:
        embedded_view.render_html('<p>other</p>',key='demo-skin')
        assert not native.called
        custom.assert_called_once()


def test_native_views_receive_updated_pose_and_keep_original_animation():
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file('app.py',default_timeout=40).run()
    assert not app.exception
    def documents():
        return [base64.b64decode(x.proto.src.split(',',1)[1]).decode() for x in app.get('iframe') if x.proto.src.startswith('data:text/html')]
    before=documents()
    assert len(before)==2
    app.button(key='three_d_close').click().run()
    assert not app.exception
    after=documents()
    assert len(after)==2
    hand=next(x for x in after if 'bio-hand' in x)
    assert hand != next(x for x in before if 'bio-hand' in x)
    assert 'motionDuration=520' in hand
    assert 'overflow:hidden' in hand
    assert 'fingerCapsules' in hand and 'previousFingerCapsules' in hand
