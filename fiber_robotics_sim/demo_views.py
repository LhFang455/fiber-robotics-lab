"""Streamlit entry points for the optional, parameter-linked demonstrations."""

import streamlit as st

from . import demo_eskin, demo_extensions, demos, embedded_view


LINKED_INSPECTORS = {"foot", "skin", "shape", "health", "distributed", "dynamic", "taxel", "pressure"}


def inspector_navigation(kind: str) -> None:
    """Same-page links leave model parameters and the current selection intact."""
    if kind not in LINKED_INSPECTORS:
        return
    st.markdown(f'<span id="inspector-{kind}"></span><a href="#demo-{kind}" target="_self">↑ 返回模型</a>', unsafe_allow_html=True)
    st.caption("这里的选择会更新上方模型；模型内选择不改写这里的数据。")


@st.cache_data(show_spinner=False, max_entries=12)
def _sequence(kind: str, parameters: dict) -> dict:
    return {"foot": demos.foot_demo, "skin": demos.skin_demo, "shape": demos.shape_demo,
            "assembly": demos.assembly_demo, "health": demos.health_demo, "fbg": demos.fbg_demo,
            "distributed": demo_extensions.distributed_demo, "tactile": demo_extensions.tactile_demo,
            "optical": demo_extensions.optical_demo, "dynamic": demo_extensions.dynamic_demo,
            "taxel": demo_eskin.taxel_demo, "pressure": demo_eskin.pressure_demo}[kind](parameters)


def render_demo(kind: str, parameters: dict) -> None:
    if kind in LINKED_INSPECTORS:
        st.markdown(f'<span id="demo-{kind}"></span>', unsafe_allow_html=True)
    display = st.container()
    key = f"show_{kind}_demo"
    visible = st.toggle("显示模型", value=True, key=key, help="关闭只隐藏模型，不影响实验计算；随时可重新打开。")
    with display:
        if visible:
            payload = render_demo_panel(kind, parameters, instance_key=kind)
            current_label = payload.get("current_label", "当前参数")
            st.caption(f"教学模拟 · 点击‘{current_label}’回到本页设置；播放不改写实验参数。")
        else:
            st.caption("模型已隐藏，可用下方“显示模型”开关恢复。实验数据仍可查看。")
        if kind in LINKED_INSPECTORS:
            st.markdown(f'<a href="#inspector-{kind}" target="_self">查看详细数据 ↓</a>', unsafe_allow_html=True)


def render_demo_panel(
    kind: str,
    parameters: dict,
    *,
    instance_key: str,
    context: str | None = None,
) -> dict:
    """Render one always-visible demo instance and return its generated payload."""
    payload = dict(_sequence(kind, parameters))
    if context:
        payload["presentation_context"] = context
    embedded_view.render_html(
        demos.demo_html(payload),
        key=f"demo-{instance_key}",
        title=payload["title"],
    )
    return payload
