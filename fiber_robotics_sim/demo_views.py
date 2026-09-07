"""Streamlit entry points for the optional, parameter-linked demonstrations."""

import streamlit as st

from . import demo_eskin, demo_extensions, demos, embedded_view


@st.cache_data(show_spinner=False, max_entries=12)
def _sequence(kind: str, parameters: dict) -> dict:
    return {"foot": demos.foot_demo, "skin": demos.skin_demo, "shape": demos.shape_demo,
            "assembly": demos.assembly_demo, "health": demos.health_demo, "fbg": demos.fbg_demo,
            "distributed": demo_extensions.distributed_demo, "tactile": demo_extensions.tactile_demo,
            "optical": demo_extensions.optical_demo, "dynamic": demo_extensions.dynamic_demo,
            "taxel": demo_eskin.taxel_demo, "pressure": demo_eskin.pressure_demo}[kind](parameters)


def render_demo(kind: str, parameters: dict) -> None:
    if st.toggle("显示直观模型演示", value=True, key=f"show_{kind}_demo"):
        render_demo_panel(kind, parameters, instance_key=kind)
        st.caption("按下方实验参数生成；播放只改变面板内的演示进度，不改写实验参数。点击‘当前参数’可与下方图表对照。")


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
