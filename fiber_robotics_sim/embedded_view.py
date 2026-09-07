"""Isolated generated views with source-checked automatic height messages."""

import base64
import hashlib
import json
from pathlib import Path

import streamlit as st


_RESIZE_SCRIPT = """<script>
(() => {
  const id = __FRAME_ID__;
  function start() {
    let pending = false;
    function schedule() {
      if (pending) return;
      pending = true;
      requestAnimationFrame(() => {
        pending = false;
        if (document.fullscreenElement || !document.body) return;
        const body = document.body, style = getComputedStyle(body);
        const height = Math.ceil(body.getBoundingClientRect().height +
          (parseFloat(style.marginTop) || 0) + (parseFloat(style.marginBottom) || 0));
        parent.postMessage({type: 'fiber-frame-height', id, height}, '*');
      });
    }
    new ResizeObserver(schedule).observe(document.body);
    window.addEventListener('resize', schedule);
    document.addEventListener('fullscreenchange', schedule);
    schedule();
  }
  if (document.readyState === 'loading')
    document.addEventListener('DOMContentLoaded', start, {once: true});
  else start();
})();
</script>"""


def frame_html(document: str, *, key: str, height: int | None = None,
               title: str = "仿真实验视图") -> str:
    frame_id = "fiber-frame-" + hashlib.sha256(key.encode()).hexdigest()[:16]
    document += "<style>html{background:#0e1117}</style>"
    if height is None:
        document += _RESIZE_SCRIPT.replace("__FRAME_ID__", json.dumps(frame_id))
    config = {
        "id": frame_id, "title": title, "height": height or 650,
        "autoHeight": height is None,
        "source": "data:text/html;charset=utf-8;base64," + base64.b64encode(document.encode()).decode(),
    }
    template = Path(__file__).with_name("embedded_view.html").read_text()
    return template.replace("__FRAME_ID__", frame_id).replace(
        "__CONFIG__", json.dumps(config, ensure_ascii=False).replace("<", "\\u003c"),
    )


def render_html(document: str, *, key: str, height: int | None = None,
                title: str = "仿真实验视图") -> None:
    # Only generated project HTML goes here; arbitrary uploaded HTML is not accepted.
    st.html(frame_html(document, key=key, height=height, title=title), unsafe_allow_javascript=True)
