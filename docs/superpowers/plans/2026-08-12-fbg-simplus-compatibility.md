# FBG-SimPlus Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让网站独立校验并可视化 FBG-SimPlus 风格的 COMSOL 导出文本，同时完整说明来源、许可证、引用与独立使用流程。

**Architecture:** `models.py` 只负责解析和校验空格分隔的 FEM 文本，输出与界面无关的数值数组及元数据；`visuals.py` 将其转换为三张位置曲线；`app.py` 负责上传、预览和知识产权说明。网站不复制、导入、执行或修改 FBG-SimPlus 源代码。

**Tech Stack:** Python、NumPy、Plotly、Streamlit、pytest。

## Global Constraints

- 兼容 FBG-SimPlus 公开教程的 COMSOL 文本输入，而不是复制其实现。
- 页面显著提供项目链接、GPL-3.0、Frey 等人的论文引用和独立运行流程。
- 不声明本网站生成 FBG-SimPlus 的反射谱，也不生成与原项目不一致的输出格式。
- 只运行本模块相关测试。

---

### Task 1: 独立输入解析与校验

**Files:**
- Modify: `fiber_robotics_sim/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Produces: `parse_fbg_simplus_comsol_export(text: str) -> dict[str, np.ndarray | str | int]`

- [x] **Step 1: Write the failing tests**

```python
def test_fbg_simplus_parser_reads_the_public_tutorial_column_order():
    result = models.parse_fbg_simplus_comsol_export(TUTORIAL_EXPORT)
    assert np.array_equal(result["position_m"], np.array([0.0, .001]))
    assert np.array_equal(result["temperature_k"], np.array([293.15, 294.15]))

def test_fbg_simplus_parser_rejects_invalid_column_count_and_nonmonotonic_positions():
    with pytest.raises(ValueError, match="八列"):
        models.parse_fbg_simplus_comsol_export("0 1 2")
```

- [x] **Step 2: Run tests to verify they fail**

Run: `pytest -q tests/test_models.py -k fbg_simplus_parser`

Expected: FAIL because the parser does not exist.

- [x] **Step 3: Implement the minimal parser**

```python
def parse_fbg_simplus_comsol_export(text: str) -> dict[str, np.ndarray | str | int]:
    # Ignore percent-prefixed COMSOL metadata, require eight numeric columns,
    # require finite values and strictly increasing position.
    ...
```

- [x] **Step 4: Run tests to verify they pass**

Run: `pytest -q tests/test_models.py -k fbg_simplus_parser`

Expected: PASS.

### Task 2: FEM data preview figure

**Files:**
- Modify: `fiber_robotics_sim/visuals.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: parser dictionary with `position_m`, `longitudinal_strain`, `transverse_stress_pa`, and `temperature_k`.
- Produces: `fbg_simplus_input_figure(result: dict) -> go.Figure`

- [x] **Step 1: Write the failing test**

```python
def test_fbg_simplus_input_figure_exposes_strain_stress_and_temperature():
    figure = visuals.fbg_simplus_input_figure(parsed_export)
    assert [trace.name for trace in figure.data] == ["纵向应变 εxx", "横向应力 σy / σz", "温度"]
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_models.py -k fbg_simplus_input_figure`

Expected: FAIL because the figure function does not exist.

- [x] **Step 3: Implement the minimal figure**

```python
def fbg_simplus_input_figure(result: dict) -> go.Figure:
    # Three subplots: longitudinal strain, y/z transverse stress, temperature.
    ...
```

- [x] **Step 4: Run test to verify it passes**

Run: `pytest -q tests/test_models.py -k fbg_simplus_input_figure`

Expected: PASS.

### Task 3: Upload page, attribution and use instructions

**Files:**
- Modify: `app.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: uploaded text as UTF-8 compatible bytes.
- Uses: `models.parse_fbg_simplus_comsol_export` and `visuals.fbg_simplus_input_figure`.

- [x] **Step 1: Write the failing page-content test**

```python
def test_app_exposes_the_fbg_simplus_compatibility_module_and_attribution():
    source = Path("app.py").read_text(encoding="utf-8")
    assert "FBG-SimPlus 兼容" in source
    assert "benfrey/FBG-SimPlus" in source
    assert "GPL-3.0" in source
    assert "Frey, B., Snyder, P., Ziock, K., & Passian, A. (2021)" in source
```

- [x] **Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_models.py -k fbg_simplus_compatibility_module`

Expected: FAIL because no compatibility page exists.

- [x] **Step 3: Add the Streamlit tab**

```python
with fbg_simplus_tab:
    uploaded = st.file_uploader("上传 COMSOL 导出文本", type=["txt"])
    # Explain export columns, provide template download, render validated input.
    # State the independent-use boundary and link source, GPL, and paper citation.
```

- [x] **Step 4: Run page test and the two model tests**

Run: `pytest -q tests/test_models.py -k 'fbg_simplus_parser or fbg_simplus_input_figure or fbg_simplus_compatibility_module'`

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add app.py fiber_robotics_sim/models.py fiber_robotics_sim/visuals.py tests/test_models.py docs/superpowers/plans/2026-08-12-fbg-simplus-compatibility.md
git commit -m "feat: add FBG-SimPlus input compatibility"
```
