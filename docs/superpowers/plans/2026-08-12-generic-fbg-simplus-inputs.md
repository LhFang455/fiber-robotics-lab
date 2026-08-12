# 通用 FBG-SimPlus 输入适配实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将兼容页从 COMSOL 专用输入检查改为通用八列数据适配器，支持文本、CSV 和制表符数据的预检与标准化导出。

**Architecture:** `models.py` 接收文本、分隔符和跳过行数，统一输出八列数值数组；`app.py` 负责输入方式选择、说明、预览及下载标准化文本。原生模型/工作簿文件不读取，页面说明其外部转换路径。

**Tech Stack:** Python、NumPy、Streamlit、pytest。

## Global Constraints

- 不引入、执行、复制或修改 FBG-SimPlus 源代码。
- 仅声明原工具实际读取的八列数值文本格式和 m/mm 路径单位。
- 格式预检不得宣称完成物理模型、单位或标定验证。
- 只运行本模块相关测试。

---

### Task 1: 多分隔符解析与标准化导出

**Files:**
- Modify: `fiber_robotics_sim/models.py`
- Test: `tests/test_models.py`

- [x] 写失败测试：逗号分隔 CSV 可跳过表头并解析为八列；制表符和空白分隔也可解析；标准化导出为八列空白分隔文本。
- [x] 运行相关测试并确认失败。
- [x] 实现自动/逗号/制表符/空白分隔符、跳过前 N 行和标准化导出函数。
- [x] 运行相关测试并确认通过。

### Task 2: 通用适配页面

**Files:**
- Modify: `app.py`
- Test: `tests/test_models.py`

- [x] 写失败测试：页面列出支持 `.txt`、`.dat`、`.csv`、自动识别、跳过行数和标准化下载说明。
- [x] 运行相关测试并确认失败。
- [x] 实现上传扩展名、分隔符选择、跳过行输入、标准化下载和原生模型/Excel 转换说明。
- [x] 运行相关测试并确认通过。
