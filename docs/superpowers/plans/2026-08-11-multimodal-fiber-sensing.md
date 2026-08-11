# 多机制光纤传感实验室升级实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不改动二维、三维抓取逻辑的前提下，为实验室加入统一数据链和分布式光纤感知教学模块。

**Architecture:** `models.py` 负责各散射机制的解析教学模型和统一传感帧；`visuals.py` 负责空间—时间/空间—温度图；`app.py` 只组织页面、参数和报告。现有 FBG、足底、形状与健康数据都通过统一传感帧汇总。

**Tech Stack:** Python、NumPy、Plotly、Streamlit、pytest。

## Global Constraints

- 不改二维、三维抓取逻辑。
- 所有结果标注为教学解析模型，不能作为真实设备结论。
- 新功能先写失败测试，再实现最小代码。
- 只验证本次新增模块。

---

### Task 1: 统一传感帧与质量状态

**Files:**
- Modify: `fiber_robotics_sim/models.py`
- Test: `tests/test_models.py`

- [ ] 写测试：统一传感帧包含时间、机制、位置、原始信号、补偿信号、状态和质量字段。
- [ ] 运行测试，确认因接口不存在而失败。
- [ ] 实现 `build_sensor_frame`，接收机制、位置、原始/补偿信号和质量值。
- [ ] 运行测试，确认通过。

### Task 2: 分布式 Rayleigh、DAS、Brillouin、Raman 教学模型

**Files:**
- Modify: `fiber_robotics_sim/models.py`
- Test: `tests/test_models.py`

- [ ] 写测试：Rayleigh 输出应变—距离；DAS 输出时间—距离；Brillouin 输出可分离温度/应变；Raman 输出温度—距离。
- [ ] 运行测试，确认接口缺失。
- [ ] 实现四个最小模型及参数校验。
- [ ] 运行测试，确认通过。

### Task 3: 分布式可视化与应用页

**Files:**
- Modify: `fiber_robotics_sim/visuals.py`
- Modify: `app.py`
- Test: `tests/test_models.py`

- [ ] 写测试：四种分布式图形都有正确的 Plotly 数据维度，应用页暴露相应控件。
- [ ] 运行测试，确认失败。
- [ ] 实现空间曲线、时间—距离热图和温度/频移曲线；新增分布式感知页。
- [ ] 运行测试，确认通过。

### Task 4: 数据链汇总与报告

**Files:**
- Modify: `app.py`
- Test: `tests/test_models.py`

- [ ] 写测试：任务报告包含当前分布式机制与质量状态。
- [ ] 运行测试，确认失败。
- [ ] 将统一传感帧摘要加入解调器与实验任务页的报告。
- [ ] 运行测试，确认通过。
