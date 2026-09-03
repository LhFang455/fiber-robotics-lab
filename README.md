# 光纤机器人传感仿真实验室

一个本地运行的 Streamlit 教学实验室，覆盖点式/准分布式 FBG、分布式 Rayleigh/OFDR、φ-OTDR/DAS、Brillouin、Raman、偏振与干涉传感，并将它们用于机器人抓取、触觉、足底平衡、形状重建和结构健康监测。

## 本地打开

本项目为本地运行的 Streamlit 应用，GitHub 页面本身不承载在线演示。请在已安装 Python 3.10+ 的终端中执行。以下给出使用内置 `venv` 的完整命令；若使用其他虚拟环境工具，只需按该工具的方式创建并激活环境，再执行对应代码块中的安装和启动命令。

macOS/Linux：

```bash
git clone https://github.com/LhFang455/fiber-robotics-lab.git
cd fiber-robotics-lab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Windows PowerShell：

```powershell
git clone https://github.com/LhFang455/fiber-robotics-lab.git
cd fiber-robotics-lab
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

命令完成后，终端会显示本地访问地址；通常在浏览器打开 `http://localhost:8501` 即可。若已通过网页下载 ZIP，可解压后进入项目目录，跳过 `git clone` 命令后继续执行。

若需要复现本项目已经验证的精确 Python 包版本，可将安装命令改为：

```bash
python -m pip install -r requirements-verified.txt
```

环境激活后，上述安装与启动命令在各平台一致。浏览器打开后，可在左侧统一设置温度变化、波长测量噪声和随机种子。推荐按以下路径体验：

### 已验证环境与测试条件

本机已验证环境为 macOS 26.6.2（Apple Silicon）、Python 3.13.9、NumPy 2.4.6、Plotly 6.7.0、Streamlit 1.58.0、pytest 9.0.3。Windows 与 Linux 具备运行条件，但尚未在真实机器完成验收。

在项目根目录、已激活 Python 环境且已安装 `requirements.txt` 的前提下，可执行下列测试；测试不依赖浏览器或网络，但要求仓库中的本地 Three.js 文件保持完整：

```bash
python -m pytest tests -p no:cacheprovider
```

GitHub Actions 配置会使用 `requirements-verified.txt` 运行 Ruff 与完整测试。本仓库包含的第三方运行时及外部兼容工具边界见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。项目自身的开源授权仍需在公开分发前单独确定。

1. **基础标定与解调**：比较原始波长、温度补偿、故障诊断与解调输出。
2. **手部抓取与触觉**：完成二维/三维抓取并比较多材质接触模式。
3. **足底感知与装配**：联系六区载荷、CoP 反演与空载装配筛查。
4. **形状与结构监测**：比较连续体重建与点式异常定位。
5. **分布式光学与数据**：比较分布式、偏振/干涉机制并完成八列数据检查。
6. **电子皮肤与多模态感知**：从三轴触觉单元推进到 FBG 光学皮肤、压力场重建和动态滑移判别。

## 领域实验室地图

| 编号 | 领域实验室 | 内部实验 |
|---|---|---|
| ① | 系统总览 | 六条引导式路线、独立进度、领域目录与感知链 |
| ② | FBG 基础与解调 | 弯曲标定与诊断；解调与实验任务 |
| ③ | 手部抓取与触觉 | 二维抓取；三维抓取；材质识别 |
| ④ | 足底感知与装配 | 平衡与步态；装配校验 |
| ⑤ | 形状与结构监测 | 连续体形状；结构健康 |
| ⑥ | 分布式光学与数据 | 分布式感知；偏振与干涉；数据兼容 |
| ⑦ | 电子皮肤与多模态感知 | 三轴触觉单元；FBG 光学皮肤；稀疏压力重建；动态滑移与多模态 |

六条引导式路线共同覆盖②至⑦的全部内部实验；同一实验可在不同路线中承担基础、对照或汇总任务。每条路线都显示预计时间、前置建议、最终产物，并可下载当前学习记录。

## 用户指南

界面导航、逐页操作、演示功能速查和常见问题见 [用户指南](docs/USER_GUIDE.md)。

## 核心公式

FBG 波长变化采用：Δλ<sub>B</sub> = λ<sub>B</sub>[(1 − p<sub>e</sub>)ε + k<sub>T</sub>ΔT]。网站还以解析教学模型展示 Rayleigh/OFDR 连续应变、DAS 时空振动、Brillouin 频移、Raman 温度、Stokes 偏振态、Sagnac 相位、EFPI 干涉谱，以及电子皮肤的三轴力反演、FBG 感受野、稀疏压力重建与动态滑移判别。

## 模型边界

这是教学与方案验证工具，并非 COMSOL 的替代品。对于软材料大变形、粘接层、滞后、谱形失真、复杂接触、温度梯度或封装不对称，应以有限元或实验标定建立更精确的模型。应用中的“接触位置/力”尤其是简化的高斯传递模型；它用于理解传感器布置与可辨识性，不应直接作为真实硬件的最终标定公式。

## 迁移、公开部署与 FBG-SimPlus

从新电脑迁移、部署公开版本，以及独立安装和使用 FBG-SimPlus 的完整复制粘贴步骤见：[迁移与 FBG-SimPlus 使用说明](docs/MIGRATION_AND_FBG_SIMPLUS.md)。本仓库不包含 FBG-SimPlus 源码；兼容页处理其所需的通用八列文本数据，并保留原作者署名与 GPL-3.0 许可边界。
