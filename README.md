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

环境激活后，上述安装与启动命令在各平台一致。浏览器打开后，可在左侧统一设置温度变化、波长测量噪声和随机种子。推荐按以下路径体验：

### 已验证环境与测试条件

本机已验证环境为 macOS 26.6.2（Apple Silicon）、Python 3.13.9、NumPy 2.4.6、Plotly 6.7.0、Streamlit 1.58.0、pytest 9.0.3。Windows 与 Linux 具备运行条件，但尚未在真实机器完成验收。

在项目根目录、已激活 Python 环境且已安装 `requirements.txt` 的前提下，可执行下列测试；测试不依赖浏览器或网络，但要求仓库中的本地 Three.js 文件保持完整：

```bash
python -m pytest tests/test_models.py -p no:cacheprovider
```

1. 在 FBG 标定与诊断页比较原始波长、温度补偿和冗余故障诊断。
2. 在二维/三维抓取、多材质触觉与足底平衡页观察多通道接触信息。
3. 在连续体、分布式光纤、偏振与干涉页比较不同光学机制的数据形态。
4. 在解调器与实验任务页查看温补、控制输出与多模态任务报告。

## 模块地图

| 编号 | 模块 | 一句话 |
|---|---|---|
| ① | 系统总览 | 模块目录、当前配置、感知链与推荐实验路径 |
| ② | FBG 标定与诊断 | 弯曲标定、温度补偿、冗余通道故障隔离 |
| ③ | 二维手部抓取 | 平面姿态、接触与五指＋掌心六路 FBG 抓取判定 |
| ④ | 三维抓取传感 | 独立三维接触、握持稳定度与手臂/手掌/手指光纤 |
| ⑤ | 多材质触觉识别 | 五指/掌心接触分布与材料模式分类 |
| ⑥ | 足底平衡与步态 | 六区载荷、温补 CoP 与地形/相位影响 |
| ⑦ | 连续体形状重建 | 三芯光纤曲率、方向、扭转和中心线 |
| ⑧ | 机械臂健康监测 | 局部异常应变、定位区间与报警 |
| ⑨ | 分布式光纤感知 | Rayleigh、DAS、Brillouin、Raman 的空间测量 |
| ⑩ | 偏振与干涉传感 | Stokes 偏振态、Sagnac 陀螺与 EFPI 干涉谱 |
| ⑪ | 解调器与实验任务 | 波长流、滤波温补、控制输出与实验报告 |
| ⑫ | 可更换足底装配校验 | 空载温补基线、压入不足与单侧错位预测 |
| ⑬ | FBG-SimPlus 兼容 | 检查通用八列导出数据并标准化 |

标签顺序即推荐学习顺序：先基础标定，再机械交互与触觉，然后连续体/结构/分布式/偏振，最后用解调器集成；装配校验与 FBG-SimPlus 兼容作为辅助工具。

## 用户指南

界面导航、逐页操作、演示功能速查和常见问题见 [用户指南](docs/USER_GUIDE.md)。

## 核心公式

FBG 波长变化采用：`ΔλB = λB[(1 − pe)ε + kTΔT]`。网站还以解析教学模型展示 Rayleigh/OFDR 连续应变、DAS 时空振动、Brillouin 频移、Raman 温度、Stokes 偏振态、Sagnac 相位与 EFPI 干涉谱。

## 模型边界

这是教学与方案验证工具，并非 COMSOL 的替代品。对于软材料大变形、粘接层、滞后、谱形失真、复杂接触、温度梯度或封装不对称，应以有限元或实验标定建立更精确的模型。应用中的“接触位置/力”尤其是简化的高斯传递模型；它用于理解传感器布置与可辨识性，不应直接作为真实硬件的最终标定公式。

## 迁移、公开部署与 FBG-SimPlus

从新电脑迁移、部署公开版本，以及独立安装和使用 FBG-SimPlus 的完整复制粘贴步骤见：[迁移与 FBG-SimPlus 使用说明](docs/MIGRATION_AND_FBG_SIMPLUS.md)。本仓库不包含 FBG-SimPlus 源码；兼容页处理其所需的通用八列文本数据，并保留原作者署名与 GPL-3.0 许可边界。
