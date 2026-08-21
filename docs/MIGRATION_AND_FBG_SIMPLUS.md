# 迁移与 FBG-SimPlus 使用说明

本文用于将“光纤机器人传感仿真实验室”迁移到另一台电脑或公开部署环境，并指导使用者独立安装和使用 FBG-SimPlus。

## 先了解边界

- 本仓库只包含本网站的 Streamlit 代码、测试和本地 Three.js 运行时；克隆仓库后可以完整运行网站。
- 本仓库**不包含** FBG-SimPlus 的源代码、二进制、模型文件或任何私有实验数据。
- FBG-SimPlus 是 Ben Frey 等人的独立 GPL-3.0 软件。请仅从其[原始仓库](https://github.com/benfrey/FBG-SimPlus)下载、安装和运行。
- 公开网站版的上传文件仅在当前运行进程中解析预览，不提供持久化保存。请勿上传未授权公开的仿真、实验或客户数据。

---

## A. 在新电脑上迁移并运行本网站

### A1. 前置条件

准备以下软件：

- Git；
- Python 3.10+；本机已验证 macOS 26.6.2（Apple Silicon）与 Python 3.13.9。Windows 与 Linux 具备运行条件，但尚未在真实机器完成验收；
- 能访问 GitHub 的网络。

下列 `.venv` 为 Python 内置虚拟环境示例；若使用其他虚拟环境工具，请先按该工具方式激活环境，再从安装依赖步骤继续。环境激活后，各工具的安装、启动和测试命令相同。

在 macOS/Linux 终端中复制执行：

```bash
git clone https://github.com/2698685648/fiber-robotics-lab.git
cd fiber-robotics-lab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

启动后，终端会显示本地访问地址；通常为 `http://localhost:8501`。停止网站可在同一终端按 `Ctrl+C`。

Windows PowerShell 中复制执行：

```powershell
git clone https://github.com/2698685648/fiber-robotics-lab.git
cd fiber-robotics-lab
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

若 PowerShell 禁止激活脚本，可仅对当前用户执行一次：

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

然后重新执行 ` .\.venv\Scripts\Activate.ps1`。

### A2. 更新已经迁移的网站

在已克隆的目录中复制执行：

```bash
git pull --ff-only origin master
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Windows 将第二行替换为：

```powershell
.\.venv\Scripts\Activate.ps1
```

`git pull --ff-only` 只接受可快进更新；如果本地自行修改过源文件而无法更新，请先备份这些改动，不要直接覆盖。

### A3. 公开部署到 Streamlit Community Cloud

该方式适合演示网站快速公开访问。操作在网页完成：

1. 确认 GitHub 仓库 `2698685648/fiber-robotics-lab` 的 `master` 已包含需要公开的提交。
2. 打开 [Streamlit Community Cloud](https://share.streamlit.io/)，使用 GitHub 账号登录。
3. 选择 **Create app**，填写：
   - Repository：`2698685648/fiber-robotics-lab`
   - Branch：`master`
   - Main file path：`app.py`
   - Python version：选择当前受支持的 Python 版本，例如 3.11 或 3.12。
4. 选择一个公开 URL 子域名，确认 App visibility 为 **Public**，再点击部署。
5. 部署完成后访问生成的 `https://<your-name>.streamlit.app` 地址。

之后只要将更新推送到 `master`，Cloud 会重新部署。公开版不应加入密钥、账号密码、未脱敏的实验数据或受限文件。若未来需要密钥，使用平台的 Secrets 配置，不要提交 `.streamlit/secrets.toml` 到 Git。

### A4. 用 Docker 迁移到其他云或服务器（可选）

本项目当前依赖仅来自 `requirements.txt`，因此可由任意支持 Python/Streamlit 的平台运行。若目标平台要求容器，请在迁移时使用以下 Dockerfile 内容创建 `Dockerfile`，再在项目根目录执行构建和运行命令：

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . ./
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
```

```bash
docker build -t fiber-robotics-lab .
docker run --rm -p 8501:8501 fiber-robotics-lab
```

浏览器打开 `http://localhost:8501`。生产环境请由部署平台提供 HTTPS 终止、访问日志和资源限制；本网站不是用于接收或保管敏感实验数据的文件服务。

---

## B. 独立安装 FBG-SimPlus

### B1. 获取原项目

FBG-SimPlus 由 Ben Frey 等开发，原项目为：<https://github.com/benfrey/FBG-SimPlus>。

推荐从原仓库克隆，以便保留完整的许可证、文档和教程文件。复制执行：

```bash
git clone https://github.com/benfrey/FBG-SimPlus.git
cd FBG-SimPlus
```

也可以在 GitHub 页面选择 **Code → Download ZIP**，解压后进入 `FBG-SimPlus` 目录。

### B2. 创建独立 Python 3.8 环境并安装依赖

FBG-SimPlus README 指定 Python 3.8，且依赖 PyQt5、SciPy、Matplotlib、SymPy、six 和 NumPy。不要把这些依赖安装到本网站的 `.venv`；两者应保持独立。

如果系统已安装 `python3.8`，在 FBG-SimPlus 根目录复制执行：

```bash
python3.8 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install PyQt5 scipy matplotlib sympy six numpy
cd python
python run.py
```

Windows PowerShell（系统已安装 Python 3.8）中复制执行：

```powershell
py -3.8 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install PyQt5 scipy matplotlib sympy six numpy
cd python
python run.py
```

如果 `python3.8` 或 `py -3.8` 不存在，请先用你所在系统的 Python 发行方式安装 Python 3.8，再回到上面的命令。FBG-SimPlus README 明确以 Python 3.8 为目标环境；在更高版本 Python 上的兼容性不在本说明中保证。

### B3. 准备通用八列输入文本

FBG-SimPlus 不要求特定仿真软件；它实际读取八列数值文本。网站兼容页可预检空白分隔 `.txt/.dat`、逗号分隔 `.csv` 和制表符文本，并将有效数据下载为 FBG-SimPlus 所需的空白分隔 `.txt`。原生模型文件（例如 `.mph`、`.odb`、`.rst`）和 Excel `.xlsx` 不能直接读取，应先在原软件或表格软件中导出为上述文本格式。

无论数据来自 FEM 软件、实验预处理脚本还是人工构造，八列的物理含义与顺序必须固定：

| 列 | 所需量 | 单位 |
|---|---|---|
| 1 | 路径位置 | m 或 mm，需在 FBG-SimPlus 中一致选择 |
| 2 | 纵向真应变 `εxx` | 无量纲 |
| 3 | 横向真应变 `εyy` | 无量纲 |
| 4 | 横向真应变 `εzz` | 无量纲 |
| 5 | 纵向正应力 `σxx` | Pa（N/m²） |
| 6 | 横向正应力 `σyy` | Pa（N/m²） |
| 7 | 横向正应力 `σzz` | Pa（N/m²） |
| 8 | 温度 | K |

先在本网站的 **“FBG-SimPlus 兼容”** 页面选择分隔符（自动、空白、逗号或制表符）与表头跳过行数，再上传文本。页面检查八列数值、非有限值与位置递增性，并绘制应变、横向应力和温度预览。该检查不替代数据来源的物理模型验证或 FBG-SimPlus 的参数设置。

### B4. 在 FBG-SimPlus 中生成光谱

启动 `python/run.py` 后，按以下顺序操作：

1. 在 **Select Stressed/Strained Path Files** 区域点击 **Add Files**，选中从本网站下载的标准化八列 `.txt` 文件。
2. 设置 **Skip Rows**：填写文件开头的元数据/表头行数。原仓库的 `tutorial/tut-export.txt` 有 7 行以 `%` 开头的元数据，因此教程文件填写 `7`；你的文件应按实际表头行数填写。
3. 在 **Path Distance Input Units** 中选择第一列的真实单位：若第一列是米，选 `[m]`；若第一列是毫米，选 `[mm]`。
4. 设置 FBG 数量；逐一输入每个 FBG 的路径位置、初始 Bragg 波长和 FBG 长度。位置单位必须与第 3 步一致。
5. 按研究模型需求设置均匀/非均匀应变选项、温度模拟、宿主材料热膨胀系数和其他光学参数。不要把示例参数直接当作你的实验标定值。
6. 点击 **Generate** 生成模拟结果；点击 **Plot** 查看反射谱。请参阅原仓库 `documentation.pdf` 和 `tutorial/` 中的原始教程，确认每个参数的物理意义。

原作者在 README 中列出：谱图绘制可能不稳定、macOS 退出时可能需强制结束、保存图片功能可能不稳定。遇到运行时问题请优先查阅原项目文档或联系原作者提供的联系方式。

### B5. 许可证、署名与引用

FBG-SimPlus 保留 GNU General Public License v3.0。若你复制、修改或分发其源代码，应按 GPL-3.0 履行许可证义务；本网站仅做独立输入格式检查，不包含其源码。

在论文、报告或成果中使用 FBG-SimPlus 的方法或结果时，请保留原作者指定引用：

```text
Frey, B., Snyder, P., Ziock, K., & Passian, A. (2021).
Semicomputational calculation of Bragg shift in stratified materials.
Physical Review E, 104(5), 055307.
```

同时标注软件来源：Ben Frey et al., FBG-SimPlus V1.0，<https://github.com/benfrey/FBG-SimPlus>，GPL-3.0。
