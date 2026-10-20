# 企业风险大模型 / LifelongLearning-business

企业查询与 PSU 经验检索、更新的实验代码。当前处于第一阶段验证：比较 baseline（无经验）、FIFO（容量 18）和 adaptive（四组各 8，总容量 32）。FIFO 与 adaptive 从相同的 18 条已复核种子开始，容量差异为当前实验设定。

当前题库为 `enterprise_v30_120_v1_0_2_natural_v0_3_0_expanded`：60 道经验形成题、72 道测试题，另有 2 道澄清题排除在 SQL 验证之外。公共提示为精简 v3，模型需根据自然、较模糊的提问自行选择核心数据。

**克隆仓库后，还需要安装 Python 依赖、下载 BGE-M3 权重、准备 MySQL 和大模型 API，并取得实验输入文件。** 本仓库发布源码、配置、脚本和测试；模型权重、题库、合成数据、SQLite 种子库及历史运行文件位于 `data/`。当前实验输入已通过 [固定版本 Release](https://github.com/JiXX-bisl/LifelongLearning-business/releases/tag/repro-v030-20261020) 发布，安装方法见第 3 节；模型权重仍单独下载。

Windows 使用者按第 1～6 节中的 **Windows / PowerShell** 步骤操作：Conda/CUDA → 模型与材料 → CUDA 种子索引 → 原生 MySQL 与数据库验收 → API → 离线测试和实际验证。首次安装完成后的启动方式见第 6 节“Windows 日常运行”。

## 1. 环境与依赖

macOS / Linux 在终端执行：

```bash
git clone --branch dev_v0.1 https://github.com/JiXX-bisl/LifelongLearning-business.git
cd LifelongLearning-business
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-runtime.txt -r requirements-embedding.txt -r requirements-dev.txt
```

依赖文件的作用：

- [`requirements-runtime.txt`](requirements-runtime.txt)：模型 SDK、Pydantic、MySQL 驱动、YAML、NumPy，以及 SQL 解析依赖。
- [`requirements-embedding.txt`](requirements-embedding.txt)：PyTorch、Transformers、FlagEmbedding、Hugging Face 下载组件等。版本记录自原生 macOS ARM64 / Python 3.11 的现有环境；其他平台需选择与设备兼容的 PyTorch 构建，并由使用者验证。
- [`requirements-dev.txt`](requirements-dev.txt)：独立测试所需的 pytest。`requirements-psu.txt` 仅包含 SQLGlot，单独安装它不够。

### Windows + CUDA

使用已初始化 Conda 的 PowerShell 或 Anaconda PowerShell Prompt。
```powershell
Set-Location '<Your-Path>'
conda create -n lifelonglearning python=3.11 pip -y
if ($LASTEXITCODE -ne 0) { throw 'Conda 环境创建失败。' }
conda env config vars set -n lifelonglearning PYTHONUTF8=1
if ($LASTEXITCODE -ne 0) { throw 'Conda UTF-8 配置失败。' }
conda activate lifelonglearning
$env:PYTHONUTF8 = "1"
python -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu132
if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch 安装失败。' }
python -m pip install -r requirements-windows-cuda.txt
if ($LASTEXITCODE -ne 0) { throw 'Windows 依赖安装失败。' }
python -B scripts/check_windows_runtime.py
if ($LASTEXITCODE -ne 0) { throw 'Windows/CUDA 环境检查失败。' }
```


Windows 新增配置：

| 用途 | Windows 配置 |
| --- | --- |
| 正式 PSU 检索 | `src/psu_memory/retrieval/retrieval_config.windows_cuda.json` |
| 模型下载参数；通用推理工具另需额外依赖 | `configs/embedding/bge_m3_windows_cuda.yaml` |
| 原生 MySQL 配置模板 | `configs/mysql/my.windows.ini`，运行副本放在纯英文路径，见第 4 节 |
| 三组 v3 验证 | `configs/validation/enterprise_v30_120_v1_0_2_natural_coverage_v3_windows_cuda.json` |
| 三组 v2 验证 | `configs/validation/enterprise_v30_120_v1_0_2_natural_direct_v1_windows_cuda.json` |
| baseline 前 3 题 | `configs/validation/enterprise_v30_120_v1_0_2_natural_baseline3_v1_windows_cuda.json` |
| 通用业务查询（冷启动） | `configs/business/psu_query_v30_120_windows_cuda.json`，通过 `scripts/run_business_query.py --psu-config` 指定 |

Windows 验证配置预留独立种子路径 `data/derived/psu_memory/seeds/windows-cuda/seed.sqlite3` 和本机验收报告 `data/reports/enterprise_v30_120/database_acceptance_windows.json`。

## 2. 下载 embedding 模型

模型配置为 [`configs/embedding/bge_m3.yaml`](configs/embedding/bge_m3.yaml)，固定使用：

```text
模型：BAAI/bge-m3
revision：5617a9f61b028005a4858fdac845db406aefb181
本地目录：data/models/huggingface/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181
```

macOS / Linux 完成通用 embedding 依赖安装后：

```bash
python scripts/check_embedding.py --device cpu --download-only
```

### Windows 下载模型

在 `lifelonglearning` 环境、项目根目录中执行：

```powershell
python -B -c "from src.embedding import read_embedding_config, download_embedding_model; config = read_embedding_config('configs/embedding/bge_m3_windows_cuda.yaml'); print(download_embedding_model(config, download=True))"
if ($LASTEXITCODE -ne 0) { throw 'BGE-M3 下载失败，请停止后续步骤。' }
```

## 3. 取得当前实验材料

当前固定输入包为 `enterprise-v030-v1.0.2-inputs-20261020`。从 [Release 页面](https://github.com/JiXX-bisl/LifelongLearning-business/releases/tag/repro-v030-20261020) 下载下面两个附件，放到项目 `downloads/` 目录，保留文件名，不手动解压：

- [enterprise-v030-v1.0.2-inputs-20261020.zip](https://github.com/JiXX-bisl/LifelongLearning-business/releases/download/repro-v030-20261020/enterprise-v030-v1.0.2-inputs-20261020.zip)
- [enterprise-v030-v1.0.2-inputs-20261020.zip.sha256](https://github.com/JiXX-bisl/LifelongLearning-business/releases/download/repro-v030-20261020/enterprise-v030-v1.0.2-inputs-20261020.zip.sha256)

Windows 在项目根目录安装：

```powershell
$archive = Join-Path (Get-Location).Path 'downloads\enterprise-v030-v1.0.2-inputs-20261020.zip'
if (!(Test-Path -LiteralPath $archive -PathType Leaf)) { throw '缺少 ZIP 文件。' }
if (!(Test-Path -LiteralPath "$archive.sha256" -PathType Leaf)) { throw '缺少 SHA256 文件。' }
python -B scripts/enterprise_query_bench/install_reproduction_inputs.py --archive $archive
if ($LASTEXITCODE -ne 0) { throw '材料安装失败，请停止，不要复制种子或重建索引。' }
```

macOS / Linux 可使用同一 Python 安装脚本和 `--archive` 参数。。


### Windows 复制种子并重建 CUDA 索引

仅在材料安装和模型下载均成功后执行。原始 reviewed 种子保留，Windows 使用独立副本；允许目标目录已存在，但拒绝覆盖目标种子或报告：

```powershell
$sourceSeed = 'data/derived/psu_memory/seeds/reviewed-20261019-165637-416369/seed.sqlite3'
$seedDir = 'data/derived/psu_memory/seeds/windows-cuda'
$targetSeed = Join-Path $seedDir 'seed.sqlite3'
$indexReport = Join-Path $seedDir 'index_report.json'
if (!(Test-Path -LiteralPath $sourceSeed -PathType Leaf)) { throw '缺少原始种子，请检查材料安装。' }
if ((Test-Path -LiteralPath $targetSeed) -or (Test-Path -LiteralPath $indexReport)) {
    throw '目标种子或报告已存在，请保留现场，不要覆盖。'
}
New-Item -ItemType Directory -Path $seedDir -Force -ErrorAction Stop | Out-Null
Copy-Item -LiteralPath $sourceSeed -Destination $targetSeed -ErrorAction Stop
python -B -m src.psu_memory.cli.build_retrieval_index `
    --store $targetSeed `
    --namespace validation-v30-120-v1.0.2-natural-v0.3.0-split-v1 `
    --data-version synthetic_enterprise_v30_120_v1.0.2 `
    --retrieval-config src/psu_memory/retrieval/retrieval_config.windows_cuda.json `
    --report $indexReport
if ($LASTEXITCODE -ne 0) { throw 'CUDA 索引重建失败，请保留目标文件和输出。' }
```

## 4. 准备 MySQL 并验收

先安装或启动 MySQL 服务；Docker 只是可选部署方式。默认连接为 `127.0.0.1:3306`、用户 `synthetic`、库 `synthetic_enterprise_v30_120`。由数据库管理员创建账号并授予导入/查询所需权限；连接参数位于 [`configs/business/query_v30_120.yaml`](configs/business/query_v30_120.yaml)。

### macOS / Linux

以下建库、导入及验收命令均由使用者执行，示例适用于尚未创建该库的新环境。建库/建表以管理员身份运行，替换 `root` 为实际管理员账号：

```bash
mysql -h 127.0.0.1 -P 3306 -u root -p \
  < data/sql/synthetic_enterprise_v30_120_v1.0.2/create_database.sql
mysql -h 127.0.0.1 -P 3306 -u root -p synthetic_enterprise_v30_120 \
  < data/sql/synthetic_enterprise_v30_120_v1.0.2/database_schema.sql
```

通过环境变量提供 `synthetic` 账号密码，再执行数据导入与回读验收：

```bash
export ENTERPRISE_MYSQL_PASSWORD='替换为本机数据库账号密码'
python scripts/sql/build_enterprise_dataset_v30.py import \
  --baseline-dir data/sql/synthetic_enterprise_v30_120_v1.0.2 \
  --host 127.0.0.1 --port 3306 --user synthetic \
  --report data/reports/enterprise_v30_120/database_import_local.json
python scripts/sql/build_enterprise_dataset_v30.py accept \
  --baseline-dir data/sql/synthetic_enterprise_v30_120_v1.0.2 \
  --private-dir data/evaluation/synthetic_enterprise_v30_120_v1.0.2 \
  --host 127.0.0.1 --port 3306 --user synthetic \
  --report data/reports/enterprise_v30_120/database_acceptance_local.json
```

每次指定新的报告文件名，已有报告不会被覆盖。验收必须为 `PASS`；将验证配置中的 `acceptance_report` 改为本机新生成的验收报告路径。历史报告不能代替新机器的数据库验收。若更改连接地址、端口或用户，需要同时更新业务配置及导入/验收参数。

### Windows 原生 MySQL 首次部署

此流程使用 Windows 原生 MySQL。

**已完成的初始化、建表和导入步骤无需重复执行。** 已导入但尚未生成本机 `PASS` 验收报告时，从步骤（6）的 `accept` 命令继续；完成验收和 API 配置后，日常测试使用第 6 节的启动方式。

**（1）准备正式版程序并核对版本**

从 [MySQL 官方 8.0 下载页](https://dev.mysql.com/downloads/mysql/8.0.html) 获取 Windows x86 64-bit ZIP Archive 正式版。项目至少需要 8.0.16，本机本次实际使用 8.0.16；`8.0.0-dmr` 是开发预览版，不满足要求。文件夹叫 `mysql-8.0` 不代表里面的程序版本正确，必须检查 `mysqld --version`。

以下命令在一个 PowerShell 窗口中执行，后续称为“服务窗口”：

```powershell
$mysqlBase = '<Path to MySQL>\mysql-8.0'
$mysqld = Join-Path $mysqlBase 'bin\mysqld.exe'
if (!(Test-Path -LiteralPath $mysqld -PathType Leaf)) { throw '未找到 mysqld.exe，请检查解压层级。' }
$versionText = (& $mysqld --no-defaults --version | Out-String)
$versionExit = $LASTEXITCODE
Write-Output $versionText
$versionMatch = [regex]::Match($versionText, '\b8\.0\.(\d+)\b')
if ($versionExit -ne 0 -or !$versionMatch.Success -or
    [int]$versionMatch.Groups[1].Value -lt 16 -or
    $versionText -match '(?i)dmr|milestone|release.candidate') {
    throw '需要 MySQL 8.0.16 或以上的 8.0 正式版，请先更换程序。'
}
```

**（2）配置纯英文运行路径**

本机路径约定如下。更改位置时，配置和命令中的路径要同步调整；服务账号必须能读取配置并写入数据、日志目录。

| 用途 | 本机路径 |
| --- | --- |
| 程序目录 | `<Path to MySQL>\mysql-8.0` |
| 仓库配置模板 | `<Path to MySQL Config>\my.windows.ini` |
| 实际启动配置 | `<Your Path>\my.ini` |
| 8.0.16 数据目录 | `<Your Path>\data-8.0.16` |
| 错误日志 | `<Your Path>\logs\mysql-8.0.16-error.log` |

确认 [`configs/mysql/my.windows.ini`](configs/mysql/my.windows.ini) 的 `[mysqld]` 中这三项与实际目录一致，替换已有行，不追加重复项：

```ini
basedir=D:/tools/mysql-8.0
datadir=D:/lifelonglearning/mysql/data-8.0.16
log-error=D:/lifelonglearning/mysql/logs/mysql-8.0.16-error.log
```

模板已配置本机监听 `127.0.0.1:3306`、InnoDB、`utf8mb4` 和 `utf8mb4_0900_ai_ci`。复制为运行配置：

```powershell
$sourceConfig = 'configs\mysql\my.windows.ini'
$runtimeConfig = '<Your Path>\my.ini'
New-Item -ItemType Directory -Path '<Your Path>\logs' `
    -Force -ErrorAction Stop | Out-Null
if (Test-Path -LiteralPath $runtimeConfig) {
    $backup = $runtimeConfig + '.' + (Get-Date -Format 'yyyyMMdd-HHmmssfff') + '.bak'
    Copy-Item -LiteralPath $runtimeConfig -Destination $backup -ErrorAction Stop
}
Copy-Item -LiteralPath $sourceConfig -Destination $runtimeConfig -ErrorAction Stop
```

**（3）只初始化一次**

```powershell
$mysqlData = '<Your Path>\data-8.0.16'
if (Test-Path -LiteralPath $mysqlData) {
    throw '数据目录已存在，请保留并检查；已初始化的实例直接进入启动步骤。'
}
& $mysqld --no-defaults --initialize --console `
    "--basedir=$mysqlBase" "--datadir=$mysqlData"
if ($LASTEXITCODE -ne 0) { throw '初始化失败，请保留现场，不要继续启动或删除目录重试。' }
```

初始化会输出新的临时 `root@localhost` 密码。本地保存，后续首次登录使用；共享日志时隐藏密码行。该步骤只创建 MySQL 系统库，不创建实验的七张表。

**（4）启动服务并首次登录**

在服务窗口执行下面的命令，`--defaults-file` 保持为第一个参数：

```powershell
Set-Location '<Your Path>'
& 'D:\tools\mysql-8.0\bin\mysqld.exe' `
    '--defaults-file=D:/lifelonglearning/mysql/my.ini' --console
```

看到 `ready for connections`、端口为 `3306` 后保持窗口打开。另开一个 PowerShell，后续称为“项目窗口”，登录客户端：

```powershell
Set-Location '<Your Path>'
& 'D:\tools\mysql-8.0\bin\mysql.exe' --no-defaults `
    --protocol=TCP -h 127.0.0.1 -P 3306 -u root -p `
    --connect-expired-password --default-character-set=utf8mb4
```

输入临时密码。进入 `mysql>` 后替换下面的占位密码并执行，再退出客户端：

```sql
ALTER USER 'root'@'localhost' IDENTIFIED BY '替换为你设置的root密码';
EXIT;
```

**（5）创建实验库、表和业务账号**

MySQL 客户端读取 SQL 文件也使用英文路径。回到项目窗口的 PowerShell，仅复制原始建库/建表文件，数据包本身保留不变：

```powershell
$sqlSource = 'data\sql\synthetic_enterprise_v30_120_v1.0.2'
$sqlStage = '<Your Path>\bootstrap-v30'
New-Item -ItemType Directory -Path $sqlStage -Force -ErrorAction Stop | Out-Null
Copy-Item -LiteralPath "$sqlSource\create_database.sql" `
    -Destination "$sqlStage\create_database.sql" -ErrorAction Stop
Copy-Item -LiteralPath "$sqlSource\database_schema.sql" `
    -Destination "$sqlStage\database_schema.sql" -ErrorAction Stop
& 'D:\tools\mysql-8.0\bin\mysql.exe' --no-defaults `
    --protocol=TCP -h 127.0.0.1 -P 3306 -u root -p --default-character-set=utf8mb4
```

使用新 root 密码登录，在 `mysql>` **逐条执行**以下语句；任一语句报错就停止。如果库、表或账号已存在，先检查，不删除重建，也不重复执行整段：

```sql
SOURCE D:/lifelonglearning/mysql/bootstrap-v30/create_database.sql;
USE synthetic_enterprise_v30_120;
SOURCE D:/lifelonglearning/mysql/bootstrap-v30/database_schema.sql;
SHOW TABLES;
```

应看到 7 张业务表。替换下方账号密码，再执行：

```sql
CREATE USER 'synthetic'@'127.0.0.1' IDENTIFIED BY '替换为你的synthetic账号密码';
GRANT SELECT, INSERT, LOCK TABLES ON synthetic_enterprise_v30_120.*
TO 'synthetic'@'127.0.0.1';
EXIT;
```

导入使用 `SELECT ... FOR UPDATE`，因此除了读写所需的 SELECT、INSERT，还授予此库的 LOCK TABLES 权限；验收和正常查询由代码使用只读事务。项目连接使用 synthetic 账号，root 用于上述管理操作。

**（6）导入并生成本机验收报告**

在项目窗口激活环境并设置 synthetic 密码，密码仅进入当前进程环境，不写入配置或 README：

```powershell
Set-Location '<Project Path>'
conda activate lifelonglearning
$env:PYTHONUTF8 = '1'
$mysqlPassword = Read-Host '请输入 synthetic 密码' -AsSecureString
$env:ENTERPRISE_MYSQL_PASSWORD = [System.Net.NetworkCredential]::new('', $mysqlPassword).Password
Remove-Variable mysqlPassword
```

先导入，成功后再验收：

```powershell
python -B scripts/sql/build_enterprise_dataset_v30.py import `
    --baseline-dir data/sql/synthetic_enterprise_v30_120_v1.0.2 `
    --host 127.0.0.1 --port 3306 --user synthetic `
    --report data/reports/enterprise_v30_120/database_import_windows.json
if ($LASTEXITCODE -ne 0) { throw '数据导入失败，请停止后续步骤。' }
```

```powershell
python -B scripts/sql/build_enterprise_dataset_v30.py accept `
    --baseline-dir data/sql/synthetic_enterprise_v30_120_v1.0.2 `
    --private-dir data/evaluation/synthetic_enterprise_v30_120_v1.0.2 `
    --host 127.0.0.1 --port 3306 --user synthetic `
    --report data/reports/enterprise_v30_120/database_acceptance_windows.json
if ($LASTEXITCODE -ne 0) { throw '数据库验收失败，请停止后续步骤。' }
```

## 5. 配置大模型 API

macOS / Linux：

```bash
export OPENAI_API_KEY='替换为自己的API密钥'
export OPENAI_BASE_URL='替换为服务商的OpenAI兼容接口地址'
export OPENAI_MODEL_NAME='替换为真实模型名称'
```

Windows 在已激活 `lifelonglearning` 的项目 PowerShell 窗口中设置：

```powershell
$apiKey = Read-Host '请输入查询大模型 API Key' -AsSecureString
$env:OPENAI_API_KEY = [System.Net.NetworkCredential]::new('', $apiKey).Password
Remove-Variable apiKey
$env:OPENAI_BASE_URL = '替换为服务商的OpenAI兼容接口基础地址'
$env:OPENAI_MODEL_NAME = '替换为服务商实际可用的模型名称'
```

## 6. 独立测试与正式运行

当前新评分版本为 **`expected_coverage_v3`**，由使用者先运行独立测试（模拟远程模型和 embedding，部分测试写入临时 SQLite）：

```bash
bash scripts/enterprise_query_bench/run_coverage_sql_tests.sh
```

准备好依赖、权重、数据、种子、API 和本机验收报告后，进行小范围运行，默认三组各 10 题：

```bash
unset PSU_COVERAGE_RUN
bash scripts/enterprise_query_bench/run_coverage_sql_validation.sh
```

完整 72 题测试分区，三组共 216 回合：


## 7. 代码入口

| 路径 | 内容 |
| --- | --- |
| `src/chatflow_agent/` | 通用企业查询、数据库接入和公共提示 |
| `src/psu_memory/retrieval/` | BGE-M3 编码、向量索引、经验检索和选择 |
| `src/psu_memory/pool/` | FIFO / adaptive 经验成员维护及难度标注 |
| `src/psu_memory/validation/` | 实验准备、核心/补充结果比较、事实冻结和汇总 |
| `scripts/enterprise_query_bench/` | 测试、运行、形成题、种子初始化和已有结果诊断入口 |
| `configs/validation/` | 题库、种子路径、评分版本和实验条件 |
| `test/` | 使用者执行的独立测试 |
