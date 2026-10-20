# 企业风险大模型 / LifelongLearning-business

企业查询与 PSU 经验检索、更新的实验代码。当前处于第一阶段验证：比较 baseline（无经验）、FIFO（容量 18）和 adaptive（四组各 8，总容量 32）。FIFO 与 adaptive 从相同的 18 条已复核种子开始，容量差异为当前实验设定。

当前题库为 `enterprise_v30_120_v1_0_2_natural_v0_3_0_expanded`：60 道经验形成题、72 道测试题，另有 2 道澄清题排除在 SQL 验证之外。公共提示为精简 v3，模型需根据自然、较模糊的提问自行选择核心数据。

**克隆仓库后，还需要安装 Python 依赖、下载 BGE-M3 权重、安装实验材料包、准备 MySQL 和大模型 API。** 源码、配置、脚本和测试由 Git 同步；合成数据、验收材料、题库、18 条共同种子及来源证据通过 [GitHub Release 材料包](https://github.com/JiXX-bisl/LifelongLearning-business/releases/tag/repro-v030-20261020) 提供。BGE-M3 权重从官方模型仓库下载。

新机器的顺序：**克隆代码 → 安装 Python 依赖 → 下载并安装材料包（第 3 节）→ 下载 BGE-M3 → 按设备处理种子索引 → 导入 MySQL 并生成本机验收报告 → 配置 API → 由使用者测试和运行。** Docker 或本机 MySQL 的部署由使用者自行配置。Windows 使用 WSL，并优先将项目放在 WSL 的 Linux 文件系统中（例如 `~/projects/`）。

## 1. 环境与依赖

| 项目 | 当前要求 |
| --- | --- |
| 系统 | macOS / Linux；Windows 使用 WSL。部分脚本使用 Bash 和 `fcntl` |
| Python | 推荐 3.11；代码使用 `StrEnum`，最低为 3.11 |
| 下载工具 | 下方命令使用 `curl`；也可用浏览器下载 ZIP 和对应 SHA-256 文件 |
| MySQL | MySQL 8.0.16+，支持窗口函数、CHECK 约束和 `utf8mb4_0900_ai_ci`；还需 `mysql` 命令行客户端 |
| 查询/难度/经验选择模型 | 可用的 OpenAI 兼容 Chat Completions API；默认配置为 `kimi-k2.6` |
| 本地 embedding | `BAAI/bge-m3`，固定 revision，1024 维；权重需单独下载 |
| PSU 检索设备 | 当前支持 `mps` 和 `cpu`；默认 `mps`，适用于可用 MPS 的 Apple Silicon 环境 |
| SQLite | 使用 Python 标准库 `sqlite3`，无需额外部署数据库服务 |

在终端执行：

```bash
git clone --branch repro-v030-20261020 https://github.com/JiXX-bisl/LifelongLearning-business.git
cd LifelongLearning-business
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-runtime.txt -r requirements-embedding.txt -r requirements-dev.txt
```

上述命令检出固定发布标签，会处于 detached HEAD 状态，适合复现本次实验。已有仓库可在安装依赖和材料前执行 `git checkout repro-v030-20261020`；继续开发使用 `git switch dev_v0.1`。发布标签固定本次代码与材料的配套版本。

依赖文件的作用：

- [`requirements-runtime.txt`](requirements-runtime.txt)：模型 SDK、Pydantic、MySQL 驱动、YAML、NumPy，以及 SQL 解析依赖。
- [`requirements-embedding.txt`](requirements-embedding.txt)：PyTorch、Transformers、FlagEmbedding、Hugging Face 下载组件等。版本记录自原生 macOS ARM64 / Python 3.11 的现有环境；其他平台需选择与设备兼容的 PyTorch 构建，并由使用者验证。
- [`requirements-dev.txt`](requirements-dev.txt)：独立测试所需的 pytest。`requirements-psu.txt` 仅包含 SQLGlot，单独安装它不够。

当前模型适配器使用 `DefaultHttpx2Client` / `DefaultAsyncHttpx2Client`，因此锁定 `openai==3.6.0`，其依赖会自动安装 `httpx2`。不要直接替换为旧版 OpenAI SDK。

当前三组命令行实验不需要 Streamlit、Docker Python SDK 或单独安装 Vanna。可选网页入口 `scripts/run_business_query_web.py` 需要额外安装 `streamlit`；旧 DBBench 容器和数据转换入口分别还需要 `docker`、`pyarrow`。这些不属于当前三组实验的必要组件。

第三方模块的官方地址如下；安装版本以仓库的 requirements 文件为准：

| 模块 | 官方代码或安装文档 |
| --- | --- |
| FlagEmbedding | [FlagOpen/FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) |
| PyTorch | [按操作系统及设备选择安装方式](https://pytorch.org/get-started/locally/) |
| Transformers | [huggingface/transformers](https://github.com/huggingface/transformers) |
| Hugging Face Hub | [模型下载与缓存说明](https://huggingface.co/docs/huggingface_hub/guides/download) |
| OpenAI Python SDK | [openai/openai-python](https://github.com/openai/openai-python) |
| MySQL Connector/Python | [官方文档](https://dev.mysql.com/doc/connector-python/en/) |
| SQLGlot | [tobymao/sqlglot](https://github.com/tobymao/sqlglot) |
| Pydantic | [官方文档](https://docs.pydantic.dev/latest/) |

## 2. 下载 embedding 模型

模型配置为 [`configs/embedding/bge_m3.yaml`](configs/embedding/bge_m3.yaml)，固定使用：

- [BGE-M3 官方模型与权重](https://huggingface.co/BAAI/bge-m3)。
- [本项目固定 revision 的完整文件列表](https://huggingface.co/BAAI/bge-m3/tree/5617a9f61b028005a4858fdac845db406aefb181)。

```text
模型：BAAI/bge-m3
revision：5617a9f61b028005a4858fdac845db406aefb181
本地目录：data/models/huggingface/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181
```

完成依赖安装后，由使用者执行一次下载：

```bash
python scripts/check_embedding.py --device cpu --download-only
```

这里的 `--device cpu` 使下载命令不依赖 MPS；`--download-only` 只下载权重，不执行 embedding 推理，也不调用大模型 API。先安装上一节的依赖，再运行下载命令。脚本会获取权重、tokenizer 和配置文件，需要能够访问 Hugging Face，并为模型权重和缓存预留磁盘空间。当前实验没有其他需要单独下载的模型。

正式 PSU 检索使用 [`retrieval_config.example.json`](src/psu_memory/retrieval/retrieval_config.example.json) 中的路径，通过 Transformers 从本地加载模型，**不会在运行时自动下载缺失权重**。通用下载/检查脚本与 PSU 检索分别使用自己的配置；移动模型目录时须同步修改 PSU 的 `embedding_model_path`。

PSU 当前固定 1024 维、FP32、CLS pooling、L2 归一化；实验检索的最大输入长度为 1024，batch size 为 4。改用其他 embedding 模型、revision、设备或表示参数后，需要重建向量及检索索引。

### 在 CPU 设备上运行

将 `src/psu_memory/retrieval/retrieval_config.example.json` 中的 `device` 从 `mps` 改为 `cpu`。当前 PSU 配置不接受 `cuda:0`；通用 embedding 检查脚本支持 CUDA 不代表 PSU 三组入口已经支持 CUDA。

既有 18 条种子使用 MPS 的 embedding 签名，不能只改配置后直接复用旧索引。取得下一节的种子材料后，由使用者复制种子库并重建索引：

```bash
mkdir data/derived/psu_memory/seeds/cpu-local && \
cp data/derived/psu_memory/seeds/reviewed-20261019-165637-416369/seed.sqlite3 \
  data/derived/psu_memory/seeds/cpu-local/seed.sqlite3 && \
python -B -m src.psu_memory.cli.build_retrieval_index \
  --store data/derived/psu_memory/seeds/cpu-local/seed.sqlite3 \
  --namespace validation-v30-120-v1.0.2-natural-v0.3.0-split-v1 \
  --data-version synthetic_enterprise_v30_120_v1.0.2 \
  --retrieval-config src/psu_memory/retrieval/retrieval_config.example.json \
  --report data/derived/psu_memory/seeds/cpu-local/index_report.json
```

`cpu-local` 应为新目录，勿覆盖已有文件。该步骤会执行本地 embedding 推理并写入新 SQLite 库，保留种子成员和已复核难度，不请求难度模型。完成后，将所用验证配置中的 `source_store` 指向新库，并确认其 `retrieval_config` 指向上述 CPU 配置。FIFO 与 adaptive 必须使用同一个重建后的种子库。

## 3. 下载并安装当前实验材料

材料版本为 `enterprise-v030-v1.0.2-inputs-20261020`，对应 `synthetic_enterprise_v30_120_v1.0.2` 数据和 natural v0.3.0 题库。包内标明实验源码基线提交；本次发布的代码仅增加材料分发、安装工具和说明，实验逻辑沿用该基线。

包内有 173 个材料文件，原始文件总计 77,082,065 字节；ZIP 为 4,828,797 字节（约 4.8 MB），不含模型权重。

下载入口：

- [Release 说明](https://github.com/JiXX-bisl/LifelongLearning-business/releases/tag/repro-v030-20261020)。
- [材料 ZIP](https://github.com/JiXX-bisl/LifelongLearning-business/releases/download/repro-v030-20261020/enterprise-v030-v1.0.2-inputs-20261020.zip)。
- [ZIP 的 SHA-256](https://github.com/JiXX-bisl/LifelongLearning-business/releases/download/repro-v030-20261020/enterprise-v030-v1.0.2-inputs-20261020.zip.sha256)。
- [逐文件清单及版本信息](https://github.com/JiXX-bisl/LifelongLearning-business/releases/download/repro-v030-20261020/enterprise-v030-v1.0.2-inputs-20261020.manifest.json)。

在克隆后的项目根目录执行：

```bash
mkdir -p downloads
curl --fail --location \
  --output downloads/enterprise-v030-v1.0.2-inputs-20261020.zip \
  https://github.com/JiXX-bisl/LifelongLearning-business/releases/download/repro-v030-20261020/enterprise-v030-v1.0.2-inputs-20261020.zip
curl --fail --location \
  --output downloads/enterprise-v030-v1.0.2-inputs-20261020.zip.sha256 \
  https://github.com/JiXX-bisl/LifelongLearning-business/releases/download/repro-v030-20261020/enterprise-v030-v1.0.2-inputs-20261020.zip.sha256
python scripts/enterprise_query_bench/install_reproduction_inputs.py \
  --archive downloads/enterprise-v030-v1.0.2-inputs-20261020.zip
```

安装脚本只使用 Python 标准库，先校验 ZIP 的 SHA-256、完整成员清单及所有文件的大小和 SHA-256，再按原相对路径安装。已有相同文件会保留；已有不同文件会停止安装，避免覆盖历史材料。SQLite 只作为文件复制，不打开数据库；安装不请求模型，也不自动导入 MySQL。项目所在文件系统需支持硬链接，macOS 本地磁盘及 WSL Linux 文件系统可用于此流程。

包内材料及用途：

| 相对路径 | 用途 |
| --- | --- |
| `data/sql/synthetic_enterprise_v30_120_v1.0.2/` 整个目录 | 120 家企业、Schema 3.0、七张表的合成数据、建库/建表 SQL、manifest 及全部校验文件 |
| `data/evaluation/synthetic_enterprise_v30_120_v1.0.2/` 整个目录 | 数据验收所需的隐藏事实、生成配置、manifest 等；不传给查询模型 |
| `data/questions/psu_memory/enterprise_v30_120_v1_0_2_natural_v0_3_0_expanded/questions.json` | 已审核的自然问题 |
| `data/evaluation/psu_memory/enterprise_v30_120_v1_0_2_natural_v0_3_0_expanded/ground_truth.json` | 必要结果和参考 SQL，仅供评分 |
| 同目录的 `split_v1.json`、`reference_validation.json` | 固定形成/测试划分、用户参考结果核验凭据及哈希绑定 |
| `data/derived/psu_memory/seeds/reviewed-20261019-165637-416369/` | 已复核的 18 条共同种子、检索索引及 `summary.json` |
| `data/derived/psu_memory/seeds/initial-20261019-160745-357466/` | 人工复核前的种子库及初始化汇总，供来源追溯 |
| `data/experiences/psu_memory/enterprise_v30_120_v1_0_2_initial_seed_v1/seeds.json` | 18 条种子的选种清单及来源绑定 |
| `data/experiences/psu_memory/enterprise_v30_120_v1_0_2_authored_v1/` | 形成题 SQL 包及 60 份 SQL |
| `data/derived/psu_memory/formation/authored-20261019-135802-408380/` 中的汇总及结果 | 60 道形成题的真实执行结果，供来源校验 |
| 当前题库目录下的 `business_context.json` | 形成题当时使用的公共背景 |
| `data/catalog/path_map.json` | 原 Mac 绝对路径及旧目录的映射依据 |
| `provenance/` | 来源绑定的代码/配置副本、种子形成时公共提示及历史验收证据；具体路径见材料清单，不覆盖当前源码 |

安装记录写入 `data/catalog/reproduction-bundles/enterprise-v030-v1.0.2-inputs-20261020.json`。以 `split_v1.json` 和各 manifest 的来源记录为准，保留原文件字节；不要改写历史 JSON 的路径、换行或内容。ZIP 避免了 Git 文本换行转换，仓库的 `.gitattributes` 也固定了源码及配置的 LF 换行。

`expected_coverage_v3` 会从包内静态 `tables.json` 冻结补充字段依据。隐藏事实、标准结果和来源证据仅供验收/评分，不作为查询模型提示。历史数据库验收报告放在来源证据目录，安装后仍须按下一节生成本机验收报告。

此包支持**导入并验收已冻结 v1.0.2 数据、使用已复核种子启动新的 72 题实验，以及追溯种子来源**。完整重放种子初始化/人工复核尚有两个限制：历史汇总含原 Mac 绝对路径键，部分代码尚未做跨根目录适配；种子形成时公共提示与当前提示不同，包内保留当时提示副本，但不会替换当前提示。该差异不阻塞现有种子快照的实验入口。包也不包含从旧 40 家数据重新生成所有历史版本所需的完整旧输入。

维护者重新打包可使用 [`package_reproduction_inputs.py`](scripts/enterprise_query_bench/package_reproduction_inputs.py)，范围由 [`reproduction_bundle_v1.json`](configs/validation/reproduction_bundle_v1.json) 固定。该脚本只读取允许清单中的文件，拒绝覆盖已有输出，不打开 SQLite 或执行模型/数据库操作。

## 4. 准备 MySQL 并验收

先安装或启动 MySQL 服务；Docker 只是可选部署方式。默认连接为 `127.0.0.1:3306`、用户 `synthetic`、库 `synthetic_enterprise_v30_120`。由数据库管理员创建账号并授予导入/查询所需权限；连接参数位于 [`configs/business/query_v30_120.yaml`](configs/business/query_v30_120.yaml)。

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

## 5. 配置大模型 API

```bash
export OPENAI_API_KEY='替换为自己的API密钥'
export OPENAI_BASE_URL='替换为服务商的OpenAI兼容接口地址'
export OPENAI_MODEL_NAME='kimi-k2.6'
```

非空 `OPENAI_MODEL_NAME` 同时覆盖查询模型、在线难度评估模型和经验选择模型。密钥从环境读取，不写入仓库。服务应支持 Chat Completions 及当前配置使用的 JSON 响应格式；部分配置包含 `thinking` 参数，切换服务商时需要核对其兼容性。

使用托管 API 时，无需在本机下载查询大模型权重；本机需要单独下载的是 BGE-M3。三组正式运行会产生远程模型请求与相应费用。

## 6. 独立测试与正式运行

当前新评分版本为 **`expected_coverage_v3`**：

- expected 中的核心字段、值和结果行必须完整、正确，不能遗漏、补算或用错误值替代。
- 正确的补充业务列可以通过，前提是评分器有冻结的独立事实依据，且逐项匹配；补充列错误或暂无法核验时仍为 FAIL。
- 模型仍需自行识别问题所需核心数据；逐题 expected、参考 SQL 和隐藏评分事实不作为模型提示。
- 输出仅 PASS / FAIL；PASS 才能进入成功经验准入流程，但并非每个 PASS 都必然入池。

新规则的代码与测试已加入，本次发布未代使用者执行模型调用、数据库操作或独立测试；正式效果以使用者执行结果为准。

由使用者先运行独立测试（模拟远程模型和 embedding，部分测试写入临时 SQLite）：

```bash
python -B -m pytest -q test/test_reproduction_bundle_install.py
bash scripts/enterprise_query_bench/run_coverage_sql_tests.sh
```

当前两个入口的测试自行创建所需临时材料；它们不要求下载 BGE 权重。全仓库其他旧版测试可能依赖旧题库，不属于本次 72 题入口。

准备好依赖、权重、数据、种子、API 和本机验收报告后，进行小范围运行，默认三组各 10 题：

```bash
unset PSU_COVERAGE_RUN
bash scripts/enterprise_query_bench/run_coverage_sql_validation.sh
```

完整 72 题测试分区，三组共 216 回合：

```bash
unset PSU_COVERAGE_RUN
PSU_COVERAGE_CASE_LIMIT=72 bash scripts/enterprise_query_bench/run_coverage_sql_validation.sh
```

配置入口为 [`enterprise_v30_120_v1_0_2_natural_coverage_v3.json`](configs/validation/enterprise_v30_120_v1_0_2_natural_coverage_v3.json)。脚本为每次实验新建 `data/derived/psu_memory/validation/natural-v030-coverage-v3-<时间>/`，保存输入快照、经验状态与各题结果，并打印汇总。最终记录含实际 SQL、实际结果、expected、比较版本、PASS/FAIL 及原因。

若一组在查询开始后中断，应保留原目录并从原始种子建立新目录重跑，避免带入不确定的经验状态。不要对旧运行目录切换评分版本或覆盖历史结果。

旧版 `explicit_columns_aux_v2` 和 [`run_direct_sql_validation.sh`](scripts/enterprise_query_bench/run_direct_sql_validation.sh) 仍保留。v2 只允许企业名称、编号、金额单位等约定辅助列；v3 放宽正确补充业务列。正在运行的 v2 实验沿用已冻结规则，新规则需使用 v3 入口新建实验。

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

助手负责代码、文件部署和已有记录分析；模型请求、数据库操作、独立测试及正式验收由使用者执行。
