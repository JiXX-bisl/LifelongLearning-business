# dev_v1.1 第一阶段总结

阶段固定日期：2026-11-06。代码分支：`dev_v1.1`，固定标签：`dev_v1.1-20261106`。本阶段验证企业查询中的经验检索与更新机制，保留已确认的实验设定，尚未升级经验方法。

## 1. 已完成的工作

- 建立自然、较模糊的企业查询题库，供模型根据问题、有效上下文和经验判断所需数据。
- 完成 60 道经验形成题及 18 条共同种子的准备与复核，固定 72 道测试题。
- 打通 baseline、FIFO、adaptive 的查询、提交、评分、PASS 后经验更新及结果保存流程。
- 精简公共提示至 v3，删除辅助列推荐和业务指标清单。
- 将评分集中为 `expected_subset_v2`，同步在线入口与已有结果重评入口。
- 补充 macOS、Windows + CUDA 的依赖、BGE-M3 下载、材料安装、数据库验收及运行说明。
- 清理旧评分实现、淘汰入口及过程文件；保留两个核心验证文件和手动 `test_llm.py`。维护操作集中到 `scripts/manage.py`，实现放在代码模块中；三个与材料来源哈希绑定的脚本保留原件。

## 2. 固定的代码、材料和实验条件

| 项目 | 本阶段版本或设定 |
| --- | --- |
| 代码 | 分支 `dev_v1.1`；固定快照标签 `dev_v1.1-20261106` |
| 评分 | `expected_subset_v2`，核心为 `src/psu_memory/validation/comparison.py` |
| 题库 | `enterprise_v30_120_v1_0_2_natural_v0_3_0_expanded` |
| 题目划分 | 60 道经验形成题、72 道测试题；另有 2 道澄清题排除在 SQL 验证之外 |
| 合成企业数据 | `synthetic_enterprise_v30_120_v1.0.2` |
| 公共提示 | 精简 v3 |
| baseline | 无经验 |
| FIFO | 容量 18 |
| adaptive | 四组各 8，总容量 32 |
| 共同种子 | 已复核的 18 条；两个经验组从相同种子开始 |
| embedding | `BAAI/bge-m3`，revision `5617a9f61b028005a4858fdac845db406aefb181` |
| 固定输入包 | `enterprise-v030-v1.0.2-inputs-20261020` |

FIFO 与 adaptive 的容量差异是已经确认的实验条件。模型权重单独下载，输入材料继续使用原 [Release `repro-v030-20261020`](https://github.com/JiXX-bisl/LifelongLearning-business/releases/tag/repro-v030-20261020)，不改写原发行材料。

输入包包含合成企业建库建表 SQL、数据、manifest 和校验文件，私有验收材料，当前 questions、ground_truth、split、reference_validation，以及共同种子和依赖的来源材料。安装步骤见 [README](../README.md)。新机器仍需完成本机数据库验收，历史验收报告不能替代本机验收。

## 3. 当前准入规则

准入原则是：最终提交的数据包含本题必需结果，即为 PASS。必要结果错误、遗漏或提交无效，判为 FAIL。

当前实现的具体边界：

- 不要求 SQL、查询对象、列名或行列布局一致；允许额外输出，额外值的正确性不参与评分。
- 对必要值按出现次数进行包含匹配，每个实际单元格只使用一次；不从其他值计算缺失答案。
- 题目或有效上下文明确给定日历月份范围时，允许省略与该范围一致的 `business_month_count`。实际报送月份数、税额已知月份数等需要查询确定的计数继续必需。
- 对税额结果中的 `value_status`，认可“未知”与 `NULL`、“已知正数”与 `1`、“确认零”与 `0` 的限定等价表示。税额单元格不能同时充当单独的状态答案。
- 数值格式允许等价，预期月份维度允许 `YYYY-MM` 与月首日 `YYYY-MM-01`；未增加通用语义评审模型。
- 最终提交仍须指向一次成功、完整的 SQL 查询。原始 ground_truth、种子和历史运行记录保持不变。

详细说明见 [结果包含评分](psu_memory_结果包含评分.md)。

## 4. 完整运行与离线重评结果

原始运行目录：`data/derived/psu_memory/validation/natural-v030-subset-v1-20261103-093106`。三组各完成 72 题，共 216 回合，在线运行时使用 `expected_subset_v1`。

2026-11-06 使用 `expected_subset_v2` 对这些已保存结果离线重评，报告为 `data/reports/psu_memory/subset-v2-rescore-20261106.json`。

| 组别 | 原在线 PASS / 72 | 原通过率 | v2 重评 PASS / 72 | v2 重评 FAIL | v2 重评通过率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 27 | 37.50% | 31 | 41 | 43.06% |
| FIFO | 28 | 38.89% | 31 | 41 | 43.06% |
| adaptive | 41 | 56.94% | 46 | 26 | 63.89% |

本次共 12 条结果从 FAIL 改为 PASS：3 条 B2 状态等价表达、9 条 B3 已知月份数省略。没有 PASS 改为 FAIL。必要状态、查询确定的覆盖计数、金额或期间汇总遗漏仍保留 FAIL。

在重评口径下，adaptive 比 baseline 和 FIFO 各多通过 15 题，领先 20.83 个百分点。原始在线结果也呈现 adaptive 明显领先、FIFO 与 baseline 接近的关系；优势主要体现在盈利判断及利润率业务口径的满足程度。

**重评仅调整已有输出的判定，没有重放模型查询或在线经验更新。** 原运行期间的经验演化仍由 v1 的在线判定产生，因此这张表不是一轮全新 v2 在线实验的成绩。

## 5. 归档与证据范围

本次归档入口：

- [下载阶段证据 ZIP](https://github.com/JiXX-bisl/LifelongLearning-business/raw/refs/tags/dev_v1.1-20261106/archives/dev_v1.1/stage1-evidence-20261106.zip)
- [归档清单与文件校验](../archives/dev_v1.1/manifest.json)
- [结果汇总](../archives/dev_v1.1/summary.json)
- [v2 离线重评报告](../archives/dev_v1.1/subset-v2-rescore-20261106.json)

`stage1-evidence-20261106.zip` 保存 227 个原运行文件、原 v2 重评报告，以及当前题库的 `intent_contracts.json`、`bundle_manifest.json`、`expansion_map.json` 和原已确认业务口径文档。归档保留这些来源文件的原始字节，文件列表与 SHA-256 以 manifest 为准。

将 ZIP 下载到项目的 `downloads/` 目录后，解压到一个全新的独立目录阅读；目标目录已存在时更换目录名，不覆盖已有材料。

macOS / Linux：

```bash
mkdir review-dev-v1.1-20261106 && unzip downloads/stage1-evidence-20261106.zip -d review-dev-v1.1-20261106
```

Windows / PowerShell：

```powershell
$stageDir = 'review-dev-v1.1-20261106'
if (Test-Path -LiteralPath $stageDir) { throw '目标目录已存在，请更换新的目录名。' }
Expand-Archive -LiteralPath 'downloads/stage1-evidence-20261106.zip' -DestinationPath $stageDir
```

归档中的原绝对路径仅用于记录历史来源，不是另一台机器应直接使用的运行路径。原始输入包、来源 manifest 及原运行配置中的旧评分或状态元数据描述的是原封存时点；当前评分版本以 `expected_subset_v2` 为准，阶段结果应同时参照原在线结果和 v2 重评报告。归档本身不替代 README 第 3 节的实验输入安装与第 4 节的本机数据库验收。

原运行目录、原始重评报告和固定输入包保留。阶段资料与代码一并纳入 `dev_v1.1`，固定快照由 `dev_v1.1-20261106` 标签定位；后续修改应另留可追溯的提交和结果文件。

模型请求、数据库操作、独立测试及重评由使用者执行。使用者已反馈完成本轮执行，本阶段总结引用已保存的 216 条结果及 v2 重评报告；本次版本整理不新增模型、数据库或测试执行，也不将未保存的测试终端输出作为已归档证据。

## 6. 阶段结论与后续工作

第一阶段实验流程已打通，当前评分口径完成本轮校准，adaptive 在本轮实验条件下呈现明确优势。当前代码、固定输入材料和结果归档可作为下一阶段工作的起点。

接下来优先围绕现有 FAIL 做业务原因归纳：区分必要结果遗漏、历史范围未覆盖、指标口径选择及无有效提交。保持已确认的隐含业务目标，依据新的明确需求再调整代码或材料。若后续升级经验方法，或需要检验新版准入对在线经验演化及重复运行稳定性的影响，再由使用者安排新的在线实验。
