# 结果包含评分

当前唯一评分版本：`expected_subset_v2`。更新日期：2026-11-06。

**输出的数据结果完整包含本题必需结果，就判 PASS；必要结果缺失或不匹配，才判 FAIL。含义一致的已确认表达可以等价匹配。**

必要内容依据问题、有效上下文和已确认的业务目标判断。当前规则直接实现在 `src/psu_memory/validation/comparison.py`，运行和重评入口传入题目上下文；不新增逐题规则配置，也不在评分时调用模型判断题意。

- 题目已明确指定统计月份，且该日历范围与 expected 中的 `business_month_count` 一致时，该计数是已知条件，无需重复输出。例如已指定三个具体月份，营收明细及合计正确，不因缺少独立数字“3”而 FAIL。仅在能够确定固定月份范围时应用此规则；不能确定时保留原必需值。
- 统计月份需要 Agent 查询后确定时，业务月份数仍须输出。实际报送月份数、税额已知月份数、实际观测月份数、亏损月份数等独立统计不因题目指定了时间范围而免除。
- B2 税款状态 `value_status` 按本题已确认的含义认可等价表达：“未知”与 `NULL`、“已知正数”与 `1`、“确认零”与 `0`。此规则只作用于税款状态，不将所有字符串、金额、计数或任意判断统一转换。必要状态没有输出仍 FAIL。

原始 `ground_truth.json` 和参考结果保留；评分时按上述规则确定必需值，并记录实际使用的评分信息。其余既有业务目标保持不变，评分器不会从明细补算缺失的合计、人数或其他必需统计。

比较时将结果的单元格值展开，每一个必需值都须在 actual 中找到一个对应单元格。重复值仍按出现次数匹配，例如必需值为 `[0, 0]` 时，输出须至少包含两个 0。每个实际单元格最多匹配一次：同一个税额 `NULL` 不能同时充当税额和状态两个答案。列名、SQL、查询对象、行列布局、额外行列及额外输出是否正确均不作为结果准确匹配的约束，也不要求补充事实证据。

数值按精确值比较，`12`、`12.00` 等价；既有月份表示 `YYYY-MM` 与月首日 `YYYY-MM-01` 保持等价。最终提交仍引用实际执行的查询结果，提交记录缺失或结果不可读时不能完成比较。

`test/` 仅保留两个核心验证及手动模型连通性工具：

- `test_psu_submission_comparison.py`：必要值包含、已知月份数、状态等价、真实遗漏和最终提交绑定。
- `test_psu_direct_runner.py`：题目分区、共同种子、三组运行、PASS更新和历史结果保护。
- `test_llm.py`：手动验证真实模型接口，保留原样，不纳入核心测试入口。

下面入口只运行前两个文件，查询、检索、经验更新和数据库访问使用模拟对象。

macOS / Linux 独立测试：

```bash
bash scripts/enterprise_query_bench/run_values_sql_tests.sh
```

Windows / PowerShell 独立测试：

```powershell
.\scripts\enterprise_query_bench\run_values_sql_tests.ps1
```

历史结果仅从 JSON 重新评分，写入全新报告：

```bash
python -B scripts/manage.py rescore \
  --run-dir data/derived/psu_memory/validation/natural-v030-subset-v1-20261103-093106 \
  --report data/reports/psu_memory/subset-v2-rescore-20261106.json
```

已有报告不能覆盖，请使用新文件名。重评使用当前 v2 规则和历史题目上下文，仅评价已保存的最终结果；不调用模型、不查询数据库、不重放经验更新，也不改写历史结果。重评后的通过率不能视为按新准则重新开展经验学习的实验结果。

完整三组各 72 题运行：

```bash
unset PSU_VALUES_RUN
PSU_VALUES_CASE_LIMIT=72 bash scripts/enterprise_query_bench/run_values_sql_validation.sh
```

Windows / PowerShell：

```powershell
Remove-Item Env:PSU_VALUES_RUN -ErrorAction SilentlyContinue
.\scripts\enterprise_query_bench\run_values_sql_validation.ps1 -CaseLimit 72
```

运行配置为 `configs/validation/enterprise_v30_120_v1_0_2_natural_values.json`；Windows 使用同名 `_windows_cuda.json` 配置。新目录前缀为 `natural-v030-subset-v2`，环境变量仍为 `PSU_VALUES_RUN` 和 `PSU_VALUES_CASE_LIMIT`。模型、数据、种子和安装步骤见 [README](../README.md)。全部独立测试、重评、模型与数据库操作由使用者执行。
