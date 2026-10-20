#!/usr/bin/env bash
# 用户独立测试入口：模拟模型与嵌入，使用临时SQLite，不启动正式实验。
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

python -B -m pytest -q \
  test/test_psu_direct_comparison.py \
  test/test_psu_submission_comparison.py

python -B -m unittest \
  test.test_chatflow_runtime \
  test.test_psu_seed_pool \
  test.test_psu_pool_difficulty \
  test.test_psu_direct_runner \
  test.test_psu_saved_result_check \
  test.test_psu_direct_update \
  test.test_psu_query_integration \
  test.test_psu_pool_classification \
  test.test_psu_pool_selection \
  test.test_psu_active_members \
  test.test_psu_quality \
  test.test_psu_model_client \
  test.test_psu_retrieval_text \
  test.test_psu_embedding \
  test.test_psu_retrieval_index \
  test.test_psu_prefilter \
  test.test_psu_selector_prompt \
  test.test_psu_selector \
  test.test_psu_selector_acceptance \
  test.test_psu_retriever \
  test.test_psu_augmentation \
  test.test_psu_prompt_enhancement
