"""Run v2 using API environment variables from the invoking terminal."""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.lifelong_learning.language_models.openai_compatible import OpenAICompatibleLanguageModel
from src.psu_memory.retrieval.retrieval_models import RetrievalConfig
from src.psu_memory.retrieval.service import PSURetriever
from src.sql_learning.comparison import ComparisonPolicy
from src.sql_learning.experiment import run_experiment, load_comparison_policy


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--retrieval-config", type=Path,
                        default=Path("src/psu_memory/retrieval/retrieval_config.example.json"))
    parser.add_argument("--model-config", default="configs/business/query_v30_120.yaml",
                        help="Existing PSU selector model configuration (credentials from environment)")
    parser.add_argument("--seed-store", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--memory-budget", type=int, default=3000)
    parser.add_argument("--ordered", action="store_true", default=None)
    parser.add_argument("--policy-file", type=Path, help="Evaluator-only comparison policy JSON")
    parser.add_argument("--extra-body", type=json.loads, default={}, help="Optional model-specific API options as JSON")
    args = parser.parse_args(argv)
    if not os.environ.get("OPENAI_API_KEY") or not os.environ.get("OPENAI_BASE_URL"):
        parser.error("set OPENAI_API_KEY and OPENAI_BASE_URL in this terminal before running")
    config = RetrievalConfig(**json.loads(args.retrieval_config.read_text()))
    settings = {"model_name": args.model, "temperature": 0, "max_tokens": 4096,
                "timeout": 120, "max_retries": 0, "reject_truncated": True, "extra_body": args.extra_body}
    policy = load_comparison_policy(args.bundle, args.policy_file)
    if args.ordered is not None:
        policy = replace(policy, ordered=args.ordered)
    summary = run_experiment(
        args.bundle, args.output, model_factory=lambda: OpenAICompatibleLanguageModel(**settings),
        retriever_factory=lambda path: PSURetriever.from_local(path, retrieval_config=config,
                                                              model_config_path=args.model_config),
        retrieval_config=config, model_settings={**settings, "selector_model": config.selector_model_name,
                                                 "embedding_model": config.embedding_model_name,
                                                 "embedding_revision": config.embedding_revision},
        seed_store=args.seed_store, max_context_chars=args.memory_budget, limit=args.limit,
        policy=policy)
    print(json.dumps(summary["conditions"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
