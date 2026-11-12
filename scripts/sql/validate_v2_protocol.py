"""Replay evaluator SQL to verify interfaces only; this does NOT measure model accuracy."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.psu_memory.pool.pool_models import ModelReply, DifficultyUsage
from src.psu_memory.retrieval.retrieval_models import RetrievalConfig
from src.psu_memory.retrieval.service import PSURetriever
from src.sql_learning.comparison import ComparisonPolicy
from src.sql_learning.experiment import run_experiment


class ReplayModel:
    def __init__(self, statements):
        self.statements = iter(statements)

    def generate(self, chat_history, system_prompt=""):
        message = chat_history.messages[-1].content
        if message.startswith("工具返回:"):
            result = json.loads(message.split("\n", 1)[1])
            query_id = result["query_id"] if result["success"] else "null"
            return f"Action: Answer\nResult Query ID: {query_id}\nFinal Answer:\nProtocol replay."
        return "Action: Operation\n```sql\n" + next(self.statements) + "\n```"


class FixtureEncoder:
    def __init__(self, config):
        self.config = config

    def count_tokens(self, text, *, usage):
        return len(text)

    def encode(self, texts, *, usage):
        return [(1.0, 0.0) for _ in texts]


class FixtureSelector:
    signature = "protocol-fixture-only"

    def generate(self, *, system_prompt, user_text):
        return ModelReply('{"selected_id":"c1","reason":"protocol fixture"}', DifficultyUsage())

    def close(self):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--ordered", action="store_true", default=None)
    args = parser.parse_args()
    public = json.loads((args.bundle / "public.json").read_text())
    gold_sql = json.loads((args.bundle / "private_sql.json").read_text())
    statements = [gold_sql[case["id"]] for case in public["cases"]]
    config = RetrievalConfig(embedding_model_name="protocol-fixture", embedding_revision="v1",
                             embedding_dimension=2, device="cpu")
    summary = run_experiment(
        args.bundle, args.output, model_factory=lambda: ReplayModel(statements),
        retriever_factory=lambda path: PSURetriever.from_local(
            path, retrieval_config=config, encoder=FixtureEncoder(config), client_factory=FixtureSelector),
        retrieval_config=config, model_settings={"model": "reference SQL replay", "selector": "fixture",
                                                 "embedding": "constant fixture"},
        validation_kind="protocol_replay_not_model_accuracy",
        policy=ComparisonPolicy(ordered=True) if args.ordered else None)
    print(json.dumps({"validation_kind": summary["config"]["validation_kind"],
                      "conditions": summary["conditions"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
