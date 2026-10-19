"""Check dependencies and run local BGE-M3 inference (no external LLM API)."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
from time import perf_counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Keep download caches inside this project. These must precede HF imports.
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "data/models/huggingface_home"))

from src.embedding import (
    DEFAULT_CONFIG, download_embedding_model, load_bge_m3,
    read_embedding_config, select_embedding_device,
)
from src.project_paths import resolve_project_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=["auto", "mps", "cpu", "cuda:0"])
    parser.add_argument("--check-only", action="store_true", help="检查依赖和设备，不加载权重。")
    parser.add_argument("--download", action="store_true", help="允许下载模型权重。")
    parser.add_argument("--download-only", action="store_true", help="只下载权重，不运行推理。")
    parser.add_argument("--fp16", action="store_true", help="使用 FP16 推理。")
    parser.add_argument("--text", action="append", help="要编码的文本，可重复传入。")
    parser.add_argument("--output", type=Path, help="保存验证结果 JSON（不含向量）。")
    args = parser.parse_args()

    import numpy as np
    import torch
    from FlagEmbedding import BGEM3FlagModel  # Verify the full import chain.

    config = read_embedding_config(args.config)
    device = select_embedding_device(args.device or config["device"])
    result = {
        "python": sys.version.split()[0], "python_executable": sys.executable,
        "architecture": platform.machine(), "macos": platform.mac_ver()[0],
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["torch", "FlagEmbedding", "transformers", "sentence-transformers"]
        },
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(), "selected_device": device,
        "model_name": config["model_name"], "revision": config["revision"],
        "mps_cpu_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0"),
    }
    if args.download_only:
        result["model_path"] = download_embedding_model(config, download=True)
    elif not args.check_only:
        texts = args.text or [
            "企业是否存在债务违约风险？", "公司逾期未偿还银行贷款。",
            "企业出现贷款逾期，可能存在债务违约风险。", "今天的天气晴朗。",
        ]
        started = perf_counter()
        model = load_bge_m3(
            args.config, device=device,
            use_fp16=True if args.fp16 else None, download=args.download,
        )
        result["load_seconds"] = round(perf_counter() - started, 3)
        started = perf_counter()
        vectors = model.encode(
            texts, batch_size=int(config["batch_size"]), max_length=int(config["max_length"]),
            return_dense=True, return_sparse=False, return_colbert_vecs=False,
        )["dense_vecs"]
        if device == "mps":
            torch.mps.synchronize()
        result["encode_seconds"] = round(perf_counter() - started, 3)
        parameter = next(model.model.parameters())
        result.update({
            "parameter_device": str(parameter.device), "parameter_dtype": str(parameter.dtype),
            "embedding_shape": list(vectors.shape), "embedding_dtype": str(vectors.dtype),
            "all_finite": bool(np.isfinite(vectors).all()),
            "vector_norms": np.linalg.norm(vectors.astype(np.float32), axis=1).round(6).tolist(),
            "texts": texts,
            "similarity_to_first": (vectors.astype(np.float32) @ vectors[0].astype(np.float32)).round(6).tolist(),
        })
        assert parameter.device.type == device.split(":")[0], "模型设备与配置不一致。"
        assert vectors.shape == (len(texts), 1024) and result["all_finite"], "向量验证失败。"
        assert np.allclose(result["vector_norms"], 1.0, atol=0.01), "向量未正常归一化。"
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    print(serialized)
    if args.output:
        output = resolve_project_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
