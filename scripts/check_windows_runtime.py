"""Check native Windows PSU dependencies and CUDA without loading model weights."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def check_runtime() -> dict:
    if sys.platform != "win32":
        raise RuntimeError("This check is intended for native Windows.")
    if sys.version_info < (3, 11):
        raise RuntimeError("Python 3.11 or newer is required.")

    import mysql.connector
    import numpy as np
    import pydantic
    import sqlglot
    import torch
    import yaml
    from openai import DefaultAsyncHttpx2Client, DefaultHttpx2Client
    from transformers import AutoModel, AutoTokenizer, XLMRobertaModel

    from src.psu_memory.retrieval.retrieval_models import RetrievalConfig

    config = RetrievalConfig.from_json(
        "src/psu_memory/retrieval/retrieval_config.windows_cuda.json"
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment.")
    device = torch.device(config.device)
    if device.index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device does not exist: {device}")

    values = torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4)
    result = values @ values.T
    torch.cuda.synchronize(device)
    expected = np.arange(16, dtype=np.float32).reshape(4, 4)
    np.testing.assert_allclose(result.cpu().numpy(), expected @ expected.T)
    packages = (
        "torch", "transformers", "huggingface-hub", "sentencepiece", "openai",
        "pydantic", "mysql-connector-python", "PyYAML", "numpy", "sqlglot", "pytest",
    )
    return {
        "status": "PASS",
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "cuda_build": torch.version.cuda,
        "device": config.device,
        "gpu": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "compiled_architectures": torch.cuda.get_arch_list(),
        "gpu_matmul": "PASS",
        "embedding_weights_present": config.model_path.is_dir(),
        "embedding_inference_tested": False,
        "database_connection_tested": False,
        "llm_api_tested": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write a new JSON report; never overwrite.")
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error(f"Report already exists: {args.output}")
    try:
        result = check_runtime()
    except Exception as error:
        result = {"status": "FAIL", "error_type": type(error).__name__, "error": str(error)}
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    print(serialized)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized + "\n")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
