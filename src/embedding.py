"""Load the project's BGE-M3 model on Apple MPS, CUDA, or CPU."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.project_paths import PROJECT_ROOT, resolve_project_path


DEFAULT_CONFIG = PROJECT_ROOT / "configs/embedding/bge_m3.yaml"
# Download the native PyTorch model without the duplicate ONNX model or images.
MODEL_FILES = [
    "config.json", "pytorch_model.bin", "model.safetensors",
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "sentencepiece.bpe.model", "colbert_linear.pt", "sparse_linear.pt",
]


def read_embedding_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = yaml.safe_load(resolve_project_path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Embedding 配置必须是 YAML 对象。")
    if int(config["batch_size"]) < 1 or not 1 <= int(config["max_length"]) <= 8192:
        raise ValueError("batch_size 必须为正数，max_length 必须在 1–8192 之间。")
    return config


def select_embedding_device(requested: str = "auto") -> str:
    import torch

    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("当前进程无法使用 MPS；请检查原生 ARM64 PyTorch，或显式使用 cpu。")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("当前进程无法使用 CUDA。Apple GPU 请使用 mps。")
    if requested not in {"cpu", "mps"} and not requested.startswith("cuda"):
        raise ValueError(f"不支持的设备：{requested}")
    return requested


def download_embedding_model(config: dict[str, Any], *, download: bool = False) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=config["model_name"],
        revision=config["revision"],
        cache_dir=str(resolve_project_path(config["cache_dir"])),
        allow_patterns=MODEL_FILES,
        local_files_only=False if download else bool(config.get("local_files_only", True)),
        max_workers=2,
    )


def load_bge_m3(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    device: str | None = None,
    use_fp16: bool | None = None,
    download: bool = False,
):
    """Return BGEM3FlagModel; call encode(...)[\"dense_vecs\"] for vectors.

    Load once per process and reuse it. The default configuration loads only
    local weights; pass download=True on the first run to fetch them.
    """
    from FlagEmbedding import BGEM3FlagModel

    config = read_embedding_config(config_path)
    selected = select_embedding_device(device or config["device"])
    fp16 = config["use_fp16"] if use_fp16 is None else use_fp16
    return BGEM3FlagModel(
        download_embedding_model(config, download=download),
        devices=selected,
        use_fp16=bool(fp16) and selected != "cpu",
        normalize_embeddings=bool(config["normalize_embeddings"]),
        batch_size=int(config["batch_size"]),
        query_max_length=int(config["max_length"]),
        passage_max_length=int(config["max_length"]),
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
        trust_remote_code=False,
    )
