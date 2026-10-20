"""CUDA 配置、设备失败和编码语义回归；不需要 GPU 或模型权重。"""

from __future__ import annotations

import sys
import unittest
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call, patch

from src.project_paths import PROJECT_ROOT
from src.psu_memory.retrieval.embedding import (
    EmbeddingUnavailable,
    LocalBgeM3Encoder,
    embedding_signature,
)
from src.psu_memory.retrieval.retrieval_models import (
    EmbeddingUsage,
    RetrievalConfig,
)
from src.psu_memory.retrieval.vector_cache import vector_cache_key


class CudaConfigTests(unittest.TestCase):
    def test_explicit_cuda_devices_are_accepted(self):
        for device in ("cpu", "mps", "cuda:0", "cuda:1", "cuda:10"):
            with self.subTest(device=device):
                self.assertEqual(RetrievalConfig(device=device).device, device)

    def test_ambiguous_or_invalid_devices_are_rejected(self):
        for device in (
            "cuda", "cuda:-1", "cuda:01", "cuda:1.0", "cuda:x",
            "cuda:0 ", "auto", "CUDA:0", "", None, 0,
        ):
            with self.subTest(device=device):
                with self.assertRaises(ValueError):
                    RetrievalConfig(device=device)

    def test_existing_default_and_cpu_signatures_are_unchanged(self):
        self.assertEqual(RetrievalConfig().device, "mps")
        # 固定旧签名，避免引入 CUDA 时让已有 MPS / CPU 索引全部失效。
        for device in ("mps", "cpu"):
            expected = (
                '{"adapter_version":"bge_m3_dense_v1","device":"'
                + device
                + '","embedding_dimension":1024,"embedding_dtype":"float32",'
                '"embedding_max_tokens":1024,"embedding_model_name":"BAAI/bge-m3",'
                '"embedding_revision":"5617a9f61b028005a4858fdac845db406aefb181",'
                '"normalize_embeddings":true,"pooling_method":"cls",'
                '"text_budget_policy_version":"question_fallback_v1"}'
            )
            with self.subTest(device=device):
                self.assertEqual(
                    embedding_signature(RetrievalConfig(device=device)), expected
                )

    def test_cuda_keeps_representation_constraints(self):
        for values in (
            {"embedding_dtype": "float16"},
            {"pooling_method": "mean"},
            {"normalize_embeddings": False},
        ):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    RetrievalConfig(device="cuda:0", **values)

    def test_windows_configs_change_only_device(self):
        from src.embedding import read_embedding_config

        retrieval_dir = PROJECT_ROOT / "src/psu_memory/retrieval"
        existing = RetrievalConfig.from_json(
            retrieval_dir / "retrieval_config.example.json"
        )
        windows = RetrievalConfig.from_json(
            retrieval_dir / "retrieval_config.windows_cuda.json"
        )
        self.assertEqual(windows, replace(existing, device="cuda:0"))
        self.assertEqual(windows.embedding_batch_size, 4)

        existing_check = read_embedding_config()
        windows_check = read_embedding_config(
            PROJECT_ROOT / "configs/embedding/bge_m3_windows_cuda.yaml"
        )
        self.assertEqual(windows_check, {**existing_check, "device": "cuda:0"})

    def test_cuda_vectors_cannot_reuse_mps_or_cpu_cache_keys(self):
        keys = {
            vector_cache_key(
                "unit", "v1", "same-text-sha256",
                embedding_signature(RetrievalConfig(device=device)),
            )
            for device in ("mps", "cpu", "cuda:0", "cuda:1")
        }
        self.assertEqual(len(keys), 4)


class CudaEncoderTests(unittest.TestCase):
    def test_unavailable_cuda_stops_before_loading_and_never_falls_back(self):
        cuda = SimpleNamespace(
            is_available=Mock(return_value=False), device_count=Mock()
        )
        loader = Mock()
        encoder = LocalBgeM3Encoder(RetrievalConfig(device="cuda:0"))
        usage = EmbeddingUsage(embedding_input_tokens=0)
        with patch.dict(sys.modules, {
            "torch": SimpleNamespace(cuda=cuda),
            "transformers": SimpleNamespace(AutoModel=loader),
        }):
            with self.assertRaisesRegex(EmbeddingUnavailable, "^cuda_unavailable$"):
                encoder.encode(["统计企业数量"], usage=usage)

        cuda.device_count.assert_not_called()
        loader.from_pretrained.assert_not_called()
        self.assertEqual(encoder.config.device, "cuda:0")
        self.assertIsNone(encoder._model)
        self.assertEqual(usage.model_load_count, 0)

    def test_cuda_device_index_is_checked_before_loading(self):
        for device, count in (("cuda:1", 1), ("cuda:0", 0)):
            with self.subTest(device=device, count=count):
                cuda = SimpleNamespace(
                    is_available=Mock(return_value=True),
                    device_count=Mock(return_value=count),
                )
                loader = Mock()
                encoder = LocalBgeM3Encoder(RetrievalConfig(device=device))
                with patch.dict(sys.modules, {
                    "torch": SimpleNamespace(cuda=cuda),
                    "transformers": SimpleNamespace(AutoModel=loader),
                }):
                    with self.assertRaisesRegex(
                        EmbeddingUnavailable, "^cuda_device_out_of_range$"
                    ):
                        encoder.encode(
                            ["统计企业数量"],
                            usage=EmbeddingUsage(embedding_input_tokens=0),
                        )
                loader.from_pretrained.assert_not_called()
                self.assertEqual(encoder.config.device, device)

    def test_cuda_inference_keeps_fp32_cls_normalization_and_explicit_device(self):
        config = RetrievalConfig(device="cuda:1")
        cuda = SimpleNamespace(
            is_available=Mock(return_value=True),
            device_count=Mock(return_value=2),
            synchronize=Mock(),
        )
        dense = Mock()
        dense.detach.return_value.to.return_value.tolist.return_value = [
            [1.0] + [0.0] * 1023
        ]
        normalize = Mock(return_value=dense)
        torch = SimpleNamespace(
            cuda=cuda, float32=object(), inference_mode=nullcontext,
            nn=SimpleNamespace(functional=SimpleNamespace(normalize=normalize)),
        )
        model = MagicMock()
        model.config = SimpleNamespace(hidden_size=1024)
        model.to.return_value = model
        model_loader = Mock(return_value=model)
        tokenizer = Mock()
        tokenizer.model_max_length = 8192
        input_ids, attention_mask = Mock(), Mock()
        lengths = attention_mask.sum.return_value
        lengths.max.return_value.item.return_value = 4
        lengths.sum.return_value.item.return_value = 4
        tokenizer.return_value = {
            "input_ids": input_ids, "attention_mask": attention_mask
        }
        tokenizer_loader = Mock(return_value=tokenizer)
        encoder = LocalBgeM3Encoder(config)
        usage = EmbeddingUsage(embedding_input_tokens=0)

        with patch.dict(sys.modules, {
            "torch": torch,
            "transformers": SimpleNamespace(
                AutoModel=SimpleNamespace(from_pretrained=model_loader),
                AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_loader),
            ),
        }):
            vectors = encoder.encode(["统计企业数量"], usage=usage)

        model_loader.assert_called_once_with(
            str(config.model_path), local_files_only=True, trust_remote_code=False
        )
        tokenizer_loader.assert_called_once_with(
            str(config.model_path), local_files_only=True, trust_remote_code=False
        )
        model.to.assert_called_once_with(device="cuda:1", dtype=torch.float32)
        model.eval.assert_called_once_with()
        input_ids.to.assert_called_once_with("cuda:1")
        attention_mask.to.assert_called_once_with("cuda:1")
        hidden = model.return_value.last_hidden_state
        hidden.__getitem__.assert_called_once_with((slice(None), 0))
        normalize.assert_called_once_with(hidden.__getitem__.return_value, p=2, dim=1)
        dense.detach.return_value.to.assert_called_once_with(
            device="cpu", dtype=torch.float32
        )
        self.assertEqual(cuda.synchronize.call_args_list, [call("cuda:1"), call("cuda:1")])
        self.assertEqual(len(vectors[0]), 1024)
        self.assertEqual(usage.model_load_count, 1)
        self.assertEqual(usage.encoded_text_count, 1)
        self.assertEqual(usage.embedding_input_tokens, 4)


if __name__ == "__main__":
    unittest.main()
