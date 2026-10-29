"""Standalone worker: NumPy1/Torch2.2 only; never imports A2 or any labels."""

import json
import importlib.metadata
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, CLIPTextModelWithProjection, CLIPTokenizerFast
from transformers.utils import logging


def main():
    logging.set_verbosity_error()
    torch.set_num_threads(4)
    if np.__version__ != "1.26.4" or torch.__version__ != "2.2.2":
        raise RuntimeError("Isolated numeric dependencies differ")
    for name, expected in {
        "transformers": "4.44.2",
        "tokenizers": "0.19.1",
        "huggingface-hub": "0.25.2",
        "safetensors": "0.4.5",
    }.items():
        if importlib.metadata.version(name) != expected:
            raise RuntimeError("Encoder dependency differs: " + name)
    policy_file = Path(sys.argv[1]).resolve()
    p = json.loads(policy_file.read_text())
    began = time.perf_counter()
    for name in ["sbert_root", "clip_root"]:
        value = Path(os.path.expandvars(p[name])).expanduser()
        p[name] = str((policy_file.parent / value).resolve() if not value.is_absolute() else value)
    sb = AutoModel.from_pretrained(
        p["sbert_root"], local_files_only=True, use_safetensors=True, torch_dtype=torch.float32
    ).eval()
    st = AutoTokenizer.from_pretrained(p["sbert_root"], local_files_only=True)
    cp = CLIPTextModelWithProjection.from_pretrained(
        p["clip_root"], local_files_only=True, use_safetensors=True, torch_dtype=torch.float32
    ).eval()
    ct = CLIPTokenizerFast.from_pretrained(p["clip_root"], local_files_only=True)
    for model in [sb, cp]:
        model.requires_grad_(False)
    print(
        json.dumps(
            {
                "ready": True,
                "models_loaded": 2,
                "device": "cpu",
                "numpy": np.__version__,
                "torch": torch.__version__,
                "load_seconds": time.perf_counter() - began,
            }
        ),
        flush=True,
    )
    for line in sys.stdin:
        request_id = None
        try:
            r = json.loads(line)
            request_id = r["id"]
            texts = r["texts"]
            if (
                not isinstance(texts, list)
                or not 1 <= len(texts) <= 16
                or any(not isinstance(t, str) or not t.strip() or len(t) > 2048 for t in texts)
            ):
                raise ValueError("Invalid query texts")
            with torch.inference_mode():
                start = time.perf_counter()
                inputs = st(
                    texts, padding=True, truncation=True, max_length=256, return_tensors="pt"
                )
                hidden = sb(**inputs).last_hidden_state.float()
                mask = inputs["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
                s = torch.nn.functional.normalize(pooled, dim=1).numpy().astype(np.float32)
                s = s / np.linalg.norm(s, axis=1, keepdims=True)
                s_seconds = time.perf_counter() - start
                start = time.perf_counter()
                inputs = ct(
                    texts, padding=True, truncation=True, max_length=77, return_tensors="pt"
                )
                c = cp(**inputs).text_embeds.float().numpy().astype(np.float32)
                c = c / np.linalg.norm(c, axis=1, keepdims=True)
                c_seconds = time.perf_counter() - start
            print(
                json.dumps(
                    {
                        "id": request_id,
                        "vectors": {"sbert": s.tolist(), "clip": c.tolist()},
                        "timings": {"sbert_seconds": s_seconds, "clip_seconds": c_seconds},
                    },
                    allow_nan=False,
                ),
                flush=True,
            )
        except Exception as exc:
            print("Encoder request failed: " + type(exc).__name__, file=sys.stderr, flush=True)
            print(json.dumps({"id": request_id, "error": "encoding_failed"}), flush=True)


if __name__ == "__main__":
    main()
