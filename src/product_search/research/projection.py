"""Evaluated but not adopted: lightweight, frozen-CLIP contrastive heads.

This module is never imported by serving. Only explicit Warm training content
and independent Warm alignment validation pairs are permitted; Cold products
cannot enter training. Retained to explain the reported adaptation study.
"""

from pathlib import Path
import time
import numpy as np
from ..schema import KEY
from .ranking import _tensor, _numpy
from .evaluation import bootstrap
from .io import write_json as atomic_json


def validate_alignment_roles(products, train, val, seen):
    def keys(f):
        return set(f[KEY].itertuples(index=False, name=None))

    if train.empty or val.empty or keys(train) & keys(val):
        raise ValueError("Independent nonempty alignment train/validation required")
    pm = products.set_index(KEY)
    for f in [train, val]:
        if f.duplicated(KEY).any() or not f.product_split.eq("warm").all():
            raise ValueError("Warm unique pairs only; Cold pairs forbidden")
        actual = pm.loc[list(f[KEY].itertuples(index=False, name=None))]
        if not actual.product_split.eq("warm").all() or not actual.has_image_embedding.all():
            raise ValueError("Invalid Warm image role")
        if not (actual.embedding_row.to_numpy() == f.embedding_row.to_numpy()).all():
            raise ValueError("Embedding row mismatch")
    if not keys(train) <= keys(seen) or keys(val) & keys(seen):
        raise ValueError("Warm-Val must be unseen in Train supervision")
    for c in ["title_normalized", "image_url", "image_vector_hash"]:
        a = set(pm.loc[list(keys(train)), c].dropna()) - {""}
        b = set(pm.loc[list(keys(val)), c].dropna()) - {""}
        if a & b:
            raise ValueError("Alignment holdout content overlap")


def projection_config():
    return {
        "seed": 20261015,
        "dimension": 768,
        "epochs": 5,
        "batch_size": 256,
        "lr": 0.0001,
        "weight_decay": 0.0001,
        "temperature": 0.07,
        "device": "auto",
        "eval_chunk": 256,
    }


class Heads:
    def __init__(self, dim):
        import torch

        self.text = torch.nn.Linear(dim, dim, bias=False)
        self.image = torch.nn.Linear(dim, dim, bias=False)
        with torch.no_grad():
            self.text.weight.copy_(torch.eye(dim))
            self.image.weight.copy_(torch.eye(dim))

    def to(self, device):
        self.text.to(device)
        self.image.to(device)
        return self

    def parameters(self):
        return list(self.text.parameters()) + list(self.image.parameters())

    def numeric(self):
        return {"text": _numpy(self.text.weight).copy(), "image": _numpy(self.image.weight).copy()}


def project(array, weight, batch=2048):
    a = np.asarray(array, np.float32)
    out = np.empty((len(a), len(weight)), np.float32)
    for start in range(0, len(a), batch):
        z = a[start : start + batch] @ weight.T
        n = np.linalg.norm(z, axis=1, keepdims=True)
        if np.any(n < 1e-10):
            raise ValueError("Degenerate projected nonmissing vector")
        out[start : start + batch] = z / n
    return out


def symmetric_infonce(text, image, temperature=0.07):
    import torch

    if temperature <= 0 or text.shape != image.shape or len(text) < 2:
        raise ValueError("Paired batch>=2 and positive temperature required")
    logits = text @ image.T / temperature
    target = torch.arange(len(text), device=text.device)
    return 0.5 * (
        torch.nn.functional.cross_entropy(logits, target)
        + torch.nn.functional.cross_entropy(logits.T, target)
    )


def alignment(text, image, device="cpu", chunk=256, confidence=True):
    import torch

    t = _tensor(text).to(device)
    im = _tensor(image).to(device)
    result = {}
    per = {}
    with torch.no_grad():
        for name, q, g in [("text_to_image", t, im), ("image_to_text", im, t)]:
            hits = {k: [] for k in [1, 5, 10]}
            for start in range(0, len(q), chunk):
                ranks = torch.topk(q[start : start + chunk] @ g.T, k=min(10, len(g)), dim=1).indices
                truth = torch.arange(start, min(start + chunk, len(q)), device=device)[:, None]
                for k in hits:
                    hits[k].extend(_numpy((ranks[:, : min(k, len(g))] == truth).any(1)).tolist())
            per[name] = hits
            result[name] = {f"recall@{k}": float(np.mean(v)) for k, v in hits.items()}
            result[name]["ci95"] = (
                {f"recall@{k}": bootstrap(v) for k, v in hits.items()} if confidence else {}
            )
    result["selection_mean_recall@10"] = float(np.mean([result[n]["recall@10"] for n in per]))
    result["products"] = len(t)
    result["confidence_scope"] = "Resample paired product IDs, gallery fixed; no relevance labels"
    return result


def train_heads(products, train, validation, seen, vectors, output, cfg=None, progress=print):
    import os

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    validate_alignment_roles(products, train, validation, seen)
    cfg = cfg or projection_config()
    if len(train) < 2 or cfg["dimension"] != vectors["clip_product_text"].shape[1]:
        raise ValueError("At least two training pairs and matching head dimension required")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = "cuda" if cfg["device"] == "auto" and torch.cuda.is_available() else cfg["device"]
    device = "cpu" if device == "auto" else device
    tr = train.sort_values(["product_locale", "product_id"]).embedding_row.to_numpy()
    va = validation.sort_values(["product_locale", "product_id"]).embedding_row.to_numpy()
    t = _tensor(np.asarray(vectors["clip_product_text"][tr])).to(device)
    im = _tensor(np.asarray(vectors["clip_product_image"][tr])).to(device)
    vt = np.asarray(vectors["clip_product_text"][va])
    vi = np.asarray(vectors["clip_product_image"][va])
    h = Heads(t.shape[1]).to(device)
    optim = torch.optim.AdamW(h.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    rng = np.random.default_rng(cfg["seed"])
    start = time.perf_counter()
    baseline = alignment(vt, vi, device, cfg["eval_chunk"])
    best = -np.inf
    best_epoch = 0
    history = []
    for epoch in range(1, cfg["epochs"] + 1):
        order = rng.permutation(len(tr))
        losses = []
        for a in range(0, len(order), cfg["batch_size"]):
            ids = order[a : a + cfg["batch_size"]]
            if len(ids) < 2:
                continue
            ids = _tensor(ids).to(device)
            tv = torch.nn.functional.normalize(h.text(t[ids]), dim=-1)
            iv = torch.nn.functional.normalize(h.image(im[ids]), dim=-1)
            loss = symmetric_infonce(tv, iv, cfg["temperature"])
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            losses.append(float(loss.detach().cpu()))
            if (a // cfg["batch_size"] + 1) % 100 == 0:
                progress(f"Projection epoch {epoch}: {a + len(ids)}/{len(tr)} pairs")
        w = h.numeric()
        scores = alignment(
            project(vt, w["text"]),
            project(vi, w["image"]),
            device,
            cfg["eval_chunk"],
            confidence=False,
        )
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "alignment": scores,
            "elapsed_seconds": time.perf_counter() - start,
        }
        history.append(row)
        np.savez(root / f"epoch_{epoch}.npz", **w)
        atomic_json(root / "training_curve.json", history)
        progress(
            f"Projection epoch {epoch}/{cfg['epochs']}: loss={row['loss']:.5f}, Warm-Val bidirectional R@10={scores['selection_mean_recall@10']:.5f}"
        )
        if scores["selection_mean_recall@10"] > best:
            best = scores["selection_mean_recall@10"]
            best_epoch = epoch
            np.savez(root / "heads.npz", **w)
    with np.load(root / "heads.npz", allow_pickle=False) as f:
        weights = {k: f[k] for k in f.files}
    final = alignment(
        project(vt, weights["text"]), project(vi, weights["image"]), device, cfg["eval_chunk"]
    )
    torch.save(
        {"text_weight": _tensor(weights["text"]), "image_weight": _tensor(weights["image"])},
        root / "heads.pt",
    )
    result = {
        "config": cfg,
        "device": device,
        "frozen_clip": True,
        "encoder_downloaded": False,
        "supervised_labels_used": False,
        "train_products": len(tr),
        "alignment_validation_products": len(va),
        "query_projection": "Same text head as product text; normalized",
        "selected_epoch": best_epoch,
        "selection": "Warm-Val mean bidirectional Recall@10 only",
        "frozen": baseline,
        "adapted": final,
        "seconds": time.perf_counter() - start,
        "history": history,
    }
    atomic_json(root / "report.json", result)
    return result
