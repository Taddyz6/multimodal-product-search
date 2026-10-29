"""Original three ranking objectives; training is explicit and offline only.

Only Train/Warm supervision is accepted. Model selection uses the same full
fixed Validation candidate pool with non-Warm target masking. Existing outputs
are protected; callers must choose a fresh experiment directory.
"""

import json
from pathlib import Path
import time
import numpy as np
from ..schema import PAIR_KEY, check_labels
from ..ranking import ListNetRanker
from .evaluation import evaluate_pool
from .io import write_json as atomic_json

KINDS = ["xgboost", "lambdamart", "listnet"]


def model_config():
    return {
        "seed": 20261014,
        "threads": 4,
        "tree_rounds": 600,
        "check_every": 20,
        "patience": 5,
        "xgboost": {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "max_depth": 6,
            "eta": 0.05,
            "min_child_weight": 10,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "lambda": 10,
            "alpha": 0,
        },
        "lambdamart": {
            "objective": "lambdarank",
            "metric": "None",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "min_data_in_leaf": 20,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
            "label_gain": [0, 1, 3, 7],
            "lambdarank_truncation_level": 10,
            "deterministic": True,
            "force_col_wise": True,
            "verbosity": -1,
        },
        "listnet": {
            "hidden": [64, 32],
            "lr": 0.001,
            "weight_decay": 1e-05,
            "batch_queries": 64,
            "epochs": 30,
            "device": "auto",
        },
    }


class WarmMetric:
    def __init__(self, pairs, labels):
        dummy = pairs.copy()
        dummy["score"] = 0.0
        evaluate_pool(dummy, labels, pairs)
        self.canonical = pairs.reset_index(drop=True).sort_values(PAIR_KEY).index.to_numpy()
        f = pairs.iloc[self.canonical].reset_index(drop=True)
        target = {
            (r.query_id, r.product_locale, r.product_id): 2 ** int(r.grade) - 1
            for r in labels.itertuples()
        }
        positive = set(
            labels.loc[labels.label.isin(["E", "S"]), PAIR_KEY].itertuples(index=False, name=None)
        )
        keys = list(f[PAIR_KEY].itertuples(index=False, name=None))
        gain = np.array([target.get(k, 0) for k in keys], float)
        self.groups = []
        for idx in f.groupby("query_id", sort=False).indices.values():
            if not any((keys[i] in positive for i in idx)):
                continue
            g = gain[idx]
            d = 1 / np.log2(np.arange(2, len(idx) + 2))
            ideal = np.sort(g)[::-1][:10] @ d[:10]
            self.groups.append((idx, g, d, ideal))

    def __call__(self, scores):
        s = np.asarray(scores)[self.canonical]
        if not np.isfinite(s).all():
            raise ValueError("Nonfinite validation scores")
        return float(
            np.mean(
                [
                    g[np.argsort(-s[idx], kind="stable")][:10] @ d[:10] / ideal
                    for idx, g, d, ideal in self.groups
                ]
            )
        )


def _tensor(x):
    import torch

    a = np.asarray(x)
    try:
        return torch.from_numpy(a)
    except RuntimeError as e:
        if "numpy" not in str(e).lower() or a.size > 1000000:
            raise
        return torch.tensor(
            a.tolist(),
            dtype=torch.bool
            if a.dtype == bool
            else torch.long
            if np.issubdtype(a.dtype, np.integer)
            else torch.float32,
        )


def _numpy(x):
    try:
        return x.detach().cpu().numpy()
    except RuntimeError as e:
        if "numpy" not in str(e).lower():
            raise
        return np.asarray(x.detach().cpu().tolist(), np.float32)


def listnet_loss(logits, grades, mask):
    import torch

    floor = torch.finfo(logits.dtype).min
    target = torch.softmax(grades.masked_fill(~mask, floor), dim=1)
    log_prob = torch.log_softmax(logits.masked_fill(~mask, floor), dim=1)
    return -(target * log_prob).sum(dim=1).mean()


def load_predictor(root):
    root = Path(root)
    meta = json.loads((root / "model.json").read_text())
    kind = meta["kind"]
    if kind == "xgboost":
        import xgboost as xgb

        m = xgb.Booster()
        m.load_model(root / "model.xgb.json")
        m.set_param({"nthread": meta["config"]["threads"]})

        def scorer(x):
            return m.predict(xgb.DMatrix(x, feature_names=meta["features"]))
    elif kind == "lambdamart":
        import lightgbm as lgb

        m = lgb.Booster(model_file=str(root / "model.txt"))

        def scorer(x):
            return m.predict(x, num_threads=meta["config"]["threads"])
    elif kind == "listnet":
        with np.load(root / "model.npz", allow_pickle=False) as f:
            w = {k: f[k] for k in f.files}
        ranker = ListNetRanker(w)

        def scorer(x):
            return ranker.forward(ranker.scale(x))

    else:
        raise ValueError("Unknown ranking model kind")

    def result(X):
        X = np.ascontiguousarray(X, np.float32)
        if X.ndim != 2 or X.shape[1] != len(meta["features"]):
            raise ValueError("Feature dimension mismatch")
        return scorer(X)

    return result


def train(
    kind, train_labels, X, pairs, V, validation_labels, names, output, config=None, progress=print
):
    cfg = config or model_config()
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    check_labels(train_labels, "train")
    if kind not in KINDS or len(X) != len(train_labels) or len(V) != len(pairs):
        raise ValueError("Invalid model inputs")
    if set(train_labels.query_id) & set(pairs.query_id):
        raise ValueError("Train/Validation Query overlap")
    if (
        not train_labels[PAIR_KEY]
        .reset_index(drop=True)
        .equals(train_labels.sort_values(PAIR_KEY)[PAIR_KEY].reset_index(drop=True))
    ):
        raise ValueError("Train Query lists must be sorted")
    X = np.ascontiguousarray(X, np.float32)
    V = np.ascontiguousarray(V, np.float32)
    y = train_labels.grade.to_numpy(np.float32)
    metric = WarmMetric(pairs, validation_labels)
    start = time.perf_counter()
    history = []
    best = {"score": -np.inf, "step": 0}
    stale = 0

    def observe(step, scores):
        nonlocal stale
        value = metric(scores)
        history.append({"step": step, "warm_ndcg@10": value})
        progress(f"{kind} step {step}: Warm NDCG@10={value:.6f}")
        if value > best["score"] + 1e-12:
            best.update(score=value, step=step)
            stale = 0
            return True
        stale += 1
        return False

    if kind == "xgboost":
        import xgboost as xgb

        dm = xgb.DMatrix(X, label=y, feature_names=names)
        dv = xgb.DMatrix(V, feature_names=names)

        class Callback(xgb.callback.TrainingCallback):
            def after_iteration(self, model, epoch, evals_log):
                if (epoch + 1) % cfg["check_every"] == 0 or epoch + 1 == cfg["tree_rounds"]:
                    observe(epoch + 1, model.predict(dv))
                return stale >= cfg["patience"]

        params = dict(cfg[kind], seed=cfg["seed"], nthread=cfg["threads"])
        m = xgb.train(params, dm, num_boost_round=cfg["tree_rounds"], callbacks=[Callback()])
        m = m[: best["step"]]
        m.save_model(root / "model.xgb.json")
    elif kind == "lambdamart":
        import lightgbm as lgb

        groups = train_labels.groupby("query_id", sort=False).size().to_numpy()
        dataset = lgb.Dataset(X, label=y.astype(int), group=groups, feature_name=names)

        def callback(env):
            step = env.iteration + 1
            if step % cfg["check_every"] == 0 or step == cfg["tree_rounds"]:
                observe(step, env.model.predict(V, num_threads=cfg["threads"]))
            if stale >= cfg["patience"]:
                raise lgb.callback.EarlyStopException(
                    best["step"] - 1, [("warm", "ndcg@10", best["score"], True)]
                )

        params = dict(cfg[kind], seed=cfg["seed"], num_threads=cfg["threads"])
        m = lgb.train(params, dataset, num_boost_round=cfg["tree_rounds"], callbacks=[callback])
        m.save_model(str(root / "model.txt"), num_iteration=best["step"])
    else:
        import os

        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch

        torch.manual_seed(cfg["seed"])
        torch.set_num_threads(cfg["threads"])
        lc = cfg["listnet"]
        device = "cuda" if lc["device"] == "auto" and torch.cuda.is_available() else lc["device"]
        device = "cpu" if device == "auto" else device
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        if device == "cuda":
            torch.cuda.manual_seed_all(cfg["seed"])
        mean = X.mean(0, dtype=np.float64)
        std = X.std(0, dtype=np.float64)
        std[std < 1e-08] = 1
        norm = np.asarray((X - mean) / std, np.float32)
        features = _tensor(norm).to(device)
        grades = _tensor(y).to(device)
        h1, h2 = lc["hidden"]
        m = torch.nn.Sequential(
            torch.nn.Linear(len(names), h1),
            torch.nn.ReLU(),
            torch.nn.Linear(h1, h2),
            torch.nn.ReLU(),
            torch.nn.Linear(h2, 1),
        ).to(device)
        optim = torch.optim.Adam(m.parameters(), lr=lc["lr"], weight_decay=lc["weight_decay"])
        counts = train_labels.groupby("query_id", sort=False).size().to_numpy()
        starts = np.r_[0, counts.cumsum()[:-1]]
        rng = np.random.default_rng(cfg["seed"])
        weights = None
        for epoch in range(1, lc["epochs"] + 1):
            m.train()
            order = rng.permutation(len(counts))
            losses = []
            for a in range(0, len(order), lc["batch_queries"]):
                batch = order[a : a + lc["batch_queries"]]
                position = np.arange(counts[batch].max())[None, :]
                mask = position < counts[batch, None]
                ids = np.minimum(starts[batch, None] + position, len(X) - 1)
                it = _tensor(ids).to(device)
                mt = _tensor(mask).to(device)
                loss = listnet_loss(m(features[it]).squeeze(-1), grades[it], mt)
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()
                losses.append(float(loss.detach().cpu()))
            current = {"mean": mean, "std": std}
            for j, index in enumerate([0, 2, 4], 1):
                current[f"w{j}"] = _numpy(m[index].weight).copy()
                current[f"b{j}"] = _numpy(m[index].bias).copy()
            if observe(epoch, ListNetRanker(current).forward(ListNetRanker(current).scale(V))):
                weights = current
                torch.save(
                    {k: v.detach().cpu() for k, v in m.state_dict().items()}, root / "model.pt"
                )
            history[-1]["mean_loss"] = float(np.mean(losses))
            if stale >= cfg["patience"]:
                break
        np.savez(root / "model.npz", **weights)
    seconds = time.perf_counter() - start
    meta = {
        "kind": kind,
        "features": names,
        "config": cfg,
        "best_step": best["step"],
        "warm_ndcg@10": best["score"],
        "seconds": seconds,
        "training_queries": train_labels.query_id.nunique(),
        "training_pairs": len(train_labels),
        "training_scope": "Train/Warm frozen LTR view",
        "selection": "Full fixed candidate pool Warm Validation NDCG@10",
        "listnet_target": "softmax(human grade) top-one cross entropy, masked padding",
        "history": history,
    }
    atomic_json(root / "model.json", meta)
    replay = metric(load_predictor(root)(V))
    if abs(replay - best["score"]) > 1e-10:
        raise ValueError(f"Saved model validation replay disagreement: {replay}, {best}")
    return meta
