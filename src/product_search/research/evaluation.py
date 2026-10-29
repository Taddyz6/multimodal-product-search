"""Frozen evaluation definitions for external, scoped human labels.

Fixed-pool NDCG keeps every competitor position; zero outside the chosen scope
is a target mask, never a fabricated I label. Catalog evaluation reports only
known-positive recall/hit/judgement coverage, not unjudged-catalog NDCG.
No dataset or final-test file is opened by this module.
"""

import numpy as np
import pandas as pd
from ..schema import KEY, PAIR_KEY, check_labels

GRADES = {"E": 3, "S": 2, "C": 1, "I": 0}


def ordered(frame):
    if frame.duplicated(PAIR_KEY).any() or not np.isfinite(frame.score).all():
        raise ValueError("Unique prediction keys and finite scores required")
    return frame.sort_values(
        ["query_id", "score", *KEY], ascending=[True, False, True, True], kind="stable"
    )


def evaluate_pool(predictions, warm_labels, expected_candidates=None):
    try:
        check_labels(warm_labels, "validation")
    except ValueError as exc:
        raise ValueError("Only Warm Validation labels allowed") from exc
    ranking = ordered(predictions)
    identities = set(ranking[PAIR_KEY].itertuples(index=False, name=None))
    if expected_candidates is not None:
        expected = set(expected_candidates[PAIR_KEY].itertuples(index=False, name=None))
        if identities != expected:
            raise ValueError("Predictions differ from full frozen candidate identities")
    wanted = set(warm_labels[PAIR_KEY].itertuples(index=False, name=None))
    if not wanted <= identities:
        raise ValueError("Predictions missing labelled candidates")
    warm_ids = set(
        ranking.loc[ranking.product_split.eq("warm"), PAIR_KEY].itertuples(index=False, name=None)
    )
    if warm_ids != wanted:
        raise ValueError("Extra/missing Warm candidate identities")
    targets = warm_labels.groupby("query_id", sort=False)
    rows = []
    for qid, f in ranking.groupby("query_id", sort=False):
        if qid not in targets.groups:
            rows.append(
                dict(
                    query_id=qid,
                    warm_eligible=False,
                    warm_es_count=0,
                    warm_ndcg_at10=None,
                    warm_map_at10=None,
                    warm_mrr=None,
                )
            )
            continue
        lab = targets.get_group(qid)
        positive = lab.loc[lab.label.isin(["E", "S"])]
        gains = {
            (r.product_locale, r.product_id): 2 ** GRADES[r.label] - 1 for r in lab.itertuples()
        }
        pos = set(positive[KEY].itertuples(index=False, name=None))
        keys = list(f[KEY].itertuples(index=False, name=None))
        eligible = bool(pos)
        discounts = 1 / np.log2(np.arange(2, len(f) + 2))
        g = np.array([gains.get(k, 0) for k in keys], dtype=float)
        ideal = np.sort(np.array(list(gains.values())))[::-1][:10]
        idcg = float(ideal @ discounts[: len(ideal)])
        hits = np.array([k in pos for k in keys])
        hit10 = hits[:10]
        places = np.flatnonzero(hits)
        rows.append(
            dict(
                query_id=qid,
                warm_eligible=eligible,
                warm_es_count=len(pos),
                warm_ndcg_at10=float(g[:10] @ discounts[:10] / idcg) if eligible and idcg else None,
                warm_map_at10=float(
                    (np.cumsum(hit10) / np.arange(1, len(hit10) + 1) * hit10).sum()
                    / min(10, len(pos))
                )
                if eligible
                else None,
                warm_mrr=float(1 / (places[0] + 1))
                if eligible and len(places)
                else 0.0
                if eligible
                else None,
            )
        )
    perquery = pd.DataFrame(rows)
    eligible = perquery.loc[perquery.warm_eligible]
    summary = {
        "warm_ndcg@10": float(eligible.warm_ndcg_at10.mean()) if len(eligible) else None,
        "warm_map@10": float(eligible.warm_map_at10.mean()) if len(eligible) else None,
        "warm_mrr": float(eligible.warm_mrr.mean()) if len(eligible) else None,
        "eligible_queries": len(eligible),
        "total_queries": len(perquery),
        "scope": "Warm Validation E/S-eligible queries; all original candidate positions preserved; non-Warm relevance masked, not labelled I",
    }
    return (summary, perquery)


def known_positive_metrics(predictions, warm_labels, ks=(10, 50, 100)):
    check_labels(warm_labels, "validation")
    f = ordered(predictions)
    positive = warm_labels.loc[warm_labels.label.isin(["E", "S"])]
    positives = {
        q: set(g[KEY].itertuples(index=False, name=None)) for q, g in positive.groupby("query_id")
    }
    retrieved = {
        q: list(g[KEY].itertuples(index=False, name=None)) for q, g in f.groupby("query_id")
    }
    known = {
        q: set(g[KEY].itertuples(index=False, name=None))
        for q, g in warm_labels.groupby("query_id")
    }
    out = {
        "eligible_queries": len(positives),
        "known_positive_pairs": len(positive),
        "scope": "Full frozen catalog; only known Warm Validation E/S positives in denominator; unjudged products receive no relevance label",
    }
    for k in ks:
        recalls = []
        hits = []
        coverage = []
        for q, p in positives.items():
            top = retrieved.get(q, [])[:k]
            n = len(p & set(top))
            recalls.append(n / len(p))
            hits.append(bool(n))
            coverage.append(len(set(top) & known[q]) / len(top) if top else 0.0)
        out[f"known_positive_recall@{k}"] = float(np.mean(recalls)) if recalls else None
        out[f"known_positive_hit@{k}"] = float(np.mean(hits)) if hits else None
        out[f"warm_judged_fraction@{k}"] = float(np.mean(coverage)) if coverage else None
    return out


def scoped_metrics(pred, labels, scope, catalog=False, target_keys=None):
    if scope not in ["warm", "cold", "overall"]:
        raise ValueError("Unknown scope")
    if (
        labels.empty
        or not labels.query_split.eq("validation").all()
        or (not labels.product_split.isin(["warm", "cold_val"]).all())
    ):
        raise ValueError("Only Warm/Cold-Val Validation relevance permitted")
    if (
        not labels.grade.eq(labels.label.map({"E": 3, "S": 2, "C": 1, "I": 0})).all()
        or labels.duplicated(PAIR_KEY).any()
    ):
        raise ValueError("Invalid human grades")
    if "query_split" in pred and (not pred.query_split.eq("validation").all()):
        raise ValueError("Validation predictions only")
    if pred.duplicated(PAIR_KEY).any() or not np.isfinite(pred.score).all():
        raise ValueError("Invalid predictions")
    scoped = (
        labels
        if scope == "overall"
        else labels.loc[labels.product_split.eq("warm" if scope == "warm" else "cold_val")]
    )
    if target_keys is not None:
        scoped = scoped.loc[
            [k in target_keys for k in scoped[KEY].itertuples(index=False, name=None)]
        ]
    ordered = pred.sort_values(["query_id", "score", *KEY], ascending=[True, False, True, True])
    rg = {
        q: list(f[KEY].itertuples(index=False, name=None)) for q, f in ordered.groupby("query_id")
    }
    out = []
    if not catalog and (
        not set(labels[PAIR_KEY].itertuples(index=False, name=None))
        <= set(pred[PAIR_KEY].itertuples(index=False, name=None))
    ):
        raise ValueError("Incomplete fixed candidate pool")
    for q, lab in scoped.groupby("query_id"):
        pos = set(lab.loc[lab.label.isin(["E", "S"]), KEY].itertuples(index=False, name=None))
        if not pos:
            continue
        keys = rg.get(q, [])
        known = set(lab[KEY].itertuples(index=False, name=None))
        row = {"query_id": q, "positives": len(pos)}
        for k in [1, 10, 50, 100]:
            top = set(keys[:k])
            row[f"known_recall@{k}"] = len(pos & top) / len(pos)
            row[f"hit@{k}"] = float(bool(pos & top))
            row[f"coverage@{k}"] = len(known & top) / len(top) if top else 0.0
        if not catalog:
            gain = {
                (r.product_locale, r.product_id): 2 ** int(r.grade) - 1 for r in lab.itertuples()
            }
            top = keys[:10]
            g = [gain.get(k, 0) for k in top]
            ideal = sorted(gain.values(), reverse=True)[:10]
            disc = 1 / np.log2(np.arange(2, 12))
            row["ndcg@10"] = float(np.dot(g, disc[: len(g)]) / np.dot(ideal, disc[: len(ideal)]))
            hits = np.array([k in pos for k in top])
            row["map@10"] = float(
                np.sum(np.cumsum(hits) / np.arange(1, len(hits) + 1) * hits) / min(10, len(pos))
            )
            places = [i + 1 for i, k in enumerate(keys) if k in pos]
            row["mrr"] = 1 / places[0] if places else 0.0
        out.append(row)
    return (
        pd.DataFrame(out).set_index("query_id")
        if out
        else pd.DataFrame(index=pd.Index([], name="query_id"))
    )


def bootstrap(values, seed=20261015, replicates=1000):
    x = np.asarray(values, float)
    if not len(x):
        return [None, None]
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0, replicates, 100):
        means.extend(x[rng.integers(len(x), size=(min(100, replicates - start), len(x)))].mean(1))
    return np.quantile(means, [0.025, 0.975]).tolist()


def summary(per):
    out = {
        "effective_queries": len(per),
        "confidence_scope": "Query bootstrap, fixed selected model; validation evidence only",
        "ci95": {},
    }
    for c in per:
        if c == "positives":
            out["known_es_pairs"] = int(per[c].sum())
            continue
        out[c] = float(per[c].mean())
        out["ci95"][c] = bootstrap(per[c])
    return out


def paired(a, b):
    if set(a.index) != set(b.index):
        raise ValueError("Paired Query populations differ")
    out = {}
    for c in a:
        if c == "positives" or c not in b:
            continue
        delta = a[c] - b[c].reindex(a.index)
        out[c] = {"candidate_minus_baseline": float(delta.mean()), "ci95": bootstrap(delta)}
    return out
