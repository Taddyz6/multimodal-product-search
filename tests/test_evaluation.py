"""Scope masks retain competitors and catalog recall never invents I labels."""

import numpy as np
import pandas as pd
import pytest
from product_search.research.evaluation import (
    evaluate_pool,
    known_positive_metrics,
    scoped_metrics,
    bootstrap,
)
from product_search.research.ranking import WarmMetric
from product_search.schema import check_labels


def fixture():
    pred = pd.DataFrame(
        [
            dict(
                query_id="q",
                product_locale="us",
                product_id="cold",
                product_split="cold_val",
                score=2.0,
            ),
            dict(
                query_id="q",
                product_locale="us",
                product_id="warm",
                product_split="warm",
                score=1.0,
            ),
        ]
    )
    labels = (
        pred.loc[pred.product_split.eq("warm")]
        .drop(columns="score")
        .assign(query_split="validation", label="E", grade=3)
    )
    return pred, labels


def test_mask_preserves_competitor_rank():
    pred, labels = fixture()
    metrics, per = evaluate_pool(pred, labels, pred)
    assert metrics["warm_ndcg@10"] == pytest.approx(1 / np.log2(3))
    assert metrics["warm_mrr"] == 0.5
    assert WarmMetric(pred, labels)(pred.score.to_numpy()) == pytest.approx(metrics["warm_ndcg@10"])
    cold = pred.iloc[[0]].drop(columns="score").assign(query_split="validation", label="E", grade=3)
    both = pd.concat([labels, cold])
    assert scoped_metrics(pred, both, "cold").iloc[0]["ndcg@10"] == 1


def test_unknown_catalog_results_are_not_negative_labels():
    pred, labels = fixture()
    unknown = pred.iloc[[0]].assign(product_id="unjudged")
    recall = known_positive_metrics(unknown, labels, ks=(10,))
    assert recall["known_positive_recall@10"] == 0
    assert recall["warm_judged_fraction@10"] == 0
    assert "ndcg@10" not in recall
    with pytest.raises(ValueError):
        evaluate_pool(unknown, labels)


def test_training_scope_and_bootstrap():
    _, labels = fixture()
    with pytest.raises(ValueError):
        check_labels(labels, "train")
    bad = labels.assign(query_split="train", product_split="cold_val")
    with pytest.raises(ValueError):
        check_labels(bad, "train")
    assert bootstrap([0.25] * 4) == [0.25, 0.25]
