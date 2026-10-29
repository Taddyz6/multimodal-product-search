"""Small deterministic regressions; synthetic fixtures are not latency evidence."""

import numpy as np
import pandas as pd
from product_search.lexical import SparseLexicalIndex
from product_search.features import FeatureBuilder
from product_search.schema import FEATURES
from product_search.ranking import ListNetRanker, ordered_indices
from product_search.retrieval import FusionRetriever


def test_original_scaling_precision_and_forward():
    rng = np.random.default_rng(41)
    weights = {"mean": rng.normal(size=28), "std": rng.uniform(0.2, 2, size=28)}
    for j, shape in [(1, (64, 28)), (2, (32, 64)), (3, (1, 32))]:
        weights[f"w{j}"] = rng.normal(size=shape).astype(np.float32)
        weights[f"b{j}"] = rng.normal(size=shape[0]).astype(np.float32)
    x = rng.normal(size=(100, 28)).astype(np.float32)
    expected = np.asarray((x - weights["mean"]) / weights["std"], np.float32)
    ranker = ListNetRanker(weights)
    np.testing.assert_array_equal(ranker.scale(x), expected)
    for j in [1, 2]:
        expected = np.maximum(expected @ weights[f"w{j}"].T + weights[f"b{j}"], 0)
    expected = (expected @ weights["w3"].T + weights["b3"]).reshape(-1)
    np.testing.assert_array_equal(ranker.forward(ranker.scale(x)), expected)


def test_joint_key_ties_preserve_strings():
    keys = [("us", "0002"), ("us", "0001"), ("uk", "Ω")]
    assert ordered_indices(np.ones(3, np.float32), keys).tolist() == [2, 1, 0]


def test_sparse_lexical_features_keep_original_definitions():
    p = pd.DataFrame(
        [
            dict(
                product_locale="us",
                product_id="0001",
                product_title="USB Cable 2M",
                document="USB Cable 2M BrandA",
                product_brand="BrandA",
                product_color="",
                product_bullet_point="",
                product_description="",
                embedding_row=0,
                has_image_embedding=False,
            )
        ]
    )
    index = SparseLexicalIndex().fit(p)
    vectors = {
        "sbert_product": np.array([[1.0, 0]], np.float32),
        "clip_product_text": np.array([[1.0, 0]], np.float32),
        "clip_product_image": np.array([[0.0, 0]], np.float32),
    }
    params = {
        "text": {"mean": 0.2, "std": 0.1},
        "image": {"mean": 0.1, "std": 0.05},
        "missing_text": {"mean": 0.3, "std": 0.2},
        "correlation": 0.0,
    }
    builder = FeatureBuilder(p, index, vectors, params)
    x = builder.build(
        "USB Cable 2M", [("us", "0001")], np.array([1, 0], np.float32), np.array([1, 0], np.float32)
    )
    assert x.shape == (1, 28) and x.dtype == np.float32
    assert x[0, 3] == 1 and x[0, 9] == 1 and x[0, 10] == 1 and x[0, 11] == 1
    assert x[0, 16] == x[0, 17] == x[0, 19] == 1
    assert x[0, 22] == 0 and x[0, 23] == 0
    assert len(FEATURES) == 28


def test_flat_boundary_tie_does_not_drop_canonical_item():
    import faiss

    p = pd.DataFrame(
        {
            "product_locale": ["us"] * 105,
            "product_id": [f"{i:04}" for i in range(105)],
            "embedding_row": range(105),
        }
    )
    index = faiss.IndexFlatIP(5)
    index.add(np.ones((105, 5), np.float32))
    r = FusionRetriever(
        index, p, list(p[["product_locale", "product_id"]].itertuples(index=False, name=None))
    )
    result = r.search(np.ones(2, np.float32), np.ones(2, np.float32))
    assert result["candidate_rows"].tolist() == list(range(100))
    assert result["keys"][0] == ("us", "0000")


def test_legacy_pickle_class_is_mapped_without_legacy_package(tmp_path):
    from product_search.lexical import load_lexical_index

    path = tmp_path / "index.pkl"
    # Protocol0 GLOBAL names the original class but stores no model or dataset.
    path.write_bytes(b"cproject.esci_sqid.src.week2_lexical\nSparseLexicalIndex\n(tR.")
    assert isinstance(load_lexical_index(path), SparseLexicalIndex)
