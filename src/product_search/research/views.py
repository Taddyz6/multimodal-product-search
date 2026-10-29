"""Frozen LTR eligibility and dependency-aware feature group deletion."""

import numpy as np
from ..schema import FEATURES, LEXICAL_FEATURES as FEATURE_NAMES, PAIR_KEY, check_labels

DEPENDENCIES = {n: ["Lexical"] if i < 14 else ["Metadata"] for i, n in enumerate(FEATURE_NAMES)}

DEPENDENCIES.update(
    sbert_similarity=["Semantic"],
    clip_text_similarity=["Semantic"],
    clip_image_similarity=["Image"],
    image_available=["Image"],
    calibrated_text_fusion_score=["Semantic"],
    calibrated_fusion_score=["Semantic", "Image"],
    brand_length=["Metadata"],
    color_length=["Metadata"],
)

for _name in ["brand_overlap", "color_overlap", "brand_phrase_match"]:
    DEPENDENCIES[_name] = ["Metadata"]


def feature_columns(remove=None):
    if remove not in [None, "Lexical", "Semantic", "Image", "Metadata"]:
        raise ValueError("Unknown feature group")
    return [n for n in FEATURES if remove not in DEPENDENCIES[n]]


def training_view(labels, eligibility, expected=(6171, 100856)):
    check_labels(labels, "train")
    if eligibility.query_id.duplicated().any():
        raise ValueError("Duplicate frozen eligibility")
    ids = set(eligibility.loc[eligibility.ltr_eligible, "query_id"])
    selected = labels.loc[labels.query_id.isin(ids)].sort_values(PAIR_KEY).reset_index(drop=True)
    if (
        selected.query_id.nunique() != expected[0]
        or len(selected) != expected[1]
        or set(selected.query_id) != ids
    ):
        raise ValueError("Frozen LTR counts/identities mismatch")
    if "candidates" in eligibility:
        counts = selected.groupby("query_id").size()
        wanted = eligibility.set_index("query_id").loc[list(counts.index), "candidates"]
        if not np.array_equal(counts, wanted):
            raise ValueError("Incomplete frozen LTR Query lists")
    return selected
