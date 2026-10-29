"""Shared identities and the frozen feature order; no data or model imports."""

KEY = ["product_locale", "product_id"]
PAIR_KEY = ["query_id", *KEY]
LABEL_FIELDS = {"label", "esci_label", "grade", "gain", "is_positive"}
LEXICAL_FEATURES = [
    "bm25",
    "tfidf",
    "title_tfidf",
    "query_coverage",
    "title_coverage",
    "brand_overlap",
    "color_overlap",
    "bullet_coverage",
    "description_coverage",
    "digit_coverage",
    "has_digits",
    "title_phrase_match",
    "brand_phrase_match",
    "query_length",
    "title_length",
    "document_length",
    "missing_description",
    "missing_bullet",
    "missing_brand",
    "missing_color",
]
FEATURES = LEXICAL_FEATURES + [
    "sbert_similarity",
    "clip_text_similarity",
    "clip_image_similarity",
    "image_available",
    "calibrated_text_fusion_score",
    "calibrated_fusion_score",
    "brand_length",
    "color_length",
]


def check_labels(frame, query_split):
    """Offline supervision is restricted to the declared Query/Warm role."""
    if (
        frame.empty
        or not frame.query_split.eq(query_split).all()
        or not frame.product_split.eq("warm").all()
    ):
        raise ValueError(f"Only {query_split}/Warm labels allowed")
    if not set(frame.label) <= {"E", "S", "C", "I"} or frame.duplicated(PAIR_KEY).any():
        raise ValueError("Invalid human labels or duplicate pair keys")
    if (
        "grade" in frame
        and not frame.grade.eq(frame.label.map({"E": 3, "S": 2, "C": 1, "I": 0})).all()
    ):
        raise ValueError("Human label/grade mismatch")
