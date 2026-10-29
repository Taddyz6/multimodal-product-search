"""One production path: exact retrieval → Prepared28 → scaler → ListNet."""

import time
import numpy as np
import pandas as pd
from .features import FeatureBuilder
from .ranking import ListNetRanker, ordered_indices
from .retrieval import FusionRetriever
from .schema import KEY, LABEL_FIELDS, FEATURES


class ProductRepository:
    """Only actual display fields are returned; absent fields remain null."""

    def __init__(self, products):
        if LABEL_FIELDS & set(products):
            raise ValueError("Relevance labels cannot enter the product repository")
        self._products = {}
        frame = products.reindex(columns=[*KEY, "product_title", "image_url"])
        for locale, pid, title, image in frame.itertuples(index=False, name=None):
            key = (locale, pid)
            if any(not isinstance(value, str) for value in key) or key in self._products:
                raise ValueError("Invalid or duplicate product key")
            self._products[key] = {
                "product_locale": locale,
                "product_id": pid,
                "title": None if pd.isna(title) or title == "" else title,
                "image_url": None if pd.isna(image) or image == "" else image,
            }

    def get(self, key):
        return dict(self._products[key])


class SearchPipeline:
    def __init__(self, products, lexical, vectors, calibration, weights, index):
        self.retriever = FusionRetriever(index, products, lexical.keys)
        self.builder = FeatureBuilder(products, lexical, vectors, calibration)
        self.ranker = ListNetRanker(weights)
        self.products = ProductRepository(products)
        self.dimensions = (vectors["sbert_product"].shape[1], vectors["clip_product_text"].shape[1])

    def search(self, query, sbert, clip):
        for vector, dimension in zip([sbert, clip], self.dimensions):
            if (
                not isinstance(vector, np.ndarray)
                or vector.shape != (dimension,)
                or vector.dtype != np.float32
                or not np.isfinite(vector).all()
            ):
                raise ValueError("Query vectors violate the frozen dimensions/dtype")
        beginning = time.perf_counter()
        answer = self.retriever.search(sbert, clip)
        started = time.perf_counter()
        features = self.builder.build(query, answer["keys"], sbert, clip)
        feature_seconds = time.perf_counter() - started
        started = time.perf_counter()
        scaled = self.ranker.scale(features)
        scale_seconds = time.perf_counter() - started
        started = time.perf_counter()
        scores = self.ranker.forward(scaled)
        forward_seconds = time.perf_counter() - started
        started = time.perf_counter()
        order = ordered_indices(scores, answer["keys"])
        sorting_seconds = time.perf_counter() - started
        if (
            features.shape != (100, 28)
            or not np.isfinite(features).all()
            or not np.isfinite(scores).all()
        ):
            raise ValueError("Invalid Top100 feature or ranking output")
        # Retained arrays support exact engineering regression, not relevance input.
        return {
            **answer,
            "x": features,
            "scaled": scaled,
            "scores": scores,
            "order": order,
            "columns": FEATURES,
            "features_seconds": feature_seconds,
            "scaler_seconds": scale_seconds,
            "forward_seconds": forward_seconds,
            "sorting_seconds": sorting_seconds,
            "total_seconds": time.perf_counter() - beginning,
        }
