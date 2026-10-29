"""Exact FlatIP retrieval with immutable row-aligned product keys."""

import time
import numpy as np
from .schema import KEY, LABEL_FIELDS


def search_flat(index, queries, k):
    """Canonical ID tie break including ties crossing the Top-K boundary."""
    q = np.ascontiguousarray(queries, np.float32)
    if q.ndim != 2 or q.shape[1] != index.d or k < 1 or (index.ntotal < 1):
        raise ValueError("Invalid Flat search")
    k = min(k, index.ntotal)
    take = min(k + 1, index.ntotal)
    scores, ids = index.search(q, take)
    out_i = np.empty((len(q), k), np.int64)
    out_d = np.empty((len(q), k), np.float32)
    for row in range(len(q)):
        d, i = (scores[row], ids[row])
        if take > k and d[k - 1] == d[k]:
            radius = float(np.nextafter(np.float32(d[k - 1]), np.float32(-np.inf)))
            _, rd, ri = index.range_search(q[row : row + 1], radius)
            if len(ri) >= k:
                d, i = (rd, ri)
        order = np.lexsort((i, -d))[:k]
        out_i[row] = i[order]
        out_d[row] = d[order]
    return (out_d, out_i)


class FusionRetriever:
    """Map exact FAISS rows to original joint keys without Pandas per request."""

    def __init__(self, index, products, lexical_keys):
        if LABEL_FIELDS & set(products):
            raise ValueError("Relevance inputs are not part of retrieval")
        if len(products) != index.ntotal:
            raise ValueError("Index and catalog populations differ")
        keys = list(products[KEY].itertuples(index=False, name=None))
        if keys != list(lexical_keys) or keys != sorted(keys) or len(set(keys)) != len(keys):
            raise ValueError("Canonical catalog and lexical order must agree")
        if any(not isinstance(value, str) for key in keys for value in key):
            raise ValueError("Product keys must remain original strings")
        if not np.array_equal(products.embedding_row, np.arange(len(products))):
            raise ValueError("Catalog rows must align with embedding rows")
        self.index = index
        # Object arrays preserve leading zeros, Unicode and long IDs verbatim.
        self.keys = np.asarray(keys, dtype=object)
        self.keys.setflags(write=False)

    def search(self, sbert, clip):
        started = time.perf_counter()
        query = np.concatenate([sbert, clip, [1.0]]).astype(np.float32)[None, :]
        raw_scores, raw_rows = search_flat(self.index, query, 100)
        raw_scores, raw_rows = raw_scores[0], raw_rows[0]
        if len(raw_rows) != 100 or len(set(raw_rows)) != 100 or np.any(raw_rows < 0):
            raise ValueError("Retrieval must return exactly100 distinct catalog rows")
        selected = self.keys[raw_rows]
        order = np.lexsort((selected[:, 1], selected[:, 0], -raw_scores))
        return {
            "keys": [tuple(key) for key in selected[order].tolist()],
            "candidate_rows": raw_rows[order],
            "retrieval_scores": raw_scores[order],
            "raw_faiss_rows": raw_rows,
            "raw_faiss_scores": raw_scores,
            "retrieval_seconds": time.perf_counter() - started,
        }
