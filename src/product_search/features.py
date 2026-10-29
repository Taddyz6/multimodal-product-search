"""Exact 28-feature construction with a bounded product-token cache.

Sparse multiplication/sum, einsum and float64 calibration retain the original
conversion points. Equivalent-looking dot products can change rounding.
"""

from functools import lru_cache
import numpy as np
from .schema import KEY, LABEL_FIELDS
from .lexical import tokens
from .calibration import calibrated_clip_scores


class FeatureBuilder:
    def __init__(self, products, lexical, vectors, params, token_cache_size=8192):
        if LABEL_FIELDS & set(products):
            raise ValueError("No relevance input allowed")
        if products.duplicated(KEY).any():
            raise ValueError("Duplicate product keys")
        self.keys = list(lexical.keys)
        self.lookup = {k: i for i, k in enumerate(self.keys)}
        if len(self.lookup) != len(products):
            raise ValueError("Lexical/catalog keys differ")
        p = products.set_index(KEY).loc[self.keys]
        self.lexical = lexical
        self.bm = lexical.bm25.tocsr()
        self.dt = lexical.tfidf.tocsr()
        self.tt = lexical.title_tfidf.tocsr()
        self.lengths = np.asarray(lexical.lengths)
        fields = [
            "product_title",
            "document",
            "product_brand",
            "product_color",
            "product_bullet_point",
            "product_description",
        ]
        self.values = {n: p[n].fillna("").astype(str).to_numpy() for n in fields}
        self.embedding_rows = p.embedding_row.to_numpy(dtype=int)
        self.mask = p.has_image_embedding.to_numpy(dtype=bool)
        self.vectors = vectors
        self.params = params

        @lru_cache(maxsize=int(token_cache_size))
        def prepared(row):
            lists = [tokens(self.values[n][row]) for n in fields]
            return (
                tuple((frozenset(x) for x in lists)),
                " ".join(lists[0]),
                " ".join(lists[2]),
                np.log1p(len(lists[2])),
                np.log1p(len(lists[3])),
            )

        self._prepared = prepared

    def cache_info(self):
        return self._prepared.cache_info()

    def rows(self, keys):
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate candidates")
        return np.fromiter((self.lookup[k] for k in keys), dtype=np.int64, count=len(keys))

    def _lexical(self, query, rows):
        if not isinstance(query, str):
            raise ValueError("Query must be text")
        n = len(rows)
        out = np.zeros((n, 20), np.float32)
        if not n:
            return out
        qc = self.lexical.vectorizer.transform([query])
        qt = self.lexical.tfidf_transformer.transform(qc).tocsr()
        qm = qc.copy()
        qm.data[:] = 1
        repeated = np.zeros(n, dtype=int)
        out[:, 0] = np.asarray(self.bm[rows].multiply(qm[repeated]).sum(axis=1)).ravel()
        out[:, 1] = np.asarray(self.dt[rows].multiply(qt[repeated]).sum(axis=1)).ravel()
        out[:, 2] = np.asarray(self.tt[rows].multiply(qt[repeated]).sum(axis=1)).ravel()
        qtokens = tokens(query)
        qs = set(qtokens)
        denom = max(1, len(qs))
        phrase = " ".join(qtokens)
        digit = {t for t in qs if any((c.isdigit() for c in t))}
        dd = max(1, len(digit))
        qlength = np.log1p(len(qs))
        for j, row in enumerate(rows):
            fields, title_phrase, brand_phrase, _, _ = self._prepared(int(row))
            title, doc, brand, color, bullet, desc = fields
            out[j, 3:] = [
                len(qs & doc) / denom,
                len(qs & title) / denom,
                len(qs & brand) / denom,
                len(qs & color) / denom,
                len(qs & bullet) / denom,
                len(qs & desc) / denom,
                len(digit & doc) / dd,
                bool(digit),
                bool(phrase and " " + phrase + " " in " " + title_phrase + " "),
                bool(brand_phrase and " " + brand_phrase + " " in " " + phrase + " "),
                qlength,
                np.log1p(len(title)),
                np.log1p(self.lengths[row]),
                not bool(desc),
                not bool(bullet),
                not bool(brand),
                not bool(color),
            ]
        return out

    def append_semantic(self, lexical20, keys, sbert, text, image):
        """Append normalized-vector similarities, mask, calibrated fusion and lengths."""
        rows = self.rows(keys)
        mask = self.mask[rows]
        s = np.asarray(sbert, np.float32)
        t = np.asarray(text, np.float32)
        raw = np.asarray(image, np.float32).copy()
        raw[~mask] = 0
        if any((x.shape != (len(rows),) or not np.isfinite(x).all() for x in [s, t, raw])):
            raise ValueError("Invalid branch scores")
        content = [self._prepared(int(i)) for i in rows]
        extra = np.column_stack(
            [
                s,
                t,
                raw,
                mask.astype(np.float32),
                0.5 * s + 0.5 * t,
                0.5 * s + 0.5 * calibrated_clip_scores(t, raw, mask, self.params, 0.5, True),
                [x[3] for x in content],
                [x[4] for x in content],
            ]
        )
        out = np.column_stack([np.asarray(lexical20, np.float32), extra]).astype(np.float32)
        if out.shape != (len(rows), 28) or not np.isfinite(out).all():
            raise ValueError("Invalid feature matrix")
        return out

    def build(self, query, keys, sbert_query, clip_query):
        rows = self.rows(keys)
        pr = self.embedding_rows[rows]
        sq, cq = (np.asarray(sbert_query, np.float32), np.asarray(clip_query, np.float32))
        v = self.vectors
        if sq.shape != (v["sbert_product"].shape[1],) or cq.shape != (
            v["clip_product_text"].shape[1],
        ):
            raise ValueError("Query vector dimensions differ")
        if not np.isfinite(sq).all() or not np.isfinite(cq).all() or np.linalg.norm(sq) == 0:
            raise ValueError("Invalid query vectors")
        sq = np.repeat(sq[None, :], len(rows), axis=0)
        cq = np.repeat(cq[None, :], len(rows), axis=0)
        s = np.einsum("ij,ij->i", v["sbert_product"][pr], sq)
        t = np.einsum("ij,ij->i", v["clip_product_text"][pr], cq)
        im = np.einsum("ij,ij->i", v["clip_product_image"][pr], cq)
        return self.append_semantic(self._lexical(query, rows), keys, s, t, im)
