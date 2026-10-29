"""BM25/TF-IDF and lexical features shared by offline research and serving.

The vocabulary, retained-token lengths and sparse reduction order belong to the
frozen feature contract. Never refit a loaded artifact during online startup.
"""

from functools import lru_cache
from pathlib import Path
import pickle
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from .schema import KEY, LEXICAL_FEATURES as FEATURE_NAMES

TOKEN = re.compile(r"(?u)\b\w+\b")


def tokens(text):
    return TOKEN.findall(str(text).casefold())


def top_indices(scores, k):
    scores = np.asarray(scores)
    if scores.ndim != 1 or not np.isfinite(scores).all() or k < 1:
        raise ValueError("Finite 1D scores and positive k required")
    k = min(k, len(scores))
    if k == 0:
        return np.array([], dtype=int)
    threshold = np.partition(scores, len(scores) - k)[len(scores) - k]
    higher = np.flatnonzero(scores > threshold)
    tied = np.flatnonzero(scores == threshold)[: k - len(higher)]
    selected = np.concatenate([higher, tied])
    return selected[np.lexsort((selected, -scores[selected]))]


class SparseLexicalIndex:
    def __init__(self, max_features=150000, k1=1.5, b=0.75):
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("Invalid BM25 parameters")
        self.config = dict(max_features=max_features, k1=k1, b=b)

    def fit(self, products):
        if products.empty or products.duplicated(KEY).any():
            raise ValueError("Unique nonempty catalog required")
        p = products.sort_values(KEY).reset_index(drop=True)
        self.keys = list(p[KEY].itertuples(index=False, name=None))
        self.lookup = {k: i for i, k in enumerate(self.keys)}
        self.vectorizer = CountVectorizer(
            lowercase=True,
            token_pattern="(?u)\\b\\w+\\b",
            max_features=self.config["max_features"],
            dtype=np.float32,
        )
        counts = self.vectorizer.fit_transform(p.document.fillna("").astype(str)).tocsr()
        self.lengths = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
        self.avgdl = max(float(self.lengths.mean()), 1.0)
        n = len(p)
        df = np.asarray((counts > 0).sum(axis=0)).ravel()
        self.idf = np.log1p((n - df + 0.5) / (df + 0.5)).astype(np.float32)
        self.bm25 = counts.copy()
        row = np.repeat(np.arange(n), np.diff(counts.indptr))
        norm = self.config["k1"] * (
            1 - self.config["b"] + self.config["b"] * self.lengths[row] / self.avgdl
        )
        self.bm25.data = (
            self.idf[counts.indices] * counts.data * (self.config["k1"] + 1) / (counts.data + norm)
        )
        self.tfidf_transformer = TfidfTransformer(norm="l2", smooth_idf=True, sublinear_tf=False)
        self.tfidf = self.tfidf_transformer.fit_transform(counts).tocsr()
        self.title_tfidf = self.tfidf_transformer.transform(
            self.vectorizer.transform(p.product_title.fillna(""))
        ).tocsr()
        self.bm25 = self.bm25.tocsc()
        self.tfidf = self.tfidf.tocsc()
        self.title_tfidf = self.title_tfidf.tocsr()
        return self

    def scores(self, query, method):
        qc = self.vectorizer.transform([query])
        if method == "bm25":
            qc.data[:] = 1
            matrix = self.bm25
        elif method == "tfidf":
            qc = self.tfidf_transformer.transform(qc)
            matrix = self.tfidf
        else:
            raise ValueError("Method must be bm25 or tfidf")
        return np.asarray((qc @ matrix.T).toarray()).ravel().astype(np.float32)

    def features(self, pairs, queries, products, chunk_size=512, progress=None):
        pairs = pairs.reset_index(drop=True)
        if pairs.duplicated(["query_id", *KEY]).any():
            raise ValueError("Duplicate feature pairs")
        q = queries[["query_id", "query"]].drop_duplicates("query_id").reset_index(drop=True)
        qmap = {v: i for i, v in enumerate(q.query_id)}
        pi = np.array(
            [self.lookup[k] for k in pairs[KEY].itertuples(index=False, name=None)], dtype=int
        )
        qi = np.array([qmap[v] for v in pairs.query_id], dtype=int)
        qc = self.vectorizer.transform(q["query"].astype(str))
        qt = self.tfidf_transformer.transform(qc).tocsr()
        qm = qc.copy()
        qm.data[:] = 1
        bm = self.bm25.tocsr()
        dt = self.tfidf.tocsr()
        p = products.set_index(KEY).loc[self.keys].reset_index()
        fields = [
            "product_title",
            "document",
            "product_brand",
            "product_color",
            "product_bullet_point",
            "product_description",
        ]
        values = {c: p[c].fillna("").astype(str).to_numpy() for c in fields}

        @lru_cache(maxsize=4096)
        def sets(i):
            return tuple((set(tokens(values[c][i])) for c in fields))

        qsets = [set(tokens(v)) for v in q["query"]]
        out = np.zeros((len(pairs), len(FEATURE_NAMES)), np.float32)
        for a in range(0, len(pairs), chunk_size):
            z = min(a + chunk_size, len(pairs))
            px, qx = (pi[a:z], qi[a:z])
            out[a:z, 0] = np.asarray(bm[px].multiply(qm[qx]).sum(axis=1)).ravel()
            out[a:z, 1] = np.asarray(dt[px].multiply(qt[qx]).sum(axis=1)).ravel()
            out[a:z, 2] = np.asarray(self.title_tfidf[px].multiply(qt[qx]).sum(axis=1)).ravel()
            for j, (ip, iq) in enumerate(zip(px, qx), a):
                title, doc, brand, color, bullet, desc = sets(int(ip))
                qs = qsets[iq]
                denom = max(1, len(qs))
                digit = {t for t in qs if any((c.isdigit() for c in t))}
                phrase = " ".join(tokens(q.iloc[iq]["query"]))
                ts = " ".join(tokens(values["product_title"][ip]))
                bs = " ".join(tokens(values["product_brand"][ip]))
                out[j, 3:] = [
                    len(qs & doc) / denom,
                    len(qs & title) / denom,
                    len(qs & brand) / denom,
                    len(qs & color) / denom,
                    len(qs & bullet) / denom,
                    len(qs & desc) / denom,
                    len(digit & doc) / max(1, len(digit)),
                    bool(digit),
                    bool(phrase and " " + phrase + " " in " " + ts + " "),
                    bool(bs and " " + bs + " " in " " + phrase + " "),
                    np.log1p(len(qs)),
                    np.log1p(len(title)),
                    np.log1p(self.lengths[ip]),
                    not bool(desc),
                    not bool(bullet),
                    not bool(brand),
                    not bool(color),
                ]
            if progress and (z == len(pairs) or z % (chunk_size * 20) == 0):
                progress(f"Features {z:,}/{len(pairs):,}")
        if not np.isfinite(out).all():
            raise ValueError("Nonfinite features")
        return pd.DataFrame(out, columns=FEATURE_NAMES)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            pickle.dump(self, stream, protocol=5)


class _IndexUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Preserve frozen bytes without shipping the entire old project package.
        if (module, name) == ("project.esci_sqid.src.week2_lexical", "SparseLexicalIndex"):
            return SparseLexicalIndex
        return super().find_class(module, name)


def load_lexical_index(path):
    """Load an artifact whose SHA256 was checked by the asset loader."""
    with Path(path).open("rb") as stream:
        return _IndexUnpickler(stream).load()
