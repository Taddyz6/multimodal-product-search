"""Exact cached-vector regression against external locked, unlabeled A2 arrays.

This is a correctness check, not a timing benchmark. No relevance labels are used.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from product_search.artifacts import load_pipeline
from product_search.schema import LABEL_FIELDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["config", "queries", "sbert-query", "clip-query", "reference", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    queries = pd.read_parquet(args.queries)
    if LABEL_FIELDS & set(queries):
        raise ValueError("Unlabeled query content only")
    if queries.query_id.duplicated().any():
        raise ValueError("Duplicate query ID")
    queries = queries.assign(query_id=queries.query_id.astype(str)).set_index("query_id")
    sb = np.load(args.sbert_query, mmap_mode="r", allow_pickle=False)
    cp = np.load(args.clip_query, mmap_mode="r", allow_pickle=False)
    files = sorted(args.reference.glob("query_*.npz"))[: args.count]
    if len(files) != args.count:
        raise ValueError("Insufficient locked reference records")
    if args.output.exists():
        raise FileExistsError("Choose a new regression output file")
    pipeline, limiter = load_pipeline(args.config)
    report = {"relevance_labels_read": False, "performance_benchmark": False, "queries": []}
    try:
        for file in files:
            qid = file.stem.removeprefix("query_")
            q = queries.loc[qid]
            if q.query_split != "validation":
                raise ValueError("Unlabeled Validation queries only")
            answer = pipeline.search(q["query"], sb[q.embedding_row], cp[q.embedding_row])
            with np.load(file, allow_pickle=False) as ref:
                fields = {k: bool(np.array_equal(np.asarray(answer[k]), ref[k])) for k in ref.files}
            report["queries"].append(
                {"query_id": qid, "exact": all(fields.values()), "fields": fields}
            )
            if not all(fields.values()):
                raise AssertionError("Core differs from the locked reference: " + qid)
        report["all_exact"] = True
    finally:
        limiter.restore_original_limits()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
