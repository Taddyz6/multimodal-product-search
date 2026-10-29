"""Verify external assets before deserialization; never fit or replace them."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path


def sha256_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def resolve_paths(spec, base):
    """Relative paths are relative to the configuration file, not the process CWD."""

    def path(value):
        p = Path(os.path.expandvars(value)).expanduser()
        return str((base / p).resolve() if not p.is_absolute() else p.resolve())

    return {name: path(value) for name, value in spec.items()}


def verify_assets(expected, paths):
    if set(expected) != set(paths):
        raise ValueError("Asset roles differ from the frozen contract")
    for name, record in expected.items():
        if not Path(paths[name]).is_file():
            raise FileNotFoundError("Missing asset: " + name)
        if sha256_file(paths[name]) != record["sha256"]:
            raise ValueError("SHA256 mismatch: " + name)


def verify_encoder_assets(policy_file):
    p = Path(policy_file).resolve()
    policy = read_json(p)
    if policy.get("user_approved_compatibility") is not True:
        raise ValueError("An approved encoder compatibility policy is required")
    contract = read_json(Path(__file__).with_name("frozen_contract.json"))["encoders"]
    if any(policy.get(k) != v for k, v in contract.items()):
        raise ValueError("Encoder checkpoint/tokenizer contract differs")
    roots = resolve_paths(
        {k: policy[k] for k in ["clip_root", "sbert_root", "dependencies", "hf_home"]}, p.parent
    )
    policy.update(roots)
    for folder, entries in policy["checkpoint_files"].items():
        for filename, expected in entries.items():
            target = Path(roots[folder + "_root"]) / filename
            if sha256_file(target) != expected:
                raise ValueError("Encoder SHA256 mismatch: " + filename)
    return policy


def load_pipeline(config_file):
    """Load one core with the original four-thread numerical policy."""
    for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
        os.environ[name] = "4"
    import faiss
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    from threadpoolctl import threadpool_limits
    from .lexical import load_lexical_index
    from .pipeline import SearchPipeline
    from .schema import KEY, LABEL_FIELDS, FEATURES

    limiter = threadpool_limits(limits=4)
    faiss.omp_set_num_threads(4)
    try:
        file = Path(config_file).resolve()
        config = read_json(file)
        contract_path = Path(__file__).with_name("frozen_contract.json")
        contract = read_json(contract_path)
        for name, version in contract["numeric_environment"].items():
            if importlib.metadata.version(name) != version:
                raise ValueError("Pinned numerical dependency differs: " + name)
        paths = resolve_paths(config["assets"], file.parent)
        verify_assets(contract["assets"], paths)
        schema = pq.ParquetFile(paths["products"]).schema_arrow.names
        if LABEL_FIELDS & set(schema):
            raise ValueError("Product content must not contain relevance labels")
        products = pd.read_parquet(paths["products"]).sort_values(KEY).reset_index(drop=True)
        if len(products) != 164900:
            raise ValueError("Frozen catalog must contain 164,900 products")
        vectors = {}
        for name in ["sbert_product", "clip_product_text", "clip_product_image"]:
            value = np.load(paths[name], mmap_mode="r", allow_pickle=False)
            if (
                value.shape != (164900, 384 if name == "sbert_product" else 768)
                or value.dtype != np.float32
            ):
                raise ValueError("Frozen embedding shape/dtype mismatch: " + name)
            for start in range(0, len(value), 4096):
                if not np.isfinite(value[start : start + 4096]).all():
                    raise ValueError("Nonfinite catalog vectors")
            vectors[name] = value
        lexical = load_lexical_index(paths["lexical"])
        lexical.bm25 = lexical.bm25.tocsr()
        lexical.tfidf = lexical.tfidf.tocsr()
        index = faiss.read_index(paths["fusion_index"])
        if not isinstance(index, faiss.IndexFlatIP) or index.d != 1153 or index.ntotal != 164900:
            raise ValueError("Frozen Fusion FlatIP index differs")
        metadata = read_json(paths["listnet_json"])
        if metadata["kind"] != "listnet" or metadata["features"] != FEATURES:
            raise ValueError("Frozen ListNet feature order differs")
        with np.load(paths["listnet_npz"], allow_pickle=False) as archive:
            weights = {name: archive[name] for name in archive.files}
        expected = {
            "mean": (28,),
            "std": (28,),
            "w1": (64, 28),
            "b1": (64,),
            "w2": (32, 64),
            "b2": (32,),
            "w3": (1, 32),
            "b3": (1,),
        }
        if set(weights) != set(expected) or any(
            weights[k].shape != shape or not np.isfinite(weights[k]).all()
            for k, shape in expected.items()
        ):
            raise ValueError("Frozen ListNet weights differ")
        if np.any(weights["std"] <= 0):
            raise ValueError("Invalid frozen scaler")
        for name in ["mean", "std"]:
            if (
                hashlib.sha256(weights[name].tobytes()).hexdigest()
                != contract["ranker"][name + "_sha256"]
            ):
                raise ValueError("Frozen scaler identity differs")
        pipeline = SearchPipeline(
            products, lexical, vectors, read_json(paths["calibration"]), weights, index
        )
        return pipeline, limiter
    except BaseException:
        limiter.restore_original_limits()
        raise
