"""Natural-language search over one frozen core and one persistent encoder."""

from pathlib import Path
import threading
import time

from .errors import QueryProcessor, Unavailable, Busy, EncodeFailure


class SearchService:
    def __init__(self, pipeline, encoder, limiter=None):
        self.pipeline = pipeline
        self.encoder = encoder
        self.limiter = limiter
        self.processor = QueryProcessor()
        self._slot = threading.Lock()

    def status(self):
        state = self.encoder.status
        core = self.pipeline is not None
        ready = core and state.sbert_loaded and state.clip_loaded and self.encoder.compatible
        return {
            "ready": bool(ready),
            "components": {
                "core": core,
                "index": core,
                "ranker": core,
                "sbert": state.sbert_loaded,
                "clip": state.clip_loaded,
                "encoder_compatibility_verified": self.encoder.compatible,
                "historical_sqid_identity_verified": state.identity_certified,
            },
            "reason": None if ready else state.reason or "Required search assets are unavailable.",
            "candidate_budget": 100,
            "product_cache": self.pipeline.builder.cache_info()._asdict() if core else None,
        }

    def search(self, query, top_k=10):
        query = self.processor.validate(query, top_k)
        if not self.status()["ready"]:
            raise Unavailable(self.status()["reason"])
        if not self._slot.acquire(blocking=False):
            raise Busy()
        began = time.perf_counter()
        try:
            try:
                encoded = self.encoder.encode(query)
            except Exception as exc:
                raise EncodeFailure("The query encoders could not complete this search.") from exc
            encoding_seconds = time.perf_counter() - began
            answer = self.pipeline.search(query, encoded.sbert, encoded.clip)
            results = []
            for rank, position in enumerate(answer["order"][:top_k], 1):
                item = self.pipeline.products.get(answer["keys"][position])
                item.update(rank=rank, score=float(answer["scores"][position]))
                results.append(item)
            times = {
                k.removesuffix("_seconds"): float(v) * 1000
                for k, v in answer.items()
                if k.endswith("_seconds")
            }
            times.update(
                {k.removesuffix("_seconds"): float(v) * 1000 for k, v in encoded.timings.items()}
            )
            times.update(
                encoding=encoding_seconds * 1000, service_total=(time.perf_counter() - began) * 1000
            )
            return dict(
                query=query,
                top_k=top_k,
                candidate_budget=100,
                mode="online_vector_compatible",
                timings_ms=times,
                results=results,
            )
        finally:
            self._slot.release()

    def close(self):
        try:
            self.encoder.close()
        finally:
            self.pipeline = None
            if self.limiter is not None:
                self.limiter.restore_original_limits()
                self.limiter = None


def load_service(config_file):
    from .artifacts import load_pipeline, read_json, resolve_paths
    from .encoders import ProcessEncoder

    file = Path(config_file).resolve()
    config = read_json(file)
    paths = resolve_paths({k: config[k] for k in ["encoder_policy", "runtime_dir"]}, file.parent)
    Path(paths["runtime_dir"]).mkdir(parents=True, exist_ok=True)
    pipeline, limiter = load_pipeline(file)
    try:
        encoder = ProcessEncoder(paths["encoder_policy"], paths["runtime_dir"])
        return SearchService(pipeline, encoder, limiter)
    except BaseException:
        limiter.restore_original_limits()
        raise
