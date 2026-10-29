"""Persistent isolated SBERT/CLIP worker using the approved compatibility path.

NumPy1/Torch dependencies stay in the worker; the ranking core keeps its own
numeric environment. All checkpoint/tokenizer hashes are checked at startup.
"""

import json
import os
from pathlib import Path
import select
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
import numpy as np
from .artifacts import verify_encoder_assets


@dataclass(frozen=True)
class EncoderStatus:
    sbert_loaded: bool
    clip_loaded: bool
    identity_certified: bool
    reason: str | None


@dataclass(frozen=True)
class EncodedQuery:
    sbert: np.ndarray
    clip: np.ndarray
    timings: dict


def decode_vectors(values, count):
    arrays = []
    for key, dimension in [("sbert", 384), ("clip", 768)]:
        a = np.asarray(values[key], dtype=np.float32)
        if a.shape != (count, dimension) or not np.isfinite(a).all():
            raise ValueError("Invalid encoder vector shape/value")
        if np.any(np.abs(np.linalg.norm(a, axis=1) - 1) > 5e-06):
            raise ValueError("Nonunit encoder output")
        arrays.append(a)
    return tuple(arrays)


class JsonWorker:
    def __init__(self, command, *, env, log, timeout=120):
        self.timeout = timeout
        self.lock = threading.Lock()
        self.log = Path(log).open("a")
        self.buffer = bytearray()
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=env,
        )
        try:
            ready = self._read()
            if ready.get("ready") is not True:
                raise RuntimeError("Encoder worker not ready")
        except BaseException:
            self.close()
            raise

    @property
    def alive(self):
        return self.process.poll() is None

    def _read(self):
        deadline = time.monotonic() + self.timeout
        while True:
            end = self.buffer.find(b"\n")
            if end >= 0:
                if end > 2 * 1024**2:
                    raise RuntimeError("Encoder worker response too large")
                line = bytes(self.buffer[:end])
                del self.buffer[: end + 1]
                return json.loads(line.decode("utf-8"))
            if len(self.buffer) > 2 * 1024**2:
                raise RuntimeError("Encoder worker response too large")
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                self.close()
                raise RuntimeError("Encoder worker timed out")
            chunk = os.read(self.process.stdout.fileno(), 65536)
            if not chunk:
                raise RuntimeError("Encoder worker exited before complete response")
            self.buffer.extend(chunk)

    def request(self, body):
        with self.lock:
            if not self.alive:
                raise RuntimeError("Encoder worker exited")
            try:
                request_id = uuid.uuid4().hex
                self.process.stdin.write(
                    json.dumps({**body, "id": request_id}, ensure_ascii=False) + "\n"
                )
                self.process.stdin.flush()
                result = self._read()
                if result.get("id") != request_id or result.get("error"):
                    raise RuntimeError("Encoder response failed identity/error check")
                return result
            except BaseException:
                self.close()
                raise

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        for stream in [self.process.stdin, self.process.stdout, self.log]:
            if stream and (not stream.closed):
                stream.close()


class ProcessEncoder:
    def __init__(self, policy_file, output):
        self.policy = verify_encoder_assets(policy_file)
        p = self.policy
        env = dict(os.environ)
        env.update(
            PYTHONPATH=p["dependencies"],
            HF_HOME=p["hf_home"],
            TRANSFORMERS_OFFLINE="1",
            HF_HUB_OFFLINE="1",
            TOKENIZERS_PARALLELISM="false",
            OMP_NUM_THREADS="4",
            OPENBLAS_NUM_THREADS="4",
            MKL_NUM_THREADS="4",
            PYTHONNOUSERSITE="1",
        )
        self.worker = JsonWorker(
            [
                sys.executable,
                "-u",
                str(Path(__file__).with_name("encoder_worker.py")),
                str(policy_file),
            ],
            env=env,
            log=Path(output) / "encoder_worker.log",
        )
        # This approved checkpoint/preprocessing pair passed vector/result
        # compatibility checks. It does not establish SQID's historical revision.
        self.compatible = True

    @property
    def status(self):
        alive = self.worker.alive
        return EncoderStatus(alive, alive, False, None if alive else "Text encoder worker exited.")

    def encode_many(self, texts):
        try:
            result = self.worker.request({"texts": texts})
            sb, cp = decode_vectors(result["vectors"], len(texts))
            times = result["timings"]
            if any(
                (
                    not isinstance(v, (float, int)) or not np.isfinite(v) or v < 0
                    for v in times.values()
                )
            ):
                raise ValueError("Invalid encoder timings")
            return (sb, cp, times)
        except BaseException:
            self.close()
            raise

    def encode(self, text):
        sb, cp, timing = self.encode_many([text])
        return EncodedQuery(sb[0], cp[0], timing)

    def close(self):
        self.worker.close()
