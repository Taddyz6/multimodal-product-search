"""Small display-image cache. No model imports or relevance inputs."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
from pathlib import Path
import ssl
import threading
import time
from urllib.parse import urlsplit
import urllib.request
import uuid

ALLOWED_HOSTS = {
    "m.media-amazon.com",
    "images-na.ssl-images-amazon.com",
    "images-eu.ssl-images-amazon.com",
    "images-fe.ssl-images-amazon.com",
    "images.amazon.com",
}
MAX_BYTES = 8 * 1024**2


def tls_context():
    import certifi

    return ssl.create_default_context(cafile=certifi.where())


def image_key(url):
    if not isinstance(url, str) or len(url) > 4096:
        raise ValueError("Invalid image URL")
    u = urlsplit(url)
    if (
        u.scheme != "https"
        or u.hostname not in ALLOWED_HOSTS
        or u.username
        or u.password
        or u.port
        or u.fragment
    ):
        raise ValueError("Unregistered image host")
    return hashlib.sha256(url.encode()).hexdigest()


def store_image(root, url, data):
    from PIL import Image, ImageOps

    key = image_key(url)
    if not data or len(data) > MAX_BYTES:
        raise ValueError("Image size refused")
    try:
        with Image.open(io.BytesIO(data)) as source:
            if source.width * source.height > 30_000_000:
                raise ValueError("Oversized decoded image")
            im = ImageOps.exif_transpose(source).convert("RGB")
            im.thumbnail((480, 480))
    except Exception as exc:
        raise ValueError("Invalid image bytes") from exc
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / (key + ".jpg")
    temporary = root / (key + "." + uuid.uuid4().hex + ".tmp")
    im.save(temporary, format="JPEG", quality=86)
    temporary.replace(target)
    return target


def image_display(cache, url):
    path = cache.path(url) if url else None
    if not path and url:
        cache.submit(url)
        path = cache.path(url)
    if path:
        return path, "ready"
    state = cache.state(url)
    return None, "loading" if state == "ready" else state


class ImageCache:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="product-images")
        self.lock = threading.RLock()
        self.pending = {}
        self.failed = {}

    def path(self, url):
        try:
            p = self.root / (image_key(url) + ".jpg")
        except ValueError:
            return None
        return p if p.is_file() else None

    def submit(self, url):
        try:
            key = image_key(url)
        except ValueError:
            return
        with self.lock:
            if (
                self.path(url)
                or key in self.pending
                or time.monotonic() - self.failed.get(key, -1e9) < 600
            ):
                return
            if len(self.pending) >= 100:
                return
            future = self.pool.submit(self._download, url)
            self.pending[key] = future

            def done(f):
                with self.lock:
                    self.pending.pop(key, None)
                    if f.exception() is not None:
                        self.failed[key] = time.monotonic()

            future.add_done_callback(done)

    def _download(self, url):
        image_key(url)

        class Redirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                image_key(newurl)
                return super().redirect_request(req, fp, code, msg, headers, newurl)

        opener = urllib.request.build_opener(
            Redirect(), urllib.request.HTTPSHandler(context=tls_context())
        )
        with opener.open(
            urllib.request.Request(url, headers={"User-Agent": "ProductSearchDemo/1.0"}), timeout=5
        ) as r:
            data = r.read(MAX_BYTES + 1)
        return store_image(self.root, url, data)

    def state(self, url):
        if not url:
            return "missing"
        if self.path(url):
            return "ready"
        try:
            key = image_key(url)
        except ValueError:
            return "unavailable"
        with self.lock:
            return "unavailable" if key in self.failed else "loading"
