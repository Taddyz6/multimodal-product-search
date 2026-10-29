"""HTTP-only frontend client; it does not import inference modules."""

import time
import httpx


class ApiError(RuntimeError):
    pass


class ApiClient:
    def __init__(self, base_url, client=None):
        self.base = base_url.rstrip("/")
        self.client = client or httpx.Client(timeout=httpx.Timeout(30.0, connect=5.0))
        self.owns = client is None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.owns:
            self.client.close()

    def _call(self, method, path, body=None):
        start = time.perf_counter()
        try:
            response = self.client.request(method, self.base + path, json=body)
            response.raise_for_status()
            result = response.json()
        except httpx.HTTPStatusError as exc:
            try:
                error = exc.response.json()
                message = error.get("error", {}).get(
                    "message", error.get("detail", "API request rejected")
                )
            except ValueError:
                message = "API request rejected"
            raise ApiError(f"{exc.response.status_code}: {message}") from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise ApiError(
                "The search service could not be reached or returned an invalid response."
            ) from exc
        if path.endswith("/search"):
            result["client_http_ms"] = (time.perf_counter() - start) * 1000

        return result

    def health(self):
        return self._call("GET", "/health")

    def search(self, query, top_k):
        return self._call("POST", "/search", {"query": query, "top_k": top_k})
