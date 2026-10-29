"""The HTTP boundary is tested without loading data or encoders."""

import pytest
from fastapi.testclient import TestClient
from product_search.api import create_app


class Service:
    closed = False

    def status(self):
        return {"ready": True, "components": {"core": True}}

    def close(self):
        self.closed = True

    def search(self, query, top_k):
        return dict(
            query=query,
            top_k=top_k,
            candidate_budget=100,
            mode="online_vector_compatible",
            timings_ms={"service_total": 1.0},
            results=[
                dict(
                    product_id="0001",
                    product_locale="us",
                    title=None,
                    image_url=None,
                    score=0.5,
                    rank=1,
                )
            ],
        )


def test_search_lifecycle_and_public_routes():
    service = Service()
    with TestClient(create_app(lambda: service)) as client:
        assert client.get("/ready").status_code == 200
        r = client.post("/search", json={"query": " cable 2m ", "top_k": 1})
        assert r.status_code == 200 and r.json()["query"] == " cable 2m "
        assert r.json()["results"][0]["product_id"] == "0001"
        assert r.headers["X-Request-Coalesced"] == "false"
        assert client.get("/internal/replay/queries").status_code == 404
    assert service.closed


@pytest.mark.parametrize(
    "body",
    [
        {"query": ""},
        {"query": "   "},
        {"query": "a", "top_k": True},
        {"query": "a", "top_k": 101},
        {"query": "a", "extra": "x"},
        {"query": 1},
        {"query": "a\u0000"},
    ],
)
def test_invalid_inputs(body):
    with TestClient(create_app(Service)) as client:
        assert client.post("/search", json=body).status_code == 422


def test_missing_assets_does_not_report_ready():
    def broken():
        raise FileNotFoundError("external asset missing")

    with TestClient(create_app(broken)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        assert client.post("/search", json={"query": "headphones"}).status_code == 503
