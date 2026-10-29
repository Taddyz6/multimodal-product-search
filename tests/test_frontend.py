"""Render the actual Streamlit script with an HTTP-client fixture, no model load."""

from pathlib import Path
from streamlit.testing.v1 import AppTest
import product_search.ui.client as client_module


class Client:
    ready = True

    def __init__(self, base):
        pass

    def health(self):
        return {"ready": self.ready, "components": {"core": self.ready}}

    def search(self, query, top_k):
        return {
            "query": query,
            "mode": "online_vector_compatible",
            "client_http_ms": 10.0,
            "timings_ms": {"service_total": 9.0},
            "results": [
                {
                    "product_id": "0001",
                    "product_locale": "us",
                    "title": "Fixture cable",
                    "image_url": None,
                    "rank": 1,
                    "score": 0.5,
                }
            ],
        }


def test_frontend_form_result_and_missing_image(monkeypatch, tmp_path):
    monkeypatch.setenv("SEARCH_IMAGE_CACHE", str(tmp_path))
    monkeypatch.setattr(client_module, "ApiClient", Client)
    app = Path(client_module.__file__).with_name("app.py")
    at = AppTest.from_file(str(app)).run()
    assert not at.exception
    at.text_input(key="search_query").set_value("cable")
    at.button(key="FormSubmitter:natural_search-Search").click().run()
    assert not at.exception
    content = " ".join(x.value for x in at.markdown)
    assert "Fixture cable" in content and "0001" in content and "No image available" in content


def test_frontend_unavailable_state(monkeypatch, tmp_path):
    class Unavailable(Client):
        ready = False

    monkeypatch.setenv("SEARCH_API_URL", "http://unavailable.invalid")
    monkeypatch.setenv("SEARCH_IMAGE_CACHE", str(tmp_path))
    monkeypatch.setattr(client_module, "ApiClient", Unavailable)
    at = AppTest.from_file(str(Path(client_module.__file__).with_name("app.py"))).run()
    assert not at.exception
    assert at.text_input(key="search_query").disabled
    assert any("unavailable" in x.value.lower() for x in at.warning)
