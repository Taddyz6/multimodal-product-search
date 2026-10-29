import numpy as np
import pytest
from product_search.artifacts import resolve_paths, verify_assets, sha256_file
from product_search.encoders import decode_vectors
from product_search.ui.presentation import title_markup, results_markup, validate_query
from product_search.ui.images import image_key


def test_hash_and_relative_path_contract(tmp_path):
    asset = tmp_path / "a"
    asset.write_bytes(b"content")
    paths = resolve_paths({"a": "a"}, tmp_path)
    assert paths["a"] == str(asset)
    expected = {"a": {"sha256": sha256_file(asset)}}
    verify_assets(expected, paths)
    asset.write_bytes(b"changed")
    with pytest.raises(ValueError):
        verify_assets(expected, paths)


def test_encoder_vectors_shape_norm_and_nonfinite():
    values = {"sbert": np.eye(1, 384).tolist(), "clip": np.eye(1, 768).tolist()}
    sb, cp = decode_vectors(values, 1)
    assert sb.dtype == cp.dtype == np.float32
    values["clip"][0][0] = 2
    with pytest.raises(ValueError):
        decode_vectors(values, 1)
    with pytest.raises(ValueError):
        decode_vectors({"sbert": [], "clip": []}, 1)


def test_ui_escapes_content_and_validates_query():
    assert "<script>" not in title_markup("<script>")
    result = {"query": "<script>", "results": [], "client_http_ms": 10}
    assert "<script>" not in results_markup(result)
    assert validate_query("   ") and validate_query("camera", 101)
    assert validate_query(" camera ", 10) is None


def test_display_images_only_use_actual_allowed_https_hosts():
    assert len(image_key("https://m.media-amazon.com/images/I/example.jpg")) == 64
    for value in [
        "http://m.media-amazon.com/a",
        "https://localhost/a",
        "https://user@m.media-amazon.com/a",
    ]:
        with pytest.raises(ValueError):
            image_key(value)
