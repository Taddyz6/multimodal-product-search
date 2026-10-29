"""Escape product/query text; keep presentation separate from ranking."""

import html
import math
from ..errors import QueryProcessor, InvalidInput


def validate_query(query, top_k=10):
    try:
        QueryProcessor().validate(query, top_k)
    except InvalidInput as exc:
        return str(exc)
    return None


def error_message(error):
    status = str(error).split(":", 1)[0]
    return {
        "429": "Search is busy right now. Please try again in a moment.",
        "504": "This search took too long. Please try again.",
        "503": "Search is temporarily unavailable. Please try again shortly.",
        "502": "We could not complete your search. Please try again.",
        "422": "Check your search text and result count, then try again.",
    }.get(status, "We could not reach the search service. Please try again shortly.")


def title_markup(title):
    text = title if isinstance(title, str) and title else "Title unavailable"
    safe = html.escape(text, quote=True)
    return f'<div class="product-title" title="{safe}">{safe}</div>'


def latency_text(value):
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        return "—"
    return f"{value:.0f} ms" if value < 1000 else f"{value / 1000:.2f} s"


def results_markup(result):
    query = html.escape(result["query"], quote=True)
    mode = "Live search"
    scope = "Request time includes query encoding, retrieval and ranking; excludes images."
    count = len(result["results"])
    noun = "product" if count == 1 else "products"
    return f'<div class="results-bar"><div><h2>Results for <span>“{query}”</span></h2>\n    <p>{count} {noun} · Ordered by relevance</p></div><div class="results-summary">\n    <span class="mode-pill">{mode}</span><span class="latency-pill" title="{scope}">{latency_text(result.get("client_http_ms"))}</span></div></div>'
