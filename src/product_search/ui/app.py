"""English storefront. All searches go through the existing FastAPI service."""

import html
import os
from pathlib import Path
import streamlit as st
from product_search.ui.client import ApiClient, ApiError
from product_search.ui.images import ImageCache

from product_search.ui.images import image_display
from product_search.ui.presentation import (
    validate_query,
    error_message,
    title_markup,
    results_markup,
)


@st.cache_resource
def api_client(base):
    return ApiClient(base)


@st.cache_resource
def image_cache(root):
    return ImageCache(root)


EXAMPLES = [
    ("Noise-cancelling headphones", "wireless noise cancelling headphones for travel"),
    ("Hiking backpacks", "compact waterproof daypack for hiking"),
    ("Insulated water bottles", "stainless steel insulated water bottle"),
    ("Laptop sleeves", "laptop sleeve for 13 inch notebook"),
]


def use_example(text):
    st.session_state["search_query"] = text


@st.fragment(run_every="1s")
def cards(items, root, show_scores):
    cache = image_cache(root)
    with st.container(key="results_grid"):
        for start in range(0, len(items), 3):
            columns = st.columns(3, gap="medium")
            for column, item in zip(columns, items[start : start + 3]):
                rank = item["rank"]
                with column, st.container(key=f"product_card_{rank}"):
                    st.markdown(
                        f'<div class="card-topline"><span class="rank-badge">#{rank:02d}</span><span>{html.escape(item["product_locale"].upper())} catalog</span></div>',
                        unsafe_allow_html=True,
                    )
                    path, state = image_display(cache, item.get("image_url"))
                    if path:
                        st.image(str(path), width="stretch")
                    else:
                        label = {
                            "loading": "Loading image…",
                            "missing": "No image available",
                            "unavailable": "Image unavailable",
                        }[state]
                        st.markdown(
                            f'<div class="image-placeholder"><span class="placeholder-icon">▧</span><span>{label}</span></div>',
                            unsafe_allow_html=True,
                        )
                    st.markdown(title_markup(item["title"]), unsafe_allow_html=True)
                    st.markdown(
                        '<div class="product-id">Product ID: '
                        + html.escape(item["product_id"])
                        + "</div>",
                        unsafe_allow_html=True,
                    )
                    if show_scores:
                        st.markdown(
                            f'<div class="model-score">Ranking score: {item["score"]:.5f}</div>',
                            unsafe_allow_html=True,
                        )


def main():
    st.set_page_config(
        page_title="Search Studio | Product Search", page_icon="⌕", layout="wide", menu_items={}
    )
    st.markdown(
        "<style>" + Path(__file__).with_name("theme.css").read_text() + "</style>",
        unsafe_allow_html=True,
    )
    base = os.environ.get("SEARCH_API_URL", "http://127.0.0.1:8000")
    images = os.environ.get("SEARCH_IMAGE_CACHE", str(Path.home() / ".cache/product-search/images"))
    try:
        client = api_client(base)
        health = client.health()
    except (ApiError, OSError, ValueError):
        client = None
        health = {"ready": False, "components": {}}
    online = bool(health.get("ready"))
    status = "Search is ready" if online else "Search unavailable"
    status_class = "" if online else " offline"
    result = st.session_state.get("last_result")
    st.markdown(
        f"""<div class="brand-row"><div class="wordmark"><span class="brand-mark">⌕</span>Search Studio<span class="brand-subtitle">Product search</span></div><span class="status-pill{status_class}"><span class="status-dot"></span>{status}</span></div>""",
        unsafe_allow_html=True,
    )
    if not result:
        st.markdown(
            """<div class="hero"><h1>Find what you’re looking for.</h1><p>Explore 164,900 products. Start with a product, a brand, or a few details.</p></div>""",
            unsafe_allow_html=True,
        )
    if not online:
        st.warning("Search is temporarily unavailable. Check the service status below.")
    with st.container(key="search_panel"):
        with st.form("natural_search"):
            left, number, action = st.columns([5.8, 1.15, 1.3], vertical_alignment="bottom")
            left.text_input(
                "Search products",
                key="search_query",
                placeholder="What are you looking for?",
                max_chars=2048,
                disabled=not online,
            )
            count = number.selectbox("Results", [5, 10, 20, 50, 100], index=1, key="display_count")
            submitted = action.form_submit_button(
                "Search", type="primary", disabled=not online, width="stretch"
            )
    with st.container(key="suggestions", horizontal=True):
        st.caption("Try:", width="content")
        for i, (label, text) in enumerate(EXAMPLES):
            st.button(
                label, key=f"example_{i}", on_click=use_example, args=(text,), disabled=not online
            )
    if submitted:
        query = st.session_state.get("search_query", "")
        error = validate_query(query, count)
        if error:
            st.warning(error)
        else:
            try:
                with st.spinner("Finding products…"):
                    st.session_state["last_result"] = client.search(query, count)
                st.rerun()
            except ApiError as exc:
                st.error(error_message(exc))
    result = st.session_state.get("last_result")
    # Reserve the result area so display options below can update the same cards.
    result_area = st.container()
    with st.container(key="demo_tools"), st.expander("Search details", expanded=False):
        show_scores = st.checkbox("Show ranking scores", value=False)
        if st.button("Refresh service status"):
            st.rerun()
        st.caption(
            "Live searches encode your query, retrieve 100 candidates and rank them using the frozen Fusion → Prepared28 → Scaler → ListNet pipeline. Product images load separately."
        )
        st.caption(
            "The online encoders passed vector and result compatibility checks. The exact historical SQID checkpoint revision remains unverified."
        )
        st.markdown("**Service components**")
        st.json(health.get("components", {}), expanded=False)
        if result:
            st.markdown("**Request timing (ms)**")
            st.json(result["timings_ms"], expanded=False)
    with result_area:
        if result:
            st.markdown(results_markup(result), unsafe_allow_html=True)
            if result["results"]:
                cards(result["results"], images, show_scores)
            else:
                st.info("No products found. Try a broader description or different keywords.")
        else:
            st.markdown(
                '<div class="empty-state"><div class="empty-icon">⌕</div><h3>A little detail goes a long way.</h3><p>Try a product name, a size, or how you plan to use it.</p></div>',
                unsafe_allow_html=True,
            )
    st.markdown(
        '<div class="page-footer"><span>Search Studio</span><span>164,900 products · Product search demo</span></div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
