<div align="center">

<h1>Multimodal Product Search</h1>

<p>Multimodal retrieval, learning to rank, and an interactive product search demo.</p>

<p>
  <a href="https://github.com/Taddyz6/multimodal-product-search/actions/workflows/tests.yml"><img src="https://github.com/Taddyz6/multimodal-product-search/actions/workflows/tests.yml/badge.svg?branch=main" alt="Tests"></a>
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&amp;logoColor=white" alt="Python 3.12">
  <img src="https://img.shields.io/badge/Inference-CPU-0F766E" alt="CPU inference">
</p>

<p><strong>English</strong> · <a href="README.zh-CN.md">简体中文</a></p>

<p>
  <a href="docs/Technical_Report.pdf">Technical Report</a> ·
  <a href="#architecture">Architecture</a> ·
  <a href="#results">Results</a> ·
  <a href="#quick-start">Quick Start</a>
</p>

</div>

---

## Overview

Search a catalog of **164,900 products** using natural-language queries, frozen SBERT/CLIP representations, and a learned ranker. The project connects relevance experiments on **Amazon ESCI + SQID** with a working FastAPI backend and an English Streamlit interface.

<table>
  <tr>
    <td align="center"><strong>164,900</strong><br>catalog products</td>
    <td align="center"><strong>Top100</strong><br>exact retrieval candidates</td>
    <td align="center"><strong>28 features</strong><br>lexical, semantic, image and metadata</td>
    <td align="center"><strong>CPU demo</strong><br>live query encoding and ranking</td>
  </tr>
</table>

- **Multimodal retrieval:** train-calibrated text/image fusion with an explicit missing-image fallback.
- **Learning to rank:** a frozen 28→64→32→1 ListNet using lexical, semantic, visual and metadata signals.
- **Interactive search:** free-text queries, real product titles and images, ranking scores and request timings.
- **Verified inference preparation:** row-aligned candidate lookup and cached product tokens preserve the original computation while removing repeated catalog processing.

## Search demo

![English product search interface](docs/images/search_demo.png)

The interface accepts a product name, brand or description and displays ranked results from the real catalog. Images load separately; unavailable images receive a placeholder. Missing prices, ratings and other absent fields remain missing.

## Architecture

![Retrieval and ranking architecture](docs/images/architecture.png)

```text
Natural-language query
        │
        ▼
Frozen SBERT + compatible CLIP text encoder
        │
        ▼
Calibrated + mask-aware Fusion · FAISS FlatIP · Top100
        │
        ▼
Prepared28 features → original scaler → frozen ListNet
        │
        ▼
Ranked products → FastAPI → Streamlit
```

Fusion uses **α = β = 0.5**. The catalog, index, feature order, scaler and ranking weights remain fixed. The online encoders passed vector and result compatibility checks; the exact historical SQID CLIP revision remains unverified. RRF and projection adaptation are documented research alternatives, while Fusion→ListNet remains the default system.

## Results

### Relevance evaluation

The one-time Final Test uses human ESCI judgments and frozen query/product splits. NDCG is computed in the **fixed judged candidate pool**, preserving all competitor positions and masking targets outside the reported scope. Cold-Test measures generalization to product IDs excluded from supervised training.

| Model | Warm NDCG@10 | Cold-Test NDCG@10 | Overall NDCG@10 |
|---|---:|---:|---:|
| BM25 | 0.6660 | 0.3090 | 0.7234 |
| Week2 XGBoost | 0.6740 | 0.3193 | 0.7339 |
| LambdaMART · 28 features | 0.7131 | **0.3320** | 0.7758 |
| ListNet · 28 features · default | **0.7163** | 0.3247 | **0.7776** |

Bold marks the highest point estimate in each column. ListNet was selected on Validation before the final evaluation; the Test results do not switch the default model. Warm, Cold-Test and Overall use distinct target scopes, so Overall is not their arithmetic average.

On Warm Test, XGBoost improves NDCG@10 by **1.20%** over BM25; LambdaMART and ListNet improve by **5.80%** and **6.28%** over Week2 XGBoost. ListNet's Cold-Test gain is **1.69%**, with a paired confidence interval crossing zero. The original ranking improvement targets were not met.

For full-catalog retrieval, calibrated Fusion achieves Warm **known-positive Recall@50 = 0.5449**, compared with SBERT's **0.4765**, a **14.35% relative improvement**. This measures coverage of known E/S positives; unjudged products are not assigned negative labels, and this recall is kept separate from pool NDCG.

The [38-page technical report](docs/Technical_Report.pdf) includes confidence intervals, ablations, hard cases and unsuccessful experiments. Historical metadata-based results are reported separately because their queries, labels and evaluation setup differ.

### Local CPU performance

Recorded on an **Intel Core i5-1038NG7**, macOS 14.8.7, four numerical threads and one API worker. Each boundary contains 4,000 measurements across four repeat observations, with models loaded and the product LRU independently conditioned by two warmup passes. B/C execute live encoders for every request.

| Timing boundary | Mean (ms) | P95 (ms) |
|---|---:|---:|
| **A** · cached query vectors → retrieval and ranking | 102.159 | 204.994 |
| **B** · live dual encoders → ranking and product fields | 164.026 | 337.103 |
| **C** · client HTTP request → complete response body | 184.525 | 370.082 |

Means average the four round means; P95 is recomputed from pooled same-condition requests. Some 64-client workloads produced queueing timeouts, and two Broad sustained windows ended early. Different timing boundaries and hardware sessions are reported separately. See [performance notes](docs/PERFORMANCE.md) for burst/sustained definitions, cache preparation, request coalescing and incomplete scenarios.

## Quick start

The verified demo runs on **Intel macOS CPU with Python 3.12**. Model weights, embeddings and indices are external assets and are not included in the repository. Prepare the registered files using the [asset guide](docs/ASSETS.md) before starting the service.

### 1. Install and configure

```bash
git clone https://github.com/Taddyz6/multimodal-product-search.git
cd multimodal-product-search

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --target .encoder-deps -r requirements-encoder.txt

cp configs/service.example.json configs/service.json
cp configs/encoder_policy.example.json configs/encoder_policy.json
```

Update the asset paths in both configuration files. Relative paths resolve against the configuration file. Encoder dependencies stay isolated from the NumPy2 ranking core.

```bash
export PYTHONPATH="$PWD/src"
python tools/verify_assets.py --config configs/service.json
```

### 2. Start the API

```bash
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export SEARCH_CONFIG="$PWD/configs/service.json"

python -m uvicorn product_search.api:application \
  --factory --host 127.0.0.1 --port 8000 --workers 1
```

### 3. Start the interface

In a second terminal, activate the same virtual environment and run:

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD/src"
export SEARCH_API_URL=http://127.0.0.1:8000

python -m streamlit run src/product_search/ui/app.py \
  --server.address 127.0.0.1 --server.port 8501
```

Open **http://127.0.0.1:8501**.

<details>
<summary><strong>API endpoints and request example</strong></summary>

| Endpoint | Purpose |
|---|---|
| `GET /health` | Service, asset and dispatcher status |
| `GET /ready` | Readiness; returns 503 when required components are unavailable |
| `POST /search` | Natural-language search with `top_k` from 1 to 100 |

```bash
curl http://127.0.0.1:8000/ready

curl -X POST http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"wireless headphones for travel","top_k":10}'
```

Retrieval always keeps Top100, regardless of the display count. Startup checks the registered assets and never silently substitutes models or rebuilds the index.

</details>

## Project structure

```text
src/product_search/
├── artifacts.py       # external asset identity and loading
├── encoders.py        # persistent isolated encoder worker
├── retrieval.py       # exact FAISS search and aligned product keys
├── lexical.py         # BM25, TF-IDF and lexical definitions
├── features.py        # Prepared28 and bounded product-token LRU
├── calibration.py     # score calibration and missing-image fallback
├── ranking.py         # original scaler and ListNet inference
├── pipeline.py        # retrieval → features → ranking
├── service.py         # natural-language search and product fields
├── dispatch.py        # bounded admission and in-flight coalescing
├── api.py             # FastAPI application
├── ui/                # English Streamlit interface and HTTP client
└── research/          # offline baselines, metrics and projection study

configs/               # portable asset configuration examples
docs/                  # report, architecture, setup and verification
tests/                 # small correctness and lifecycle tests
tools/                 # asset checks and exact regression replay
```

<details>
<summary><strong>Research code and serving configuration</strong></summary>

The offline package retains the Week2 lexical/pointwise baseline, the Week5 XGBoost/LambdaMART/ListNet comparison, feature group ablations, scoped evaluation and the frozen-CLIP projection study. These modules are separate from serving. See [research usage](docs/RESEARCH.md) and the [28-feature definitions](docs/FEATURES.md).

The dispatcher uses one active computation, a queue capacity and waiter limit of 64, a 9.5 s queue deadline, a 10 s request deadline and an 8,192-product token LRU. Identical in-flight `(query, top_k)` requests may share computation. Completed answers and query vectors are not cached; an HTTP timeout does not cancel an already running CPU job.

</details>

## Verification

**45 tests passed locally.** Exact refactor checks cover 20 unlabeled Validation queries, including candidates, all 28 features, scaler output, ListNet scores and complete ranking. Three same-condition live-query checks also matched the approved encoder implementation.

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
ruff check src tests tools
ruff format --check src tests tools
```

Unit tests use small fixtures and require no catalog, relevance labels or checkpoints. See [verification details](docs/REFACTOR_VERIFICATION.md) and the optional [external-asset regression tool](tools/verify_regression.py).

## Documentation

| Document | Contents |
|---|---|
| [Technical report](docs/Technical_Report.pdf) | Methods, final evaluation, hard cases and system performance |
| [Asset setup](docs/ASSETS.md) | Required files, hashes, checkpoints and configuration |
| [Feature reference](docs/FEATURES.md) | Definitions and order of the original 28 features |
| [Research guide](docs/RESEARCH.md) | Baselines, ablations and evaluation boundaries |
| [Performance notes](docs/PERFORMANCE.md) | Timing scopes, workload and cache protocols |
| [Refactor verification](docs/REFACTOR_VERIFICATION.md) | Exactness checks and deployment scope |

Data and model terms apply separately to Amazon ESCI, SQID, SBERT and CLIP. Full datasets, checkpoints, relevance labels and private configuration paths are excluded from this source repository. See [source and asset attribution](NOTICE.md).
