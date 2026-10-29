# Essential research code

The `research/` package retains the central algorithms behind the current ESCI+SQID experiments. It has no import from the archived weekly runners, and serving does not import it. No training is started when this repository is opened or the API is launched.

| Module | Retained purpose |
|---|---|
| `lexical.py` in the main package | Catalog-fitted BM25/TF-IDF, deterministic Top-K selection and original 20 lexical/metadata features |
| `research/pointwise.py` | Week2 20-feature pointwise XGBoost; Train/Warm grades, Warm Validation RMSE early stopping |
| `research/views.py` | Original 6,171-query/100,856-pair LTR view and Lexical/Semantic/Image/Metadata deletion groups |
| `research/ranking.py` | Same-feature pointwise XGBoost, LambdaMART and masked top-one ListNet; full-pool Warm NDCG selection |
| `research/evaluation.py` | Fixed-pool ranking, Warm/Cold scope masks, known-positive catalog recall and paired query bootstrap |
| `research/projection.py` | Frozen-CLIP linear projection heads and symmetric InfoNCE; evaluated alternative, not deployed |

For new offline training, `requirements-research.txt` targets Linux/Colab with a NumPy2-compatible Torch build. The Intel Mac demo uses the separate NumPy1/Torch2.2 encoder worker. Training entrypoints were retained and syntax/unit checked here; no models were retrained during cleanup.

## Lexical baseline

```python
from product_search.lexical import SparseLexicalIndex, top_indices

index = SparseLexicalIndex().fit(products)
index.save("lexical_index.pkl")
rows = top_indices(index.scores("wireless headphones", "bm25"), 100)
keys = [index.keys[i] for i in rows]
```

The online system must load the original Week5 index. Fitting this example creates a separate research artifact; it does not replace the production asset. The retained `SparseLexicalIndex.features()` implements the original offline 20-column builder. Serving uses the exactly checked Prepared28 builder to avoid whole-catalog Pandas work per request.

## Ranking comparisons

Prepare external feature tables from frozen query/product splits. Keep Train/Warm labels separate from unlabeled candidate features; retain every candidate's competition position. Feature tables must use the same lexical index and semantic score definitions as the model being studied.

```python
from product_search.research.ranking import train
from product_search.schema import FEATURES

# Arrays and metadata are supplied explicitly, in aligned frozen order.
train("listnet", train_warm_labels, X_train, validation_candidates,
      X_validation, warm_validation_labels, FEATURES, "outputs/new_listnet")
```

`train()` requires sorted Train/Warm lists and disjoint Train/Validation queries. It refuses an existing output directory. It saves model weights, configuration, selection history and training time, then checks the saved predictor against its selected score. `model_config()` holds the original fixed budgets; this export does not rerun them or adopt later unsuccessful optimization branches.

Week2 is a distinct original baseline:

```python
from product_search.research.pointwise import train_pointwise

model, metadata = train_pointwise(train_features, validation_features,
                                 warm_validation_labels,
                                 "outputs/new_week2", device="cpu")
```

It uses the original 20 features and RMSE early stopping. Do not substitute the 28-feature Week5 XGBoost when reporting the original Week2 comparison.

`research.views.feature_columns(remove="Image")` removes every feature depending on the image group, including calibrated fusion. Other supported groups are `Lexical`, `Semantic`, and `Metadata`. Such ablations require separate retraining and are not runtime switches for the frozen ranker.

## Evaluation boundaries

- `evaluate_pool()` verifies the full fixed candidate identities. Only Warm E/S-eligible queries enter the summary; non-Warm target gain is masked while their rank positions remain.
- `scoped_metrics(..., scope="cold", catalog=False)` evaluates Cold-Val targets at the original full-pool positions. It does not filter Warm competitors out of the ranked list.
- `scoped_metrics(..., catalog=True)` omits NDCG/MAP/MRR and reports only known-positive recall, hit and judged coverage. Unjudged products are never assigned I.
- `bootstrap()` resamples queries, and `paired(candidate, baseline)` requires the same query population. Product or request repeats are not independent query samples.

These helpers accept explicitly provided scoped Validation frames. The sealed one-time Final Test execution code and its label files are intentionally kept in the original archive rather than exposed as an everyday training command.

Projection training checks Warm-only, image availability, Train-Seen roles and content overlap in the alignment holdout. The projection module is retained for the reported study; it is never loaded by the production pipeline. The selected system remains frozen CLIP+ListNet.
