# External asset setup

The source repository contains code and small hash contracts. Keep the frozen catalog, indices, embeddings and checkpoints outside Git. Do not copy relevance labels into the serving bundle.

`configs/service.example.json` describes ten production assets:

| Role | Expected content |
|---|---|
| `products` | 164,900 canonical locale/product-ID rows, content, embedding row and image-availability mask |
| `sbert_product` | float32 `(164900, 384)` product vectors |
| `clip_product_text` | float32 `(164900, 768)` SQID product text vectors |
| `clip_product_image` | float32 `(164900, 768)` SQID image vectors, including original missing rows |
| `lexical` | Original Week5 BM25/TF-IDF vocabulary and statistics |
| `fusion_index` | Original mask-aware, calibrated FAISS FlatIP index, 164,900 rows, 1,153 dimensions |
| `calibration` | Train-only text/image/missing-text score statistics |
| `fusion_recipe` | Original α=0.5, β=0.5 index recipe |
| `listnet_npz` | Original scaler and 28→64→32→1 weights |
| `listnet_json` | Original feature order and model metadata |

Every file hash must match `src/product_search/frozen_contract.json`. The loader checks hashes before deserializing the lexical pickle. Its single legacy class name is mapped to the clean package class; the pickle itself is not edited or refitted.

The product Parquet must include the original fields used by Prepared28: `product_locale`, `product_id`, `product_title`, `document`, `product_brand`, `product_color`, `product_bullet_point`, `product_description`, `embedding_row`, and `has_image_embedding`. `image_url` is used for display when present. Joint keys stay as original strings, with no stripping or numeric conversion.

## Text encoders

`configs/encoder_policy.example.json` contains the approved checkpoint/tokenizer hashes. Its roots point to external SBERT and CLIP checkpoint folders, the isolated dependency folder and a local Hugging Face cache. No network download occurs during service startup.

- SBERT: `sentence-transformers/all-MiniLM-L6-v2`, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`; attention-mask mean pooling, max length 256, Torch L2 followed by original float32 NumPy L2.
- CLIP: `openai/clip-vit-large-patch14`, compatible revision `32bd64288804d66eefd0ccbe215aa642df71cc41`; projected text output, max length 77, float32 NumPy L2.

These are the accepted free-text demo assets. Compatibility is not proof of the exact historical SQID revision. The approved preprocessing is retained in `encoder_worker.py`.

Install encoder packages into the policy's `dependencies` directory using the **same Python3.12 interpreter** as the backend. They are loaded by a persistent subprocess; do not activate a NumPy1 environment over the ranking core. The verified runtime is CPU-only on Intel macOS. Other operating systems or hardware require their own numerical and resource checks; no portable latency guarantee is implied.

## Verify paths

```bash
export PYTHONPATH="$PWD/src"
python tools/verify_assets.py --config configs/service.json
```

Paths may be absolute, use `${ENVIRONMENT_VARIABLES}`, or be relative to the config file. If a file differs, restore the registered asset; do not replace it with a same-name model or rebuild the index silently. Startup also validates dimensions, feature order, scaler hashes and the numerical dependency versions.

Only the service configuration and encoder policy need local paths. Keep those files ignored by Git. The example files and frozen hashes are safe to track.
