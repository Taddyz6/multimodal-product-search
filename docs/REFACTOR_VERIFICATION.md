# Refactor verification

This export consolidates the accepted A2 engineering path into `product_search`. The original experiment directories, assets and sealed evaluation outputs are retained separately and were not overwritten.

## Exact core check

Twenty fixed, unlabeled Validation queries were replayed using the actual 164,900-product catalog, original Week5 lexical pickle, Fusion index, calibration, scaler and ListNet weights. All input files passed their frozen SHA256 checks.

For all 20 queries, the following matched the locked local A2 reference arrays exactly: raw FAISS rows/scores, canonical candidate rows/keys/scores, all 28 feature columns, scaled features, ListNet scores, complete final order and feature names. Missing-image candidates were present in the checked queries. See `CORE_REGRESSION.json` for per-query fields and maximum errors.

These are correctness checks in the same local numerical environment, not performance benchmarks or a new relevance evaluation. No Final Test or Cold-Test labels were read. The optional `tools/verify_regression.py` can repeat the check with externally supplied query vectors and reference arrays.

## Online encoder check

Three unlabeled queries were run through the original approved worker and the cleaned worker with matching single-query batch sizes, the same checkpoints, tokenizers, dependency isolation and four threads. SBERT/CLIP outputs and final Top10 keys matched exactly. Both workers were closed and waited for. See `ONLINE_REGRESSION.json`.

An initial diagnostic compared a three-query batch against single-query calls and failed bitwise equality. That comparison mixed padding/batch conditions and was retained in the internal cleanup logs. The accepted regression compares the same invocation conditions. It does not establish the exact historical SQID revision or certify arbitrary batch-size equivalence.

## Tests and boundaries

All 45 tests passed locally, and Ruff reported no unused imports or local variables.

Asset-free tests cover lexical definitions, masks, scaler precision, ranking/tie handling, legacy pickle identity mapping, dispatcher timeouts/cancellation/coalescing/drain, HTTP schemas/readiness/errors, label scope masking, display escaping and the actual Streamlit form/result/error states. Optional small Torch tests cover the original ListNet mask and symmetric contrastive loss. Synthetic fixtures are not catalog quality or latency evidence.

The original offline model configurations, full-pool target masking and key algorithms were retained. Training workflows were simplified to explicit function calls and fresh output directories; no new training or statistical model selection was performed. Full historical notebooks, optimization failures, observer protocols and raw performance records remain in the original archive rather than being duplicated in this engineering repository.

The checked deployment remains CPU-only. A different CPU/BLAS environment can produce floating-point differences and needs its own regression run. The export does not imply new hardware-independent timing or exact historical online-encoder provenance.
