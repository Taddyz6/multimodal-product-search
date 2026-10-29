# Frozen feature order

All28 output columns are float32. The scaler subtracts its original float64 mean and divides by its original float64 standard deviation, then casts to float32 before the64→32 ReLU MLP. Changing conversion points can change scores.

Let Q be the set of case-folded query tokens, F a field-token set, and D the query tokens containing at least one digit. Coverage uses `|Q∩F|/max(1,|Q|)`; digit coverage uses `|D∩document|/max(1,|D|)`. Phrase tests use the original space-delimited token strings.

| Index | Feature | Definition |
|---:|---|---|
|0|bm25|Frozen BM25 document weights summed over binary query vocabulary terms|
|1|tfidf|Frozen normalized document/query TF-IDF sparse inner product|
|2|title_tfidf|Frozen title/query TF-IDF sparse inner product|
|3|query_coverage|Query token coverage in document|
|4|title_coverage|Query token coverage in title|
|5|brand_overlap|Query token coverage in brand|
|6|color_overlap|Query token coverage in color|
|7|bullet_coverage|Query token coverage in bullet text|
|8|description_coverage|Query token coverage in description|
|9|digit_coverage|Digit-bearing query token coverage in document|
|10|has_digits|Whether D is nonempty|
|11|title_phrase_match|Entire normalized query phrase occurs in title|
|12|brand_phrase_match|Entire normalized brand phrase occurs in query|
|13|query_length|log1p of unique query token count|
|14|title_length|log1p of unique title token count|
|15|document_length|log1p of frozen retained-vocabulary document token count|
|16–19|missing_description / missing_bullet / missing_brand / missing_color|Whether the corresponding token set is empty|
|20|sbert_similarity|Original normalized SBERT product/query inner product|
|21|clip_text_similarity|Original normalized CLIP product-text/query inner product|
|22|clip_image_similarity|CLIP image/query inner product; zero for missing-image rows|
|23|image_available|Original SQID image-availability mask|
|24|calibrated_text_fusion_score|0.5×SBERT similarity +0.5×raw CLIP text similarity|
|25|calibrated_fusion_score|0.5×SBERT similarity +0.5×calibrated, mask-aware CLIP score|
|26|brand_length|log1p of brand token count, including repeated tokens|
|27|color_length|log1p of color token count, including repeated tokens|

For text and image scores, calibration forms train-fitted standardized values `zt` and `zi`. Available-image rows use `0.5zt+0.5zi`. Missing-image rows use the original missing-text standardization multiplied by the mixture standard deviation, then return to the reference text score scale. It does not substitute a higher-scale text cosine directly or remove a product.

`document_length` deliberately uses the stored float32 count and original `np.log1p` evaluation. TF-IDF features keep the sparse multiply/sum reduction. Semantic features keep `einsum`. These details and canonical product-key tie breaking remain explicit in the implementation because apparently equivalent arithmetic can alter rounding.
