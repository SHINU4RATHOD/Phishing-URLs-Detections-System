# Why We Upgraded to DataPrep9 / Version 9

To optimize the Hybrid GLU Fusion model, we upgraded to Version 9, which focuses on optimizing the heuristic feature space and dropping redundant, collinear, or leaky features.

## Dropped Features & Rationales

### 1. `h_primary_category` (Target Leakage / String Type)
*   **Reason for Dropping:** It is a string/categorical value representing the predicted category of the threat. String values are not suitable for direct input into the Heuristic MLP without encoding. Furthermore, it represents a direct heuristic output that can cause target leakage and prevent the model from learning organic boundaries.

### 2. `h_flags_count` & `h_severity_score` (Collinearity Reduction)
*   **Reason for Dropping:** Both are linear combinations of the individual binary flags (`hF_*`). 
    *   `h_flags_count` is simply the sum of active flags.
    *   `h_severity_score` is a weighted sum of the active flags.
*   Feeding these aggregate features alongside the raw binary indicators (`hF_*`) creates high collinearity in the MLP layers. Dropping them forces the MLP to learn cleaner, non-linear boundaries directly from the individual binary features.

### 3. `h_has_https` (Text Redundancy)
*   **Reason for Dropping:** The text encoder (MiniLM) receives the fully cleaned `canonical_url` as its input. In WordPiece tokenization, the scheme (`http` vs `https`) is the first token in the sequence. Since the transformer already embeds this signal with 100% certainty, including `h_has_https` as a separate heuristic feature is redundant.

## Tokenizer Experiment 2: Custom URL-trained Byte-Level BPE

To further improve our URL classification models, we introduced the custom URL-trained Byte-Level BPE (Byte-Pair Encoding) tokenizer.

### Rationale & Benefits
1. **Zero Out-Of-Vocabulary (OOV) / Unknown Tokens:** Pretrained tokenizers (like WordPiece in MiniLM) map unknown characters or rare substrings to `[UNK]`. For complex URLs containing specialized query parameters, paths, or custom symbols, this can cause significant signal loss. A Byte-Level BPE operates directly on raw bytes (0-255), meaning *any* sequence of characters can be represented as a sequence of byte tokens without ever emitting a `[UNK]` token.
2. **Domain-Specific Vocabulary:** Pretrained tokenizers are trained on general corpus texts (Wikipedia, Books). Our custom BPE tokenizer is trained directly on the training dataset URLs, producing a vocabulary tailored explicitly to phishing and benign URL patterns (e.g. common protocol substrings, subdomains, domains, and top-level domain patterns).
3. **Scaling with large datasets (50M URLs):** With a very large corpus, a small vocab size leads to excessive sequence fragmentation (shorter token chunks). By configuring a vocab size of `50000` instead of standard `30000`, the model captures longer, more meaningful semantic blocks (such as entire domain/subdomain tokens) from the 50M dataset, improving representation density and training efficiency.

### Config Integration
* **Switch:** Configured via `use_custom_tokenizer_byte-level_BPE` under the `model:` block in the configuration YAML files (set to `true` to enable).

## Tokenizer Experiment 3: Custom URL-trained SentencePiece Unigram

To evaluate different tokenization behaviors on URLs, we introduced the custom URL-trained SentencePiece Unigram model.

### Rationale & Benefits
1. **Unigram Vocabulary Selection:** Rather than building a vocabulary bottom-up (like BPE), Unigram starts with a large vocabulary of subwords/characters and iteratively prunes the least useful ones according to a language model likelihood objective. This produces a highly optimized subword set.
2. **Multiple Segmentations (Probabilistic):** Unigram tokenization allows sampling different tokenizations for the same URL, which acts as a regularizer during training (similar to subword regularization).
3. **SentencePiece Metaspace Pre-Tokenization:** By replacing traditional whitespace pre-tokenizers with SentencePiece's Metaspace pre-tokenizer (which replaces spaces with `_`), we preserve exact character sequences without losing spatial alignment, which is critical for parsing URL paths.

### Config Integration
* **Switch:** Configured via `use_custom_tokenizer_SentencePiece_Unigram_BPE` under the `model:` block in the configuration YAML files (set to `true` to enable).

## 🛡️ Default Tokenizer Fallback Behavior

If both configuration switches (`use_custom_tokenizer_byte-level_BPE` and `use_custom_tokenizer_SentencePiece_Unigram_BPE`) are set to `false` (default), the pipeline cleanly falls back to the default pretrained MiniLM WordPiece tokenizer (`microsoft/MiniLM-L12-H384-uncased`) without any modifications.
