# 🔬 Comprehensive Tokenizer & Gating Rationale Report (DataPrep12)

This report provides the technical rationale, architectural trade-offs, and empirical benchmarks comparing **WordPiece**, **Byte-Level BPE**, **SentencePiece Unigram**, and **Character-Level** tokenization schemes on the phishing URL detection corpus. It also documents the fusion gating upgrades (GLU vs. SwiGLU) and feature space optimizations implemented in the **DataPrep12** workspace.

---

## 1. Tokenization Methodologies for URL Classification

URLs are highly structured strings that do not conform to standard natural language grammar rules. They contain scheme qualifiers (`https://`), domain hierarchies (`sub.domain.tld`), port mappings, path structures, and parameterized query strings (`?id=123&token=abc`). Standard natural language tokenizers often perform suboptimally on URLs due to vocabulary mismatch and sequence fragmentation.

Below is an overview of the four tokenization approaches evaluated:

### A. WordPiece (Default Pretrained MiniLM)
* **Pretraining Alignment:** Native tokenizer for the `microsoft/MiniLM-L12-H384-uncased` backbone.
* **Mechanism:** A likelihood-ratio-based subword extraction method. It builds vocabulary bottom-up, selecting subword units that maximize the likelihood of the training data under a unigram language model constraint.
* **Vocab Size:** `30,522` tokens (general language domain: English Wikipedia + BookCorpus).

### B. Custom URL-Trained Byte-Level BPE
* **Zero Out-of-Vocabulary (OOV):** By mapping the base vocabulary to raw bytes (0–255), this tokenizer eliminates the `[UNK]` token entirely. Any unseen string can be encoded as a sequence of bytes.
* **Mechanism:** Co-occurrence-based merge operations trained directly on our URL dataset.
* **Vocab Size:** `50,000` tokens (tailored specifically to domain and path patterns).

### C. Custom URL-Trained SentencePiece Unigram
* **Subword Regularization:** Employs a pruning-based vocabulary builder. Starting with a large candidate set, it iteratively discards low-probability subwords to maximize text likelihood.
* **Metaspace Pre-Tokenization:** Replaces traditional whitespace separators with a meta-symbol (`_`), preserving precise spatial alignment and character sequences across URL boundaries.
* **Vocab Size:** `50,000` tokens (built natively on our URL training corpus).

### D. Character-Level Tokenizer (Theoretical Baseline)
* **Fine-Grained Parsing:** Treats every character (or byte) as an individual token.
* **Mechanism:** Direct character-to-index mapping without subword chunking.
* **Vocab Size:** `< 256` tokens (representing basic ASCII and standard Unicode ranges).

---

## 2. Empirical Performance Evaluation

Both Subproject 1 (Raw & Canonical Model) and Subproject 2 (Hybrid Gating Model) were evaluated across the three active tokenization strategies using a test corpus of **497,095 URLs** (Subproject 1) and **496,017 URLs** (Subproject 2).

### Subproject 1: Raw & Canonical Model (MiniLM v3 Base)
Evaluated on raw URLs using a 5-layer classification head (`[384, 512, 256, 128, 64]`).

| Tokenization Scheme | Accuracy | Precision | Recall | F1-Score | AUC-ROC | FPR | FNR | Test Loss | Optimal Threshold |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **WordPiece (WPE)** | **92.89%** | **97.23%** | **88.22%** | **92.51%** | **97.93%** | **2.49%** | **11.78%** | **0.0316** | **0.530** |
| **Byte-Level BPE** | 92.14% | 95.66% | 88.19% | 91.77% | 97.51% | 3.96% | 11.81% | 0.0323 | 0.490 |
| **SP Unigram** | 90.60% | 92.69% | 88.05% | 90.31% | 96.60% | 6.87% | 11.95% | 0.0354 | 0.475 |

### Subproject 2: Hybrid Gating Model (MiniLM Hybrid GLU Fusion)
Evaluated end-to-end on combined URL text features and a Z-score normalized 87-feature heuristic space.

| Tokenization Scheme | Accuracy | Precision | Recall | F1-Score | AUC-ROC | FPR | FNR | Test Loss | Optimal Threshold |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **WordPiece (WPE)** | **92.21%** | **95.83%** | **88.16%** | **91.84%** | **96.99%** | **3.79%** | **11.84%** | **0.0375** | **0.490** |
| **Byte-Level BPE** | 90.66% | 92.89% | 87.94% | 90.35% | 95.95% | 6.65% | 12.06% | 0.0427 | 0.440 |
| **SP Unigram** | 90.26% | 92.92% | 87.02% | 89.88% | 95.59% | 6.55% | 12.98% | 0.0422 | 0.465 |

---

## 3. Tokenizer Pros, Cons & Suitability Matrix

| Tokenizer Method | Vocab Size | Avg. Tokens per URL | OOV / Unknown (`[UNK]`) Rate | Pros for URL Detection | Cons for URL Detection | Suitability |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: |
| **WordPiece (WPE)** | `30,522` | ~24.1 | Low (< 1.5% due to percent-encoding) | • Perfect alignment with pretrained backbone weights.<br>• Faster convergence under LoRA / adapter tuning.<br>• Stable gradient updates. | • Replaces custom/rare characters with `[UNK]`. | **Highly Suitable (Recommended)** |
| **Byte-Level BPE** | `50,000` | ~18.4 | **0.0% (Zero OOV)** | • Custom vocabulary maps complete subdomains and query structures.<br>• Highly compact sequence lengths.<br>• Immune to unseen character errors. | • **Vocabulary mismatch:** Scrambles pretrained MiniLM embedding weights.<br>• Poor compatibility with low-rank PEFT updates. | **Moderately Suitable** (Requires full-parameter training) |
| **SP Unigram** | `50,000` | ~19.8 | Low (< 0.2%) | • Probabilistic subword regularization mitigates overfitting.<br>• Preserves character offsets via Metaspace tokens. | • Highest training loss and validation loss.<br>• Suffers from embedding misalignment under LoRA limits. | **Low Suitability** |
| **Character-Level** | `< 256` | ~78.5 | **0.0%** | • Highly robust to typosquatting and character obfuscations.<br>• Very small embedding parameter count. | • Extreme sequence fragmentation.<br>• Heavy computation/attention overhead.<br>• Fails to capture high-level domain semantics. | **Unsuitable** (for Transformer transfer learning) |

---

## 4. Key Engineering Insights: The Embedding Alignment Mismatch

The empirical data demonstrates that **WordPiece (pretrained)** consistently outperforms the custom-trained URL tokenizers (BPE & Unigram), achieving **+0.75% to +2.29% higher Accuracy** and **substantially lower False Positive Rates (FPR)** across both subprojects. 

### Why Custom Tokenizers Underperformed Under LoRA:
1. **Scrambled Token ID Space:** A custom tokenizer (BPE or Unigram) creates a vocabulary specific to URLs. A token index like `4213` which represents `"https"` in custom BPE might represent a word like `"kitchen"` in standard MiniLM. 
2. **LoRA Parameter Constraint:** When using Parameter-Efficient Fine-Tuning (PEFT/LoRA), only a tiny fraction of the model parameters (`1.23%` to `1.49%`) are updated. The primary embedding layer (`model.embeddings.word_embeddings.weight`) remains **completely frozen**.
3. **Semantic Disconnect:** Because the embedding layer is frozen, the model maps the custom token IDs to the old pretrained general English representations. Since the LoRA adapters are not large enough to realign the entire representation space of the scrambled token IDs, the network's capacity is severely bottlenecked.
4. **Conclusion:** To extract the true benefit of custom URL-trained tokenizers, **full-parameter fine-tuning (especially of the embedding layer)** must be enabled, which increases training cost and GPU memory utilization. If computing resources are constrained (e.g., RTX A4000 16GB), staying with the native **WordPiece** tokenizer is the optimal choice.

---

## 5. Switchable SwiGLU Gating Integration

To enhance multi-modal feature fusion, Subproject 2 supports **SwiGLU (Swish Gated Linear Unit)** as a switchable alternative to standard **GLU (Gated Linear Unit)**.

### Mathematical Formulation
* **Standard GLU:**
  $$\text{GLU}(x) = \tanh(x W_1 + b_1) \otimes \sigma(x W_2 + b_2)$$
* **SwiGLU Gating:**
  $$\text{SwiGLU}(x) = (x W_1 + b_1) \otimes \text{SiLU}(x W_2 + b_2)$$
  *Where $\text{SiLU}(z) = z \cdot \sigma(z)$ is the Swish activation function.*

### Benefits of SwiGLU:
1. **Improved Gradient Flow:** The SiLU activation function features a smooth, non-monotonic profile that prevents gradient vanishing/saturation in the gating tower, stabilizing the fusion of text embeddings ($384\text{D}$) and heuristic arrays ($87\text{D}$).
2. **Non-linearity Representational Capacity:** Removing the monotonic squashing bounds of $\tanh$ enables the bottleneck layer to scale representation values dynamically, matching state-of-the-art architectures like LLaMA.

---

## 6. Dropped Features & Rationales

To prevent data leakage and multicollinearity, the heuristic feature extraction pipeline dropped three legacy columns:

1. **`h_primary_category` (Target Leakage / String Type):**
   * *Reason:* Directly encodes the predicted category of the threat. String values are incompatible with MLP architectures, and their inclusion causes target leakage, preventing the transformer from learning organic classification boundaries.
2. **`h_flags_count` & `h_severity_score` (Collinearity Reduction):**
   * *Reason:* Both are linear combinations of individual binary flags (`hF_*`). Feeding these aggregate features alongside raw binary indicators creates high collinearity. Dropping them forces the MLP to learn cleaner, non-linear boundaries directly from individual features.
3. **`h_has_https` (Text Redundancy):**
   * *Reason:* The text encoder receives fully cleaned canonical URLs. The protocol scheme (`http` vs `https`) is parsed as the very first token in WordPiece. Since the transformer encodes this signal with 100% certainty, including it as a heuristic feature is redundant.

---

## 7. Classification Layer Analysis: Sigmoid vs Softmax Output Heads

To optimize gradient convergence and inference execution speed, we conducted a comparative analysis of the classification layer output head, implementing support for both **Sigmoid (1-class output)** and **Softmax (2-class output)** activation schemes.

### A. Mathematical and Architectural Configurations

| Feature / Metric | Sigmoid Output Head | Softmax Output Head (Default Baseline) |
| :--- | :--- | :--- |
| **Output Dimension** | `(batch_size, 1)` | `(batch_size, 2)` |
| **Probability Function** | $p = \sigma(z) = \frac{1}{1 + e^{-z}}$ | $p_1 = \text{Softmax}(z)_1 = \frac{e^{z_1}}{e^{z_0} + e^{z_1}}$ |
| **Loss Function** | Binary Focal Loss + BCE With Logits Loss | Multiclass Focal Loss + Cross Entropy |
| **Parameter Efficiency** | projection weights size = `(in_dim, 1)` (reduces final head parameters by 50%) | projection weights size = `(in_dim, 2)` |
| **Log-Odds Extraction** | $z$ (directly obtained from single-logit output) | $z_1 - z_0$ (computed as the difference between class logits) |
| **Temperature Scaling** | $\tilde{p} = \sigma(\frac{z}{T})$ | $\tilde{p}_1 = \text{Softmax}(\frac{z}{T})_1$ |

### B. Analytical Comparison of Loss Functions

1. **Multiclass Focal Loss (Softmax):**
   - Models benign and malicious predictions as mutually exclusive categories.
   - Cross-entropy calculations apply label smoothing ($0.05$) to penalize overconfidence, helping generalize prediction boundaries during temperature calibration.
   
2. **Binary Focal Loss (Sigmoid):**
   - Focuses solely on the probability of a URL being malicious.
   - Computes gradients using `binary_cross_entropy_with_logits` against target classes cast to float.
   - The lack of explicit label smoothing in the binary loss formulation results in more aggressive gradient updates, which can accelerate initial convergence on unbalanced datasets.

### C. Full Production-Scale (5M Dataset) Performance Comparison

After training on the production corpus of **~4.9 million samples**, the empirical performance landscape changes significantly. Below is the side-by-side performance of both activation heads across the test datasets:

| Subproject / Model | Output Activation | Test Accuracy | Test Precision | Test Recall | Test F1-Score | AUC-ROC | False Positive Rate (FPR) | False Negative Rate (FNR) | Test Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Subproject 1 (Raw) | Softmax | 92.9953% | 97.6124% | 88.0728% | 92.5976% | 98.1698% | 2.1323% | 11.9272% | 0.028309 |
| | **Sigmoid** | **93.0094%** | **97.6405%** | **88.0753%** | **92.6115%** | **98.1832%** | **2.1067%** | **11.9247%** | **0.026744** |
| Subproject 2 (Hybrid) | Softmax | 92.2418% | 96.2633% | 87.8003% | 91.8372% | 96.8604% | 3.3684% | 12.1997% | 0.040578 |
| | **Sigmoid** | **92.3688%** | **96.3317%** | **87.9986%** | **91.9768%** | **97.0178%** | **3.3119%** | **12.0014%** | **0.036327** |

#### 🔬 Scale-Tuning Insights:
1. **Sigmoid Outperforms at Scale:** On the full 4.9M sample dataset, the **Sigmoid** configuration outperforms the Softmax configuration across **all metrics** in both subprojects. It achieves higher Accuracy, higher Precision, higher Recall, and lower Test Loss.
2. **Gradient Focus:** Sigmoid maps outputs directly to threat probability, simplifying the learning objective. In contrast, Softmax models the two classes as mutually exclusive categories, creating competing gradient paths that can bottleneck representational convergence at massive scales.
3. **Logit Calibration Quality:** At scale, Sigmoid outputs generate extremely smooth log-odds. The test loss for Sigmoid (Subproject 1: `0.0267`, Subproject 2: `0.0363`) is consistently lower than Softmax, indicating superior model calibration and higher confidence margins on correct classifications.

### E. Suitability Recommendation

1. **Sigmoid (Production Standard):** Recommended for both backend and resource-constrained edge-device deployments. At scale, it delivers superior accuracy, lower loss, and reduces classification layer parameters by 50% (projecting to 1 logit instead of 2).
2. **Softmax:** Only recommended for legacy systems where multi-class extensions are anticipated, or for small-scale datasets where class-competing gradients help mitigate initial convergence volatility.


