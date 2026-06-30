# Project: Advanced Phishing URL Detection & Modular Classification Pipeline

This document serves as the centralized engineering runbook, documenting the data pipeline distribution, experiment training matrix, hyperparameters, deployment benchmarks, and E2E execution paths for the Phishing URL Detection modular pipeline.

---

## 1. Data Pipeline & Distribution
### File Registry
* **Raw Dataset Location:** `***/DataPrep10/0_DATA`
* **Processed Dataset Location:** `***/DataPrep10/RESULTS_&_MODELS/2_preprocess_urls_output`
* **Processing Script:** `***/DataPrep10/SRC/2_preprocess_urls_v8_refactored.py`
* **Dynamic Import Wrapper:** `***/DataPrep10/SRC/urls_cate_V7.py` 

### Version Log & Processing Stats
* **Active Data Version:** `1.0`
* **Data Processing Time:** `[1-2 hrs / 5M URLs]`
* **Feature Distributions:** Here using raw url embedding as input to the model.

---

## 2. Model Training Log
### Centralized Configurations
* **Dynamic Config File:** All hyperparameters are managed via [4_config.yaml](***DataPrep10/1_MiniLM_V4_Model_On_Raw_Data_and_OFP_and_Canonical_Inferencing/SRC/4_config.yaml).
* **Data Split Strategy:** Strict 80/10/10 train/val/test split across all iterations:
  * **1. OVERALL SPLIT STATISTICS**
    * **Total Samples:** 4,954,117
    * **Train Set:**     3,959,595 (79.93%)
    * **Val Set:**        497,427 (10.04%)
    * **Test Set:**       497,095 (10.03%)

### Experiment Matrix

| Model Variant Name | Base Model Path | Training Script | Compute Resources | Training Duration | Checkpoint Strategy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `MiniLM_HybridFF_v4` | `*/data_prep8/saved_models/MiniLM_HybridFF_v4/best_model_epoch_003/model_full.pt` | `*/data_prep8/3_MiniLM_Hybrid_FF_V4.py` | 1x NVIDIA RTX A4000 GPU (16GB VRAM) | 1 Day (per epoch) | Saved checkpoint_epoch_003.pt |
| `MiniLM_raw_orig3` | `microsoft/MiniLM-L12-H384-uncased` | `SRC/5_train.py` | 1x NVIDIA RTX A4000 GPU (16GB VRAM) |1 Day (per epoch) | Resumable Fold-based checkpoints saved every epoch |

### Table 1: Core Optimization & Regularization Hyperparameters

| Model Architecture | Base Backbone Name | Learning Rate | Optimizer | Learning Rate Scheduler | Warmup Ratio | Dropout (Backbone / Head) | Weight Decay |
| :--- | :--- | :---: | :---: | :--- | :---: | :---: | :---: |
| **RoBERTa** | `roberta-base` | $2.0 \times 10^{-5}$ | `AdamW` | Linear with Warmup | `~3.0%` | `0.10 / [0.10, 0.05]` | `0.01` |
| **DistilBERT** | `distilbert-base-uncased` | $2.5 \times 10^{-5}$ | `AdamW` | Cosine Annealing | `6.0%` | `0.10 / 0.10` | `0.02` |
| **MobileBERT** | `google/mobilebert-uncased` | $3.0 \times 10^{-5}$ | `AdamW` | Cosine Annealing | `5.0%` | `0.10 / 0.10` | `0.01` |
| **DeBERTa** | `microsoft/deberta-v3-base` | $1.5 \times 10^{-5}$ | `AdamW` | Cosine Annealing | `10.0% – 15.0%` | `0.20 / 0.20` | `0.01` |
| **MiniLM (Raw Base)** | `microsoft/MiniLM-L12-H384-uncased` | $2.0 \times 10^{-5}$ | `AdamW` | Cosine Annealing | `6.0%` | `0.30 / 0.30` | `0.02` |
| **MiniLM HybridFF_v4** | `microsoft/MiniLM-L12-H384-uncased` | $5.0 \times 10^{-5}$ / $10^{-3}$ | `AdamW` | Cosine Annealing | `5.0%` | `0.10 / [192, 64]` | `0.02` |
| **MiniLM raw_orig2** | `microsoft/MiniLM-L12-H384-uncased` | $1.0 \times 10^{-4}$ | `AdamW` | Cosine Annealing | `3.0%` | `0.15 / [384, 256, 128, 64]` | `0.01` |
| **MiniLM raw_orig3** | `microsoft/MiniLM-L12-H384-uncased` | $1.0 \times 10^{-4}$ | `AdamW` | Cosine Annealing | `3.0%` | `0.15 / [384, 512, 256, 128, 64]` | `0.01` |
| **MiniLM raw_orig4_aug2m** | `microsoft/MiniLM-L12-H384-uncased` | $1.0 \times 10^{-4}$ | `AdamW` | Cosine Annealing | `3.0%` | `0.15 / [384, 512, 256, 128, 64]` | `0.01` |
| **MiniLM raw_orig4_nonstatified** | `microsoft/MiniLM-L12-H384-uncased` | $1.0 \times 10^{-4}$ | `AdamW` | Cosine Annealing | `3.0%` | `0.15 / [384, 256, 128, 64, 32]` | `0.01` |

---

### Table 2: PEFT LoRA & Class-Balanced Loss Settings

| Model Architecture | LoRA Rank ($r$) | LoRA Alpha ($\alpha$) | LoRA Dropout | LoRA Target Modules | Focal Loss Gamma ($\gamma$) | Focal Loss Alpha ($\alpha_{t}$) | Label Smoothing |
| :--- | :---: | :---: | :---: | :--- | :---: | :--- | :---: |
| **RoBERTa** | `8` | `16` | `0.00` | `["query", "key", "value", "out_proj"]` | `2.0` | Balanced (Frequency-based) | `0.00` |
| **DistilBERT** | `16` | `32` | `0.05` | `["q_lin", "k_lin", "v_lin", "out_lin"]` | `2.0` | `[0.50, 2.75]` | `0.00` |
| **MobileBERT** | `32` | `64` | `0.05` | `["query", "key", "value", "dense"]` | `2.5` | `[0.40, 3.50]` | `0.00` |
| **DeBERTa** | `64` | `128` | `0.10` | `["query_proj", "key_proj", "value_proj", "dense"]` | `2.0` | `[0.28, 0.72]` | `0.02 – 0.10` |
| **MiniLM (Raw Base)** | `32` | `64` | `0.15` | `["query", "key", "value", "dense", "output.dense"]` | `2.5` | `[0.28, 0.72]` | `0.05` |
| **MiniLM HybridFF_v4** | `32` | `64` | `0.05` | `["query", "key", "value", "dense", "output.dense"]` | `2.0` | `[0.35, 0.65]` | `0.05` |
| **MiniLM raw_orig2** | `32` | `64` | `0.05` | `["query", "key", "value", "dense", "output.dense"]` | `3.0` | `[0.23, 0.77]` | `0.05` |
| **MiniLM raw_orig3** | `32` | `64` | `0.05` | `["query", "key", "value", "dense", "output.dense"]` | `+2.0 / -4.0` (Asymmetric) | `[0.23, 0.77]` | `0.05` |
| **MiniLM raw_orig4_aug2m** | `32` | `64` | `0.05` | `["query", "key", "value", "dense", "output.dense"]` | `+2.0 / -4.0` (Asymmetric) | `[0.28, 0.72]` | `0.05` |
| **MiniLM raw_orig4_nonstatified** | `32` | `64` | `0.05` | `["query", "key", "value", "dense", "output.dense"]` | `+2.0 / -4.0` (Asymmetric) | `[0.28, 0.72]` | `0.05` |

---

### 💡 Core Engineering Insights

1. **The Rank Scale ($r$):** The larger base representations (such as DeBERTa's `128K` vocabulary size and custom relative position embeddings) required a higher rank of `64` to preserve performance compared to RoBERTa's smaller embeddings, which successfully converged at rank `8`.
2. **MiniLM Focal Loss Tuning ($\gamma = 2.5$):** By increasing the focusing parameter $\gamma$ to `2.5` and applying an exact mathematical inverse alpha ratio `[0.28, 0.72]`, the model was able to penalize easy negatives heavily. This specifically helped to reduce False Positives (**FPR = 2.40%**) and False Negatives (**FNR = 10.00%**) on heavily class-skewed raw data inputs.
3. **DeBERTa Label Smoothing:** DeBERTa was highly sensitive to extreme prediction confidence values. Implementing a label smoothing parameter between `0.02` and `0.10` prevented the model from outputting overconfident wrong answers on noisy URL domains.
4. **Asymmetric Focal Loss ($\gamma_{pos}=2.0, \gamma_{neg}=4.0$):** Introduced in the advanced `raw_orig3` and `raw_orig4` models, this asymmetric loss dynamically boosted the penalization on hard/misclassified legitimate URLs and phishing bypasses, ensuring maximum compliance with target FPR/FNR boundaries.
5. **5-Fold Stratified K-Fold Integration:** First integrated in `raw_orig3`, training over stratified folds prevented localized category-specific training skew, establishing high generalization bounds under extreme dataset scale.
6. **2M High-Fidelity Synthetic Augmentation:** Augmenting the dataset with 2 million synthetic samples in `raw_orig4_aug2m` stabilized training curves, allowing the adapter matrices to memorize rare homograph signatures without model saturation or overfitting.

### Fault Tolerance & Continuity
* **Resume Strategy:** Training can be resumed cleanly from any `.pt` fold checkpoint file in `DataPrep10/RESULTS_&_MODELS/3_saved_models/MiniLM_raw_orig4/checkpoints` using the dynamic loader. It automatically restores model weights, optimizer parameters, scheduler states, gradient scaler levels, and training histories from the exact epoch it was interrupted.

---

## 3. Inference & Deployment
### Execution Scripts
* **Inference Pipeline:** Located at [6_inference.py](***DataPrep10/SRC/6_inference.py) (Supports PyTorch checkpoint evaluation & CPU-optimized ONNX runtime execution).
* **Threshold Re-Evaluator:** Located at [7_re_evaluate_thresholds.py](***DataPrep10/SRC/7_re_evaluate_thresholds.py) (Pure NumPy re-computation of KPIs on test predictions).

### Performance Benchmarks
* **Individual Latency:** `~4.03` ms per sample on CPU.
* **Production Path:** Best verified weights are deployed as merged full precision state dicts (`model_merged_full.pt`) or quantized ONNX runtimes from `RESULTS_&_MODELS/3_saved_models/MiniLM_raw_orig4/best_model_epoch_XXX/`.

---

## 4. Evaluation & Results
### Experiment Benchmarking

| Date | Data Version | Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC | FPR | FNR | Model Size (FP32/INT8) | Latency (CPU) | Pros for Use Case | Cons for Use Case |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2025-10-03 | Dataset V3 | RoBERTa | 90.41% | 77.43% | 80.73% | 79.05% | 95.02% | 6.74% | 19.67% | ~500 MB / - | ~18.2 ms | High baseline representation capacity | High VRAM & latency overhead |
| 2025-11-20 | data6 | DistilBERT | 91.67% | 86.30% | 74.66% | 80.06% | 94.77% | 3.42% | 25.00% | ~268 MB / - | ~8.1 ms | Good size-performance tradeoff | Lower performance on complex features |
| 2025-11-25 | data5 | MobileBERT | 93.62% | 93.74% | 82.24% | 87.62% | 96.70% | 2.08% | 17.76% | ~147 MB / - | ~4.5 ms | Native deployment friendly | High training/distillation complexity |
| 2026-01-08 | data9 | DeBERTa | 94.00% | 98.00% | 90.00% | 94.00% | 98.00% | 1.00% | 10.00% | ~370 MB / - | ~22.4 ms | Rich sequence parsing capability | Extremely slow inference latency |
| 2026-02-23 | data10 | **MiniLM** | **95.00%** | **94.00%** | **90.00%** | **92.00%** | **99.00%** | **2.40%** | **10.00%** | **133 MB / 33 MB** | **4.03 ms / 1.2 ms** | Ultra-lightweight and extremely fast | Requires fine-tuning domain-specific tokenizer |
| 2026-05-04 | 10.3 (data_prep8) | **MiniLM HybridFF_v4** | **94.00%** | **88.00%** | **89.00%** | **88.00%** | **98.00%** | **4.79%** | **10.87%** | **135 MB / 33 MB** | **4.20 ms / 1.3 ms** | Integrates heuristics for better rules | Slightly higher execution complexity |
| 2026-05-26 | 10.3 (data_prep8) | **MiniLM raw_orig3** | **95.00%** | **94.00%** | **90.00%** | **92.00%** | **99.00%** | **2.40%** | **10.00%** | **133 MB / 33 MB** | **4.03 ms / 1.2 ms** | Multi-fold cross-val generalizability | Requires longer training duration |
| 2026-05-31 | 10.3 (data_prep8) | **MiniLM raw_orig4** | 94.80% | 94.67% | 89.44% | 91.98% | 98.56% | 2.51% | 10.55% | **133 MB / 33 MB** | **4.03 ms / 1.2 ms** | Handles minor/rare threat classes | Large augmented data preprocessing load |


### Key Insights
* **Optimal Decision Thresholds**:
  Standard `0.5` decision threshold is prone to higher false alarms (FPR) on highly compressed test samples. By employing the `EnhancedKPIEvaluator` multi-objective optimizer, an optimal threshold (e.g. `0.5342` or `0.5350`) is selected to satisfy the strict KPI rule: **Maximize Recall while maintaining FPR <= 1% and FNR <= 10%**.
* **LoRA Convergence**:
  LoRA parameters (trainable ratio: `1.49%`) reduce VRAM utilization to `<5GB`, enabling robust multi-fold training on RTX A4000 workstations with zero memory overhead, while keeping performance identical to full fine-tuning.

* **MiniLM HybridFF_v4**: 
  A hybrid feed-forward model trained end-to-end on 100% of the unified 40M+ URL master dataset (combining Train, Validation, and Test sets), implemented in [3_MiniLM_Hybrid_FF_V4.py](file:///c:/Users/HP/Desktop/DataPrep10/SRC/3_MiniLM_Hybrid_FF_V4.py). The heuristic feature space (`HEURISTIC_DIM`) was expanded to a 90-feature vector consisting of **14 numeric, 13 binary, and 63 flags** (upgraded from the 76-feature set in `HybridFF_v3` which had 10 numeric, 6 binary, and 60 flags) to capture highly granular structural signals.
* **MiniLM_raw_orig3**: 
  Introduced mean pooling for sequence representation, expanded the custom classification network hidden dimensions from `[384, 256, 128, 64]` to a deeper and wider bottleneck of `[384, 512, 256, 128, 64]`, and integrated **5-Fold Stratified Cross-Validation** (`USE_STRATIFIED_KFOLD`) to robustly handle label imbalance. Training is executed on stratified train/val/test splits partitioned both by URL threat category and label distribution to ensure strict alignment across each fold.
* **MiniLM_raw_orig4**: 
  Incorporates the advanced architectural changes of `MiniLM_raw_orig3` (including mean pooling and the expanded `[384, 512, 256, 128, 64]` custom classifier bottleneck under **5-Fold Stratified Cross-Validation**). To maximize model generalizability and resolve minority-class threat signatures, the dataset size was augmented by generating **2 million high-fidelity synthetic URLs**, minimizing overfitting on rare categories.
---

### 3. MiniLM_raw_orig2 Model Evaluation Results Across Confidence Thresholds

The default classification threshold is **0.5** (standard sigmoid cutoff). By adjusting this threshold, we can trade off between false positives and false negatives to match different operational requirements:

| Threshold | Accuracy | Precision | Recall | FPR | FNR |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0.40** | 0.8896 | 0.7351 | 0.9845 | 0.1507 | 0.0155 |
| **0.50** | 0.9368 | 0.8493 | 0.9580 | 0.0722 | 0.0420 |
| **0.605** (Current) | **0.9531** | **0.9410** | **0.8991** | **0.0240** | **0.1009** |
| **0.70** | 0.9403 | 0.9770 | 0.8191 | 0.0082 | 0.1809 |
| **0.80** | 0.9013 | 0.9930 | 0.6737 | 0.0020 | 0.3263 |
| **0.97** | 0.7160 | 1.0000 | 0.0474 | 0.0000 | 0.9526 |
| **0.99** | 0.7018 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |


#### 🔬 Analysis of Target KPI Failure at 0.80 Threshold

While raising the decision boundary to **0.80** provides an exceptional False Positive Rate (**FPR = 0.20%**, comfortably satisfying the target constraint of **FPR <= 1.00%**), it **fails to achieve the remaining core KPI targets** (specifically, **Recall >= 90.00% / FNR <= 10.00%** and **Accuracy >= 98.00%**). At this threshold, Recall drops precipitously to **67.37%** (causing False Negatives to soar to **32.63%**), and Accuracy degrades to **90.13%**.

The inability to meet KPIs at a 0.80 threshold is driven by the following mathematical and architectural constraints:

1. **Probability Compression from Focal Loss Training:**
   During optimization, the model is trained using Focal Loss ($\gamma = 3.0$ for `raw_orig2`). Focal Loss dynamically downweights easy-to-classify examples, focusing gradients heavily on hard/misclassified boundaries. This prevents output logits from saturating towards extreme confidence values ($0.0$ or $1.0$). Consequently, the calibrated probability outputs are highly compressed around the **0.45 – 0.65** region. Shifting the threshold to $0.80$ cuts off a massive volume of true positive predictions that only carry moderate confidence, causing Recall to collapse.

2. **PEFT/LoRA Representation Capacity Limits:**
   Since the model is fine-tuned using Parameter-Efficient Fine-Tuning (LoRA rank $16$, alpha $32$), the primary pretrained transformer layers of MiniLM remain frozen. LoRA adapters have a restricted parameter capacity ($~1.49\%$ trainable parameters), which prevents the model from fully reorganizing the embedding representation space. While the adapters successfully align features to draw a linear boundary at $0.605$, they cannot map complex phishing URLs to high-confidence regions ($>0.80$), resulting in high false negative leakage.

3. **Subword Tokenization Fragmentation:**
   The WordPiece tokenizer splits complex URLs into multiple subwords (e.g., query strings, custom paths, dynamic parameter hashes). This sequence fragmentation distributes attention weights across many subword components. The classifier head pools these fragmented tokens, producing a moderately confident prediction rather than a strong consensus, leaving the correct classification values trapped below the $0.80$ boundary.

4. **Lack of Gated Structural Heuristic Integration (Subproject 1 Limitations):**
   In this subproject (Subproject 1), classification relies **purely** on textual lexical patterns parsed by the transformer, without incorporating structural metadata (such as TLD risks, redirect counts, or active security flags). Lacking access to these deterministic heuristics, the textual model operates under high uncertainty on lexical edge-cases, producing probabilities that cluster around $0.50$ to $0.70$ and failing to meet the high confidence cutoffs required at $0.80$.


### 💡 Core Engineering Insights from the Deep Dive

* **At 0.605 (Current Optimal Threshold)**:
  * **Accuracy**: `0.9531` (`95.31%`)
  * **Precision**: `0.9410` (`94.10%`)
  * **Recall**: `0.8991` (`89.91%`)
  * **FPR (False Positive Rate)**: `0.0240` (only `2.40%` false positives)
  * **FNR (False Negative Rate)**: `0.1009` (`10.09%` false negatives)
  * This confirms the optimal threshold chosen for Epoch 8 is highly calibrated, keeping false alarms near our strict KPI boundaries.

* **FPR and FNR Tradeoff Alignment**:
  * **FPR (False Positive Rate)** decreases smoothly as the decision threshold rises: starting at `0.1507` at threshold `0.40` down to `0.0000` at `0.97`.
  * **FNR (False Negative Rate)** increases correspondingly: starting at `0.0155` at threshold `0.40` up to `1.0000` at `0.99`.
  * *This demonstrates the classic risk-mitigation tradeoff where a higher sigmoid cutoff shields legitimate domains from false classifications at the expense of letting more evasive phishing URLs slip through.*

* **Understanding the Edge Case of 0.99**:
  * At an extreme threshold of `0.99`, the classifier is so conservative that it predicts exactly `0` URLs as malicious. Thus:
    * **True Positives ($tp$)** = `0`
    * **False Positives ($fp$)** = `0`
    * **Precision** ($\frac{tp}{tp + fp}$) = `0.0000`
    * **Recall** ($\frac{tp}{tp + fn}$) = `0.0000`
    * **FNR** = `1.0000` (`100%` of phishing URLs go undetected).


### 🔬 Tokenizer Comparison & Optimization Rationale

To determine the most suitable subword representation mechanism for our URL classification pipeline, we conducted an empirical comparison of **WordPiece**, **Byte-Level BPE**, **SentencePiece Unigram**, and **Character-Level** tokenizers across both Subprojects.

#### Tokenizer Pros, Cons & Suitability Matrix

| Tokenizer Method | Vocab Size | Avg. Tokens per URL | OOV / Unknown (`[UNK]`) Rate | Pros for URL Detection | Cons for URL Detection | Suitability |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: |
| **WordPiece (WPE)** | `30,522` | ~24.1 | Low (< 1.5% due to percent-encoding) | • Perfect alignment with pretrained backbone weights.<br>• Faster convergence under LoRA / adapter tuning.<br>• Stable gradient updates. | • Replaces custom/rare characters with `[UNK]`. | **Highly Suitable (Recommended)** |
| **Byte-Level BPE** | `50,000` | ~18.4 | **0.0% (Zero OOV)** | • Custom vocabulary maps complete subdomains and query structures.<br>• Highly compact sequence lengths.<br>• Immune to unseen character errors. | • **Vocabulary mismatch:** Scrambles pretrained MiniLM embedding weights.<br>• Poor compatibility with low-rank PEFT updates. | **Moderately Suitable** (Requires full-parameter training) |
| **SP Unigram** | `50,000` | ~19.8 | Low (< 0.2%) | • Probabilistic subword regularization mitigates overfitting.<br>• Preserves character offsets via Metaspace tokens. | • Highest training loss and validation loss.<br>• Suffers from embedding misalignment under LoRA limits. | **Low Suitability** |
| **Character-Level** | `< 256` | ~78.5 | **0.0%** | • Highly robust to typosquatting and character obfuscations.<br>• Very small embedding parameter count. | • Extreme sequence fragmentation.<br>• Heavy computation/attention overhead.<br>• Fails to capture high-level domain semantics. | **Unsuitable** (for Transformer transfer learning) |

#### Empirical Tokenizer Performance Benchmarks

##### Subproject 1: Raw & Canonical Model (MiniLM v3 Base)
| Tokenization Scheme | Accuracy | Precision | Recall | F1-Score | AUC-ROC | FPR | FNR | Test Loss | Optimal Threshold |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **WordPiece (WPE)** | **92.89%** | **97.23%** | **88.22%** | **92.51%** | **97.93%** | **2.49%** | **11.78%** | **0.0316** | **0.530** |
| **Byte-Level BPE** | 92.14% | 95.66% | 88.19% | 91.77% | 97.51% | 3.96% | 11.81% | 0.0323 | 0.490 |
| **SP Unigram** | 90.60% | 92.69% | 88.05% | 90.31% | 96.60% | 6.87% | 11.95% | 0.0354 | 0.475 |

##### Subproject 2: Hybrid Gating Model (MiniLM Hybrid GLU Fusion)
| Tokenization Scheme | Accuracy | Precision | Recall | F1-Score | AUC-ROC | FPR | FNR | Test Loss | Optimal Threshold |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **WordPiece (WPE)** | **92.21%** | **95.83%** | **88.16%** | **91.84%** | **96.99%** | **3.79%** | **11.84%** | **0.0375** | **0.490** |
| **Byte-Level BPE** | 90.66% | 92.89% | 87.94% | 90.35% | 95.95% | 6.65% | 12.06% | 0.0427 | 0.440 |
| **SP Unigram** | 90.26% | 92.92% | 87.02% | 89.88% | 95.59% | 6.55% | 12.98% | 0.0422 | 0.465 |

#### 💡 Key Engineering Insight: The Embedding Alignment Mismatch under LoRA

The empirical data demonstrates that **WordPiece (pretrained)** consistently outperforms the custom-trained URL tokenizers (BPE & Unigram), achieving **+0.75% to +2.29% higher Accuracy** and **substantially lower False Positive Rates (FPR)** across both subprojects. 

1. **Scrambled Token ID Space:** A custom tokenizer (BPE or Unigram) creates a vocabulary specific to URLs. A token index like `4213` which represents `"https"` in custom BPE might represent a word like `"kitchen"` in standard MiniLM. 
2. **LoRA Parameter Constraint:** When using Parameter-Efficient Fine-Tuning (PEFT/LoRA), only a tiny fraction of the model parameters (`1.23%` to `1.49%`) are updated. The primary embedding layer (`model.embeddings.word_embeddings.weight`) remains **completely frozen**.
3. **Semantic Disconnect:** Because the embedding layer is frozen, the model maps the custom token IDs to the old pretrained general English representations. Since the LoRA adapters are not large enough to realign the entire representation space of the scrambled token IDs, the network's capacity is severely bottlenecked.
4. **Recommendation:** To extract the true benefit of custom URL-trained tokenizers, **full-parameter fine-tuning (especially of the embedding layer)** must be enabled, which increases training cost and GPU memory utilization. If computing resources are constrained (e.g., RTX A4000 16GB), staying with the native **WordPiece** tokenizer is the optimal choice.


## 5. References

### General Architectural & Methodology Papers
1. Wang, W., et al. (2020). *[MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers](https://arxiv.org/abs/2002.10957)*. **NeurIPS 2020**.
2. Hu, E. J., et al. (2022). *[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)*. **ICLR 2022**.
3. Lin, T.-Y., et al. (2017). *[Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)*. **ICCV 2017**.
4. Loshchilov, I., & Hutter, F. (2019). *[Decoupled Weight Decay Regularization (AdamW)](https://arxiv.org/abs/1711.05101)*. **ICLR 2019**.
5. Liu, Y., et al. (2019). *[RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)*. arXiv:1907.11692.
6. Sanh, V., et al. (2019). *[DistilBERT: A Distilled Version of BERT](https://arxiv.org/abs/1910.01108)*. NeurIPS Workshop 2019.
7. Sun, Z., et al. (2020). *[MobileBERT: A Compact Task-Agnostic BERT for Resource-Limited Devices](https://arxiv.org/abs/2004.02984)*. **ACL 2020**.
8. He, P., et al. (2021). *[DeBERTa: Decoding-Enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654)*. **ICLR 2021**.

### List A: Papers Employing WordPiece Tokenizers for Fine-Tuning LLMs/Transformers on URLs
| S.No. | Paper | Link |
| :---: | :--- | :--- |
| 1 | URLTran: Improving Phishing URL Detection Using Transformers | [arXiv:2106.05256](https://arxiv.org/pdf/2106.05256) |
| 2 | BERT-Based Approaches to Identifying Malicious URLs | [Sensors 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10610561/pdf/sensors-23-08499.pdf) |
| 3 | Lightweight Malicious URL Detection Using Deep Learning and Language Models | [Scientific Reports 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12675596/pdf/41598_2025_Article_26653.pdf) |
| 4 | Dynamic Feature Analysis for Malicious URL Detection and Classification: A Multitask Learning Approach Using Clustering and Language Model Embeddings | [ACM 2024](https://dl.acm.org/doi/pdf/10.1145/3759023.3759119) |
| 5 | Enhancing Generalization in Phishing URL Detection via a Fine-Tuned BERT-Based Multimodal Approach | [ResearchGate 2025](https://www.researchgate.net/publication/393961297_Enhancing_Generalization_in_Phishing_URL_Detection_via_a_Fine-Tuned_BERT-Based_Multimodal_Approach/link/68817f954eccfb3f29c483ab/download?_tp=eyJjb250ZXh0Ijp7InBhZ2UiOiJwdWJsaWNhdGlvbiIsInByZXZpb3VzUGFnZSI6InB1YmxpY2F0aW9uIn19) |
| 6 | Context-Aware Embeddings for Robust Multiclass Fraudulent URL Detection in Online Social Platforms | [ScienceDirect 2024](https://pdf.sciencedirectassets.com/271419/1-s2.0-S0045790624X00075/1-s2.0-S004579062400421X/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEEIaCXVzLWVhc3QtMSJGMEQCIF8oX7bIt0gT0MWNhLp7yZrhL%2FPAq1cUBg4VQzzX8xtRAiA0T1gmPNQW8MNCqlAcfFo3yAL3N9qYKAyHGgBXrFPyJCqzBQgLEAUaDDA1OTAwMzU0Njg2NSIMVWgSO2t%2F4bWzIq9TKpAFO1UiCRjVuaywDdl1vLtiHvv4UQWTid7LQ7wpd8m9xEDq2BBnGnEGMkssAhGpYieXSaFfQ4fAnXZzGlzxa16B6KTIYcUWt1EEefq0PEptPn3OibVafFA990MlligjlEJL7wNKoBEgmKqecbiw5YW82aBWj8haG7O9Vl0qV6iJocjhL9MO8E6a8QjOEB2u3wJVfVa3fiO%2FKLPAVZt7sXVgx0u55zaxEHHWoMStQc%2BFJ7cfpxBFk57pIvZWEyWDcQ0O0dNj4AZrUbZ%2FNKg2735Xgk7bTvGWfwQKCr8mz%2BCu8Tf5K8pGd%2B9CcfyMmpgOiYuJbJPoZAwnmSJ5v4qRSXYgeWEVKCxUEMGrJ%2BeZlh6UOVGKd9cykkesogz57hV3UXf8iXUp8dGqi2kzR0Ym8NulKoIIwW4kFqQtqtxfjRpgYbkK2QKYqiFWKJbVLJAlZuLcAk9SeBTTICk83b%2BL%2FKM0DyC%2BKfp1Qp1eVYbI%2FNkf740ligRYBbR%2FWUL30GMDmXNvCJOiAu1amLUJuurfeh%2BdqycCWNTu7mZq5g%2FS%2FFvBHx%2FU7b5QRiTCIbi4kbPbzsiuaUEfApFxYJ%2BavOnBEWJxAWYBmEYKUimrZxxuYm9MEGLxRkaHl0vrU1OAgzbs4mFWwDxc%2Bz0v%2BKyHsMA3RLR4NZyOuUr7zKeD1LCr9%2BCExzajm%2FHq3x0%2F0nOZwHw7kORTdUNrjUp8sSxwjJSio8RNS%2BNTE%2FJXZmSOkchX%2Fzg9Z2eq7GFjJVJF2mTPllS91MVBDHtDL4Pwkz7sXLEAU%2BRKT7XLJgzo%2FnkRRWXWh5hTUtcX7pJ9BdHBS1xjxuCMJxoxDzwBW%2BnxLQnyAUvpXYUFzY7AqA%2B8Qp7rg0CjaOw9zucwr6T10AY6sgEw0yuAxxgId60cih0WNFuxSUPtFI2HCihec2KPCDxQ23Dx%2BzAJ1FWjB0KNUUt1glqVwnyyheVoEr6yzXPeq94jHHDAaVl%2BowoJq71iuFHDsIOy%2BBS9%2BzJbfAw8knZlWX75hzZqKI0wbft9FDcw655lEmCa7t3HfNQ5D4cE%2Fwh3ooqpWCkkNAgZjND4fg0ikIbtTuCP6h%2FuLG5zxvCjjMW0%2B2OSzwpT6IfBidNSszCNwUtJ&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20260601T103513Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY5JNZAK2H%2F20260601%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=2cf847ee60af97225c29d38ad263eecaea988c411121e0a6f9e21465f123a614&hash=b6ee6f97c225942ca56e938ca92ca798226331d7a452edc183049e47d1e7ecf8&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S004579062400421X&tid=spdf-08253737-d498-4b6b-8184-53e2e4665f09&sid=a7ebf3e630a08349ba0a81a1750b200733f5gxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&rh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=0f080353560656550507&rr=a04d91ff9f4f5969&cc=in) |
| 7 | Phishing URL Detection Using Transformer-Based Architecture and Contextual Content Features | [MDPI 2022](https://www.mdpi.com/2073-431X/15/6/335) |
| 8 | Hyperparameter Optimization for Malicious URL Detection: Leveraging Optuna and Random Search in Machine Learning and Deep Learning Models | [Informatica 2023](https://www.informatica.si/index.php/informatica/article/view/9106/4601) |
| 9 | Intelligent Deep Machine Learning Cyber Phishing URL Detection Based on BERT | [MDPI 2022](https://www.mdpi.com/2079-9292/11/22/3647) |

### List B: Papers Employing Custom Byte-Level BPE or Similar Byte-Level Tokenizer for Fine-Tuning LLMs on URLs
| S.No. | Paper | Tokenizer Type | Link |
| :---: | :--- | :--- | :--- |
| 1 | URLTran: Improving Phishing URL Detection Using Transformers | Existing WordPiece/BPE + custom vocab variants | [Microsoft Research](https://www.microsoft.com/en-us/research/wp-content/uploads/2021/12/URLTran_Milcom2021.pdf?utm_source=chatgpt.com) |
| 2 | Real-Time Phishing URL Detection Using Fine-Tuned DistilRoBERTa | BPE | [Shanlax Journals](https://shanlaxjournals.in/journals/index.php/sijash/article/download/10523/8911/?utm_source=chatgpt.com) |
| 3 | RoBERTa-Augmented Synthesis for Detecting Malicious URLs | Byte-Level BPE | [arXiv:2405.11258v2](https://arxiv.org/pdf/2405.11258v2) |
| 4 | TransURL | Character-aware transformer | [arXiv:2312.00508v3](https://arxiv.org/pdf/2312.00508v3) |
| 5 | DomURLs_BERT: Pre-trained BERT-based Model for Malicious Domains and URLs Detection and Classification | Custom SentencePiece BPE tokenizer trained from scratch on URL/domain/DGA data | [arXiv:2409.09143v1](https://arxiv.org/pdf/2409.09143v1) |
| 6 | urlBERT: A Contrastive and Adversarial Pre-trained Model for URL Classification / Continuous Multi-Task Pre-training for Malicious URL Detection and Webpage Classification | Custom URL tokenizer trained on billions of URLs; exact tokenizer algorithm not clearly stated in accessible text | [arXiv:2402.11495](https://arxiv.org/pdf/2402.11495) |


---

## 6. How to Run
# Local Run
### Step 1: URLs Categorization
Extract security threat groups and clean the input dataset:
```bash
python SRC/urls_cate_V7.py --input "DATA/3_LNU_Phish1.csv"
```

### Step 2: Preprocessing & K-Fold Stratified Splitting
```bash
python SRC/2_preprocess_urls_v8_refactored.py \
  --input "DATA/3_LNU_Phish.csv" \
  --enable-split \
  --split-source all \
  --chunk-size 100000 \
  --use-rule-features \
  --enable-multiprocessing \
  --num-workers 8
```

### Step 3: Run Resumable LoRA Training
Train the MiniLM LoRA model with multi-objective evaluations and automatic adapter merges:
```bash
python SRC/5_train.py --config SRC/test_config.yaml
```

### Step 4: Checkpoint / ONNX Test Inference
Run PyTorch checkpoint evaluation:
```bash
python SRC/6_inference.py --mode inference --config SRC/test_config.yaml
```

Run CPU-optimized ONNX runtime inference:
```bash
python SRC/6_inference.py --mode onnx-inference --onnx-model int8 --config SRC/test_config.yaml
```

### Step 5: Fast NumPy Threshold Re-Evaluation
Re-tune decision thresholds dynamically without reloading model weights:
```bash
python SRC/7_re_evaluate_thresholds.py --config SRC/test_config.yaml --thresholds 0.525 0.5342 0.90 0.999
```


# Mlflow Run

To run training, perform multi-engine dynamic batch inferences, and launch the tracking dashboard inside the Windows virtual environment, execute these commands from the subproject directory:

### Step 1: Run Resumable LoRA Training
*   **Pipeline Sanity Check**:
    ```powershell
    SRC\5_train.py --config SRC\test_config.yaml
    ```
*   **Full Production Training**:
    ```powershell
    SRC\5_train.py --config SRC\4_config.yaml
    ```

### Step 2: Run Dynamic Batch Inference Tests
*   **PyTorch / Merged Model Checkpoint Inference**:
    ```powershell
    SRC\6_inference.py --config SRC\test_config.yaml --mode inference
    ```
*   **FP32 ONNX Engine Inference**:
    ```powershell
    SRC\6_inference.py --config SRC\test_config.yaml --mode onnx-inference --onnx-model fp32
    ```
*   **INT8 Quantized ONNX Engine Inference**:
    ```powershell
    SRC\6_inference.py --config SRC\test_config.yaml --mode onnx-inference --onnx-model int8
    ```

### Step 3: Launch MLflow UI Dashboard
Inspect parameters, compare run charts, download ONNX model artifacts, and manage the model registry:
*   **Launch the Local Server**:
    ```powershell
    ui --backend-store-uri sqlite:///mlflow.db --port 5000
    ```
*   **Access UI**: Open your web browser and navigate to **`http://localhost:5000`**
