# Project: Advanced Hybrid GLU Fusion Phishing URL Detection & Modular Classification Pipeline

This document serves as the centralized engineering runbook, documenting the multi-modal data pipeline, dual-tower Gated Linear Unit (GLU) fusion architecture, temperature scaling logit calibration, experiment tracking matrix, production benchmarks, and E2E execution paths for the Hybrid GLU Fusion Phishing URL Detection pipeline.

---

## 1. Data Pipeline & Distribution
### File Registry
* **Raw Dataset Location:** `DATA/v2.0/` (Master 50M+ URL preprocessed corpus)
* **Processed Dataset Location:** `RESULTS_&_MODELS/2_preprocess_urls_output/`
* **Train / Val / Test Splits:** `urls_hybrid_train.csv` (80%), `urls_hybrid_val.csv` (10%), `urls_hybrid_test.csv` (10%)
* **Processing Script:** `SRC/3_MiniLM_Hybrid_FF_V4.py` (Hybrid model)
* **Refactored Codebase Library:** Modular package under `SRC/core/` and orchestrator scripts inside `SRC/`.

### Version Log & Processing Stats
* **Active Data Version:** `10.3`
* **Data Processing Time:** `[6-7 hrs / 37887508 URLs]`
* **Feature Distributions:** 
### Heuristic Feature Space & Normalization Z-Scores
The heuristic feature space consists of a highly granular **90-feature vector** specifically mapped to capture structural and behavioral cyber-intelligence indicators:
* **Numeric Features (14 columns)**: `h_flags_count`, `h_severity_score`, `h_entropy_url`, `h_digit_ratio`, `h_path_depth`, `h_url_length`, `h_domain_length`, `h_subdomain_count`, etc. 
* **Binary Features (13 columns)**: `h_is_ip_host`, `h_has_https`, `h_tld_risk_normal`, `h_tld_risk_high`, `h_tld_risk_critical`, `h_has_punycode`, `h_mixed_script`, etc.
* **Category Flags (63 columns)**: Heuristics flags (`hF_NONE`, `hF_TYPO`, `hF_SHOR`, `hF_OBFU`, `hF_DATA`, `hF_NEST`, `hF_CAPT`, etc.) auto-detected starting with prefix `hF_`.
* **Tabular Z-Score Normalization**: Continuous numeric features are normalized via Z-score parameters ($\mu$, $\sigma$) computed **strictly on the training split** to prevent data leakage. These stats are persisted to [normalization_stats.json](file:///c:/Users/HP/Desktop/DataPrep8/2_MiniLM_Hybrid_FF_V4/RESULTS_&_MODELS/3_saved_models/MiniLM_HybridFF_v4_Sanity/best_model_epoch_001/normalization_stats.json) for runtime loading during inference.
* **Leaky Categorical Dropout**: Leak-prone categories (e.g. `h_primary_category`) are cleanly excluded during training.

---

## 2. Model Training Log
### Centralized Configurations
* **Dynamic Config File**: All hyperparameters and hardware settings are managed dynamically via [4_config.yaml](file:///c:/Users/HP/Desktop/DataPrep8/2_MiniLM_Hybrid_FF_V4/SRC/4_config.yaml) (Production) and [test_config.yaml](file:///c:/Users/HP/Desktop/DataPrep8/2_MiniLM_Hybrid_FF_V4/SRC/test_config.yaml) (CPU Sanity Test).

* **Data Split Strategy:** Strict 80/10/10 train/val/test split across all iterations:
  * **1. OVERALL SPLIT STATISTICS**
    * **Total Samples:** 37,887,508
    * **Train Set:**     30,276,686 (79.91%)
    * **Val Set:**        3,806,230 (10.05%)
    * **Test Set:**       3,804,592 (10.04%)
### Architectural Breakdown
* **Tower A (Text Encoder)**: Processes raw URL text via `MiniLM-L12-H384-uncased` with PEFT LoRA (Rank `32`, Alpha `64`, Dropout `0.05`) targeting query, key, value, and projection modules. Reduces trainable parameters to `1.23%` (`421,762` trainable parameters out of 34M total). Produces a dense `384-dimensional` embedding.
* **Tower B (Heuristics MLP)**: Processes the Z-score normalized continuous and binary 90-feature heuristic array through a compact network (`90 → 256 → LayerNorm → GELU → Dropout → 192`), outputting a `192-dimensional` continuous semantic vector.
* **Gated GLU Fusion Gate**: Fuses semantic representations. Operates in logit space:
  $$concat = [text\_emb \; ; \; feat\_emb]$$
  $$gate = \sigma(W_{gate} \cdot concat + b_{gate}) \in \mathbb{R}^{384}$$
  $$value = \tanh(W_{val} \cdot concat + b_{val}) \in \mathbb{R}^{384}$$
  $$fused\_output = LayerNorm(gate \odot value) \in \mathbb{R}^{384}$$
* **Bottleneck Classifier Head**: Maps the fused `384-dimensional` GLU vector through dense bottlenecks (`384 → 192 → LayerNorm → GELU → Dropout → 64 → 2`) to output binary classification logits.
* **Stable Focal Loss**: Handled via numerically-clamped Focal Loss:
  $$FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
  Stabilized with label smoothing ($0.05$) and logit clamping (`[-10.0, 10.0]`) to protect against gradient explosions under FP16 Mixed Precision (AMP) scaling.

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
* **Resume Strategy**: Intermediate training checkpoints are serialized as `checkpoint_epoch_XXX.pt`. Resumption automatically restores model states, Peft layers, AdamW weights, learning rate cosine schedulers, gradient scalers, and full metrics history.
* **MLOps continuity**: The active `mlflow_run_id` is saved inside checkpoints to allow training pipelines to cleanly resume logging directly into the same tracking session upon hardware restarts.

---

## 3. Inference & Deployment
### Execution Scripts
* **Inference Pipeline**: Located at [6_inference.py](file:///c:/Users/HP/Desktop/DataPrep8/2_MiniLM_Hybrid_FF_V4/SRC/6_inference.py) (Supports PyTorch checkpoint evaluation & CPU-optimized ONNX runtime execution).
* **Threshold Re-Evaluator**: Located at [7_re_evaluate_thresholds.py](file:///c:/Users/HP/Desktop/DataPrep8/2_MiniLM_Hybrid_FF_V4/SRC/7_re_evaluate_thresholds.py) (Pure NumPy re-computation of KPIs on test predictions).

### Multi-Engine Production Exporter
* **PyTorch PEFT Weight Merge**: Peft adapters are merged directly into transformer weights (`model_merged_full.pt`: `129.70 MB`), enabling fast dependency-free standalone CPU deployment.
* **3-Input Dynamic ONNX**:
  Traces three inputs simultaneously: `input_ids` `(batch, seq)`, `attention_mask` `(batch, seq)`, and `heuristic_features` `(batch, 90)`. Generates dynamic-batch graphs.
* **Dynamic INT8 CPU Quantization**:
  Compresses parameters via 8-bit dynamic quantization (`model_quant_8bit.onnx`: **`32.83 MB`**, a **`74.6%`** size reduction), successfully satisfying our <= 40MB production target with minimal accuracy loss.

---

## 4. Evaluation & Results
### Calibration & Decision Engine (SDE)
To bridge smooth probabilities with hard cybersecurity constraints in the wild, the modular evaluator employs:
1. **L-BFGS Temperature scaling**: Learns validation logit temperature calibration $T$ to minimize Expected Calibration Error (ECE) and NLL:
   $$calibrated\_logits = logits / T$$
2. **Dynamic Risk-logit Boosting ($\lambda$)**: Boosts threat indicators dynamically for high-severity features (Credential harvesting forms, obfuscation, typosquatting) while applying contextual offsets to minimize False Positives (FPR) on trusted categories.
3. **Joint ($\lambda$, base_threshold) Grid Search**: Computes optimized operating points over 1,210 grid points to satisfy strict KPIs: **FPR <= 1% and FNR <= 10%**.

### Benchmarks (Sanity CPU Verification)

| Model Engine | Sample Latency | Accuracy | Precision | Recall | F1-Score | FPR | FNR | Model Size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PyTorch Merged** | `~30.0 ms` | **94.00%** | **88.00%** | **89.00%** | **88.00%** | **4.79%%** | **10.87%** | `129.70 MB` |
| **Quantized INT8 ONNX** | **`19.21 ms`** | **94.00%** | **88.00%** | **89.00%** | **88.00%** | **4.79%%** | **10.87%** | **`32.83 MB`** |

*Throughput Rate on INT8 CPU*: **`52 URLs/sec`** on a single CPU core.

---

## 5. References
1. Wang, W., et al. (2020). *[MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers](https://arxiv.org/abs/2002.10957)*. **NeurIPS 2020**.
2. Hu, E. J., et al. (2022). *[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)*. **ICLR 2022**.
3. Lin, T.-Y., et al. (2017). *[Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)*. **ICCV 2017**.
4. Loshchilov, I., & Hutter, F. (2019). *[Decoupled Weight Decay Regularization (AdamW)](https://arxiv.org/abs/1711.05101)*. **ICLR 2019**.
5. Liu, Y., et al. (2019). *[RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)*. arXiv:1907.11692.
6. Sanh, V., et al. (2019). *[DistilBERT: A Distilled Version of BERT](https://arxiv.org/abs/1910.01108)*. NeurIPS Workshop 2019.
7. Sun, Z., et al. (2020). *[MobileBERT: A Compact Task-Agnostic BERT for Resource-Limited Devices](https://arxiv.org/abs/2004.02984)*. **ACL 2020**.
8. He, P., et al. (2021). *[DeBERTa: Decoding-Enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654)*. **ICLR 2021**.
9. Maneriker, A., et al. (2021). *[URLTran: Improving Phishing URL Detection Using Transformers](https://arxiv.org/pdf/2106.05256)*. **arXiv:2106.05256**.
10. Aljofey, A., et al. (2023). *[BERT-Based Approaches to Identifying Malicious URLs](https://pmc.ncbi.nlm.nih.gov/articles/PMC10610561/pdf/sensors-23-08499.pdf)*. **Sensors 2023**.
11. Abed, A., et al. (2025). *[Lightweight Malicious URL Detection Using Deep Learning and Language Models](https://pmc.ncbi.nlm.nih.gov/articles/PMC12675596/pdf/41598_2025_Article_26653.pdf)*. **Scientific Reports 2025**.
12. Sathish, S., et al. (2024). *[Dynamic Feature Analysis for Malicious URL Detection and Classification: A Multitask Learning Approach Using Clustering and Language Model Embeddings](https://dl.acm.org/doi/pdf/10.1145/3759023.3759119)*. **ACM 2024**.
13. Rathod, S., et al. (2025). *[Enhancing Generalization in Phishing URL Detection via a Fine-Tuned BERT-Based Multimodal Approach](https://www.researchgate.net/publication/393961297_Enhancing_Generalization_in_Phishing_URL_Detection_via_a_Fine-Tuned_BERT-Based_Multimodal_Approach/link/68817f954eccfb3f29c483ab/download?_tp=eyJjb250ZXh0Ijp7InBhZ2UiOiJwdWJsaWNhdGlvbiIsInByZXZpb3VzUGFnZSI6InB1YmxpY2F0aW9uIn19)*. **ResearchGate 2025**.
14. Kumar, A., et al. (2024). *[Context-Aware Embeddings for Robust Multiclass Fraudulent URL Detection in Online Social Platforms](https://www.sciencedirect.com/science/article/pii/S004579062400421X)*. **ScienceDirect 2024**.
15. Butnaru, A., et al. (2022). *[Phishing URL Detection Using Transformer-Based Architecture and Contextual Content Features](https://www.mdpi.com/2073-431X/15/6/335)*. **Computers 2022**.
16. Tufail, M., et al. (2023). *[Hyperparameter Optimization for Malicious URL Detection: Leveraging Optuna and Random Search in Machine Learning and Deep Learning Models](https://www.informatica.si/index.php/informatica/article/view/9106/4601)*. **Informatica 2023**.
17. Al-Sarem, M., et al. (2022). *[Intelligent Deep Machine Learning Cyber Phishing URL Detection Based on BERT](https://www.mdpi.com/2079-9292/11/22/3647)*. **Electronics 2022**.

---

## 6. How to Run

# Local Run

### Step 1: URLs Categorization
Extract security threat groups and clean the input dataset:
```bash
python SRC/urls_cate_V7.py --input "DATA/3_LNU_Phish.csv"
```

### Step 2: Preprocessing & K-Fold Stratified Splitting
```bash
python SRC/2_preprocess_urls_v8_refactored.py \
  --input "DATA/3_LNU_Phish.csv" \
  --enable-split \
  --split-source hybrid \
  --chunk-size 100000 \
  --use-rule-features \
  --enable-multiprocessing \
  --num-workers 8
```

### Step 3: Run LoRA Hybrid Training
Train the multi-modal classification towers, run logit calibration, optimize decision grids, and compile production models:
```bash
python SRC/5_train.py --config SRC/test_config.yaml
```

### Step 4: Run Multi-Engine Test Inference
**Mode 1**: PyTorch Merged FP32 checkpoint
Evaluate the production-ready LoRA merged model in PyTorch:
```bash
python SRC/6_inference.py --mode inference --config SRC/test_config.yaml
```
**Mode 2**: ONNX FP32
```bash
python SRC/6_inference.py --mode onnx-inference --onnx-model fp32 --config SRC/test_config.yaml
```
**Mode 3**: ONNX INT8 Quantized
```bash
python SRC/6_inference.py --mode onnx-inference --onnx-model int8 --config SRC/test_config.yaml
```

### Step 5: Fast NumPy Joint Operating Point Calibration
Re-evaluate and search threshold-lambda parameters offline using saved log-odds sheets:
```bash
python SRC/7_re_evaluate_thresholds.py --config SRC/test_config.yaml --thresholds 0.45 0.50 0.55 --lambdas 0.0 0.5 1.0
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
*   **Access UI**: Open web browser and navigate to **`http://localhost:5000`**
