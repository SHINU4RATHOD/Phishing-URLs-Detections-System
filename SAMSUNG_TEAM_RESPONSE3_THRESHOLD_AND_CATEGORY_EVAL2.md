# 📋 PhishGuard — Model Evaluation Report: Threshold Analysis & Category-Wise Performance

> **Prepared by:** IIT Ropar — Phishing URL Detection Research Team  
> **Date:** April 13, 2026  
> **Model:** MiniLM-L12-H384-uncased + LoRA (Epoch 3 · Preprocessed Canonical URLs · Dataset V10)  
> **Test Dataset:** 3,507,694 samples (2,516,634 Benign · 991,060 Malicious)

---

## Table of Contents

| # | Agenda Item | Status |
| :--- | :--- | :--- |
| [1](#-item-1--model-evaluation-with-updated-decision-thresholds) | Model Evaluation with Updated Decision Thresholds (0.5, 0.90, 0.999) | 📊 Results Provided |
| [2](#-item-2--category-wise-model-evaluation) | Category-Wise Model Evaluation (19 Categories) | 📊 Results Provided |

---

## 📌 Item 1 — Model Evaluation with Updated Decision Thresholds

### 1.1 Background

A decision threshold determines the probability cutoff at which the model classifies a URL as **malicious**. By default, binary classifiers use `0.5`, but this value can be adjusted to tune the trade-off between **False Positives (FP)** and **False Negatives (FN)**.

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│  HOW THE DECISION THRESHOLD WORKS                                            │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  The model outputs a probability P(malicious) ∈ [0.0, 1.0] for each URL.     │
│                                                                              │
│  Classification Rule:                                                        │
│    IF   P(malicious) ≥ threshold  →  Predict MALICIOUS (Label = 1)           │
│    ELSE                           →  Predict BENIGN    (Label = 0)           │
│                                                                              │
│  Trade-off:                                                                  │
│    ┌─────────────────────┬──────────────────────┬───────────────────────┐    │
│    │  Threshold Level    │  FPR (False Alarm)   │  FNR (Missed Attack)  │    │
│    ├─────────────────────┼──────────────────────┼───────────────────────┤    │
│    │  LOW    (e.g. 0.3)  │  ↑ HIGH (many FPs)   │  ↓ LOW  (few FNs)     │    │
│    │  MEDIUM (e.g. 0.5)  │  ■ BALANCED          │  ■ BALANCED           │    │
│    │  HIGH   (e.g. 0.9)  │  ↓ LOW  (few FPs)    │  ↑ HIGH (many FNs)    │    │
│    │  EXTREME(e.g. 0.999)│  ≈ ZERO              │  ≈ TOTAL (all miss)   │    │
│    └─────────────────────┴──────────────────────┴───────────────────────┘    │
│                                                                              │
│  Probability distribution of current model:                                  │
│    Range:  [0.029, 0.970]                                                    │
│    Mean:   0.378                                                             │
│    Median: 0.285                                                             │
│                                                                              │
│  ⚠ NOTE: Max probability = 0.970 means the model NEVER outputs P ≥ 0.999.   │
│  Therefore, threshold = 0.999 results in ZERO detections by design.          │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

### 1.2 Evaluation Results at Three Thresholds

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                THRESHOLD RE-EVALUATION — COMPLETE RESULTS                    │
│                Test Dataset: 3,507,694 samples                               │
│                Benign: 2,516,634 │ Malicious: 991,060                        │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┬──────────────┬──────────────┬──────────────┐               │
│  │   Metric     │  t = 0.500   │  t = 0.900   │  t = 0.999   │               │
│  ├──────────────┼──────────────┼──────────────┼──────────────┤               │
│  │  Recall      │    91.05%    │    44.81%    │     0.00%    │               │
│  │  FPR         │  ❌ 5.54%    │  ✅ 0.05%   │  ✅ 0.00%    │               │
│  │  FNR         │  ✅ 8.95%    │  ❌ 55.19%  │  ❌ 100.0%   │               │
│  ├──────────────┼──────────────┼──────────────┼──────────────┤               │
│  │  TN          │  2,377,175   │  2,515,386   │  2,516,634   │               │
│  │  FP          │    139,459   │      1,248   │          0   │               │
│  │  FN          │     88,739   │    546,917   │    991,060   │               │
│  │  TP          │    902,321   │    444,143   │          0   │               │
│  ├──────────────┼──────────────┼──────────────┼──────────────┤               │
│  │  KPI Pass    │    ❌ NO     │    ❌ NO     │    ❌ NO    │               │
│  └──────────────┴──────────────┴──────────────┴──────────────┘               │
│                                                                              │
│  KPI Targets: Accuracy ≥ 98%, Precision ≥ 95%, Recall ≥ 95%,                 │
│               FPR ≤ 1%, FNR ≤ 10%                                            │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

### 1.3 Detailed Analysis Per Threshold

#### Threshold = 0.50 (Standard Binary Decision)

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│  THRESHOLD = 0.50 — STANDARD CLASSIFICATION BOUNDARY                         │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  STRENGTHS:                                                                  │
│  ✅ FNR = 8.95% — Meets the ≤10% FNR target                                  │
│  ✅ Recall = 91.05% — Strong phishing detection rate                         │
│                                                                              │
│  CONFUSION MATRIX:                                                           │
│  ┌───────────────────────────────────────────────────┐                       │
│  │                   Predicted Benign  Pred Malicious│                       │
│  │  Actual Benign     2,377,175 (TN)    139,459 (FP) │                       │
│  │  Actual Malicious     88,739 (FN)    902,321 (TP) │                       │
│  └───────────────────────────────────────────────────┘                       │
│                                                                              │
│  INTERPRETATION:                                                             │
│  This is the most balanced threshold. It catches 91% of phishing attacks     │
│                                                                              │
│  The FPR of 5.54% is the primary blocker for KPI compliance at this          │
│  threshold.                                                                  │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

#### Threshold = 0.90 (High-Confidence Detection)

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│  THRESHOLD = 0.90 — HIGH-CONFIDENCE CLASSIFICATION                           │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  STRENGTHS:                                                                  │
│  ✅ Precision = 99.72% — Near-perfect; when the model says "phishing",       │
│     it is correct 99.72% of the time                                         │
│  ✅ FPR = 0.05% — Only 1,248 benign URLs are incorrectly flagged             │
│     (far below the 1% KPI target)                                            │
│                                                                              │
│  WEAKNESSES:                                                                 │
│  ❌ FNR = 55.19% — 546,917 phishing URLs evade detection                     │
│  ❌ Recall = 44.81% — More than half of attacks are missed                   │
│                                                                              │
│  CONFUSION MATRIX:                                                           │
│  ┌───────────────────────────────────────────────────┐                       │
│  │                   Predicted Benign  Pred Malicious│                       │
│  │  Actual Benign     2,515,386 (TN)      1,248 (FP) │                       │
│  │  Actual Malicious    546,917 (FN)    444,143 (TP) │                       │
│  └───────────────────────────────────────────────────┘                       │
│                                                                              │
│  INTERPRETATION:                                                             │
│  At t=0.90, the model becomes extremely conservative — it only flags URLs    │
│  that it is ≥90% confident are phishing. This virtually eliminates false     │
│  alarms (FPR = 0.05%) but at a catastrophic cost: more than half of all      │
│  phishing URLs are missed (FNR = 55.19%).                                    │
│                                                                              │
│  This threshold is appropriate ONLY for a first-pass "high confidence"       │
│  filter in a multi-stage detection pipeline, where a secondary system        │
│  catches the remaining 55% of attacks.                                       │
│                                                                              │
│  KEY INSIGHT:                                                                │
│  The model's max probability is 0.970 (not 1.0). This means the gap          │
│  between the model's confidence ceiling and the 0.90 threshold is only       │
│  0.07 — very narrow. Many genuinely malicious URLs have P values in the      │
│  0.50–0.89 range and are being missed.                                       │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

#### Threshold = 0.999 (Ultra-Conservative / Infeasible)

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│  THRESHOLD = 0.999 — ULTRA-CONSERVATIVE (INFEASIBLE FOR THIS MODEL)          │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  RESULT: TOTAL DETECTION FAILURE                                             │
│                                                                              │
│  ❌ Recall    = 0.00%  — Zero phishing URLs detected                         │
│  ❌ Precision = 0.00%  — Undefined (no positive predictions made)            │
│  ❌ FNR       = 100%   — All 991,060 phishing URLs evade detection           │
│  ✅ FPR       = 0.00%  — Zero false alarms (trivially achieved)              │
│                                                                              │
│  CONFUSION MATRIX:                                                           │
│  ┌──────────────────────────────────────────────────┐                        │
│  │                   Predicted Benign  Pred Malicious│                       │
│  │  Actual Benign     2,516,634 (TN)          0 (FP) │                       │
│  │  Actual Malicious    991,060 (FN)          0 (TP) │                       │
│  └──────────────────────────────────────────────────┘                        │
│                                                                              │
│  ROOT CAUSE:                                                                 │
│  The model's maximum predicted probability is 0.970313.                      │
│  Since 0.970 < 0.999, NO URL in the entire test set exceeds this threshold.  │
│  The model effectively predicts EVERYTHING as benign — it becomes a          │
│  "pass-all" gate.                                                            │
│                                                                              │
│  CONCLUSION:                                                                 │
│  Threshold 0.999 is INFEASIBLE for this model architecture. The model's      │
│  probability calibration does not produce values near 1.0, making this       │
│  threshold operationally equivalent to disabling the detector entirely.      │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

### 1.5 Key Findings & Recommendations

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│  KEY FINDINGS                                                                │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. NO SINGLE THRESHOLD meets all KPIs simultaneously.                       │
│     • t=0.5 meets FNR target but fails FPR                                   │
│     • t=0.9 meets FPR target but fails FNR                                   │
│     • t=0.999 is infeasible (model max probability = 0.970)                  │
│                                                                              │
│  2. The AUC-ROC = 98.02% is CONSTANT across all thresholds (as expected).    │
│     This confirms the model has strong class separability; the challenge     │
│     is finding the right operating point on the ROC curve.                   │
│                                                                              │
│  3. The probability distribution reveals a calibration gap:                  │
│     • Mean P = 0.378, Median P = 0.285                                       │
│     • Max P = 0.970 (never reaches near 1.0)                                 │
│     • This suggests the model is UNDER-CONFIDENT on phishing URLs.           │
│                                                                              │
│  4. The optimal threshold for this model lies in the range [0.50, 0.55].     │
│     • The training pipeline found t=0.525 as the best compromise             │
│     • At t=0.525: FPR=4.52%, FNR=10.16%                                      │
│                                                                              │
├──────────────────────────────────────────────────────────────────────────────┤
│  RECOMMENDATIONS                                                             │
├──────────────────────────────────────────────────────────────────────────────┤
│  1. HYBRID GLU FUSION MODEL (Next-Generation Architecture)                   │
│     The current model uses ONLY text embeddings (MiniLM). The hybrid model   │
│     fuses text + 76 heuristic features via GLU gate. This wider decision     │
│     boundary is expected to improve probability calibration and push         │
│     both FPR and FNR below their targets simultaneously.                     |
|                                                                              │
│  1. TEMPERATURE SCALING (Post-hoc Calibration)                               │
│     Apply Platt scaling or temperature scaling to the logits so that         │
│     P(malicious) values fully span [0.0, 1.0], enabling finer threshold      │
│     tuning without the current 0.970 ceiling.                                │
│                                                                              │
│  2. HARD NEGATIVE MINING                                                     │
│     Fine-tune specifically on the 139,459 false positives and 88,739         │
│     false negatives from t=0.5 to sharpen the decision boundary for          │
│     these edge cases.                                                        │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 📌 Item 2 — Category-Wise Model Evaluation

### 2.1 Overview

The model was evaluated on **17 URL categories** using stratified samples from the test set. Each category highlights a specific phishing tactic or URL structure, enabling identification of per-category strengths and weaknesses.

> **Model:** MiniLM-L12-H384 + LoRA · Epoch 3 · Preprocessed Canonical URLs · Dataset V10  
> **Sample Size:** Up to 10,000 benign + 10,000 malicious per category (where available)

---

### 2.2 Category-Wise Performance Table

```text
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              CATEGORY-WISE MODEL EVALUATION — COMPLETE RESULTS                                       │
│                                                                                                                      │
│  Model: MiniLM-L12-H384 + LoRA · Epoch 3 · Preprocessed Canonical URLs · Dataset V10                                 │
│  Threshold: 0.525 (Optimal from training)                                                                            │
├──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                                      │
│  #    Category (Benign/Malicious)                 Recall   FNR     FPR     TN          FP        FN        TP        │
│  ───  ─────────────────────────────────────────  ──────  ──────  ──────  ──────────  ────────  ────────  ──────────  │
│   1   Cloud_Hosting_Abuse_URL (7599/10000)          96%      4%      9%       6,912       687       433       9,567  │
│   2   HasExcessiveParams (10000/10000)              98%      2%      1%       9,911        89       169       9,831  │
│   3   HasRepeatedSubdomain (501/10000)              93%      7%      1%       9,946        54        37         464  │
│   4   Hex_Encoded_URL (10000/10000)                 99%      1%      3%       9,702       298        77       9,923  │
│   5   IsBrandImpersonation (10000/10000)            95%      5%      4%       9,573       427       531       9,469  │
│   6   IsLanguageSpecific (7154/10000)               97%      3%     12%       6,296       858       286       9,714  │
│   7   IsObfuscatedURL                               99%      1%      7%       9,253       747       130       9,870  │
│   8   IsSessionBased (7394/10000)                   99%      1%      4%       7,123       271       118       9,882  │
│   9   IsSuspiciousKeyword (10000/10000)             97%      3%      6%       9,398       602       304       9,696  │
│  10   Redirect_URL_Open_Redirect (3212/7869)        98%      2%     12%       2,834       378       162       7,707  │
│  11   Structural_Malformation_URL (10000/10000)     99%      1%      4%       9,586       414       126       9,874  │
│  12   Suspicious_TLD_URL                            91%      9%     10%       9,006       994       887       9,113  │
│  13   Suspiciously_Long_Complex_URL                 99%      1%      5%       9,499       501       128       9,872  │
│  14   TypoSquatting_URL                             87%     13%      2%       9,756       244     1,278       8,722  │
│  15   IsDynamicQuery (6577/10000)                   99%      1%      4%       6,310       267        97       9,903  │
│  ───  ─────────────────────────────────────────  ──────  ──────  ──────  ──────────  ────────  ────────  ──────────  │
│  U    Unicode_URL (2/2158)                          89%     11%   100%           0         2       230       1,928   │
│  ───  ─────────────────────────────────────────  ──────  ──────  ──────  ──────────  ────────  ────────  ──────────  │
│  ALL  Full Test Set 99:1 (2519011/25444)            90%     10%      3%   2,438,884    80,127     2,574      22,870  │
│                                                                                                                      │
├──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                                      │
│  COLUMN LEGEND:                                                                                                      │
│    Recall     = TP / (TP + FN)  — How many actual phishing URLs were correctly detected                              │
│    FNR        = FN / (FN + TP)  — False Negative Rate (phishing URLs missed)                                         │
│    FPR        = FP / (FP + TN)  — False Positive Rate (benign URLs wrongly flagged)                                  │
│    TN (L→L)   = True Negatives  — Benign correctly classified as Benign                                              │
│    FP (L→M)   = False Positives — Benign incorrectly classified as Malicious                                         │
│    FN (M→L)   = False Negatives — Malicious incorrectly classified as Benign                                         │
│    TP (M→M)   = True Positives  — Malicious correctly classified as Malicious                                        │
│                                                                                                                      │
│  NOTES:                                                                                                              │
│    • Unicode_URL has only 2 benign samples → FPR = 100% is statistically unreliable                                  │
│    • Full Test Set 99:1 row reflects real-world class imbalance (99% benign : 1% malicious)                          │
│                                                                                                                      │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### 2.3 Performance Tier Classification

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│  TIER 1 — EXCELLENT PERFORMERS (F1 ≥ 97%)                                    │
│  ═══════════════════════════════════════════                                  │
│                                                                              │
│  Category                          F1     FNR     FPR    Notes               │
│  ─────────────────────────────── ────── ─────── ──────── ──────────────────  │
│  HasExcessiveParams                99%    2%       1%    ★ Best overall      │
│  Hex_Encoded_URL                   98%    1%       3%    Near-perfect        │
│  IsSessionBased                    98%    1%       4%    Strong detection    │
│  Structural_Malformation_URL       97%    1%       4%    Strong detection    │
│  Suspiciously_Long_Complex_URL     97%    1%       5%    Strong detection    │
│  Redirect_URL_Open_Redirect        97%    2%      12%    Strong detection    │
│                                                                              │
│  ASSESSMENT: These categories have mature, well-separated distributions.     │
│  The model confidently identifies both benign and malicious patterns.        │
│                                                                              │
├──────────────────────────────────────────────────────────────────────────────┤
│  TIER 2 — GOOD PERFORMERS (93% ≤ F1 < 97%)                                  │
│  ════════════════════════════════════════════                                 │
│                                                                              │
│  Category                          F1     FNR     FPR    Notes               │
│  ─────────────────────────────── ────── ─────── ──────── ──────────────────  │
│  IsObfuscatedURL                   96%    1%       7%    Minor FPR issue     │
│  IsSuspiciousKeyword               96%    3%       6%    Balanced            │
│  Cloud_Hosting_Abuse_URL           94%    4%       9%    FPR slightly high   │
│  IsLanguageSpecific                94%    3%      12%    FPR elevated        │
│  IsBrandImpersonation              95%    5%       4%    Balanced            │
│                                                                              │
│  ASSESSMENT: Solid performance with minor FPR elevation. These categories    │
│  contain URLs that share structural similarities between benign and          │
│  malicious variants.                                                         │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

### 2.4 FPR and FNR Heatmap

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│  FPR BY CATEGORY (Sorted Worst → Best)                                       │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  IsLanguageSpecific        ██████████████                       12%  ❌     │
│  Redirect_Open_Redirect    ██████████████                       12%  ❌     │
│  Suspicious_TLD_URL        ████████████                         10%  ❌     │
│  Cloud_Hosting_Abuse       ███████████                           9%         │
│  IsObfuscatedURL           ████████▌                             7%         │
│  IsSuspiciousKeyword       ███████                               6%         │
│  Suspiciously_Long_Complex ██████                                5%         │
│  IsBrandImpersonation      █████                                 4%         │
│  Structural_Malformation   █████                                 4%         │
│  IsSessionBased            █████                                 4%         │
│  Hex_Encoded_URL           ███▌                                  3%         │
│  Punycode_URL              ██▌                                   2%         │
│  TypoSquatting_URL         ██▌                                   2%         │
│  HasExcessiveParams        █▎                                    1%  ✅     │
│  HasRepeatedSubdomain      █▎                                    1%  ✅     │
│                                                                              │
│  Target: FPR ≤ 1%          ↑ (only 2 categories meet the strict KPI)        │
│                                                                              │
├──────────────────────────────────────────────────────────────────────────────┤
│  FNR BY CATEGORY (Sorted Worst → Best)                                       │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  TypoSquatting_URL         ████████████████                     13%  ❌      │
│  Suspicious_TLD_URL        ███████████                           9%          │
│  HasRepeatedSubdomain      █████████                             7%          │
│  IsBrandImpersonation      ██████▌                               5%          │
│  Cloud_Hosting_Abuse       █████                                 4%          │
│  IsLanguageSpecific        ███▌                                  3%          │
│  IsSuspiciousKeyword       ███▌                                  3%          │
│  HasExcessiveParams        ██▌                                   2%          │
│  Redirect_Open_Redirect    ██▌                                   2%          │
│  Hex_Encoded_URL           █▎                                    1%         │
│  Credential_Harvesting     █▎                                    1%         │
│  IsObfuscatedURL           █▎                                    1%         │
│  IsSessionBased            █▎                                    1%         │
│  IsSuspiciousFileType      █▎                                    1%         │
│  IsWebAppPath              █▎                                    1%         │
│  Structural_Malformation   █▎                                    1%         │
│  Suspiciously_Long_Complex █▎                                    1%         │
│  IsMaliciousPattern        ▏                                     0%  ✅     │
│                                                                              │
│  Target: FNR ≤ 10%         ↑ (16 of 17 categories meet the FNR KPI)          │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```
---

## 📊 Combined Insights & Next Steps

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│  COMBINED FINDINGS — THRESHOLD + CATEGORY ANALYSIS                           │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. THRESHOLD ANALYSIS confirms that FPR is the primary bottleneck:          │
│     • t=0.5:   FPR=5.54% (too high)   │ FNR=8.95% (OK)                       │
│     • t=0.9:   FPR=0.05% (excellent)  │ FNR=55.19% (catastrophic)            │
│     • t=0.999: Model cannot produce probabilities this high                  │
│                                                                              │
│  2. CATEGORY ANALYSIS confirms the same pattern:                             │
│     • 15/17 categories pass FNR ≤ 10% — the model rarely misses attacks      │
│     • Only 2/17 categories pass FPR ≤ 1% — the model over-flags benign URLs  │
│                                                                              │
│  SOLUTION: HYBRID GLU FUSION (Already Built — Ready for Training)            │
│     • Fuses 384-dim MiniLM text embeddings with 76-dim heuristic vector      │
│     • GLU gate learns optimal text-vs-heuristic weighting per sample         │
│     • Expected to dramatically reduce FPR                                    │
│     • Training pipeline: 3_MiniLM_V2_hybrid_FF.py                            │
│                                                                              │
├──────────────────────────────────────────────────────────────────────────────┤
│  IMMEDIATE NEXT STEPS                                                        │
│  ──────────────────────                                                      │
│  Priority  Action                                     Expected Impact        │
│  ────────  ─────────────────────────────────────────  ─────────────────      │
│  P0        Train Hybrid GLU Fusion model              FPR → <1%              │
│  P1        Apply Temperature Scaling for calibration  Threshold range ↑      │
│  P2        Hard Negative Mining on FP/FN edge cases   Precision ↑            │
│  P3        Dedicated category wise/Typosquat fine-tuning   Category FNR ↓    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

*PhishGuard Research Team — IIT Ropar · April 2026*
