# 📁 High-Level Project Structure Overview (DataPrep10)

Below is the directory and file organization of the **DataPrep10** workspace, which contains both Subproject 1 (Raw & Canonical Model) and Subproject 2 (Hybrid SwiGLU/GLU Gated Fusion Model) for phishing URL detection.

```
D:\IIT ROPAR\phishing URL Detection\01_Research Tracker\2_Model_Building\PhishURLDetect-with-LLMS\2_Model _Preprocessed_data\Data_Preprocessing\DataPrep10\
│
├── 📁 0_DATA/                                              # Unified raw & sanity datasets
│   ├── 📁 1_Feature_Distribution/                          # Feature distribution outputs (reports & dashboards)
│   │   ├── 📁 OUTPUT_5M_RAW/                               # Feature distribution run files and outputs
│   │   └── 📁 SRC/                                         # Feature distribution source code/scripts
│   └── 📄 test_dataset.csv                                 # Local sanity verification dataset (50 samples)
│
├── 📁 1_MiniLM_V4_Model_On_Raw_Data_and_OFP_and_Canonical_Inferencing/ # 🚀 Subproject 1: Raw & Canonical Model
│   ├── 📁 RESULTS_&_MODELS/                                # Centralised destination for Subproject 1 outputs
│   │   ├── 📁 1_url_cate_data10_output/                    # Categorized threat groups
│   │   ├── 📁 2_preprocess_urls_output/                    # Preprocessed splits (raw_orig_train/val/test.csv)
│   │   └── 📁 3_saved_models/                              # Models, ONNX exports, and evaluations
│   ├── 📁 SRC/                                             # Core engineering codebase
│   │   ├── 📄 2_preprocess_urls_v8_refactored.py           # Preprocessing & stratified split orchestrator
│   │   ├── 📄 3_MiniLM_Hybrid_FF_V4.py                     # Monolithic training script
│   │   ├── 📄 4_config.yaml                                # Hyperparameters (WordPiece vs SP Unigram)
│   │   ├── 📄 5_train.py                                   # Training and checkpointing manager
│   │   ├── 📄 6_inference.py                               # E2E PyTorch and ONNX inference evaluator
│   │   ├── 📄 7_re_evaluate_thresholds.py                 # NumPy threshold tuning utility
│   │   ├── 📄 test_config.yaml                             # Sanity training configuration
│   │   ├── 📄 urls_cate_V7.py                              # URL Threat Categorization orchestrator
│   │   └── 📁 core/                                        # Shared modeling and utility packages
│   ├── 📁 mlruns/                                          # MLflow local run tracking artifacts directory
│   ├── 📄 mlflow.db                                        # SQLite database storing MLflow runs and metrics
│   ├── 📄 high-level-structure-overview.md                 # Subproject 1 structure overview
│   └── 📄 readme-format.md                                 # Formatting guidelines
│
├── 📁 2_MiniLM_Hybrid_FF_V4/                               # 🚀 Subproject 2: Hybrid SwiGLU Gated Fusion Model
│   ├── 📁 RESULTS_&_MODELS/                                # Centralised destination for Subproject 2 outputs
│   │   ├── 📁 1_url_cate_data10_output/                    # Categorized threat groups
│   │   ├── 📁 2_preprocess_urls_output/                    # Preprocessed splits (urls_hybrid_train/val/test.csv)
│   │   └── 📁 3_saved_models/                              # Models, ONNX exports, and evaluations
│   ├── 📁 SRC/                                             # Core hybrid codebase
│   │   ├── 📄 2_preprocess_urls_v8_refactored.py           # Preprocessing & stratified split orchestrator
│   │   ├── 📄 3_MiniLM_Hybrid_FF_V4.py                     # Monolithic training script
│   │   ├── 📄 4_config.yaml                                # Hyperparameters (GLU vs SwiGLU)
│   │   ├── 📄 5_train.py                                   # Training and checkpointing manager
│   │   ├── 📄 6_inference.py                               # E2E PyTorch and ONNX inference evaluator
│   │   ├── 📄 7_re_evaluate_thresholds.py                 # NumPy threshold tuning utility
│   │   ├── 📄 test_config.yaml                             # Sanity training configuration
│   │   ├── 📄 urls_cate_V7.py                              # URL Threat Categorization orchestrator
│   │   └── 📁 core/                                        # Shared library with SwiGLU Gating support
│   ├── 📁 mlruns/                                          # MLflow local run tracking artifacts directory
│   ├── 📄 mlflow.db                                        # SQLite database storing MLflow runs and metrics
│   ├── 📄 high-level-structure-overview.md                 # Subproject 2 structure overview (this file)
│   └── 📄 readme-format.md                                 # Formatting guidelines
│
├── 📄 requirements.txt                                     # Project python dependencies
└── 📄 why10.md                                             # Centralized rationale for tokenization & gating upgrades
```

---

## 🔑 Design Highlights

1. **Multi-Subproject Isolation**:
   The project is split into two distinct subprojects (`1_MiniLM_V4_Model_On_Raw_Data_and_OFP_and_Canonical_Inferencing` and `2_MiniLM_Hybrid_FF_V4`) sharing a unified data root (`0_DATA`). This prevents cross-contamination of datasets and models while keeping the codebases organized.

2. **Portable Configuration (`core/config.py`)**:
   All paths in `4_config.yaml` are resolved dynamically relative to the subproject root folder. This guarantees that model training, ONNX exports, and INT8 quantization work natively out-of-the-box on both Windows and Linux hosts.
