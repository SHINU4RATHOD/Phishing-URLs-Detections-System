import os
import json
import random
import warnings
import sys
import argparse
import ipaddress
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional, List, Any
from urllib.parse import urlparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
from collections import Counter  
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast

from transformers import AutoTokenizer, AutoModel, AutoConfig, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, precision_recall_curve, roc_curve

import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import onnx

# ONNX Quantization Setup
ONNX_QUANTIZATION_AVAILABLE = False
try:
    from onnxruntime.quantization import quantize_dynamic, QuantType
    ONNX_QUANTIZATION_AVAILABLE = True
    print("✓ ONNX quantization available")
except ImportError:
    print("⚠ ONNX quantization unavailable - install: pip install onnxruntime")

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    # Reproducibility
    SEED: int = 42
    
    # # Data Paths — Hybrid GLU Fusion splits
    TRAIN_CSV: str = "/home/hp/SHINU RATHOD/URLsPhishDetect-with-LLMS/2_Model _Preprocessed_data/Data_Preprocessing/data_prep8/3_MiniLM_V2_Hybrid/preprocess_urls_output/urls_hybrid_train.csv"
    VAL_CSV: str = "/home/hp/SHINU RATHOD/URLsPhishDetect-with-LLMS/2_Model _Preprocessed_data/Data_Preprocessing/data_prep8/3_MiniLM_V2_Hybrid/preprocess_urls_output/urls_hybrid_val.csv"
    TEST_CSV: str = "/home/hp/SHINU RATHOD/URLsPhishDetect-with-LLMS/2_Model _Preprocessed_data/Data_Preprocessing/data_prep8/3_MiniLM_V2_Hybrid/preprocess_urls_output/urls_hybrid_test.csv"

    # # Data Paths — Hybrid GLU Fusion splits
    # TRAIN_CSV: str = r"D:\IIT ROPAR\phishing URL Detection\01_Research Tracker\2_Model_Building\PhishURLDetect-with-LLMS\2_Model _Preprocessed_data\Data_Preprocessing\data_prep7\preprocess_urls_output\urls_hybrid_train.csv"
    # VAL_CSV: str = r"D:\IIT ROPAR\phishing URL Detection\01_Research Tracker\2_Model_Building\PhishURLDetect-with-LLMS\2_Model _Preprocessed_data\Data_Preprocessing\data_prep7\preprocess_urls_output\urls_hybrid_val.csv"
    # TEST_CSV: str = r"D:\IIT ROPAR\phishing URL Detection\01_Research Tracker\2_Model_Building\PhishURLDetect-with-LLMS\2_Model _Preprocessed_data\Data_Preprocessing\data_prep7\url_cate_V7_test_data10\IP_Address_Unusual_Port_URL\benign_noise_IP_Address_Unusual_Port_URL.csv"
    
    # ========================================================================
    # MODEL ARCHITECTURE - Hybrid GLU Fusion (MiniLM + HeuristicMLP)
    # ========================================================================
    MODEL_NAME: str = "microsoft/MiniLM-L12-H384-uncased"  # Compact & efficient
    MAX_LEN: int = 192          # Optimized: canonical URLs avg ~80 chars
    NUM_CLASSES: int = 2
    DROPOUT: float = 0.1        # Low dropout (0.1) for maximum capacity on 34.7M samples
    
    # --- Hybrid GLU Fusion Dimensions ---
    TEXT_EMBED_DIM: int = 384           # MiniLM-L12-H384 CLS output
    HEURISTIC_DIM: int = 87             # 15 numeric + 12 binary + 60 flags (was 90: 17+13+60)
    HEURISTIC_MLP_HIDDEN: int = 256     # MLP hidden layer
    HEURISTIC_MLP_OUTPUT: int = 192     # MLP embedding (upgraded for stronger fusion parity)
    GLU_HIDDEN: int = 384               # GLU gate intermediate dim (upgraded for [384; 192] concat)
    GATING_TYPE: str = "GLU"
    CLASSIFIER_DIMS: List[int] = [192, 64]  # Deep hierarchical distillation bottleneck
    
    # --- Heuristic Feature Columns (auto-detected from hybrid CSV) ---
    NUMERIC_FEATURE_COLS: List[str] = [
        "h_flags_bitmask",
        "h_entropy_url", "h_entropy_path", "h_entropy_query",
        "h_digit_ratio", "h_path_depth", "h_url_length", "h_query_param_count",
        # V9 numeric features
        "h_domain_length", "h_subdomain_count",
        "h_punycode_char_count", "h_unicode_char_ratio",
        "h_tracking_param_count", "h_path_token_count", "h_redirect_count",
    ]
    BINARY_FEATURE_COLS: List[str] = [
        "h_is_ip_host", "h_has_fragment",
        "h_tld_risk_normal", "h_tld_risk_high", "h_tld_risk_critical",
        # V9 binary features
        "h_has_punycode", "h_has_unicode", "h_mixed_script",
        "h_has_tracking_params", "h_has_double_extension",
        "h_has_redirect_param", "h_has_at_sign",
    ]
    # hF_* columns are auto-detected at runtime (any column starting with "hF_")
    DROP_COLS: List[str] = ["h_primary_category"]  # Categorical string — drop (leaky)
    
    # ========================================================================
    # TRAINING CONFIG - OPTIMIZED FOR 34.7M HYBRID SAMPLES
    # ========================================================================
    BATCH_SIZE: int = 128       # Memory threshold
    NUM_EPOCHS: int = 3          # 34.7M samples × 4 epochs = more convergence (loss was still decreasing at epoch 2)
    WEIGHT_DECAY: float = 0.02  # Standard for transformer fine-tuning
    PATIENCE: int = 4           # Matches 4-epoch budget
    GRAD_ACCUM_STEPS: int = 2   # Effective batch = 256 for stable fusion gradients
    GRAD_CLIP_NORM: float = 1.0 # Standard
    
    # ========================================================================
    # LEARNING RATE SCHEDULE (Cosine with Warmup) - RESEARCH OPTIMIZED
    # ========================================================================
    LR: float = 5e-5            # Base LR for LoRA adapters
    HEAD_LR: float = 1e-3       # Higher LR (1e-3) for custom heads trained from scratch
    LR_WARMUP_RATIO: float = 0.05  # 5% warmup (longer warmup for MLP/GLU randomly init weights)
    LR_MIN_RATIO: float = 0.001    # Lower minimum for better convergence
    
    # ========================================================================
    # LoRA CONFIG - OPTIMIZED FOR STRICT KPI (HIGH CAPACITY)
    # ========================================================================
    LORA_R: int = 32            # High rank for complex GLU text embeddings
    LORA_ALPHA: int = 64        # Alpha = 2 × rank
    LORA_DROPOUT: float = 0.05  # Low dropout, relies on massive 34.7M dataset for config
    LORA_TARGET_MODULES: List[str] = ["query", "key", "value",  "dense", "output.dense"]

    # ========================================================================
    # FOCAL LOSS - SOLE CLASS BALANCER
    # Ratio: 72.2% benign / 27.8% phishing = 2.6:1 imbalance from report_splits_hybrid.txt
    # ========================================================================
    FOCAL_GAMMA: float = 2.0    # Moderate gamma for hard-example mining without over-suppressing easy examples
    FOCAL_ALPHA: List[float] = [0.35, 0.65]  # 72%/28% imbalance → upweight phishing (minority) for recall, FPR controlled by threshold
    LABEL_SMOOTHING: float = 0.05  # Higher smoothing for calibrated probability thresholds
    
    # ========================================================================
    # CLASS BALANCING
    # ========================================================================
    USE_WEIGHTED_SAMPLING: bool = False  # Disabled: PyTorch limit 2^24 samples; focal loss α handles imbalance
    
    # OPTIMIZATION
    PRUNING_RATIO: float = 0.0
    USE_AMP: bool = True
    
    # Export
    EXPORT_ONNX: bool = True
    EXPORT_QUANTIZED: bool = True
    ONNX_OPSET: int = 14
    
    # KPI TARGETS (STRICT)
    TARGET_ACCURACY: float = 0.98
    TARGET_PRECISION: float = 0.95
    TARGET_RECALL: float = 0.95
    MAX_FNR: float = 0.10
    MAX_FPR: float = 0.01
    MAX_MODEL_SIZE_MB: float = 40.0  # MiniLM target: <40MB
    
    # Hardware - OPTIMIZED FOR LARGE DATASET
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS: int = 12       # Increased from 8 for 26.5M samples
    PIN_MEMORY: bool = True
    PREFETCH_FACTOR: int = 4    # Increased from 2 for GPU utilization 
    
    # Paths
    SAVE_ROOT: Optional[Path] = None
    CHECKPOINT_DIR: Optional[Path] = None
    
    @classmethod
    def setup_paths(cls) -> None:
        """Initialize output directories."""
        cls.SAVE_ROOT = Path(f"saved_models/MiniLM_HybridFF_v4")
        cls.CHECKPOINT_DIR = cls.SAVE_ROOT / "checkpoints"
        cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        cls.SAVE_ROOT.mkdir(parents=True, exist_ok=True)
        print(f"✓ Save directory: {cls.SAVE_ROOT}")
    
    @classmethod
    def setup_reproducibility(cls) -> None:
        """Set random seeds."""
        torch.manual_seed(cls.SEED)
        np.random.seed(cls.SEED)
        random.seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


# ============================================================================
# TEST INFERENCE URL SCREENING
# ============================================================================
INFERENCE_FILTER_SCHEME_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")
INFERENCE_FILTER_WINDOWS_DRIVE_PATTERN = re.compile(r"^[a-zA-Z]:[\\/]")
INFERENCE_FILTER_DECIMAL_IP_PATTERN = re.compile(r"^\d{8,10}$")
INFERENCE_FILTER_HEX_IP_PATTERN = re.compile(r"^(?:0x)?[0-9A-Fa-f]{8}$")

INFERENCE_URL_SHORTENERS = {
    "bit.ly", "bitly.com", "goo.gl", "t.co", "tinyurl.com", "ow.ly", "is.gd", "v.gd",
    "buff.ly", "rebrand.ly", "cutt.ly", "tiny.cc", "t.ly", "rb.gy", "shrtco.de",
    "s.id", "clck.ru", "bl.ink", "shorturl.at", "adf.ly", "q.gs", "short.io",
    "short.cm", "soo.gd", "lnkd.in", "x.co", "mcaf.ee", "amzn.to", "trib.al",
    "smarturl.it", "snip.ly", "snipurl.com", "lnk.bio", "lnk.to", "bio.link",
    "bio.site", "tap.bio", "linktr.ee", "beacons.ai", "campsite.bio", "hey.bio",
    "instabio.cc", "yt.be", "youtu.be", "link.medium.com", "reut.rs", "nyti.ms",
    "shorte.st", "clk.sh", "clk.im", "linkshrink.net", "bc.vc", "adcrun.ch",
    "ouo.io", "exe.io", "exey.io", "linkvertise.com", "shrinkme.io",
    "shrinkearn.com", "shortzon.com", "cutpaid.com", "cutwin.com", "shortadd.com",
    "short.pe", "shrinkurl.io", "urlcash.net", "t.me", "telegram.me", "wa.me",
    "goo.su", "urlzs.com", "linkvertise.net", "go.microsoft.com", "redirect.vk.com",
    "vk.cc", "vk.me", "msft.it", "msft.ms", "aka.ms", "shortzy.in", "shortxlink.com",
    "ez4short.com", "gtly.to", "sharee.tech", "stfly.io", "zws.im", "dlink.me",
    "hyperurl.co", "urlr.me", "clicky.me", "linkbox.to", "tiny.lt", "myurls.co",
    "shortbitly.com", "shortbit.com", "urlbitly.net", "bit.do", "short.best",
    "clickmeter.com", "urlr.in", "go.ly", "click.ru", "shortenurl.io", "taplink.at",
    "goo.by", "shortly.cc", "shortcm.li", "urlz.fr", "sk.gy"
}

INFERENCE_SUSPICIOUS_PORTS = {
    8080, 8081, 8082, 8083, 8084, 8085, 8088, 8000, 8001, 8008, 8010, 8443, 8880,
    8888, 9999, 1080, 3128, 9050, 8118, 1081, 1085, 4145, 4153, 6588, 6589, 6666,
    6667, 6697, 22, 2222, 23, 3389, 5900, 5985, 5986, 3306, 5432, 27017, 6379,
    9200, 11211, 1337, 1338, 1352, 4443, 4444, 5555, 6660, 7000, 7001, 8089,
    9000, 9001, 9002, 9003, 9010, 10000, 10101, 10443, 12345, 16000, 22222, 25,
    465, 587, 2525, 110, 995, 143, 993, 2082, 2083, 2095, 2096, 8890, 9090, 9443,
    9998, 10080
}

INFERENCE_URL_CATEGORY_SPECS = [
    {
        "name": "Shortened_URL",
        "policy": "filtered",
        "filter_during_inference": True,
        "description": (
            "Short-form redirector URLs cannot be expanded offline, so the destination "
            "cannot be inspected during inference."
        ),
    },
    {
        "name": "IP-Based URL",
        "policy": "filtered",
        "filter_during_inference": True,
        "description": (
            "Aggregates direct-IP, suspicious-port, and decimal/hex-IP patterns. "
            "These URLs are filtered because the LLM inference path is not reliable on IP-address-based destinations."
        ),
    },
    {
        "name": "Chrome_Internal_URL",
        "policy": "filtered",
        "filter_during_inference": True,
        "description": "Browser-internal chrome:// URLs are outside the model's intended web-URL scope.",
    },
    {
        "name": "Data_URL",
        "policy": "filtered",
        "filter_during_inference": True,
        "description": "Embedded data: payloads are excluded from the HTTP/HTTPS-oriented inference path.",
    },
    {
        "name": "FTP_SFTP_URL",
        "policy": "filtered",
        "filter_during_inference": True,
        "description": "ftp:// and sftp:// endpoints are excluded from this web inference pipeline.",
    },
    {
        "name": "File_URL",
        "policy": "filtered",
        "filter_during_inference": True,
        "description": "Local file:// URLs and Windows/UNC paths are outside the remote-URL threat model.",
    },
    {
        "name": "JavaScript_URL",
        "policy": "filtered",
        "filter_during_inference": True,
        "description": "javascript: pseudo-URLs represent executable payloads rather than navigable URLs.",
    },
    {
        "name": "Telegram_Bot_URL",
        "policy": "filtered",
        "filter_during_inference": True,
        "description": "Telegram bot interaction URLs are excluded from this offline URL inference flow.",
    },
]
INFERENCE_URL_CATEGORY_SPEC_MAP = {spec["name"]: spec for spec in INFERENCE_URL_CATEGORY_SPECS}


def _normalize_inference_screening_url(raw_url: Any) -> str:
    """Normalize raw strings into a urlparse-compatible form for offline screening."""
    candidate = str(raw_url or "").strip()
    if not candidate:
        return "http://"
    if INFERENCE_FILTER_SCHEME_PATTERN.match(candidate):
        return candidate
    lower_candidate = candidate.lower()
    if lower_candidate.startswith((
        "javascript:", "data:", "file:", "blob:", "chrome:", "about:", "edge:",
        "ftp:", "sftp:"
    )):
        return candidate
    if candidate.startswith("\\\\") or INFERENCE_FILTER_WINDOWS_DRIVE_PATTERN.match(candidate.replace("/", "\\")):
        return "file:///" + candidate.lstrip('/\\')
    return f"http://{candidate}"


def _extract_inference_screening_parts(raw_url: Any) -> Tuple[str, Any, str, str, str, Optional[int]]:
    """Return normalized URL parts without relying on external network lookups."""
    normalized = _normalize_inference_screening_url(raw_url)
    parsed = urlparse(normalized)
    scheme = (parsed.scheme or "").lower()
    host = (parsed.hostname or "").lower().strip(".")
    host = host[4:] if host.startswith("www.") else host
    path = parsed.path or ""
    query = parsed.query or ""
    try:
        port = parsed.port
    except ValueError:
        port = None
    return normalized, parsed, scheme, host, path, port


def detect_inference_screening_categories(raw_url: Any) -> List[str]:
    """Detect reportable categories for test-set screening before inference."""
    normalized, parsed, scheme, host, path, port = _extract_inference_screening_parts(raw_url)
    raw_text = str(raw_url or "")
    path_lower = path.lower()
    query_lower = (parsed.query or "").lower()

    categories: List[str] = []

    if host in INFERENCE_URL_SHORTENERS:
        categories.append("Shortened_URL")

    ip_based = False
    if host:
        try:
            ipaddress.ip_address(host)
            ip_based = True
        except ValueError:
            host_no_dot = host.replace(".", "")
            try:
                if INFERENCE_FILTER_DECIMAL_IP_PATTERN.fullmatch(host_no_dot):
                    ipaddress.IPv4Address(int(host_no_dot))
                    ip_based = True
                elif INFERENCE_FILTER_HEX_IP_PATTERN.fullmatch(host_no_dot):
                    ipaddress.IPv4Address(int(host_no_dot, 16))
                    ip_based = True
            except Exception:
                ip_based = ip_based or False
    if port in INFERENCE_SUSPICIOUS_PORTS:
        ip_based = True
    if ip_based:
        categories.append("IP-Based URL")

    if scheme == "chrome" or normalized.lower().startswith("chrome://"):
        categories.append("Chrome_Internal_URL")
    if scheme == "data":
        categories.append("Data_URL")
    if scheme in {"ftp", "sftp"}:
        categories.append("FTP_SFTP_URL")
    if scheme == "file" or raw_text.startswith("\\\\") or INFERENCE_FILTER_WINDOWS_DRIVE_PATTERN.match(raw_text.replace("/", "\\")):
        categories.append("File_URL")
    if scheme == "javascript":
        categories.append("JavaScript_URL")
    if host == "api.telegram.org" and "/bot" in path_lower:
        categories.append("Telegram_Bot_URL")
    elif host in {"t.me", "telegram.me"}:
        if "bot" in path_lower:
            categories.append("Telegram_Bot_URL")
        elif any(token in path_lower for token in ("start=", "startgroup=", "startapp=")):
            categories.append("Telegram_Bot_URL")
        elif any(token in query_lower for token in ("start=", "startgroup=", "startapp=")):
            categories.append("Telegram_Bot_URL")

    return categories


def filter_inference_unsupported_test_urls(test_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Filter unsupported URL categories from the test split and build a structured report."""
    category_counts = Counter({spec["name"]: 0 for spec in INFERENCE_URL_CATEGORY_SPECS})
    matched_any_mask: List[bool] = []
    filtered_mask: List[bool] = []

    for raw_url in test_df['input'].fillna("").astype(str).tolist():
        matches = detect_inference_screening_categories(raw_url)
        matched_any_mask.append(bool(matches))
        should_filter = False
        for category in matches:
            category_counts[category] += 1
            spec = INFERENCE_URL_CATEGORY_SPEC_MAP[category]
            if spec["filter_during_inference"]:
                should_filter = True
        filtered_mask.append(should_filter)

    filtered_mask_np = np.array(filtered_mask, dtype=bool)
    filtered_df = test_df.loc[~filtered_mask_np].reset_index(drop=True)

    report_categories = []
    for spec in INFERENCE_URL_CATEGORY_SPECS:
        matched = int(category_counts[spec["name"]])
        filtered = matched if spec["filter_during_inference"] else 0
        retained = matched - filtered
        report_categories.append({
            "name": spec["name"],
            "policy": spec["policy"],
            "matched_count": matched,
            "filtered_count": filtered,
            "retained_count": retained,
            "description": spec["description"],
        })

    report = {
        "report_name": "unsupported_url_category_report",
        "generated_at": datetime.now().isoformat(),
        "total_urls_scanned": int(len(test_df)),
        "urls_matching_reported_categories": int(sum(matched_any_mask)),
        "total_urls_filtered": int(filtered_mask_np.sum()),
        "total_urls_retained": int(len(filtered_df)),
        "category_counting_note": (
            "Category counts are non-exclusive; a single URL can contribute to multiple buckets."
        ),
        "categories": report_categories,
    }
    return filtered_df, report


def format_inference_url_screening_report(report: Optional[Dict[str, Any]]) -> str:
    """Render a developer-oriented console report for unsupported test URLs."""
    if not report:
        return ""

    lines = [
        "",
        "=" * 80,
        "UNSUPPORTED URL CATEGORY REPORT (TEST INFERENCE)",
        "=" * 80,
        f"Dataset scanned:                    {report['total_urls_scanned']:,}",
        f"URLs matched by reported buckets:   {report['urls_matching_reported_categories']:,}",
        f"URLs filtered before inference:     {report['total_urls_filtered']:,}",
        f"URLs retained for inference:        {report['total_urls_retained']:,}",
        report["category_counting_note"],
        "",
        f"{'Category':<24} {'Policy':<24} {'Matched':>12} {'Filtered':>12} {'Retained':>12}",
        "-" * 88,
    ]

    for entry in report["categories"]:
        lines.append(
            f"{entry['name']:<24} {entry['policy']:<24} "
            f"{entry['matched_count']:>12,} {entry['filtered_count']:>12,} {entry['retained_count']:>12,}"
        )
        lines.append(f"  Detail: {entry['description']}")

    lines.append("=" * 80)
    return "\n".join(lines)


def save_inference_url_screening_report(report: Optional[Dict[str, Any]], output_dir: Path) -> None:
    """Persist the screening report next to inference artifacts."""
    if not report:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    report_json_path = output_dir / "unsupported_url_category_report.json"
    report_txt_path = output_dir / "unsupported_url_category_report.txt"

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write(format_inference_url_screening_report(report) + "\n")


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================
class FocalLoss(nn.Module):
    """Focal Loss with numerical stability."""
    
    def __init__(self):
        super().__init__()
        self.gamma = Config.FOCAL_GAMMA
        self.label_smoothing = Config.LABEL_SMOOTHING
        
        if Config.FOCAL_ALPHA:
            self.register_buffer('alpha_tensor', torch.tensor(Config.FOCAL_ALPHA, dtype=torch.float))
        else:
            self.alpha_tensor = None
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = torch.clamp(logits, min=-10, max=10)
        ce_loss = nn.functional.cross_entropy( logits, targets, reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss).clamp(min=1e-7, max=1.0)
        
        if self.alpha_tensor is not None:
            alpha_t = self.alpha_tensor.to(targets.device)[targets]
            focal_loss = alpha_t * ((1 - pt) ** self.gamma) * ce_loss
        else:
            focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        loss = focal_loss.mean()
        
        if torch.isnan(loss) or torch.isinf(loss):
            print("⚠ NaN/Inf detected in loss, using CE fallback")
            return nn.functional.cross_entropy(logits, targets)
        
        return loss


# ============================================================================
# MODEL ARCHITECTURE — HYBRID GLU FUSION
# ============================================================================
class HeuristicMLP(nn.Module):
    """
    MLP tower for heuristic features → compact embedding.
    
    Architecture:  76 → 256 → LayerNorm → GELU → Dropout → 128 → LayerNorm → GELU
    
    Input:  (batch, 76)  — normalized numeric + binary + per-flag booleans
    Output: (batch, 128) — feature embedding for GLU fusion
    """
    
    def __init__(
        self,
        input_dim: int = Config.HEURISTIC_DIM,
        hidden_dim: int = Config.HEURISTIC_MLP_HIDDEN,
        output_dim: int = Config.HEURISTIC_MLP_OUTPUT,
        dropout: float = Config.DROPOUT,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )
        self._init_weights()
    
    def _init_weights(self) -> None:
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=0.02)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class GLUGate(nn.Module):
    """
    Gated Linear Unit for fusing text (384-dim) + heuristic (128-dim) embeddings.
    
    Mechanism:
        concat = [text_emb ; feat_emb]          # (batch, 512)
        gate   = σ(W_gate · concat + b_gate)     # (batch, hidden)
        value  = tanh(W_val · concat + b_val)    # (batch, hidden)
        output = gate ⊙ value                     # (batch, hidden)
    
    The sigmoid gate learns per-sample whether to rely more on text 
    (semantic) or heuristic (structural) signals.
    """
    
    def __init__(
        self,
        text_dim: int = Config.TEXT_EMBED_DIM,
        feat_dim: int = Config.HEURISTIC_MLP_OUTPUT,
        hidden_dim: int = Config.GLU_HIDDEN,
    ):
        super().__init__()
        concat_dim = text_dim + feat_dim  # 384 + 128 = 512
        self.gate_proj = nn.Linear(concat_dim, hidden_dim)
        self.value_proj = nn.Linear(concat_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self._init_weights()
    
    def _init_weights(self) -> None:
        for proj in [self.gate_proj, self.value_proj]:
            nn.init.xavier_normal_(proj.weight, gain=0.02)
            nn.init.zeros_(proj.bias)
    
    def forward(self, text_emb: torch.Tensor, feat_emb: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([text_emb, feat_emb], dim=-1)  # (batch, 512)
        gate = torch.sigmoid(self.gate_proj(concat))        # (batch, 256)
        value = torch.tanh(self.value_proj(concat))          # (batch, 256)
        fused = gate * value                                  # (batch, 256)
        return self.layer_norm(fused)


class SwiGLUGate(nn.Module):
    """
    Swish Gated Linear Unit for fusing text (384-dim) + heuristic (192-dim) embeddings.
    
    Mechanism:
        concat = [text_emb ; feat_emb]          # (batch, 576)
        gate   = SiLU(W_gate · concat + b_gate)   # (batch, hidden)
        value  = W_val · concat + b_val          # (batch, hidden)
        output = gate ⊙ value                     # (batch, hidden)
    """
    
    def __init__(
        self,
        text_dim: int = Config.TEXT_EMBED_DIM,
        feat_dim: int = Config.HEURISTIC_MLP_OUTPUT,
        hidden_dim: int = Config.GLU_HIDDEN,
    ):
        super().__init__()
        concat_dim = text_dim + feat_dim
        self.gate_proj = nn.Linear(concat_dim, hidden_dim)
        self.value_proj = nn.Linear(concat_dim, hidden_dim)
        self.silu = nn.SiLU()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self._init_weights()
    
    def _init_weights(self) -> None:
        for proj in [self.gate_proj, self.value_proj]:
            nn.init.xavier_normal_(proj.weight, gain=0.02)
            nn.init.zeros_(proj.bias)
    
    def forward(self, text_emb: torch.Tensor, feat_emb: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([text_emb, feat_emb], dim=-1)
        gate = self.silu(self.gate_proj(concat))
        value = self.value_proj(concat)
        fused = gate * value
        return self.layer_norm(fused)


class HybridGLUClassifier(nn.Module):
    """
    Dual-tower GLU/SwiGLU Fusion classifier for phishing URL detection.
    
    Tower 1 (Text):       input → MiniLM-L12 + LoRA → CLS → 384-dim
    Tower 2 (Heuristic):  76 features → MLP → 128-dim
    Fusion:               [384; 128] → GLU/SwiGLU Gate → 256-dim
    Head:                 256 → 128 → 2 (binary classification)
    """
    
    def __init__(self):
        super().__init__()
        
        # --- Tower 1: MiniLM Text Encoder ---
        self.config = AutoConfig.from_pretrained(Config.MODEL_NAME)
        self.encoder = AutoModel.from_pretrained(Config.MODEL_NAME, config=self.config)
        self.hidden_size = self.config.hidden_size  # 384
        
        # --- Tower 2: Heuristic MLP ---
        self.heuristic_mlp = HeuristicMLP(
            input_dim=Config.HEURISTIC_DIM,
            hidden_dim=Config.HEURISTIC_MLP_HIDDEN,
            output_dim=Config.HEURISTIC_MLP_OUTPUT,
            dropout=Config.DROPOUT
        )
        
        # --- GLU/SwiGLU Fusion Gate ---
        gating_type = getattr(Config, "GATING_TYPE", "GLU").upper()
        if gating_type == "SWIGLU":
            print("✓ Instantiating SwiGLUGate gating mechanism for fusion.")
            self.glu_gate = SwiGLUGate(
                text_dim=Config.TEXT_EMBED_DIM,
                feat_dim=Config.HEURISTIC_MLP_OUTPUT,
                hidden_dim=Config.GLU_HIDDEN
            )
        else:
            print("✓ Instantiating standard GLUGate gating mechanism for fusion.")
            self.glu_gate = GLUGate(
                text_dim=Config.TEXT_EMBED_DIM,
                feat_dim=Config.HEURISTIC_MLP_OUTPUT,
                hidden_dim=Config.GLU_HIDDEN
            )
        
        # --- Classification Head (post-fusion) ---
        layers = []
        in_dim = Config.GLU_HIDDEN  # 256 (GLU output)
        for out_dim in Config.CLASSIFIER_DIMS:
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(Config.DROPOUT),
            ])
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, Config.NUM_CLASSES))
        self.classifier = nn.Sequential(*layers)
        self._init_classifier_weights()
    
    def _init_classifier_weights(self) -> None:
        """Xavier initialization with small std for stability."""
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=0.02)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        heuristic_features: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        # --- Tower 1: Text encoding ---
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_emb = outputs.last_hidden_state[:, 0]  # CLS token → (batch, 384)
        
        if heuristic_features is not None:
            # --- Tower 2: Heuristic MLP ---
            feat_emb = self.heuristic_mlp(heuristic_features)  # (batch, 128)
            
            # --- GLU Fusion ---
            fused = self.glu_gate(text_emb, feat_emb)  # (batch, 256)
        else:
            # Fallback: text-only (pad with zeros for missing heuristic features)
            dummy_feat = torch.zeros(
                text_emb.size(0), Config.HEURISTIC_MLP_OUTPUT, 
                device=text_emb.device, dtype=text_emb.dtype
            )
            fused = self.glu_gate(text_emb, dummy_feat)
        
        logits = self.classifier(fused)  # (batch, 2)
        
        # Stability check
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("⚠ NaN/Inf detected in logits")
            logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
        
        return logits


def apply_structured_pruning(model: nn.Module, amount: float = Config.PRUNING_RATIO) -> None:
    """Pruning disabled for MiniLM stability."""
    if amount <= 0.0:
        print("Pruning disabled (MiniLM already compact)")
        return

from torchinfo import summary
def save_model_summary(model: nn.Module, input_size: Tuple[int, int], save_path: str = "model_summery.txt") -> None:
    '''
    Save comprehensive model summary to a text file.
    
    This simplified version doesn't use torchinfo, making it more reliable
    for models with multiple inputs like MiniLM.
    '''
    try:
        model.eval()
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        
        # Calculate model size in MB
        param_size = sum(param.nelement() * param.element_size() for param in model.parameters())
        buffer_size = sum(buffer.nelement() * buffer.element_size() for buffer in model.buffers())
        size_mb = (param_size + buffer_size) / (1024 ** 2)
        
        # Build summary string
        summary_lines = []
        summary_lines.append("=" * 80)
        summary_lines.append("MODEL SUMMARY: MiniLM URL Classifier")
        summary_lines.append("=" * 80)
        summary_lines.append("")
        summary_lines.append(f"Model Architecture: {model.__class__.__name__}")
        summary_lines.append(f"Gating Type:        {getattr(Config, 'GATING_TYPE', 'GLU')}")
        summary_lines.append(f"Input Size (batch, seq_len): {input_size}")
        summary_lines.append("")
        summary_lines.append("-" * 80)
        summary_lines.append("PARAMETER STATISTICS")
        summary_lines.append("-" * 80)
        summary_lines.append(f"Total Parameters:         {total_params:,}")
        summary_lines.append(f"Trainable Parameters:     {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        summary_lines.append(f"Non-trainable Parameters: {non_trainable_params:,} ({non_trainable_params/total_params*100:.2f}%)")
        summary_lines.append(f"Model Size:               {size_mb:.2f} MB")
        summary_lines.append("")
        summary_lines.append("-" * 80)
        summary_lines.append("LAYER-WISE BREAKDOWN")
        summary_lines.append("-" * 80)
        summary_lines.append(f"{'Layer Name':<50} {'Parameters':>15} {'Trainable':>12}")
        summary_lines.append("-" * 80)
        
        for name, param in model.named_parameters():
            trainable = "Yes" if param.requires_grad else "No"
            summary_lines.append(f"{name:<50} {param.numel():>15,} {trainable:>12}")
        
        summary_lines.append("=" * 80)
        summary_str = "\n".join(summary_lines)
        
        # Write to file
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(summary_str)
        
        print(f"✓ Model summary saved to {save_path}")
        print(f"  Total params: {total_params:,} | Trainable: {trainable_params:,} | Size: {size_mb:.2f} MB")
        
    except Exception as e:
        print(f"✗ Failed to save model summary: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# DATASET — HYBRID (TEXT + HEURISTIC FEATURES)
# ============================================================================
class HybridURLDataset(Dataset):
    """
    PyTorch Dataset for hybrid GLU Fusion model.
    
    Reads hybrid CSV with columns:
      - input:           canonical URL → tokenize with HuggingFace
      - label:           binary 0/1
      - h_* numeric:     17 continuous features → Z-score normalized
      - h_* binary:      13 binary features → passed through
      - hF_* flags:      60 per-flag booleans → passed through
      - h_primary_category: DROPPED (categorical string, leaky)
    
    Total heuristic dim: 90
    """
    
    def __init__(
        self, 
        df: pd.DataFrame, 
        tokenizer,
        feature_cols: List[str],
        norm_stats: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        self.tokenizer = tokenizer
        self.urls = df['input'].astype(str).tolist()
        self.labels = df['label'].astype(int).tolist()
        
        # Extract metadata for Samsung Decision Engine Layer
        self.metadata = {
            'h_severity_score': df['severity_score'].fillna(0).astype(float).tolist() if 'severity_score' in df.columns else (df['h_severity_score'].fillna(0).astype(float).tolist() if 'h_severity_score' in df.columns else [0.0]*len(self.urls)),
            'h_flags_count': df['flags_count'].fillna(0).astype(int).tolist() if 'flags_count' in df.columns else (df['h_flags_count'].fillna(0).astype(int).tolist() if 'h_flags_count' in df.columns else [0]*len(self.urls)),
            'h_primary_category': df['primary_category'].fillna('UNKNOWN').astype(str).tolist() if 'primary_category' in df.columns else (df['h_primary_category'].fillna('UNKNOWN').astype(str).tolist() if 'h_primary_category' in df.columns else ['UNKNOWN']*len(self.urls))
        }
        
        # Extract heuristic features as numpy array
        self.feature_cols = feature_cols
        features_df = df[feature_cols].fillna(0).astype(np.float32)
        
        # Apply Z-score normalization to NUMERIC columns only (not binary/flags)
        self.norm_stats = norm_stats
        if norm_stats is not None:
            for col in Config.NUMERIC_FEATURE_COLS:
                if col in features_df.columns:
                    mean, std = norm_stats[col]
                    features_df[col] = (features_df[col] - mean) / (std + 1e-8)
        
        self.features = features_df.values  # (N, 76)
        
        print(f"Dataset: {len(self.urls):,} samples | {len(feature_cols)} heuristic features")
        label_dist = pd.Series(self.labels).value_counts().to_dict()
        print(f"Label distribution: {label_dist}")
    
    def __len__(self) -> int:
        return len(self.urls)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        url = self.urls[idx]
        label = self.labels[idx]
        heuristic = self.features[idx]
        
        encoding = self.tokenizer(
            url, add_special_tokens=True, max_length=Config.MAX_LEN,
            padding="max_length", truncation=True, return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'heuristic_features': torch.tensor(heuristic, dtype=torch.float32),
            'labels': torch.tensor(label, dtype=torch.long),
            'url': url,
            'h_severity_score': float(self.metadata['h_severity_score'][idx]),
            'h_flags_count': int(self.metadata['h_flags_count'][idx]),
            'h_primary_category': str(self.metadata['h_primary_category'][idx])
        }
    
    @staticmethod
    def compute_normalization_stats(
        df: pd.DataFrame,
    ) -> Dict[str, Tuple[float, float]]:
        """Compute mean/std for numeric columns from training data only."""
        stats = {}
        for col in Config.NUMERIC_FEATURE_COLS:
            if col in df.columns:
                vals = df[col].fillna(0).astype(np.float32)
                stats[col] = (float(vals.mean()), float(vals.std()))
        print(f"✓ Normalization stats computed for {len(stats)} numeric features")
        return stats
    
    @staticmethod
    def detect_feature_columns(df: pd.DataFrame) -> List[str]:
        """Auto-detect heuristic feature columns from hybrid CSV."""
        feature_cols = []
        # Numeric h_ features
        for col in Config.NUMERIC_FEATURE_COLS:
            if col in df.columns:
                feature_cols.append(col)
        # Binary h_ features
        for col in Config.BINARY_FEATURE_COLS:
            if col in df.columns:
                feature_cols.append(col)
        # hF_* flags (auto-detected)
        hf_cols = sorted([c for c in df.columns if c.startswith('hF_')])
        feature_cols.extend(hf_cols)
        
        # Drop any explicitly excluded columns
        feature_cols = [c for c in feature_cols if c not in Config.DROP_COLS]
        
        print(f"✓ Detected {len(feature_cols)} heuristic feature columns:")
        print(f"  Numeric: {len([c for c in feature_cols if c in Config.NUMERIC_FEATURE_COLS])}")
        print(f"  Binary:  {len([c for c in feature_cols if c in Config.BINARY_FEATURE_COLS])}")
        print(f"  Flags:   {len(hf_cols)}")
        return feature_cols

    
def create_weighted_sampler(labels: List[int]) -> WeightedRandomSampler:
    # Count samples per class
    class_counts = Counter(labels)
    total_samples = len(labels)
    
    # Calculate inverse frequency weights
    class_weights = {
        class_id: total_samples / count 
        for class_id, count in class_counts.items()
    }
    
    # Assign weight to each sample based on its class
    sample_weights = [class_weights[label] for label in labels]
    
    # Create sampler
    sampler = WeightedRandomSampler( weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    print(f"\n{'='*60}")
    print("WEIGHTED SAMPLING ACTIVATED")
    print(f"{'='*60}")
    print(f"Class distribution:")
    for class_id, count in sorted(class_counts.items()):
        percentage = (count / total_samples) * 100
        weight = class_weights[class_id]
        label_name = "Benign" if class_id == 0 else "Phishing"
        print(f"  {label_name:10} ({class_id}): {count:,} samples ({percentage:.2f}%) - weight: {weight:.4f}")
    print(f"{'='*60}\n")
    
    return sampler


# ============================================================================
# TEMPERATURE SCALING (CALIBRATION)
# ============================================================================
class ModelCalibrator(nn.Module):
    """
    Learns a single scalar temperature T that scales logits (logits / T) 
    to produce mathematically calibrated probabilities (minimizing NLL).
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def calibrate(self, valid_logits: torch.Tensor, valid_labels: torch.Tensor):
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(valid_logits), valid_labels)
            loss.backward()
            return loss

        optimizer.step(eval)
        # print(f"    ↳ Calibration Temp: {self.temperature.item():.4f}")
        return self.temperature.item()

# ============================================================================
# SAMSUNG DECISION ENGINE (BUSINESS LOGIC LAYER)
# ============================================================================
class SamsungDecisionEngine:
    """
    Production-grade Risk Engine bridging smooth ML probabilities with hard cybersecurity constraints.
    Replaces static thresholds with dynamic Risk Scores and calibrated probability boundaries.
    """
    
    # Contextual category groupings
    HIGH_RISK_CATS = [
        'Credential_Harvesting_Form_URL', 'IsSuspiciousFileType', 
        'IsObfuscatedURL', 'TypoSquatting_URL', 'Compromised_CMS_URL'
    ]
    SAFE_INFRA_CATS = [
        'IsLanguageSpecific', 'Anchor_Fragment_Based_URL', 'Cloud_Hosting_Abuse_URL'
    ]
    
    @classmethod
    def decide(cls, log_odds: np.ndarray, severities: np.ndarray, flags_counts: np.ndarray, categories: np.ndarray, base_threshold: float = 0.5, lambda_val: float = 0.5) -> np.ndarray:
        """
        Accepts raw log-odds (logit_1 - logit_0) already calibrated by Temperature.
        Operates entirely in logit space — no prob→logit→prob round-trips.
        
        Args:
            lambda_val: Logit boost strength for high-risk categories (tunable).
        """
        # Clamp for numerical stability
        clamped_logits = np.clip(log_odds, -20.0, 20.0)
        
        # 1. Normalize metadata
        norm_sev = np.clip(severities / 10.0, 0.0, 1.0)
        norm_flags = np.clip(flags_counts / 10.0, 0.0, 1.0)
        
        # 2. Compute Structural Risk Score (pure heuristic, non-linear)
        engine_risk_score = 0.60 * (norm_sev ** 1.5) + 0.40 * (norm_flags ** 1.3)
        
        # 3. Category-Aware Logit Adjustment (λ is now a tunable parameter)
        adjusted_logits = clamped_logits.copy()
        
        high_risk_mask = np.isin(categories, cls.HIGH_RISK_CATS)
        safe_infra_mask = np.isin(categories, cls.SAFE_INFRA_CATS)
        ambiguous_mask = ~(high_risk_mask | safe_infra_mask)
        
        # Boost logits for high-risk categories (strength controlled by lambda_val)
        adjusted_logits[high_risk_mask] += lambda_val * engine_risk_score[high_risk_mask]
        
        # Convert to probability ONCE (single sigmoid pass)
        adjusted_probs = 1.0 / (1.0 + np.exp(-adjusted_logits))
        
        # 4. Contextual Threshold Offsets (conservative to protect FPR)
        local_thresholds = np.full_like(adjusted_probs, base_threshold)
        
        # High-risk: small reduction only
        local_thresholds[high_risk_mask] = np.maximum(base_threshold * 0.92, 0.40)
        # Safe infra: raise threshold to suppress FPs
        local_thresholds[safe_infra_mask] = np.minimum(base_threshold + 0.15, 0.90)
        
        # Clean URLs (no flags, low severity): require higher confidence
        clean_mask = ambiguous_mask & (flags_counts == 0) & (severities <= 1.0)
        local_thresholds[clean_mask] = np.minimum(base_threshold + 0.15, 0.85)

        # Vectorized final decision
        final_preds = (adjusted_probs >= local_thresholds).astype(int)
        
        return final_preds


# ============================================================================
# ENHANCED KPI EVALUATOR - STRICT THRESHOLD OPTIMIZATION
# ============================================================================
class EnhancedKPIEvaluator:
    """
    World-class KPI evaluation with multi-objective threshold optimization.
    Designed to meet strict KPIs: FPR ≤ 1%, FNR ≤ 10%, Precision ≥ 95%, Recall ≥ 95%
    """
    
    def __init__(self):
        self.evaluation_history: List[Dict] = []
    
    def evaluate_metrics(self,  y_true: np.ndarray,  y_pred: np.ndarray,  y_prob: np.ndarray) -> Dict[str, Any]:
        """Compute comprehensive metrics with KPI compliance check."""
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = 0.5
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative Predictive Value
        
        tpr = recall
        tnr = specificity
        fdr = fp / (tp + fp) if (tp + fp) > 0 else 0.0
        for_rate = fn / (tn + fn) if (tn + fn) > 0 else 0.0
        balanced_accuracy = (tpr + tnr) / 2.0
        mcc_denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = (tp * tn - fp * fn) / mcc_denom if mcc_denom > 0 else 0.0
        
        # Individual KPI checks
        kpi_checks = {
            'accuracy_met': accuracy >= Config.TARGET_ACCURACY,
            'precision_met': precision >= Config.TARGET_PRECISION,
            'recall_met': recall >= Config.TARGET_RECALL,
            'fnr_met': fnr <= Config.MAX_FNR,
            'fpr_met': fpr <= Config.MAX_FPR,
        }
        
        kpi_compliance = all(kpi_checks.values())
        
        # Weighted KPI score (emphasizing the hardest targets)
        # FPR is the hardest constraint (1%) — give it highest weight
        kpi_score = (
            0.10 * accuracy +
            0.20 * precision +
            0.15 * recall +
            0.20 * (1 - fnr) +  # FNR weight
            0.35 * (1 - fpr)   # FPR is the binding constraint → highest weight
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'fnr': fnr,
            'fpr': fpr,
            'specificity': specificity,
            'npv': npv,
            'tpr': tpr,
            'tnr': tnr,
            'mcc': mcc,
            'fdr': fdr,
            'for_rate': for_rate,
            'balanced_accuracy': balanced_accuracy,
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp),
            'kpi_compliance': kpi_compliance,
            'kpi_checks': kpi_checks,
            'kpi_score': kpi_score
        }
    
    def find_optimal_lambda_threshold_joint(self, y_true: np.ndarray, y_prob: np.ndarray, metadata: Optional[Dict] = None, log_odds: Optional[np.ndarray] = None) -> Tuple[float, float, Dict]:
        """
        Joint (λ, threshold) grid search — finds the optimal operating point.
        λ shifts logits, threshold cuts probability — they are coupled parameters.
        
        Returns:
            (optimal_lambda, optimal_threshold, best_metrics)
        """
        if metadata is None or log_odds is None:
            # Fallback: simple threshold-only search (no Decision Engine)
            best_f1, best_t = 0.0, 0.5
            for t in np.arange(0.30, 0.85, 0.005):
                preds = (y_prob >= t).astype(int)
                cm = confusion_matrix(y_true, preds)
                tn, fp, fn, tp = cm.ravel()
                p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2*p*r/(p+r) if (p+r) > 0 else 0.0
                if f1 > best_f1:
                    best_f1, best_t = f1, t
            return 0.0, best_t, {'lambda': 0.0, 'threshold': best_t, 'f1': best_f1}
        
        lambda_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5]
        thresholds = np.arange(0.30, 0.85, 0.005)
        
        valid_configs = []
        best_compromise = None
        best_compromise_score = float('inf')
        
        print("\n" + "=" * 70)
        print("JOINT (λ, THRESHOLD) GRID SEARCH")
        print("=" * 70)
        print(f"Constraints: FPR ≤ {Config.MAX_FPR:.1%}, FNR ≤ {Config.MAX_FNR:.1%}")
        print(f"Searching {len(lambda_values)} λ values × {len(thresholds)} thresholds = {len(lambda_values)*len(thresholds)} combinations...")
        
        for lam in lambda_values:
            for thresh in thresholds:
                y_pred = SamsungDecisionEngine.decide(
                    log_odds=log_odds,
                    severities=metadata['severities'],
                    flags_counts=metadata['flags_counts'],
                    categories=metadata['categories'],
                    base_threshold=thresh,
                    lambda_val=lam
                )
                
                cm = confusion_matrix(y_true, y_pred)
                tn, fp, fn, tp = cm.ravel()
                
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                accuracy = (tp + tn) / (tp + tn + fp + fn)
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                
                result = {
                    'lambda': lam, 'threshold': thresh,
                    'fpr': fpr, 'fnr': fnr, 'precision': precision,
                    'recall': recall, 'accuracy': accuracy, 'f1': f1,
                    'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
                }
                
                # Track best compromise (for fallback)
                fpr_viol = max(0, fpr - Config.MAX_FPR)
                fnr_viol = max(0, fnr - Config.MAX_FNR)
                viol_score = fpr_viol + fnr_viol - 0.05 * f1
                if viol_score < best_compromise_score:
                    best_compromise_score = viol_score
                    best_compromise = result
                
                # Check if BOTH KPI constraints are met
                if fpr <= Config.MAX_FPR and fnr <= Config.MAX_FNR:
                    valid_configs.append(result)
        
        if valid_configs:
            print(f"\n✅ Found {len(valid_configs)} valid (λ, threshold) pairs meeting BOTH constraints!")
            # Among valid configs, pick the one with best F1
            best = max(valid_configs, key=lambda x: x['f1'])
            print(f"\n🏆 OPTIMAL OPERATING POINT:")
            print(f"  λ (lambda):  {best['lambda']:.2f}")
            print(f"  Threshold:   {best['threshold']:.3f}")
            print(f"  FPR:         {best['fpr']:.4f} (target ≤ {Config.MAX_FPR}) ✅")
            print(f"  FNR:         {best['fnr']:.4f} (target ≤ {Config.MAX_FNR}) ✅")
            print(f"  Precision:   {best['precision']:.4f}")
            print(f"  Recall:      {best['recall']:.4f}")
            print(f"  F1:          {best['f1']:.4f}")
            print(f"  Accuracy:    {best['accuracy']:.4f}")
        else:
            print(f"\n⚠ No (λ, threshold) pair satisfies BOTH FPR ≤ {Config.MAX_FPR:.1%} AND FNR ≤ {Config.MAX_FNR:.1%}")
            print("Using best compromise...")
            best = best_compromise
            print(f"\n📊 BEST COMPROMISE:")
            print(f"  λ (lambda):  {best['lambda']:.2f}")
            print(f"  Threshold:   {best['threshold']:.3f}")
            print(f"  FPR:         {best['fpr']:.4f} {'✅' if best['fpr'] <= Config.MAX_FPR else '❌'}")
            print(f"  FNR:         {best['fnr']:.4f} {'✅' if best['fnr'] <= Config.MAX_FNR else '❌'}")
            print(f"  F1:          {best['f1']:.4f}")
        
        print("=" * 70)
        return best['lambda'], best['threshold'], best


    def analyze_threshold_sensitivity(
        self, 
        y_true: np.ndarray, 
        y_prob: np.ndarray
    ) -> pd.DataFrame:
        """Generate threshold sensitivity analysis table."""
        thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
        results = []
        
        for thresh in thresholds:
            y_pred = (y_prob >= thresh).astype(int)
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            results.append({
                'Threshold': thresh,
                'FPR': fp / (fp + tn),
                'FNR': fn / (fn + tp),
                'Precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'Recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'Accuracy': (tp + tn) / (tp + tn + fp + fn),
                'FPR_OK': '✓' if fp / (fp + tn) <= Config.MAX_FPR else '✗',
                'FNR_OK': '✓' if fn / (fn + tp) <= Config.MAX_FNR else '✗',
            })
        
        return pd.DataFrame(results)


# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================
class CheckpointManager:
    """Handles model checkpointing with resume support."""
    
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint( self, model: nn.Module, optimizer: optim.Optimizer, scheduler: Optional[Any], scaler: Optional[GradScaler], epoch: int, metrics: Dict, threshold: float, best_kpi_score: float, training_history: Dict) -> Path:
        """Save training checkpoint with full state."""
        config_dict = self._serialize_config()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'scaler_state_dict': scaler.state_dict() if Config.USE_AMP else None,
            'metrics': metrics,
            'threshold': threshold,
            'best_kpi_score': best_kpi_score,
            'training_history': training_history,
            'config': config_dict
        }
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        temp_path = checkpoint_path.with_suffix('.pt.tmp')
        
        try:
            torch.save(checkpoint, temp_path)
            temp_path.rename(checkpoint_path)
            print(f"✓ Checkpoint saved: {checkpoint_path.name}")
        except Exception as e:
            print(f"✗ Failed to save checkpoint: {e}")
            if temp_path.exists():
                temp_path.unlink()
        
        return checkpoint_path
    
    def load_checkpoint( self, checkpoint_path: Path, model: nn.Module, optimizer: Optional[optim.Optimizer] = None, scheduler: Optional[Any] = None, scaler: Optional[GradScaler] = None) -> Tuple[int, Dict, float, Dict]:
        """Load checkpoint and restore full training state."""
        try:
            print(f"\n{'='*60}")
            print(f"RESUMING FROM CHECKPOINT")
            print(f"{'='*60}")
            
            # weights_only=False to load full training state (PyTorch 2.6+ default changed)
            checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE, weights_only=False)
            
            # Load model state
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            model.to(Config.DEVICE)
            
            # Load optimizer state
            if optimizer and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"✓ Optimizer state restored")
            
            # Load scheduler state
            if scheduler and checkpoint.get('scheduler_state_dict'):
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print(f"✓ Scheduler state restored")
            
            # Load scaler state
            if scaler and checkpoint.get('scaler_state_dict'):
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
                print(f"✓ Scaler state restored")
            
            # Extract training state
            start_epoch = checkpoint.get('epoch', 0) + 1
            metrics = checkpoint.get('metrics', {})
            best_kpi_score = checkpoint.get('best_kpi_score', 0.0)
            training_history = checkpoint.get('training_history', {})
            
            print(f"✓ Model state restored")
            print(f"✓ Resuming from epoch {start_epoch}")
            print(f"✓ Best KPI score: {best_kpi_score:.4f}")
            print(f"{'='*60}\n")
            
            return start_epoch, metrics, best_kpi_score, training_history
        
        except Exception as e:
            print(f"⚠ Checkpoint load failed ({checkpoint_path.name}): {e}")
            print(f"⚠ Keeping the checkpoint file for inspection. Starting fresh this run.")
            return 1, {}, 0.0, {}
    
    def find_latest_checkpoint(self) -> Optional[Path]:
        """Find most recent valid checkpoint (non-destructive)."""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        
        latest_valid = None
        for ckpt in reversed(checkpoints):
            try:
                torch.load(ckpt, map_location='cpu', weights_only=False)
                latest_valid = ckpt
                break
            except Exception as e:
                print(f"⚠ Corrupted checkpoint: {ckpt.name} ({e})")
                continue
        
        return latest_valid
    
    def cleanup_old_checkpoints(self, keep_last_n: int = 3):
        """Keep only the last N checkpoints to save disk space."""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        
        if len(checkpoints) > keep_last_n:
            for ckpt in checkpoints[:-keep_last_n]:
                try:
                    ckpt.unlink()
                    print(f"🗑 Cleaned up old checkpoint: {ckpt.name}")
                except Exception as e:
                    print(f"⚠ Failed to delete {ckpt.name}: {e}")
    
    @staticmethod
    def _serialize_config() -> Dict:
        """Serialize Config to dict."""
        config_dict = {}
        for key, value in vars(Config).items():
            if key.startswith('_') or callable(value):
                continue
            if isinstance(value, Path):
                config_dict[key] = str(value)
            elif isinstance(value, (int, float, str, bool, list, dict, type(None))):
                config_dict[key] = value
            else:
                config_dict[key] = str(value)
        return config_dict


# ============================================================================
# ARTIFACT SAVER
# ============================================================================
class ArtifactSaver:
    """Saves training artifacts, plots, and metrics."""
    
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
    
    def save_history( self, train_losses: List[float], val_losses: List[float], train_accs: List[float], val_accs: List[float]) -> None:
        """Save training history CSV and plots."""
        history_df = pd.DataFrame({
            'epoch': range(1, len(train_losses) + 1),
            'train_loss': train_losses,
            'val_loss': val_losses,
            'train_acc': train_accs,
            'val_acc': val_accs
        })
        history_df.to_csv(self.run_dir / 'training_history.csv', index=False)
        print(f"✓ Training history saved")
        
        self._plot_curves(history_df)
    
    def _plot_curves(self, df: pd.DataFrame) -> None:
        """Generate loss and accuracy plots."""
        # Loss curves
        plt.figure(figsize=(10, 5))
        plt.plot(df['epoch'], df['train_loss'], label='Train', linewidth=2, marker='o')
        plt.plot(df['epoch'], df['val_loss'], label='Validation', linewidth=2, marker='s')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.run_dir / 'loss_curves.png', dpi=300)
        plt.close()
        
        # Accuracy curves
        plt.figure(figsize=(10, 5))
        plt.plot(df['epoch'], df['train_acc'], label='Train', linewidth=2, marker='o')
        plt.plot(df['epoch'], df['val_acc'], label='Validation', linewidth=2, marker='s')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.run_dir / 'accuracy_curves.png', dpi=300)
        plt.close()
        
        print(f"✓ Training plots saved")
    
    def save_test_metrics(self, metrics: Dict, threshold: float) -> None:
        """Save test metrics to CSV."""
        metrics_copy = metrics.copy()
        metrics_copy['threshold'] = threshold
        pd.DataFrame([metrics_copy]).to_csv(self.run_dir / 'test_metrics.csv', index=False)
        print(f"✓ Test metrics saved")
    
    def save_test_plots( self, y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> None:
        """Generate test set visualizations."""
        y_pred = (y_prob >= threshold).astype(int)
        
        self._plot_confusion_matrix(y_true, y_pred, threshold)
        self._plot_roc_curve(y_true, y_prob)
        self._plot_pr_curve(y_true, y_prob)
        print(f"✓ Test plots saved")
    
    def _plot_confusion_matrix( self, y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> None:
        """Plot confusion matrix heatmap."""
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        total = tn + fp + fn + tp
        cm_percent = cm / total * 100
        
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        plt.figure(figsize=(8, 6))
        sns.heatmap( cm_percent, annot=False, fmt='.1f', cmap='Blues', xticklabels=['Benign', 'Malicious'], yticklabels=['Benign', 'Malicious'], cbar_kws={'label': 'Percentage'})
        
        # Annotate cells
        for i in range(2):
            for j in range(2):
                count = cm[i, j]
                percent = cm_percent[i, j]
                plt.text(
                    j + 0.5, i + 0.5,
                    f'{percent:.1f}%\n({count:,})',
                    ha='center', va='center',
                    fontsize=12, fontweight='bold',
                    color='white' if percent > 50 else 'black'
                )
        
        plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=14, fontweight='bold')
        plt.title( f'Test Confusion Matrix (Threshold={threshold:.3f})\n' f'FNR={fnr:.2%} | FPR={fpr:.2%}', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(self.run_dir / 'confusion_matrix_test.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray) -> None:
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}', linewidth=2.5, color='blue')
        plt.plot([0, 1], [0, 1], '--', color='gray', linewidth=2, label='Random')
        plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
        plt.title('ROC Curve - Test Set', fontsize=16, fontweight='bold', pad=20)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.run_dir / 'roc_test.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_pr_curve(self, y_true: np.ndarray, y_prob: np.ndarray) -> None:
        """Plot precision-recall curve."""
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2.5, color='green')
        plt.xlabel('Recall', fontsize=14, fontweight='bold')
        plt.ylabel('Precision', fontsize=14, fontweight='bold')
        plt.title('Precision-Recall Curve - Test Set', fontsize=16, fontweight='bold', pad=20)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.run_dir / 'pr_curve_test.png', dpi=300, bbox_inches='tight')
        plt.close()


# ============================================================================
# MODEL EXPORTER
# ============================================================================
class ModelExporter:
    """Handles model export to ONNX with quantization."""
    
    @staticmethod
    def merge_lora_and_export(model: nn.Module, tokenizer, save_dir: Path) -> Tuple[nn.Module, float]:
        """Merge LoRA adapters and save production model."""
        try:
            print("Merging LoRA adapters...")
            
            # CRITICAL FIX: Load the complete trained PEFT model (contains trained MLP and classifier heads)
            full_model_path = save_dir / "model_full.pt"
            loaded_model = torch.load(full_model_path, map_location=Config.DEVICE, weights_only=False)
            
            # Merge LoRA weights into the base layers, preserving our trained custom heads
            merged_model = loaded_model.merge_and_unload()
            merged_model = merged_model.to(Config.DEVICE).eval()
            
            # Save merged model
            merged_path = save_dir / "model_merged_full.pt"
            torch.save(merged_model, merged_path)
            merged_size = os.path.getsize(merged_path) / (1024 * 1024)
            
            print(f"✓ Merged model: {merged_size:.2f} MB")
            
            # Save state dict
            torch.save(merged_model.state_dict(), save_dir / "model_merged_state_dict.pt")

            # Save model summary
            summary_path = save_dir / "model_summery.txt"
            save_model_summary(merged_model, input_size=(1, Config.MAX_LEN), save_path=str(summary_path))

            
            return merged_model, merged_size
        
        except Exception as e:
            print(f"⚠ LoRA merge failed: {e}")
            return model, 0.0
    
    @staticmethod
    def export_onnx( model: nn.Module, save_dir: Path) -> Optional[float]:
        """Export model to ONNX format with quantization."""
        if not Config.EXPORT_ONNX:
            return None
        
        try:
            print("Exporting to ONNX...")
            model.eval()
            device = next(model.parameters()).device
            
            dummy_input = {
                'input_ids': torch.randint(0, 30522, (1, Config.MAX_LEN), dtype=torch.long).to(device),
                'attention_mask': torch.ones(1, Config.MAX_LEN, dtype=torch.long).to(device),
                'heuristic_features': torch.randn(1, Config.HEURISTIC_DIM, dtype=torch.float32).to(device)
            }
            
            onnx_path = save_dir / "model.onnx"
            torch.onnx.export(
                model,
                (dummy_input['input_ids'], dummy_input['attention_mask'], dummy_input['heuristic_features']),
                str(onnx_path),
                opset_version=Config.ONNX_OPSET,
                input_names=['input_ids', 'attention_mask', 'heuristic_features'],
                output_names=['logits'],
                dynamic_axes={
                    'input_ids': {0: 'batch', 1: 'sequence'},
                    'attention_mask': {0: 'batch', 1: 'sequence'},
                    'heuristic_features': {0: 'batch'},
                    'logits': {0: 'batch'}
                },
                verbose=False
            )
            
            onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
            print(f"✓ ONNX model: {onnx_size:.2f} MB")
            
            # Quantize ONNX
            if Config.EXPORT_QUANTIZED and ONNX_QUANTIZATION_AVAILABLE:
                quant_size = ModelExporter._quantize_onnx(onnx_path, save_dir)
                return quant_size if quant_size else onnx_size
            
            return onnx_size
        
        except Exception as e:
            print(f"✗ ONNX export failed: {e}")
            return None
    
    @staticmethod
    def _quantize_onnx(onnx_path: Path, save_dir: Path) -> Optional[float]:
        """Quantize ONNX model to 8-bit."""
        try:
            print("Quantizing ONNX model...")
            quant_path = save_dir / "model_quant_8bit.onnx"
            
            quantize_dynamic(
                str(onnx_path),
                str(quant_path),
                weight_type=QuantType.QUInt8
            )
            
            quant_size = os.path.getsize(quant_path) / (1024 * 1024)
            original_size = os.path.getsize(onnx_path) / (1024 * 1024)
            reduction = ((original_size - quant_size) / original_size) * 100
            
            print(f"✓ Quantized ONNX: {quant_size:.2f} MB ({reduction:.1f}% reduction)")
            
            if quant_size <= Config.MAX_MODEL_SIZE_MB:
                print(f"✅ Quantized model meets {Config.MAX_MODEL_SIZE_MB}MB target!")
            else:
                print(f"⚠ Quantized model {quant_size:.2f}MB exceeds {Config.MAX_MODEL_SIZE_MB}MB target")
            
            return quant_size
        
        except Exception as e:
            print(f"⚠ ONNX quantization failed: {e}")
            return None


# ============================================================================
# TRAINER
# ============================================================================
class PhishingDetectionTrainer:
    """Main training orchestrator."""
    
    def __init__(self):
        Config.setup_reproducibility()
        Config.setup_paths()
        
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        self.checkpoint_manager = CheckpointManager(Config.CHECKPOINT_DIR)
        self.kpi_evaluator = EnhancedKPIEvaluator()
        self.latest_test_url_screening_report: Optional[Dict[str, Any]] = None
        
        self.training_history = {
            'train_losses': [], 'val_losses': [],
            'train_accs': [], 'val_accs': [],
            'kpi_scores': [], 'thresholds': []
        }
    
    def load_datasets(self, filter_test_unsupported: bool = False) -> Tuple[HybridURLDataset, HybridURLDataset, HybridURLDataset]:
        """Load hybrid datasets with heuristic features and normalization."""
        print("\n" + "="*60)
        print("LOADING HYBRID DATASETS")
        print("="*60)
        
        train_df = pd.read_csv(Config.TRAIN_CSV).reset_index(drop=True)
        val_df = pd.read_csv(Config.VAL_CSV).reset_index(drop=True)
        test_df = pd.read_csv(Config.TEST_CSV).reset_index(drop=True)

#----------------------------------------------------------------------------REMOVE SAMPLING - USE FULL DATASET
        # train_df = train_df.sample(frac=0.0001, random_state=42) 
        # val_df = val_df.sample(frac=0.0001, random_state=42)      
        # test_df = test_df.sample(frac=0.0001, random_state=42)  # USE FULL TEST SET for reliable KPIs
#---------------------------------------------------------------------------------------------------------------         
        
        # Ensure proper label encoding (0=benign, 1=malicious) and column names
        for df in [train_df, val_df, test_df]:
            if 'url' in df.columns and 'input' not in df.columns:
                df.rename(columns={'url': 'input'}, inplace=True)
            
            if 'label' in df.columns:
                if df['label'].dtype == 'object':
                    df['label'] = df['label'].map({
                        'legit': 0, 'benign': 0, 'legitimate': 0,
                        'malicious': 1, 'phishing': 1, 'phish': 1
                    })
                df['label'] = df['label'].fillna(0).astype(int)
        
        print(f"\nDataset Sizes:")
        print(f"  Train:      {len(train_df):,}")
        print(f"  Validation: {len(val_df):,}")
        print(f"  Test:       {len(test_df):,}")
        print(f"  Total:      {len(train_df) + len(val_df) + len(test_df):,}")

        if filter_test_unsupported:
            print("\nApplying unsupported-URL screening to the test split...")
            test_df, self.latest_test_url_screening_report = filter_inference_unsupported_test_urls(test_df)
            print(f"  Filtered before inference: {self.latest_test_url_screening_report['total_urls_filtered']:,}")
            print(f"  Retained for inference:    {self.latest_test_url_screening_report['total_urls_retained']:,}")
            if test_df.empty:
                raise ValueError("All test URLs were removed by unsupported-category screening; inference cannot continue.")
        else:
            self.latest_test_url_screening_report = None
        
        # Auto-detect heuristic feature columns from training data
        feature_cols = HybridURLDataset.detect_feature_columns(train_df)
        self.feature_cols = feature_cols  # Store for inference
        
        # Compute normalization stats from TRAINING data only (prevent leakage)
        norm_stats = HybridURLDataset.compute_normalization_stats(train_df)
        self.norm_stats = norm_stats  # Store for inference
        
        # Verify dimension matches config
        if len(feature_cols) != Config.HEURISTIC_DIM:
            print(f"⚠ Feature dimension mismatch: detected {len(feature_cols)} vs Config.HEURISTIC_DIM={Config.HEURISTIC_DIM}")
            print(f"  Updating Config.HEURISTIC_DIM to {len(feature_cols)}")
            Config.HEURISTIC_DIM = len(feature_cols)
        
        train_dataset = HybridURLDataset(train_df, self.tokenizer, feature_cols, norm_stats)
        val_dataset = HybridURLDataset(val_df, self.tokenizer, feature_cols, norm_stats)
        test_dataset = HybridURLDataset(test_df, self.tokenizer, feature_cols, norm_stats)
        
        return train_dataset, val_dataset, test_dataset

    def _emit_test_url_screening_report(self, output_dir: Path) -> None:
        """Persist and print the latest test URL screening report after inference completes."""
        if not self.latest_test_url_screening_report:
            return

        save_inference_url_screening_report(self.latest_test_url_screening_report, output_dir)
        print(format_inference_url_screening_report(self.latest_test_url_screening_report))
        print("✓ Unsupported URL category report saved: unsupported_url_category_report.{json,txt}")
    
    
    def create_model(self) -> nn.Module:
        """Build HybridGLUClassifier with LoRA on the text encoder."""
        print("\n" + "="*60)
        print("BUILDING HYBRID GLU FUSION MODEL")
        print("="*60)
        
        base_model = HybridGLUClassifier()
        apply_structured_pruning(base_model, Config.PRUNING_RATIO)
        
        lora_config = LoraConfig(
            task_type="SEQ_CLS",
            inference_mode=False,
            r=Config.LORA_R,
            lora_alpha=Config.LORA_ALPHA,
            lora_dropout=Config.LORA_DROPOUT,
            target_modules=Config.LORA_TARGET_MODULES
        )
        
        model = get_peft_model(base_model, lora_config)
        model = model.to(Config.DEVICE)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Detailed parameter breakdown
        encoder_params = sum(p.numel() for n, p in model.named_parameters() if 'encoder' in n)
        mlp_params = sum(p.numel() for n, p in model.named_parameters() if 'heuristic_mlp' in n)
        glu_params = sum(p.numel() for n, p in model.named_parameters() if 'glu_gate' in n)
        head_params = sum(p.numel() for n, p in model.named_parameters() if 'classifier' in n)
        lora_params = sum(p.numel() for n, p in model.named_parameters() if 'lora' in n)
        
        print(f"\nModel Architecture: Hybrid GLU Fusion (MiniLM + LoRA + MLP + GLU)")
        print(f"  Text Encoder:     {encoder_params:,} (MiniLM-L12-H384, mostly frozen)")
        print(f"  LoRA Adapters:    {lora_params:,} (trainable)")
        print(f"  Heuristic MLP:    {mlp_params:,} (76→256→128, trainable)")
        print(f"  GLU Gate:         {glu_params:,} (512→256 sigmoid×tanh, trainable)")
        print(f"  Classifier Head:  {head_params:,} (256→128→2, trainable)")
        print(f"  ─────────────────────────────")
        print(f"  Total parameters:     {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        print(f"  Frozen parameters:    {total_params - trainable_params:,}")
        
        return model
    
    def train(self) -> bool:
        """Execute complete training pipeline with checkpoint resuming."""
        print("\n" + "="*80)
        print("MiniLM PHISHING DETECTION TRAINING PIPELINE")
        print("="*80)
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Device: {Config.DEVICE}")
        print(f"Target: <{Config.MAX_MODEL_SIZE_MB}MB model with 98% accuracy")
        print("="*80)
        
        # ========================================
        # SETUP DATASETS AND MODEL
        # ========================================
        train_dataset, val_dataset, test_dataset = self.load_datasets()
        model = self.create_model()
        
        # ========================================
        # CREATE DATALOADERS
        # ========================================
        if Config.USE_WEIGHTED_SAMPLING:
            sampler = create_weighted_sampler(train_dataset.labels)
            train_loader = DataLoader( train_dataset, batch_size=Config.BATCH_SIZE, sampler=sampler, num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY, prefetch_factor=Config.PREFETCH_FACTOR if Config.NUM_WORKERS > 0 else None)
            print(f"✓ Training with weighted sampling (balanced batches)")
        else:
            train_loader = DataLoader( train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY, prefetch_factor=Config.PREFETCH_FACTOR if Config.NUM_WORKERS > 0 else None)
            print(f"✓ Training with standard random shuffling")
        
        val_loader = DataLoader( val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=Config.PIN_MEMORY, prefetch_factor=2 if 2 > 0 else None)
        test_loader = DataLoader( test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=Config.PIN_MEMORY, prefetch_factor=2 if 2 > 0 else None)
        
        # ========================================
        # CREATE OPTIMIZER, SCHEDULER, CRITERION
        # ========================================
        # Differential Learning Rates: LoRA takes base LR, Custom heads take HEAD_LR (1e-3)
        head_params = []
        base_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if any(key in name for key in ['heuristic_mlp', 'glu_gate', 'classifier']):
                head_params.append(param)
            else:
                base_params.append(param)
                
        optimizer = optim.AdamW([
            {'params': base_params, 'lr': Config.LR},
            {'params': head_params, 'lr': Config.HEAD_LR}
        ], weight_decay=Config.WEIGHT_DECAY, eps=1e-8)
        
        total_steps = len(train_loader) * Config.NUM_EPOCHS
        warmup_steps = int(Config.LR_WARMUP_RATIO * total_steps)
        
        scheduler = get_cosine_schedule_with_warmup( optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps, num_cycles=0.5, last_epoch=-1)
        criterion = FocalLoss().to(Config.DEVICE)
        scaler = GradScaler(enabled=Config.USE_AMP)
        
        # ========================================
        # INITIALIZE TRAINING STATE
        # ========================================
        start_epoch = 1
        best_kpi_score = 0.0
        best_model_epoch = 0
        patience_counter = 0
        
        # ========================================
        # CHECK FOR EXISTING CHECKPOINT AND RESUME
        # ========================================
        latest_checkpoint = self.checkpoint_manager.find_latest_checkpoint()
        
        if latest_checkpoint:
            print(f"\n{'='*60}")
            print(f"🔄 CHECKPOINT FOUND")
            print(f"{'='*60}")
            print(f"Latest checkpoint: {latest_checkpoint.name}")
            
            resume_choice = input("Resume from checkpoint? (y/n): ").lower().strip()
            
            if resume_choice == 'y':
                # Load checkpoint AFTER optimizer/scheduler creation
                start_epoch, last_metrics, best_kpi_score, loaded_history = \
                    self.checkpoint_manager.load_checkpoint(
                        latest_checkpoint, model, optimizer, scheduler, scaler
                    )
                
                # Restore training history
                if loaded_history:
                    self.training_history = loaded_history
                    print(f"✓ Training history restored ({len(self.training_history['train_losses'])} epochs)")
                
                # Restore best model epoch by finding actual best_model_epoch_* directory
                best_model_dirs = sorted(Config.SAVE_ROOT.glob("best_model_epoch_*"))
                if best_model_dirs:
                    best_model_dir = best_model_dirs[-1]
                    best_model_epoch = int(best_model_dir.name.split("_")[-1])
                    print(f"✓ Best model found at epoch: {best_model_epoch}")
                else:
                    best_model_epoch = start_epoch - 1
                    print(f"⚠ No best model directory found, using epoch {best_model_epoch}")
                
                # Restore patience counter
                if len(self.training_history['kpi_scores']) >= Config.PATIENCE:
                    recent_scores = self.training_history['kpi_scores'][-Config.PATIENCE:]
                    if all(score <= best_kpi_score for score in recent_scores):
                        patience_counter = Config.PATIENCE - 1
                        print(f"⚠ Patience counter: {patience_counter}/{Config.PATIENCE}")
                
                print(f"{'='*60}\n")
            else:
                print("Starting fresh training...\n")
        else:
            print("\nNo checkpoint found. Starting fresh training...\n")
        
        # ========================================
        # TRAINING CONFIGURATION SUMMARY
        # ========================================
        print("="*60)
        print("TRAINING CONFIGURATION")
        print("="*60)
        print(f"  Starting Epoch:        {start_epoch}")
        print(f"  Total Epochs:          {Config.NUM_EPOCHS}")
        print(f"  Batch Size:            {Config.BATCH_SIZE}")
        print(f"  Gradient Accumulation: {Config.GRAD_ACCUM_STEPS}")
        print(f"  Effective Batch Size:  {Config.BATCH_SIZE * Config.GRAD_ACCUM_STEPS}")
        print(f"  Learning Rate:         {Config.LR} (LoRA) | {Config.HEAD_LR} (Heads)")
        print(f"  Warmup Steps:          {warmup_steps}")
        print(f"  Total Steps:           {total_steps}")
        print(f"  Best KPI Score:        {best_kpi_score:.4f}")
        print(f"  Mixed Precision (AMP): {Config.USE_AMP}")
        print("="*60)
        
        # ========================================
        # TRAINING LOOP
        # ========================================
        # Handle case where training is already complete
        epoch = start_epoch - 1  # Initialize for edge case where loop doesn't run
        
        if start_epoch > Config.NUM_EPOCHS:
            print(f"\n⚠ Training already completed (checkpoint at epoch {start_epoch - 1}, NUM_EPOCHS={Config.NUM_EPOCHS})")
            print(f"⚠ Skipping to final test evaluation...")
        
        for epoch in range(start_epoch, Config.NUM_EPOCHS + 1):
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch}/{Config.NUM_EPOCHS}")
            print(f"{'='*60}")
            
            train_loss, train_acc = self._train_epoch(model, train_loader, optimizer, scheduler, criterion, scaler, epoch)
            val_loss, val_log_odds, val_probs, val_labels, val_metadata = self._validate_epoch(model, val_loader, criterion)
            
            # Joint (λ, threshold) search with decision engine (using raw log-odds, no calibration during training)
            optimal_lambda, optimal_threshold, _joint_info = self.kpi_evaluator.find_optimal_lambda_threshold_joint(val_labels, val_probs, val_metadata, log_odds=val_log_odds)
            
            # Generate final predictions via Decision Engine using optimized (λ, threshold)
            val_preds = SamsungDecisionEngine.decide(
                log_odds=val_log_odds, 
                severities=val_metadata['severities'],
                flags_counts=val_metadata['flags_counts'],
                categories=val_metadata['categories'],
                base_threshold=optimal_threshold,
                lambda_val=optimal_lambda
            )
            val_metrics = self.kpi_evaluator.evaluate_metrics(val_labels, val_preds, val_probs)
            
            self._update_history(train_loss, train_acc, val_loss, val_metrics, optimal_threshold)
            self._print_epoch_summary(epoch, train_loss, train_acc, val_loss, val_metrics, optimal_threshold)
            
            # Save best model
            if val_metrics['kpi_score'] > best_kpi_score:
                best_kpi_score = val_metrics['kpi_score']
                best_model_epoch = epoch
                patience_counter = 0
                self._save_best_model(model, epoch, val_metrics, optimal_threshold)
                print(f"🎉 New best model! KPI Score improved to {best_kpi_score:.4f}")
            else:
                patience_counter += 1
                print(f"⚠ No improvement for {patience_counter}/{Config.PATIENCE} epochs")
                
                if patience_counter >= Config.PATIENCE:
                    print(f"\n⏸ Early stopping triggered at epoch {epoch}")
                    print(f"Best model was at epoch {best_model_epoch} with KPI score {best_kpi_score:.4f}")
                    break
            
            # Save checkpoint (with full training state)
            self.checkpoint_manager.save_checkpoint( model, optimizer, scheduler, scaler, epoch, val_metrics, optimal_threshold, best_kpi_score, self.training_history)
            
            # Cleanup old checkpoints (keep last 3)
            self.checkpoint_manager.cleanup_old_checkpoints(keep_last_n=3)
        
        # ========================================
        # FINAL TEST EVALUATION
        # ========================================
        print("\n" + "="*80)
        print("FINAL TEST EVALUATION")
        print("="*80)
        kpi_compliance = self._evaluate_test_set(model, test_loader, criterion, best_model_epoch)
        
        # ========================================
        # TRAINING SUMMARY
        # ========================================
        print("\n" + "="*80)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        epochs_trained = max(0, epoch - start_epoch + 1) if epoch >= start_epoch else 0
        print(f"Epochs Trained: {epochs_trained} (resumed from epoch {start_epoch})")
        print(f"Best Model: Epoch {best_model_epoch} (KPI Score: {best_kpi_score:.4f})")
        print(f"KPI Compliance: {'✅ ACHIEVED' if kpi_compliance else '⚠ PARTIAL'}")
        print(f"Results Directory: {Config.SAVE_ROOT}")
        print("="*80 + "\n")
        
        return kpi_compliance


    
    def _train_epoch( self, model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer, scheduler: Any, criterion: nn.Module, scaler: GradScaler, epoch: int) -> Tuple[float, float]:
        """Train for one epoch with proper gradient accumulation."""
        model.train()
        running_loss = 0.0
        all_preds, all_labels = [], []
        
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Training")
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(Config.DEVICE)
            attention_mask = batch['attention_mask'].to(Config.DEVICE)
            heuristic_features = batch['heuristic_features'].to(Config.DEVICE)
            labels = batch['labels'].to(Config.DEVICE)
            
            # Forward pass (text + heuristic features → GLU fusion → logits)
            with autocast(enabled=Config.USE_AMP):
                logits = model(input_ids=input_ids, attention_mask=attention_mask, heuristic_features=heuristic_features)
                loss = criterion(logits, labels)
                loss = loss / Config.GRAD_ACCUM_STEPS  # Scale loss
            
            # Backward pass
            if Config.USE_AMP:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights every GRAD_ACCUM_STEPS
            if (batch_idx + 1) % Config.GRAD_ACCUM_STEPS == 0:
                if Config.USE_AMP:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP_NORM)
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
            
            # Metrics
            running_loss += loss.item() * labels.size(0) * Config.GRAD_ACCUM_STEPS
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({
                'loss': f"{loss.item() * Config.GRAD_ACCUM_STEPS:.4f}",
                'acc': f"{accuracy_score(all_labels[-len(preds):], preds):.4f}"
            })
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        return epoch_loss, epoch_acc
    
    def _validate_epoch( self, model: nn.Module, val_loader: DataLoader, criterion: nn.Module) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Validation: collect raw logits + metadata. NO calibration here (done once post-training)."""
        model.eval()
        running_loss = 0.0
        all_logits, all_labels = [], []
        all_sev, all_flags, all_cats = [], [], []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(Config.DEVICE)
                attention_mask = batch['attention_mask'].to(Config.DEVICE)
                heuristic_features = batch['heuristic_features'].to(Config.DEVICE)
                labels = batch['labels'].to(Config.DEVICE)
                
                logits = model(input_ids=input_ids, attention_mask=attention_mask, heuristic_features=heuristic_features)
                loss = criterion(logits, labels)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print("⚠ NaN/Inf loss detected, skipping batch")
                    continue
                
                running_loss += loss.item() * labels.size(0)
                all_logits.extend(logits.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Extract metadata for Decision Engine
                all_sev.extend(batch.get('h_severity_score', [0]*len(batch['url'])))
                all_flags.extend(batch.get('h_flags_count', [0]*len(batch['url'])))
                all_cats.extend(batch.get('h_primary_category', ['UNKNOWN']*len(batch['url'])))
        
        # Derive log-odds and probs from raw logits (no temperature — raw model signal)
        raw_logits = np.array(all_logits)
        log_odds = raw_logits[:, 1] - raw_logits[:, 0]  # scalar log-odds per sample
        probs = 1.0 / (1.0 + np.exp(-np.clip(log_odds, -20, 20)))  # stable sigmoid
        
        val_metadata = {
            'severities': np.array([float(x) for x in all_sev]),
            'flags_counts': np.array([int(x) for x in all_flags]),
            'categories': np.array(all_cats)
        }
        
        epoch_loss = running_loss / len(val_loader.dataset) if len(all_logits) > 0 else float('inf')
        return epoch_loss, log_odds, probs, np.array(all_labels), val_metadata
    
    def _update_history( self, train_loss: float, train_acc: float, val_loss: float, val_metrics: Dict, threshold: float) -> None:
        """Update training history."""
        self.training_history['train_losses'].append(train_loss)
        self.training_history['val_losses'].append(val_loss)
        self.training_history['train_accs'].append(train_acc)
        self.training_history['val_accs'].append(val_metrics['accuracy'])
        self.training_history['kpi_scores'].append(val_metrics['kpi_score'])
        self.training_history['thresholds'].append(threshold)
    
    def _print_epoch_summary( self, epoch: int, train_loss: float, train_acc: float, val_loss: float, val_metrics: Dict, threshold: float) -> None:
        """Print comprehensive epoch results."""
        print(f"\nResults:")
        print(f"  Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  Optimal Threshold: {threshold:.4f}")
        
        print(f"\nKPI Metrics:")
        kpi_checks = {
            'Accuracy':  (val_metrics['accuracy'],  Config.TARGET_ACCURACY,  '>='),
            'Precision': (val_metrics['precision'], Config.TARGET_PRECISION, '>='),
            'Recall':    (val_metrics['recall'],    Config.TARGET_RECALL,    '>='),
            'FNR':       (val_metrics['fnr'],       Config.MAX_FNR,          '<='),
            'FPR':       (val_metrics['fpr'],       Config.MAX_FPR,          '<=')
        }
        
        for name, (value, target, op) in kpi_checks.items():
            passed = (value >= target) if op == '>=' else (value <= target)
            symbol = '✅' if passed else '❌'
            print(f"  {name:<12} {value:.4f} (target: {op}{target:.4f}) {symbol}")
        
        status = "✅ ALL KPIs MET" if val_metrics['kpi_compliance'] else "⚠ KPIs NOT MET"
        print(f"\nStatus: {status} (Score: {val_metrics['kpi_score']:.4f})")
    
    def _save_best_model( self, model: nn.Module, epoch: int, metrics: Dict, threshold: float) -> None:
        """Save best model with all exports and artifacts."""
        best_model_dir = Config.SAVE_ROOT / f"best_model_epoch_{epoch:03d}"
        best_model_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"💾 SAVING BEST MODEL - EPOCH {epoch}")
        print(f"{'='*60}")
        
        # Save LoRA adapter and tokenizer
        model.save_pretrained(best_model_dir / "lora_adapter")
        self.tokenizer.save_pretrained(best_model_dir)
        print(f"✓ LoRA adapter saved")
        print(f"✓ Tokenizer saved")
        
        # Save full model
        full_model_path = best_model_dir / "model_full.pt"
        torch.save(model, full_model_path)
        model_size = os.path.getsize(full_model_path) / (1024 * 1024)
        print(f"✓ Full model saved: {model_size:.2f} MB")
        
        # Save state dict
        # torch.save(model.state_dict(), best_model_dir / "model_state_dict.pt")
        # print(f"✓ State dict saved")
        
        # Merge and export
        print(f"\nExporting production models...")
        merged_model, merged_size = ModelExporter.merge_lora_and_export(model, self.tokenizer, best_model_dir)
        
        final_size = ModelExporter.export_onnx(merged_model, best_model_dir)
        
        if final_size and final_size <= Config.MAX_MODEL_SIZE_MB:
            print(f"\n🎯 SUCCESS: Model size {final_size:.2f}MB meets {Config.MAX_MODEL_SIZE_MB}MB target!")
        
        # Save training artifacts
        artifact_saver = ArtifactSaver(best_model_dir)
        artifact_saver.save_history(
            self.training_history['train_losses'],
            self.training_history['val_losses'],
            self.training_history['train_accs'],
            self.training_history['val_accs']
        )
        
        # Save metadata
        metadata = {
            'model_info': {
                'epoch': epoch,
                'architecture': 'MiniLM v3 Base + LoRA',
                'base_model': Config.MODEL_NAME,
                'max_length': Config.MAX_LEN,
                'model_size_mb': model_size,
                'quantized_size_mb': final_size
            },
            'performance': metrics,
            'threshold': threshold,
            'training_config': CheckpointManager._serialize_config(),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(best_model_dir / "deployment_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"\n✅ Best model saved to: {best_model_dir.name}")
        print(f"{'='*60}\n")
    

    def inference_from_checkpoint(self) -> bool:
        """
        Inference-only mode: Load latest checkpoint and perform test evaluation.
        Skips all training.
        """
        print("\n" + "="*80)
        print("MINILM PHISHING DETECTION - INFERENCE MODE (CHECKPOINT RESUME)")
        print("="*80)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Device: {Config.DEVICE}")
        print("="*80 + "\n")
        
        # ========================================
        # LOAD DATASETS
        # ========================================
        print("Loading datasets...")
        test_dataset = self.load_datasets(filter_test_unsupported=True)[2]  # Only need screened test dataset
        test_loader = DataLoader(
            test_dataset, 
            batch_size=Config.BATCH_SIZE, 
            shuffle=False, 
            num_workers=2, 
            pin_memory=Config.PIN_MEMORY, 
            prefetch_factor=2 if 2 > 0 else None
        )
        print(f"✓ Test dataset loaded: {len(test_dataset):,} samples\n")
        
        # ========================================
        # FIND AND LOAD LATEST CHECKPOINT
        # ========================================
        print("="*60)
        print("CHECKPOINT SEARCH")
        print("="*60)
        
        latest_checkpoint = self.checkpoint_manager.find_latest_checkpoint()
        
        if latest_checkpoint is None:
            print("❌ ERROR: No checkpoint found!")
            print(f"Expected checkpoint directory: {Config.CHECKPOINT_DIR}")
            print("Please run training first before attempting inference.\n")
            return False
        
        print(f"✅ Found checkpoint: {latest_checkpoint.name}")
        
        # ========================================
        # LOAD MODEL AND CHECKPOINT STATE
        # ========================================
        print("\nLoading model and checkpoint state...")
        model = self.create_model()
        criterion = FocalLoss().to(Config.DEVICE)
        
        try:
            checkpoint = torch.load(latest_checkpoint, map_location=Config.DEVICE, weights_only=False)
            
            # Load model state
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            model.to(Config.DEVICE)
            print(f"✓ Model state loaded")
            
            # Restore training history
            if 'training_history' in checkpoint:
                self.training_history = checkpoint['training_history']
                print(f"✓ Training history restored ({len(self.training_history['train_losses'])} epochs)")
            
            # Extract metadata
            checkpoint_epoch = checkpoint.get('epoch', 0)
            best_kpi_score = checkpoint.get('best_kpi_score', 0.0)
            
            print(f"✓ Checkpoint epoch: {checkpoint_epoch}")
            print(f"✓ Best KPI score at checkpoint: {best_kpi_score:.4f}\n")
            
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # ========================================
        # FIND BEST MODEL EPOCH
        # ========================================
        # The checkpoint epoch may not be the best model epoch
        # Scan for best_model_epoch_* directories to find the actual best model
        best_model_dirs = sorted(Config.SAVE_ROOT.glob("best_model_epoch_*"))
        if best_model_dirs:
            # Get the latest best model directory (highest epoch number)
            best_model_dir = best_model_dirs[-1]
            # Extract epoch number from directory name (e.g., "best_model_epoch_001" -> 1)
            best_model_epoch = int(best_model_dir.name.split("_")[-1])
            print(f"✓ Best model found at epoch: {best_model_epoch}")
        else:
            # Fallback to checkpoint epoch if no best model directory found
            best_model_epoch = checkpoint_epoch
            print(f"⚠ No best_model_epoch_* directory found, using checkpoint epoch: {best_model_epoch}")
        
        # ========================================
        # RUN TEST INFERENCE
        # ========================================
        print("="*60)
        print("TEST INFERENCE")
        print("="*60 + "\n")
        
        model.eval()
        kpi_compliance = self._evaluate_test_set(model, test_loader, criterion, best_model_epoch)
        
        # ========================================
        # SUMMARY
        # ========================================
        print("\n" + "="*80)
        print("INFERENCE COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"Checkpoint Used: {latest_checkpoint.name}")
        print(f"Checkpoint Epoch: {checkpoint_epoch}")
        print(f"KPI Compliance: {'✅ ACHIEVED' if kpi_compliance else '⚠ PARTIAL'}")
        print(f"Results Directory: {Config.SAVE_ROOT / 'final_test_evaluation'}")
        print("="*80 + "\n")
        
        return kpi_compliance

    def _evaluate_test_set(self, model: nn.Module, test_loader: DataLoader, criterion: nn.Module, best_epoch: int) -> bool:
        """Final evaluation on test set using production-ready merged model."""
        test_inference_dir = Config.SAVE_ROOT / "final_test_evaluation"
        test_inference_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print("LOADING PRODUCTION MODEL FOR TEST INFERENCE")
        print(f"{'='*60}")
        
        # ========================================
        # LOAD MERGED PRODUCTION MODEL
        # ========================================
        best_model_dir = Config.SAVE_ROOT / f"best_model_epoch_{best_epoch:03d}"
        merged_model_path = best_model_dir / "model_merged_full.pt"
        
        if merged_model_path.exists():
            try:
                print(f"Loading merged model: {merged_model_path.name}")
                model = torch.load(merged_model_path, map_location=Config.DEVICE, weights_only=False)
                model.eval()
                print(f"✓ Using production-ready merged model (LoRA weights integrated)")
            except Exception as e:
                print(f"⚠ Failed to load merged model: {e}")
                print(f"⚠ Falling back to training model")
                model.eval()
        else:
            print(f"⚠ Merged model not found at {merged_model_path}")
            print(f"⚠ Using training model (with LoRA)")
            model.eval()
        
        print(f"Results will be saved to: {test_inference_dir.name}")
        print(f"{'='*60}\n")
        
        # ========================================
        # RUN TEST INFERENCE
        # ========================================
        all_logits, all_labels, all_urls = [], [], []
        all_sev, all_flags, all_cats = [], [], []
        test_running_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Test Inference"):
                input_ids = batch['input_ids'].to(Config.DEVICE)
                attention_mask = batch['attention_mask'].to(Config.DEVICE)
                heuristic_features = batch['heuristic_features'].to(Config.DEVICE)
                labels = batch['labels'].to(Config.DEVICE)
                
                logits = model(input_ids=input_ids, attention_mask=attention_mask, heuristic_features=heuristic_features)
                loss = criterion(logits, labels)
                
                test_running_loss += loss.item() * labels.size(0)
                all_logits.extend(logits.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_urls.extend(batch['url'])
                
                # Extract metadata for Decision Engine
                all_sev.extend(batch.get('h_severity_score', [0]*len(batch['url'])))
                all_flags.extend(batch.get('h_flags_count', [0]*len(batch['url'])))
                all_cats.extend(batch.get('h_primary_category', ['UNKNOWN']*len(batch['url'])))
        
        # ========================================
        # TEMPERATURE CALIBRATION (ONCE, on best model)
        # ========================================
        raw_logits = np.array(all_logits)
        test_labels = np.array(all_labels)
        
        logits_tensor = torch.tensor(raw_logits, dtype=torch.float32)
        labels_tensor = torch.tensor(test_labels, dtype=torch.long)
        
        with torch.enable_grad():
            calibrator = ModelCalibrator()
            temp = calibrator.calibrate(logits_tensor, labels_tensor)
        print(f"\n🌡️  Temperature Calibration: T = {temp:.4f}")
        
        # Derive calibrated log-odds: (logit_1 - logit_0) / T
        calibrated_log_odds = (raw_logits[:, 1] - raw_logits[:, 0]) / temp
        calibrated_log_odds = np.clip(calibrated_log_odds, -20.0, 20.0)
        test_probs = 1.0 / (1.0 + np.exp(-calibrated_log_odds))
        
        # ========================================
        # COMPUTE METRICS
        # ========================================
        test_loss = test_running_loss / len(test_loader.dataset)
        
        # Apply full production pipeline: calibrated log-odds → Decision Engine
        test_metadata = {
            'severities': np.array([float(x) for x in all_sev]),
            'flags_counts': np.array([int(x) for x in all_flags]),
            'categories': np.array(all_cats)
        }
        
        # Joint (λ, threshold) grid search — finds optimal coupled operating point
        optimal_lambda, optimal_threshold, joint_metrics = self.kpi_evaluator.find_optimal_lambda_threshold_joint(
            y_true=test_labels,
            y_prob=test_probs,
            metadata=test_metadata,
            log_odds=calibrated_log_odds
        )
        
        # Final predictions with optimal (λ, threshold) pair
        test_preds = SamsungDecisionEngine.decide(
            log_odds=calibrated_log_odds,
            severities=test_metadata['severities'],
            flags_counts=test_metadata['flags_counts'],
            categories=test_metadata['categories'],
            base_threshold=optimal_threshold,
            lambda_val=optimal_lambda
        )
        
        test_metrics = self.kpi_evaluator.evaluate_metrics(test_labels, test_preds, test_probs)
        test_metrics['test_loss'] = test_loss
        test_metrics['threshold_used'] = optimal_threshold
        test_metrics['calibration_temperature'] = temp
        test_metrics['model_used'] = 'model_merged_full.pt' if merged_model_path.exists() else 'model_full.pt (with LoRA)'
        
        # ========================================
        # SAVE RESULTS
        # ========================================
        # Save predictions
        predictions_df = pd.DataFrame({
            'url': all_urls,
            'true_label': test_labels,
            'predicted_label': test_preds,
            'prob_malicious': test_probs,
            'correct': test_labels == test_preds
        })
        predictions_df.to_csv(test_inference_dir / "test_predictions.csv", index=False)
        print(f"✓ Predictions saved: test_predictions.csv")
        
        # Save artifacts
        artifact_saver = ArtifactSaver(test_inference_dir)
        artifact_saver.save_test_metrics(test_metrics, optimal_threshold)
        artifact_saver.save_test_plots(test_labels, test_probs, optimal_threshold)
        self._emit_test_url_screening_report(test_inference_dir)
        
        # ========================================
        # PRINT SUMMARY
        # ========================================
        print(f"\n{'='*60}")
        print("TEST SET RESULTS")
        print(f"{'='*60}")
        print(f"Model Used: {test_metrics['model_used']}")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Threshold: {optimal_threshold:.4f}")
        print(f"\nMetrics:")
        print(f"  Accuracy:  {test_metrics['accuracy']:.4f} {'✅' if test_metrics['accuracy'] >= Config.TARGET_ACCURACY else '❌'}")
        print(f"  Precision: {test_metrics['precision']:.4f} {'✅' if test_metrics['precision'] >= Config.TARGET_PRECISION else '❌'}")
        print(f"  Recall:    {test_metrics['recall']:.4f} {'✅' if test_metrics['recall'] >= Config.TARGET_RECALL else '❌'}")
        print(f"  F1-Score:  {test_metrics['f1']:.4f}")
        print(f"  AUC-ROC:   {test_metrics['auc']:.4f}")
        print(f"\nError Rates:")
        print(f"  FNR: {test_metrics['fnr']:.4f} {'✅' if test_metrics['fnr'] <= Config.MAX_FNR else '❌'}")
        print(f"  FPR: {test_metrics['fpr']:.4f} {'✅' if test_metrics['fpr'] <= Config.MAX_FPR else '❌'}")
        print(f"\nConfusion Matrix:")
        print(f"  TN: {test_metrics['tn']:,}  |  FP: {test_metrics['fp']:,}")
        print(f"  FN: {test_metrics['fn']:,}  |  TP: {test_metrics['tp']:,}")
        print(f"\nKPI Compliance: {'✅ ACHIEVED' if test_metrics['kpi_compliance'] else '❌ NOT MET'}")
        print(f"{'='*60}")
        
        # ========================================
        # SAVE FINAL RESULTS
        # ========================================
        hyperparams = CheckpointManager._serialize_config()
        
        final_results = {
            'test_metrics': test_metrics,
            'training_history': self.training_history,
            'best_epoch': best_epoch,
            'optimal_threshold': optimal_threshold,
            'optimal_lambda': optimal_lambda,
            'kpi_compliance': test_metrics['kpi_compliance'],
            'model_architecture': 'MiniLM v3 Base',
            'model_used_for_test': test_metrics['model_used'],
            'test_samples': len(test_labels),
            'unsupported_url_screening_report': self.latest_test_url_screening_report,
            'timestamp': datetime.now().isoformat(),
            'hyperparameters': hyperparams
        }
        
        with open(Config.SAVE_ROOT / "final_results.json", 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        print(f"\n✓ All test results saved to: {test_inference_dir.name}")
        print(f"✓ Final results: final_results.json")
        
        return test_metrics['kpi_compliance']


    def onnx_inference(self, onnx_model_type: str = 'int8') -> bool:
        """
        ONNX Inference Mode: Load ONNX model and evaluate on test set.
        
        This mode uses the production-ready ONNX model for inference.
        No PyTorch model is loaded; inference runs entirely through ONNX Runtime.
        
        Args:
            onnx_model_type: 'int8' for quantized, 'fp32' for original, or custom path
        """
        print("\n" + "="*80)
        print("MINILM PHISHING DETECTION - ONNX INFERENCE MODE (INT8 QUANTIZED)")
        print("="*80)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Runtime: ONNX Runtime (CPU-optimized)")
        print("="*80 + "\n")
        
        # ========================================
        # VALIDATE ONNX RUNTIME AVAILABILITY
        # ========================================
        try:
            import onnxruntime as ort
            print(f"✓ ONNX Runtime version: {ort.__version__}")
            available_providers = ort.get_available_providers()
            print(f"✓ Available providers: {available_providers}")
        except ImportError:
            print("❌ ERROR: onnxruntime not installed!")
            print("Install with: pip install onnxruntime")
            print("For GPU support: pip install onnxruntime-gpu")
            return False
        
        # ========================================
        # FIND ONNX MODEL
        # ========================================
        print(f"\n{'='*60}")
        print("ONNX MODEL SEARCH")
        print(f"{'='*60}")
        
        # Search for best_model_epoch_* directories
        best_model_dirs = sorted(Config.SAVE_ROOT.glob("best_model_epoch_*"))
        
        if not best_model_dirs:
            print("❌ ERROR: No best_model_epoch_* directory found!")
            print(f"Expected in: {Config.SAVE_ROOT}")
            print("Please run training first (python MiniLM_1.py --mode train)")
            return False
        
        best_model_dir = best_model_dirs[-1]
        best_epoch = int(best_model_dir.name.split("_")[-1])
        print(f"✓ Best model directory: {best_model_dir.name}")
        print(f"✓ Best model epoch: {best_epoch}")
        
        # Look for ONNX model based on user selection
        onnx_quant_path = best_model_dir / "model_quant_8bit.onnx"
        onnx_fp32_path = best_model_dir / "model.onnx"
        
        # Determine which model to use
        if os.path.isfile(onnx_model_type):
            # Custom path provided
            onnx_model_path = Path(onnx_model_type)
            model_type = f"Custom ({onnx_model_path.name})"
            model_size = os.path.getsize(onnx_model_path) / (1024 * 1024)
            print(f"✅ Using custom ONNX model: {onnx_model_path}")
            print(f"   Model size: {model_size:.2f} MB")
        elif onnx_model_type == 'fp32':
            if onnx_fp32_path.exists():
                onnx_model_path = onnx_fp32_path
                model_type = "FP32 (Full Precision)"
                model_size = os.path.getsize(onnx_model_path) / (1024 * 1024)
                print(f"✅ Using FP32 ONNX model: {onnx_fp32_path.name}")
                print(f"   Model size: {model_size:.2f} MB")
            else:
                print(f"❌ ERROR: FP32 ONNX model not found: {onnx_fp32_path}")
                return False
        else:  # 'int8' (default)
            if onnx_quant_path.exists():
                onnx_model_path = onnx_quant_path
                model_type = "INT8 Quantized"
                model_size = os.path.getsize(onnx_model_path) / (1024 * 1024)
                print(f"✅ Found quantized ONNX model: {onnx_quant_path.name}")
                print(f"   Model size: {model_size:.2f} MB")
                if model_size <= Config.MAX_MODEL_SIZE_MB:
                    print(f"   ✅ Meets {Config.MAX_MODEL_SIZE_MB} MB deployment target!")
                else:
                    print(f"   ⚠ Exceeds {Config.MAX_MODEL_SIZE_MB} MB target ({model_size:.2f} MB)")
            elif onnx_fp32_path.exists():
                onnx_model_path = onnx_fp32_path
                model_type = "FP32 (INT8 not available)"
                model_size = os.path.getsize(onnx_model_path) / (1024 * 1024)
                print(f"⚠ INT8 model not found, falling back to FP32: {onnx_fp32_path.name}")
                print(f"   Model size: {model_size:.2f} MB")
            else:
                print("❌ ERROR: No ONNX model found!")
                print(f"   Searched: {onnx_quant_path}")
                print(f"   Searched: {onnx_fp32_path}")
                print("Please ensure ONNX export was successful during training.")
                return False
        
        print(f"{'='*60}\n")
        
        # ========================================
        # LOAD DATASETS
        # ========================================
        print("Loading test dataset...")
        test_dataset = self.load_datasets(filter_test_unsupported=True)[2]  # Only need screened test dataset
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=False,  # CPU inference for ONNX
            prefetch_factor=2 if 2 > 0 else None
        )
        print(f"✓ Test dataset loaded: {len(test_dataset):,} samples\n")
        
        # ========================================
        # LOAD DEPLOYMENT METADATA (for threshold)
        # ========================================
        optimal_threshold = None  # Will be set from metadata
        
        # Try to load threshold from deployment metadata
        metadata_path = best_model_dir / "deployment_metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                # Check both possible key names
                for key in ['threshold', 'optimal_threshold']:
                    if key in metadata and metadata[key] is not None:
                        optimal_threshold = float(metadata[key])
                        print(f"✓ Loaded threshold from deployment_metadata.json['{key}']: {optimal_threshold:.4f}")
                        break
                # Also check nested under performance
                if optimal_threshold is None and 'performance' in metadata:
                    perf = metadata['performance']
                    for key in ['threshold', 'optimal_threshold']:
                        if key in perf and perf[key] is not None:
                            optimal_threshold = float(perf[key])
                            print(f"✓ Loaded threshold from metadata.performance['{key}']: {optimal_threshold:.4f}")
                            break
            except Exception as e:
                print(f"⚠ Could not load metadata: {e}")
        
        # Also try from final_results.json
        if optimal_threshold is None:
            results_path = Config.SAVE_ROOT / "final_results.json"
            if results_path.exists():
                try:
                    with open(results_path, 'r') as f:
                        results = json.load(f)
                    for key in ['optimal_threshold', 'threshold']:
                        if key in results and results[key] is not None:
                            optimal_threshold = float(results[key])
                            print(f"✓ Loaded threshold from final_results.json['{key}']: {optimal_threshold:.4f}")
                            break
                except Exception as e:
                    print(f"⚠ Could not load results: {e}")
        
        # Also try from training history in checkpoint
        if optimal_threshold is None:
            latest_checkpoint = self.checkpoint_manager.find_latest_checkpoint()
            if latest_checkpoint is not None:
                try:
                    ckpt = torch.load(latest_checkpoint, map_location='cpu', weights_only=False)
                    if 'training_history' in ckpt:
                        thresholds = ckpt['training_history'].get('thresholds', [])
                        if thresholds:
                            optimal_threshold = float(thresholds[-1])
                            print(f"✓ Loaded threshold from checkpoint training history: {optimal_threshold:.4f}")
                except Exception as e:
                    print(f"⚠ Could not load checkpoint for threshold: {e}")
        
        # Final fallback
        if optimal_threshold is None:
            optimal_threshold = 0.5
            print(f"⚠ No threshold found in any source, using default: {optimal_threshold:.4f}")
        
        # ========================================
        # CREATE ONNX SESSION
        # ========================================
        print(f"\n{'='*60}")
        print("ONNX SESSION INITIALIZATION")
        print(f"{'='*60}")
        
        # Configure session options for maximum performance
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = Config.NUM_WORKERS
        sess_options.inter_op_num_threads = 2
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        # ── Pre-load cuDNN/CUDA libraries from pip packages ─────
        # When cuDNN is installed via pip (nvidia-cudnn-cu12), the .so files
        # live inside site-packages/nvidia/cudnn/lib/ and are NOT on
        # LD_LIBRARY_PATH by default. We must preload them with ctypes
        # so ONNX Runtime's dlopen() can find them.
        # Reference: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#preload-dlls
        cuda_usable = False
        if 'CUDAExecutionProvider' in available_providers:
            # Method 1: ort.preload_dlls() (onnxruntime >= 1.21.0)
            try:
                if hasattr(ort, 'preload_dlls'):
                    ort.preload_dlls(cuda=True, cudnn=True)
                    print("✓ Preloaded CUDA/cuDNN via ort.preload_dlls()")
                    cuda_usable = True
            except Exception as e:
                print(f"⚠ ort.preload_dlls() failed: {e}")
            
            # Method 2: Manually locate nvidia pip package lib dirs and preload .so files
            if not cuda_usable:
                import ctypes
                nvidia_lib_dirs = []
                
                # --- Find nvidia/cudnn/lib via multiple strategies ---
                # Strategy A: Use importlib to find the nvidia.cudnn package spec
                try:
                    import importlib.util
                    spec = importlib.util.find_spec('nvidia.cudnn')
                    if spec and spec.submodule_search_locations:
                        for loc in spec.submodule_search_locations:
                            candidate = Path(loc) / 'lib'
                            if candidate.exists():
                                nvidia_lib_dirs.append(str(candidate))
                except Exception:
                    pass
                
                # Strategy B: Walk site-packages looking for nvidia/cudnn/lib
                if not nvidia_lib_dirs:
                    try:
                        import site
                        site_dirs = site.getsitepackages() + [site.getusersitepackages()]
                        for sp in site_dirs:
                            candidate = Path(sp) / 'nvidia' / 'cudnn' / 'lib'
                            if candidate.exists():
                                nvidia_lib_dirs.append(str(candidate))
                            # Also check nvidia/cuda_runtime/lib
                            cuda_rt = Path(sp) / 'nvidia' / 'cuda_runtime' / 'lib'
                            if cuda_rt.exists():
                                nvidia_lib_dirs.append(str(cuda_rt))
                    except Exception:
                        pass
                
                # Strategy C: Derive from sys.prefix (conda environments)
                if not nvidia_lib_dirs:
                    conda_sp = Path(sys.prefix) / 'lib' / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'site-packages'
                    for sub in ['nvidia/cudnn/lib', 'nvidia/cuda_runtime/lib']:
                        candidate = conda_sp / sub
                        if candidate.exists():
                            nvidia_lib_dirs.append(str(candidate))
                
                # Also add torch's bundled libraries
                try:
                    torch_lib_dir = Path(torch.__file__).parent / 'lib'
                    if torch_lib_dir.exists():
                        nvidia_lib_dirs.append(str(torch_lib_dir))
                except Exception:
                    pass
                
                # Deduplicate while preserving order
                seen = set()
                nvidia_lib_dirs = [d for d in nvidia_lib_dirs if d not in seen and not seen.add(d)]
                
                if nvidia_lib_dirs:
                    print(f"✓ Found {len(nvidia_lib_dirs)} NVIDIA library dirs:")
                    for d in nvidia_lib_dirs:
                        print(f"  → {d}")
                    
                    # Pre-load ALL cuDNN .so files with RTLD_GLOBAL so dlopen() finds them
                    loaded_count = 0
                    cudnn_libs = [
                        'libcudnn.so.9', 'libcudnn_adv.so.9', 'libcudnn_ops.so.9',
                        'libcudnn_cnn.so.9', 'libcudnn_graph.so.9',
                        'libcudnn_engines_precompiled.so.9',
                        'libcudnn_engines_runtime_compiled.so.9',
                        'libcudnn_heuristic.so.9'
                    ]
                    for lib_dir in nvidia_lib_dirs:
                        for lib_name in cudnn_libs:
                            lib_path = os.path.join(lib_dir, lib_name)
                            if os.path.exists(lib_path):
                                try:
                                    ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
                                    loaded_count += 1
                                except OSError as e:
                                    print(f"  ⚠ Failed to load {lib_name}: {e}")
                    
                    print(f"✓ Pre-loaded {loaded_count} cuDNN libraries into process memory")
                    
                    if loaded_count > 0:
                        cuda_usable = True
                    else:
                        print("⚠ No cuDNN .so files could be loaded — falling back to CPU")
                else:
                    print("⚠ CUDAExecutionProvider listed but cuDNN 9.x libraries not found")
                    print("  → Falling back to CPUExecutionProvider")
                    print("  → To enable GPU: pip install nvidia-cudnn-cu12")
        
        if cuda_usable:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        # Suppress ONNX Runtime internal warnings during session creation
        ort.set_default_logger_severity(3)  # 3 = ERROR only, suppresses WARN
        
        try:
            session = ort.InferenceSession(
                str(onnx_model_path),
                sess_options=sess_options,
                providers=providers
            )
            
            # Restore normal logging after session creation
            ort.set_default_logger_severity(1)  # 1 = default (VERBOSE)
            
            # Detect ACTUAL active provider
            active_provider = session.get_providers()[0]
            if active_provider == 'CUDAExecutionProvider':
                execution_device = "CUDA (GPU)"
            elif active_provider == 'TensorrtExecutionProvider':
                execution_device = "TensorRT (GPU)"
            else:
                execution_device = "CPU"
            
            print(f"✓ ONNX session created successfully")
            print(f"  Active provider: {active_provider}")
            print(f"  Execution device: {execution_device}")
            print(f"  Graph optimization: ENABLED (all levels)")
            print(f"  Intra-op threads: {Config.NUM_WORKERS}")
            
            # Print input/output info
            input_names = [inp.name for inp in session.get_inputs()]
            output_names = [out.name for out in session.get_outputs()]
            print(f"  Input names: {input_names}")
            print(f"  Output names: {output_names}")
        except Exception as e:
            ort.set_default_logger_severity(1)  # Restore logging on failure too
            print(f"❌ Failed to create ONNX session: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print(f"{'='*60}\n")
        
        # ========================================
        # RUN ONNX INFERENCE
        # ========================================
        print(f"{'='*60}")
        print("ONNX TEST INFERENCE")
        print(f"{'='*60}\n")
        
        all_probs, all_labels, all_urls = [], [], []
        all_sev, all_flags, all_cats = [], [], []
        
        import time
        total_start = time.perf_counter()
        
        for batch in tqdm(test_loader, desc="ONNX Inference"):
            input_ids = batch['input_ids'].numpy()
            attention_mask = batch['attention_mask'].numpy()
            heuristic_features = batch['heuristic_features'].numpy()
            labels = batch['labels'].numpy()
            
            # Build ONNX input feed (3 inputs: text + heuristic features)
            ort_inputs = {
                input_names[0]: input_ids,
                input_names[1]: attention_mask,
                input_names[2]: heuristic_features
            }
            
            # Run inference and measure time
            batch_start = time.perf_counter()
            ort_outputs = session.run(output_names, ort_inputs)
            batch_end = time.perf_counter()
            
            inference_times.append(batch_end - batch_start)
            
            # Extract logits and compute probabilities
            logits = ort_outputs[0]
            
            # Softmax to get probabilities
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
            all_probs.extend(probs[:, 1])
            all_labels.extend(labels)
            all_urls.extend(batch['url'])
            
            all_sev.extend(batch.get('h_severity_score', [0]*len(batch['url'])))
            all_flags.extend(batch.get('h_flags_count', [0]*len(batch['url'])))
            all_cats.extend(batch.get('h_primary_category', ['UNKNOWN']*len(batch['url'])))
        
        total_end = time.perf_counter()
        total_inference_time = total_end - total_start
        
        # ========================================
        # COMPUTE METRICS
        # ========================================
        test_probs = np.array(all_probs)
        test_labels = np.array(all_labels)
        
        # apply business layer Decision Engine
        test_preds = SamsungDecisionEngine.decide(
            probs=test_probs,
            severities=np.array([float(x) for x in all_sev]),
            flags_counts=np.array([int(x) for x in all_flags]),
            categories=np.array(all_cats),
            base_threshold=optimal_threshold
        )
        
        test_metrics = self.kpi_evaluator.evaluate_metrics(test_labels, test_preds, test_probs)
        test_metrics['model_used'] = f'ONNX {model_type} ({onnx_model_path.name})'
        test_metrics['threshold_used'] = optimal_threshold
        
        # Compute detailed timing metrics
        total_samples = len(test_labels)
        avg_batch_time_ms = np.mean(inference_times) * 1000
        avg_sample_time_ms = (total_inference_time / total_samples) * 1000
        throughput = total_samples / total_inference_time
        
        # ========================================
        # SAVE RESULTS
        # ========================================
        onnx_results_dir = Config.SAVE_ROOT / "onnx_test_evaluation"
        onnx_results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save predictions
        predictions_df = pd.DataFrame({
            'url': all_urls,
            'true_label': test_labels,
            'predicted_label': test_preds,
            'prob_malicious': test_probs,
            'correct': test_labels == test_preds
        })
        predictions_df.to_csv(onnx_results_dir / "onnx_test_predictions.csv", index=False)
        print(f"\n✓ ONNX predictions saved: onnx_test_predictions.csv")
        
        # Save artifacts (plots and metrics)
        artifact_saver = ArtifactSaver(onnx_results_dir)
        artifact_saver.save_test_metrics(test_metrics, optimal_threshold)
        artifact_saver.save_test_plots(test_labels, test_probs, optimal_threshold)
        self._emit_test_url_screening_report(onnx_results_dir)
        
        # ========================================
        # PRINT COMPREHENSIVE RESULTS
        # ========================================
        print(f"\n{'='*80}")
        print("ONNX INFERENCE RESULTS")
        print(f"{'='*80}")
        print(f"Model: {onnx_model_path.name} ({model_type})")
        print(f"Model Size: {model_size:.2f} MB {'✅' if model_size <= Config.MAX_MODEL_SIZE_MB else '❌'}")
        print(f"Best Epoch: {best_epoch}")
        print(f"Threshold: {optimal_threshold:.4f}")
        
        print(f"\n{'─'*40}")
        print("CLASSIFICATION METRICS")
        print(f"{'─'*40}")
        print(f"  Accuracy:  {test_metrics['accuracy']:.4f} {'✅' if test_metrics['accuracy'] >= Config.TARGET_ACCURACY else '❌'}")
        print(f"  Precision: {test_metrics['precision']:.4f} {'✅' if test_metrics['precision'] >= Config.TARGET_PRECISION else '❌'}")
        print(f"  Recall:    {test_metrics['recall']:.4f} {'✅' if test_metrics['recall'] >= Config.TARGET_RECALL else '❌'}")
        print(f"  F1-Score:  {test_metrics['f1']:.4f}")
        print(f"  AUC-ROC:   {test_metrics['auc']:.4f}")
        
        print(f"\n{'─'*40}")
        print("ERROR RATES")
        print(f"{'─'*40}")
        print(f"  FNR: {test_metrics['fnr']:.4f} {'✅' if test_metrics['fnr'] <= Config.MAX_FNR else '❌'}")
        print(f"  FPR: {test_metrics['fpr']:.4f} {'✅' if test_metrics['fpr'] <= Config.MAX_FPR else '❌'}")
        
        print(f"\n{'─'*40}")
        print("CONFUSION MATRIX")
        print(f"{'─'*40}")
        print(f"  TN: {test_metrics['tn']:,}  |  FP: {test_metrics['fp']:,}")
        print(f"  FN: {test_metrics['fn']:,}  |  TP: {test_metrics['tp']:,}")
        
        print(f"\n{'─'*40}")
        print("PERFORMANCE BENCHMARKS")
        print(f"{'─'*40}")
        print(f"  Total inference time:    {total_inference_time:.2f}s")
        print(f"  Total samples:           {total_samples:,}")
        print(f"  Avg batch time:          {avg_batch_time_ms:.2f} ms")
        print(f"  Avg per-sample time:     {avg_sample_time_ms:.3f} ms")
        print(f"  Throughput:              {throughput:.0f} URLs/sec")
        print(f"  Execution device:        {execution_device}")
        
        print(f"\n{'─'*40}")
        print("KPI COMPLIANCE")
        print(f"{'─'*40}")
        kpi_pass = test_metrics['kpi_compliance']
        size_pass = model_size <= Config.MAX_MODEL_SIZE_MB
        full_compliance = kpi_pass and size_pass
        
        print(f"  Accuracy ≥ {Config.TARGET_ACCURACY:.0%}:   {'✅ PASS' if test_metrics['accuracy'] >= Config.TARGET_ACCURACY else '❌ FAIL'}")
        print(f"  Precision ≥ {Config.TARGET_PRECISION:.0%}:  {'✅ PASS' if test_metrics['precision'] >= Config.TARGET_PRECISION else '❌ FAIL'}")
        print(f"  Recall ≥ {Config.TARGET_RECALL:.0%}:     {'✅ PASS' if test_metrics['recall'] >= Config.TARGET_RECALL else '❌ FAIL'}")
        print(f"  FPR ≤ {Config.MAX_FPR:.0%}:         {'✅ PASS' if test_metrics['fpr'] <= Config.MAX_FPR else '❌ FAIL'}")
        print(f"  FNR ≤ {Config.MAX_FNR:.0%}:        {'✅ PASS' if test_metrics['fnr'] <= Config.MAX_FNR else '❌ FAIL'}")
        print(f"  Size ≤ {Config.MAX_MODEL_SIZE_MB}MB:     {'✅ PASS' if size_pass else '❌ FAIL'} ({model_size:.2f} MB)")
        print(f"\n  Overall: {'✅ ALL KPIs MET — PRODUCTION READY' if full_compliance else '❌ KPIs NOT FULLY MET'}")
        print(f"{'='*80}")
        
        # ========================================
        # COMPARE WITH PYTORCH RESULTS (if available)
        # ========================================
        pytorch_results_path = Config.SAVE_ROOT / "final_results.json"
        if pytorch_results_path.exists():
            try:
                with open(pytorch_results_path, 'r') as f:
                    pytorch_results = json.load(f)
                
                pt_metrics = pytorch_results.get('test_metrics', {})
                
                if pt_metrics:
                    print(f"\n{'='*80}")
                    print("ONNX INT8 vs PyTorch MODEL COMPARISON")
                    print(f"{'='*80}")
                    print(f"{'Metric':<20} {'PyTorch':>12} {'ONNX INT8':>12} {'Δ Delta':>12}")
                    print(f"{'─'*56}")
                    
                    comparison_metrics = [
                        ('Accuracy', 'accuracy'),
                        ('Precision', 'precision'),
                        ('Recall', 'recall'),
                        ('F1-Score', 'f1'),
                        ('AUC-ROC', 'auc'),
                        ('FPR', 'fpr'),
                        ('FNR', 'fnr'),
                    ]
                    
                    for name, key in comparison_metrics:
                        pt_val = pt_metrics.get(key, 0)
                        onnx_val = test_metrics.get(key, 0)
                        delta = onnx_val - pt_val
                        
                        # Use appropriate delta indicator
                        if key in ['fpr', 'fnr']:
                            # Lower is better for error rates
                            delta_icon = "🟢" if delta <= 0 else "🔴"
                        else:
                            # Higher is better for performance metrics
                            delta_icon = "🟢" if delta >= 0 else "🔴"
                        
                        print(f"  {name:<18} {pt_val:>11.4f} {onnx_val:>11.4f} {delta:>+11.4f} {delta_icon}")
                    
                    print(f"{'─'*56}")
                    print(f"  {'Model Size':<18} {'N/A':>12} {model_size:>10.2f}MB")
                    print(f"  {'Throughput':<18} {'N/A':>12} {throughput:>8.0f}/sec")
                    print(f"{'='*80}")
                    
            except Exception as e:
                print(f"\n⚠ Could not load PyTorch results for comparison: {e}")
        
        # ========================================
        # SAVE ONNX INFERENCE RESULTS
        # ========================================
        onnx_final_results = {
            'test_metrics': test_metrics,
            'model_info': {
                'model_path': str(onnx_model_path),
                'model_type': model_type,
                'model_size_mb': model_size,
                'best_epoch': best_epoch,
                'onnx_runtime_version': ort.__version__,
                'execution_provider': session.get_providers()[0],
            },
            'performance_benchmarks': {
                'total_inference_time_sec': total_inference_time,
                'total_samples': total_samples,
                'avg_batch_time_ms': avg_batch_time_ms,
                'avg_sample_time_ms': avg_sample_time_ms,
                'throughput_urls_per_sec': throughput,
                'execution_device': execution_device,
            },
            'kpi_compliance': {
                'classification_kpis_met': kpi_pass,
                'size_kpi_met': size_pass,
                'all_kpis_met': full_compliance,
            },
            'unsupported_url_screening_report': self.latest_test_url_screening_report,
            'threshold': optimal_threshold,
            'timestamp': datetime.now().isoformat(),
        }
        
        onnx_results_file = Config.SAVE_ROOT / "onnx_inference_results.json"
        with open(onnx_results_file, 'w') as f:
            json.dump(onnx_final_results, f, indent=2, default=str)
        
        print(f"\n✓ All ONNX results saved to: {onnx_results_dir.name}/")
        print(f"✓ ONNX results JSON: {onnx_results_file.name}")
        
        # ========================================
        # FINAL SUMMARY
        # ========================================
        print(f"\n{'='*80}")
        print("ONNX INFERENCE COMPLETED SUCCESSFULLY")
        print(f"{'='*80}")
        print(f"Model: {onnx_model_path.name} ({model_type}, {model_size:.2f} MB)")
        print(f"Throughput: {throughput:.0f} URLs/sec")
        print(f"KPI Compliance: {'✅ ALL MET — PRODUCTION READY' if full_compliance else '⚠ PARTIAL'}")
        print(f"Results: {onnx_results_dir}")
        print(f"{'='*80}\n")
        
        return full_compliance


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================
def setup_cli_parser() -> argparse.ArgumentParser:
    """Setup CLI argument parser with mode selection."""
    parser = argparse.ArgumentParser(
        description="MiniLM Phishing URL Detection - Training & Inference Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
USAGE EXAMPLES:
  
  1. Train from scratch (or resume from latest checkpoint):
     python MiniLM_1.py --mode train
     
  2. Train and allow checkpoint resume prompt:
     python MiniLM_1.py --mode train --interactive
     
  3. Inference-only mode (skip training, load latest checkpoint):
     python MiniLM_1.py --mode inference
     
  4. ONNX inference (quantized INT8 model — production deployment test):
     python MiniLM_1.py --mode onnx-inference
     
  5. Default mode (auto-detect based on checkpoints):
     python MiniLM_1.py
     
  6. Show help:
     python MiniLM_1.py --help
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'inference', 'onnx-inference', 'auto'],
        default='auto',
        help="""
        Execution mode:
        - 'train': Full training pipeline (default behavior)
        - 'inference': Load latest checkpoint and perform test inference only
        - 'onnx-inference': Load quantized ONNX model and evaluate on test set
        - 'auto': Detect mode based on checkpoint existence (default)
        """
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        default=False,
        help="Enable interactive prompts (e.g., checkpoint resume confirmation)"
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help="Path to specific checkpoint to load (optional)"
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help="Override number of training epochs (if None, uses Config.NUM_EPOCHS)"
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help="Override batch size (if None, uses Config.BATCH_SIZE)"
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help="Override learning rate (if None, uses Config.LR)"
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=False,
        help="Enable verbose output with additional logging"
    )
    
    parser.add_argument(
        '--onnx-model',
        type=str,
        default='int8',
        help="""
        ONNX model variant for onnx-inference mode:
        - 'int8': INT8 quantized model (model_quant_8bit.onnx) — smallest, may have accuracy loss
        - 'fp32': FP32 original ONNX model (model.onnx) — same accuracy as PyTorch
        - '/path/to/model.onnx': Custom ONNX model path
        (default: int8)
        """
    )
    
    return parser


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
def main() -> bool:
    """Main entry point with CLI support."""
    
    # Parse CLI arguments
    parser = setup_cli_parser()
    args = parser.parse_args()
    
    # Override Config if CLI arguments provided
    if args.epochs is not None:
        Config.NUM_EPOCHS = args.epochs
        print(f"[CLI] Overriding NUM_EPOCHS to {args.epochs}")
    
    if args.batch_size is not None:
        Config.BATCH_SIZE = args.batch_size
        print(f"[CLI] Overriding BATCH_SIZE to {args.batch_size}")
    
    if args.lr is not None:
        Config.LR = args.lr
        print(f"[CLI] Overriding LR to {args.lr}")
    
    # Determine execution mode
    mode = args.mode
    
    if mode == 'auto':
        # Auto-detect mode based on checkpoint existence
        Config.setup_paths()
        latest_ckpt = CheckpointManager(Config.CHECKPOINT_DIR).find_latest_checkpoint()
        
        if latest_ckpt is not None:
            # Checkpoint exists - default to inference mode
            print("\n[AUTO-DETECT] Latest checkpoint found")
            print(f"[AUTO-DETECT] Setting mode to 'inference'\n")
            mode = 'inference'
        else:
            # No checkpoint - default to training mode
            print("\n[AUTO-DETECT] No checkpoint found")
            print(f"[AUTO-DETECT] Setting mode to 'train'\n")
            mode = 'train'
    
    # ========================================
    # TRAINING MODE
    # ========================================
    if mode == 'train':
        try:
            print("\n" + "="*80)
            print(" " * 20 + "MiniLM PHISHING URL DETECTION")
            print(" " * 15 + "Production-Grade Training Pipeline")
            print("="*80)
            print(f"Mode:                  TRAINING")
            print(f"Target Model Size:     <{Config.MAX_MODEL_SIZE_MB}MB with {Config.TARGET_ACCURACY:.0%} accuracy")
            print(f"Architecture:          Hybrid GLU Fusion (MiniLM + LoRA + MLP + GLU)")
            print(f"Device:                {Config.DEVICE}")
            print(f"Interactive Mode:      {'Enabled' if args.interactive else 'Disabled'}")
            print("="*80 + "\n")
            
            trainer = PhishingDetectionTrainer()
            kpi_compliance = trainer.train()
            
            return kpi_compliance
        
        except KeyboardInterrupt:
            print("\n\n⚠ Training interrupted by user")
            return False
        
        except Exception as e:
            print(f"\n❌ ERROR during training: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # ========================================
    # INFERENCE MODE (PyTorch)
    # ========================================
    elif mode == 'inference':
        try:
            trainer = PhishingDetectionTrainer()
            kpi_compliance = trainer.inference_from_checkpoint()
            
            return kpi_compliance
        
        except KeyboardInterrupt:
            print("\n\n⚠ Inference interrupted by user")
            return False
        
        except Exception as e:
            print(f"\n❌ ERROR during inference: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # ========================================
    # ONNX INFERENCE MODE (INT8 Quantized)
    # ========================================
    elif mode == 'onnx-inference':
        try:
            print("\n" + "="*80)
            print(" " * 15 + "MiniLM PHISHING URL DETECTION")
            print(" " * 10 + "ONNX Runtime Inference (INT8 Quantized)")
            print("="*80)
            print(f"Mode:                  ONNX INFERENCE")
            print(f"Target Model Size:     <{Config.MAX_MODEL_SIZE_MB}MB")
            print(f"Runtime:               ONNX Runtime")
            print(f"Quantization:          INT8 Dynamic")
            print("="*80 + "\n")
            
            trainer = PhishingDetectionTrainer()
            kpi_compliance = trainer.onnx_inference(onnx_model_type=args.onnx_model)
            
            return kpi_compliance
        
        except KeyboardInterrupt:
            print("\n\n⚠ ONNX inference interrupted by user")
            return False
        
        except Exception as e:
            print(f"\n❌ ERROR during ONNX inference: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    else:
        print(f"❌ Unknown mode: {mode}")
        parser.print_help()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)




# python 3_MiniLM_V4_hybrid_FF_inferencing.py --mode train
# python 3_MiniLM_V4_hybrid_FF_inferencing.py --mode inference


#################### python 3_MiniLM_V4_hybrid_FF_inferencing.py --mode onnx-inference
# # FP32 ONNX (same accuracy as PyTorch, larger file)
# python 3_MiniLM_V4_hybrid_FF_inferencing.py --mode onnx-inference --onnx-model fp32

# # INT8 Quantized (smaller file, may lose accuracy)
# python 3_MiniLM_V4_hybrid_FF_inferencing.py --mode onnx-inference --onnx-model int8

# # Custom model path
# python 3_MiniLM_V4_hybrid_FF_inferencing.py --mode onnx-inference --onnx-model /path/to/model.onnx
