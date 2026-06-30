import os
import random
import sys
from pathlib import Path
from typing import List, Optional
import torch
import numpy as np

# Try to import yaml (PyYAML). If not available, we will print a warning and fallback.
try:
    import yaml
except ImportError:
    print("[WARN] PyYAML is not installed. Run: pip install pyyaml")
    print("       Falling back to manual parsing or default config values.")
    yaml = None


class Config:
    # --------------------------------------------------------------------------
    # DEFAULT HYBRID GLU FUSION CONFIGURATIONS
    # --------------------------------------------------------------------------
    SEED: int = 42
    
    TRAIN_CSV: str = ""
    VAL_CSV: str = ""
    TEST_CSV: str = ""
    
    MODEL_NAME: str = "microsoft/MiniLM-L12-H384-uncased"
    USE_CUSTOM_TOKENIZER_BYTE_LEVEL_BPE: bool = True
    USE_CUSTOM_TOKENIZER_SENTENCEPIECE_UNIGRAM_BPE: bool = False
    CUSTOM_TOKENIZER_VOCAB_SIZE: int = 50000
    CUSTOM_TOKENIZER_PATH: str = "RESULTS_&_MODELS/custom_bpe_tokenizer.json"
    
    MAX_LEN: int = 192
    NUM_CLASSES: int = 2
    CLASSIFICATION_LAYER_TYPE: str = "softmax"
    DROPOUT: float = 0.1
    
    # --- Hybrid GLU Fusion Dimensions ---
    TEXT_EMBED_DIM: int = 384
    HEURISTIC_DIM: int = 87
    HEURISTIC_MLP_HIDDEN: int = 256
    HEURISTIC_MLP_OUTPUT: int = 192
    GLU_HIDDEN: int = 384
    GATING_TYPE: str = "GLU"
    CLASSIFIER_DIMS: List[int] = [192, 64]
    
    # --- Heuristic Feature Columns (auto-detected from hybrid CSV) ---
    NUMERIC_FEATURE_COLS: List[str] = [
        "h_flags_bitmask",
        "h_entropy_url", "h_entropy_path", "h_entropy_query",
        "h_digit_ratio", "h_path_depth", "h_url_length", "h_query_param_count",
        "h_domain_length", "h_subdomain_count",
        "h_punycode_char_count", "h_unicode_char_ratio",
        "h_tracking_param_count", "h_path_token_count", "h_redirect_count",
    ]
    BINARY_FEATURE_COLS: List[str] = [
        "h_is_ip_host", "h_has_fragment",
        "h_tld_risk_normal", "h_tld_risk_high", "h_tld_risk_critical",
        "h_has_punycode", "h_has_unicode", "h_mixed_script",
        "h_has_tracking_params", "h_has_double_extension",
        "h_has_redirect_param", "h_has_at_sign",
    ]
    DROP_COLS: List[str] = ["h_primary_category"]
    
    BATCH_SIZE: int = 128
    NUM_EPOCHS: int = 3
    WEIGHT_DECAY: float = 0.02
    PATIENCE: int = 4
    GRAD_ACCUM_STEPS: int = 2
    GRAD_CLIP_NORM: float = 1.0
    
    LR: float = 5e-5
    HEAD_LR: float = 1e-3
    LR_WARMUP_RATIO: float = 0.05
    LR_MIN_RATIO: float = 0.001
    
    LORA_R: int = 32
    LORA_ALPHA: int = 64
    LORA_DROPOUT: float = 0.05
    LORA_TARGET_MODULES: List[str] = ["query", "key", "value", "dense", "output.dense"]
    
    FOCAL_GAMMA: float = 2.0
    FOCAL_ALPHA: List[float] = [0.35, 0.65]
    LABEL_SMOOTHING: float = 0.05
    
    USE_WEIGHTED_SAMPLING: bool = False
    PRUNING_RATIO: float = 0.0
    USE_AMP: bool = True
    
    EXPORT_ONNX: bool = True
    EXPORT_QUANTIZED: bool = True
    ONNX_OPSET: int = 18
    
    TARGET_ACCURACY: float = 0.98
    TARGET_PRECISION: float = 0.95
    TARGET_RECALL: float = 0.95
    MAX_FNR: float = 0.10
    MAX_FPR: float = 0.01
    MAX_MODEL_SIZE_MB: float = 40.0
    
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS: int = 12
    PIN_MEMORY: bool = True
    PREFETCH_FACTOR: int = 4
    
    MLFLOW_ENABLED: bool = False
    MLFLOW_TRACKING_URI: str = "sqlite:///mlflow.db"
    MLFLOW_TRACKING_USERNAME: str = ""
    MLFLOW_TRACKING_PASSWORD: str = ""
    MLFLOW_DAGSHUB_REPO: str = ""
    MLFLOW_EXPERIMENT_NAME: str = "PhishURL-Detection-Hybrid"
    MLFLOW_RUN_NAME: str = "MiniLM-V4-Hybrid-Pipeline"
    MLFLOW_REGISTER_MODEL: bool = True
    MLFLOW_MODEL_NAME: str = "PhishURL-Hybrid-Classifier"
    
    SAVE_ROOT: Optional[Path] = None
    CHECKPOINT_DIR: Optional[Path] = None

    @classmethod
    def load_from_yaml(cls, yaml_path: Optional[str] = None, create_dirs: bool = True, verbose: bool = True) -> None:
        """Loads hyperparameter values from centralized YAML file."""
        if yaml_path is None:
            # Look in default locations
            possible_paths = [
                Path("4_config.yaml"),
                Path("SRC/4_config.yaml"),
                Path(__file__).resolve().parent.parent / "4_config.yaml"
            ]
            for p in possible_paths:
                if p.exists():
                    yaml_path = str(p)
                    break
        
        if yaml_path and not Path(yaml_path).exists():
            project_root = Path(__file__).resolve().parent.parent.parent
            script_parent = Path(__file__).resolve().parent.parent
            
            alternative_paths = [
                project_root / yaml_path,
                script_parent / yaml_path,
                project_root / Path(yaml_path).name,
                script_parent / Path(yaml_path).name
            ]
            for ap in alternative_paths:
                if ap.exists():
                    yaml_path = str(ap)
                    break
                    
        if not yaml_path or not Path(yaml_path).exists():
            if verbose:
                print(f"[WARN] Config YAML file not found. Using internal defaults.")
            cls.setup_device()
            cls.setup_paths(create_dirs=create_dirs)
            return
            
        if verbose:
            print(f"[OK] Loading config from {yaml_path}")
        
        if yaml is None:
            cls._manual_parse(yaml_path)
            cls.setup_device()
            cls.setup_paths(create_dirs=create_dirs)
            return

        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            if not data:
                return

            if 'reproducibility' in data:
                r = data['reproducibility']
                cls.SEED = r.get('seed', cls.SEED)
                
            if 'data_paths' in data:
                dp = data['data_paths']
                cls.TRAIN_CSV = dp.get('train_csv', cls.TRAIN_CSV)
                cls.VAL_CSV = dp.get('val_csv', cls.VAL_CSV)
                cls.TEST_CSV = dp.get('test_csv', cls.TEST_CSV)
                save_root_str = dp.get('save_root', None)
                if save_root_str:
                    cls.SAVE_ROOT = Path(save_root_str)
                    
            if 'model' in data:
                m = data['model']
                cls.MODEL_NAME = m.get('name', cls.MODEL_NAME)
                cls.MAX_LEN = m.get('max_len', cls.MAX_LEN)
                cls.CLASSIFICATION_LAYER_TYPE = m.get('classification_layer_type', cls.CLASSIFICATION_LAYER_TYPE)
                if cls.CLASSIFICATION_LAYER_TYPE == "sigmoid":
                    cls.NUM_CLASSES = 1
                else:
                    cls.NUM_CLASSES = m.get('num_classes', cls.NUM_CLASSES)
                cls.DROPOUT = m.get('dropout', cls.DROPOUT)
                cls.USE_CUSTOM_TOKENIZER_BYTE_LEVEL_BPE = m.get('use_custom_tokenizer_byte-level_BPE', cls.USE_CUSTOM_TOKENIZER_BYTE_LEVEL_BPE)
                cls.USE_CUSTOM_TOKENIZER_SENTENCEPIECE_UNIGRAM_BPE = m.get('use_custom_tokenizer_SentencePiece_Unigram_BPE', cls.USE_CUSTOM_TOKENIZER_SENTENCEPIECE_UNIGRAM_BPE)
                cls.CUSTOM_TOKENIZER_VOCAB_SIZE = m.get('custom_tokenizer_vocab_size', cls.CUSTOM_TOKENIZER_VOCAB_SIZE)
                cls.CUSTOM_TOKENIZER_PATH = m.get('custom_tokenizer_path', cls.CUSTOM_TOKENIZER_PATH)
                
                # Hybrid MLP/GLU dims
                cls.TEXT_EMBED_DIM = m.get('text_embed_dim', cls.TEXT_EMBED_DIM)
                cls.HEURISTIC_DIM = m.get('heuristic_dim', cls.HEURISTIC_DIM)
                cls.HEURISTIC_MLP_HIDDEN = m.get('heuristic_mlp_hidden', cls.HEURISTIC_MLP_HIDDEN)
                cls.HEURISTIC_MLP_OUTPUT = m.get('heuristic_mlp_output', cls.HEURISTIC_MLP_OUTPUT)
                cls.GLU_HIDDEN = m.get('glu_hidden', cls.GLU_HIDDEN)
                cls.GATING_TYPE = m.get('gating_type', cls.GATING_TYPE)
                cls.CLASSIFIER_DIMS = m.get('classifier_dims', cls.CLASSIFIER_DIMS)
                
            if 'training' in data:
                t = data['training']
                cls.BATCH_SIZE = t.get('batch_size', cls.BATCH_SIZE)
                cls.NUM_EPOCHS = t.get('num_epochs', cls.NUM_EPOCHS)
                cls.WEIGHT_DECAY = t.get('weight_decay', cls.WEIGHT_DECAY)
                cls.PATIENCE = t.get('patience', cls.PATIENCE)
                cls.GRAD_ACCUM_STEPS = t.get('grad_accum_steps', cls.GRAD_ACCUM_STEPS)
                cls.GRAD_CLIP_NORM = t.get('grad_clip_norm', cls.GRAD_CLIP_NORM)
                
            if 'learning_rate' in data:
                lr = data['learning_rate']
                cls.LR = lr.get('lr', cls.LR)
                cls.HEAD_LR = lr.get('head_lr', cls.HEAD_LR)
                cls.LR_WARMUP_RATIO = lr.get('warmup_ratio', cls.LR_WARMUP_RATIO)
                cls.LR_MIN_RATIO = lr.get('min_ratio', cls.LR_MIN_RATIO)
                
            if 'lora' in data:
                lo = data['lora']
                cls.LORA_R = lo.get('r', cls.LORA_R)
                cls.LORA_ALPHA = lo.get('alpha', cls.LORA_ALPHA)
                cls.LORA_DROPOUT = lo.get('dropout', cls.LORA_DROPOUT)
                cls.LORA_TARGET_MODULES = lo.get('target_modules', cls.LORA_TARGET_MODULES)
                
            if 'loss' in data:
                l = data['loss']
                cls.FOCAL_GAMMA = l.get('focal_gamma', cls.FOCAL_GAMMA)
                cls.FOCAL_ALPHA = l.get('focal_alpha', cls.FOCAL_ALPHA)
                cls.LABEL_SMOOTHING = l.get('label_smoothing', cls.LABEL_SMOOTHING)
                
            if 'class_balancing' in data:
                cb = data['class_balancing']
                cls.USE_WEIGHTED_SAMPLING = cb.get('use_weighted_sampling', cls.USE_WEIGHTED_SAMPLING)
                
            if 'optimization' in data:
                opt = data['optimization']
                cls.PRUNING_RATIO = opt.get('pruning_ratio', cls.PRUNING_RATIO)
                cls.USE_AMP = opt.get('use_amp', cls.USE_AMP)
                
            if 'export' in data:
                exp = data['export']
                cls.EXPORT_ONNX = exp.get('export_onnx', cls.EXPORT_ONNX)
                cls.EXPORT_QUANTIZED = exp.get('export_quantized', cls.EXPORT_QUANTIZED)
                cls.ONNX_OPSET = exp.get('onnx_opset', cls.ONNX_OPSET)
                
            if 'kpi_targets' in data:
                kpi = data['kpi_targets']
                cls.TARGET_ACCURACY = kpi.get('target_accuracy', cls.TARGET_ACCURACY)
                cls.TARGET_PRECISION = kpi.get('target_precision', cls.TARGET_PRECISION)
                cls.TARGET_RECALL = kpi.get('target_recall', cls.TARGET_RECALL)
                cls.MAX_FNR = kpi.get('max_fnr', cls.MAX_FNR)
                cls.MAX_FPR = kpi.get('max_fpr', cls.MAX_FPR)
                cls.MAX_MODEL_SIZE_MB = kpi.get('max_model_size_mb', cls.MAX_MODEL_SIZE_MB)
                
            if 'hardware' in data:
                hw = data['hardware']
                cls.NUM_WORKERS = hw.get('num_workers', cls.NUM_WORKERS)
                cls.PIN_MEMORY = hw.get('pin_memory', cls.PIN_MEMORY)
                cls.PREFETCH_FACTOR = hw.get('prefetch_factor', cls.PREFETCH_FACTOR)
                
                device_str = hw.get('device', 'auto')
                if device_str == 'cuda':
                    cls.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                elif device_str == 'cpu':
                    cls.DEVICE = torch.device("cpu")
                else:
                    cls.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            if 'mlflow' in data:
                mf = data['mlflow']
                cls.MLFLOW_ENABLED = mf.get('enabled', cls.MLFLOW_ENABLED)
                cls.MLFLOW_TRACKING_URI = mf.get('tracking_uri', cls.MLFLOW_TRACKING_URI)
                cls.MLFLOW_TRACKING_USERNAME = mf.get('username', cls.MLFLOW_TRACKING_USERNAME)
                cls.MLFLOW_TRACKING_PASSWORD = mf.get('password', cls.MLFLOW_TRACKING_PASSWORD)
                cls.MLFLOW_DAGSHUB_REPO = mf.get('dagshub_repo', cls.MLFLOW_DAGSHUB_REPO)
                cls.MLFLOW_EXPERIMENT_NAME = mf.get('experiment_name', cls.MLFLOW_EXPERIMENT_NAME)
                cls.MLFLOW_RUN_NAME = mf.get('run_name', cls.MLFLOW_RUN_NAME)
                cls.MLFLOW_REGISTER_MODEL = mf.get('register_model', cls.MLFLOW_REGISTER_MODEL)
                cls.MLFLOW_MODEL_NAME = mf.get('model_name', cls.MLFLOW_MODEL_NAME)

        except Exception as e:
            print(f"[WARN] Error loading config from YAML: {e}. Using defaults.")

        cls.setup_device()
        cls.setup_paths(create_dirs=create_dirs)

    @classmethod
    def setup_device(cls) -> None:
        """Final sanity check on PyTorch device compatibility."""
        if not isinstance(cls.DEVICE, torch.device):
            if cls.DEVICE == "cuda" or (isinstance(cls.DEVICE, str) and "cuda" in cls.DEVICE):
                cls.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                cls.DEVICE = torch.device("cpu")

    @classmethod
    def setup_paths(cls, create_dirs: bool = True) -> None:
        """Initialize and create output folders."""
        project_root = Path(__file__).resolve().parent.parent.parent
        
        def resolve_path(p):
            if not p:
                return p
            path_obj = Path(p)
            if path_obj.is_absolute():
                return path_obj
            return project_root / path_obj

        if cls.TRAIN_CSV:
            cls.TRAIN_CSV = str(resolve_path(cls.TRAIN_CSV))
        if cls.VAL_CSV:
            cls.VAL_CSV = str(resolve_path(cls.VAL_CSV))
        if cls.TEST_CSV:
            cls.TEST_CSV = str(resolve_path(cls.TEST_CSV))
        if cls.CUSTOM_TOKENIZER_PATH:
            cls.CUSTOM_TOKENIZER_PATH = str(resolve_path(cls.CUSTOM_TOKENIZER_PATH))
        
        if cls.SAVE_ROOT is None:
            cls.SAVE_ROOT = project_root / "RESULTS_&_MODELS/3_saved_models/MiniLM_HybridFF_v4"
        else:
            cls.SAVE_ROOT = resolve_path(cls.SAVE_ROOT)
            
        cls.CHECKPOINT_DIR = cls.SAVE_ROOT / "checkpoints"
        if create_dirs:
            cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            cls.SAVE_ROOT.mkdir(parents=True, exist_ok=True)

    @classmethod
    def setup_reproducibility(cls) -> None:
        """Set seeds across libraries."""
        torch.manual_seed(cls.SEED)
        np.random.seed(cls.SEED)
        random.seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @classmethod
    def _manual_parse(cls, yaml_path: str) -> None:
        """Manual line-by-line fallback parser for config values."""
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith("#") or ":" not in line:
                    continue
                parts = line.split(":", 1)
                key = parts[0].strip().upper().replace("-", "_")
                val = parts[1].strip()
                if "#" in val:
                    val = val.split("#", 1)[0].strip()
                
                if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                    val = val[1:-1]
                
                if val.startswith('[') and val.endswith(']'):
                    try:
                        import ast
                        parsed_list = ast.literal_eval(val)
                        if hasattr(cls, key):
                            setattr(cls, key, parsed_list)
                        continue
                    except:
                        pass

                if hasattr(cls, key):
                    current_type = type(getattr(cls, key))
                    if current_type is bool:
                        setattr(cls, key, val.lower() in ['true', 'yes', '1'])
                    elif current_type is int:
                        setattr(cls, key, int(val))
                    elif current_type is float:
                        setattr(cls, key, float(val))
                    elif current_type is str:
                        setattr(cls, key, val)
        except Exception as e:
            print(f"[WARN] Manual configuration parser failed: {e}")

# Load default configuration initially
Config.load_from_yaml(create_dirs=False, verbose=False)
