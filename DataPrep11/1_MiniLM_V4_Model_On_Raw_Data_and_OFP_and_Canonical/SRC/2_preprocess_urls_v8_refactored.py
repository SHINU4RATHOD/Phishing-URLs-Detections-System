from __future__ import annotations

# ============================================================================
# IMPORTS
# ============================================================================
import argparse
import csv
import sys
import ipaddress
import logging
import math
import multiprocessing
import pickle
import posixpath
import re
import socket
import unicodedata
from abc import ABC, abstractmethod
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union
from urllib.parse import parse_qs, unquote, urlencode, urlparse, urlunparse, quote, urlsplit, urlunsplit

import idna
import pandas as pd
import requests
import tldextract
from tqdm import tqdm

from urls_cate_V7 import CategoryConfig, URLAnalyzer, URLCategory, DEFAULT_URL_SHORTENERS


# ============================================================================
# CONSTANTS
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "RESULTS_&_MODELS" / "2_preprocess_urls_output"


def resolve_project_path(path: Path) -> Path:
    """Resolve relative CLI paths from the project root instead of the launch cwd."""
    return path if path.is_absolute() else PROJECT_ROOT / path


class Constants: 
    # Version identifiers
    INPUT_TEXT_VERSION: str = "compact_v8"
    BLOB_PREFIX_LEN: int = 20
    
    # Tracking parameters to filter from URLs
    TRACKING_PARAMS: set[str] = {
        "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
        "gclid", "dclid", "fbclid", "msclkid", "ttclid", "twclid",
        "li_fat_id", "rdt_cid", "epik", "scid", "ytclid",
        "mc_cid", "mc_eid", "_ga", "_gid",
        
        # HubSpot
        "__hssc", "__hstc", "hsfb", "hsCtaTracking",

        # General
        "cid", "aid", "ref", "referrer",
        "sessionid", "session_id", "phpsessid",
        "gbraid", "wbraid",
        "source", "campaign", "clickid",
        "tracking_id", "aff_id", "affiliate_id"
    }
    
    # Token keywords for fragment analysis
    TOKEN_KEYWORDS: Set[str] = {"access_token", "token", "redirect", "state", "session"}
    
    # Character sets for blob detection
    BASE64_ALLOWED: Set[str] = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=_-")
    HEX_ALLOWED: Set[str] = set("0123456789abcdefABCDEF")
    
    # Scheme family classification
    SCHEME_FAMILIES: Dict[str, Set[str]] = {
        "WEB": {"http", "https"},
        "FILE_TRANSFER": {"ftp", "sftp", "file"},
        "DATA_EMBEDDED": {"data", "blob"},
        "SCRIPT_ACTIVE": {"javascript"},
        "COMMUNICATION": {"mailto", "tel", "irc", "nntp", "ssh"},
        "SYSTEM_RESOURCE": {"chrome", "edge", "about", "resource"},
        "IDENTIFIER_OTHER": {"urn"},
    }
    
    # Primary category priority: auto-sorted by severity from URLCategory.CATEGORIES
    # CRITICAL > HIGH > MEDIUM > LOW (stable alphabetical sub-sort within each tier)
    @staticmethod
    def _build_priority() -> tuple:
        severity_rank = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        return tuple(sorted(URLCategory.CATEGORIES.keys(), key=lambda cat: (severity_rank.get(URLCategory.CATEGORIES[cat].get("severity", "LOW"), 99),cat,),))
    
    PRIMARY_CATEGORY_PRIORITY: Sequence[str] = _build_priority.__func__()
    
    # Flag order for bitmask generation
    FLAG_ORDER: List[str] = list(URLCategory.CATEGORIES.keys())
    
    # Short flag code mapping: auto-generated from 'friendly' names in URLCategory.CATEGORIES
    # Each flag gets a unique 3-4 letter uppercase code derived from its friendly name.
    # Collisions are resolved by appending a numeric suffix.
    @staticmethod
    def _build_flag_code_map() -> Dict[str, str]:
        """Auto-generate unique short flag codes from URLCategory.CATEGORIES friendly names."""
        code_map: Dict[str, str] = {}
        used_codes: set = set()
        for cat_name, cat_info in URLCategory.CATEGORIES.items():
            friendly = cat_info.get("friendly", cat_name)
            # Generate base code: uppercase first 4 chars of friendly name
            base = friendly.upper().replace(" ", "")[:4]
            if len(base) < 3:
                base = cat_name.upper().replace("_", "")[:4]
            code = base
            suffix = 2
            while code in used_codes:
                code = f"{base}{suffix}"
                suffix += 1
            used_codes.add(code)
            code_map[cat_name] = code
        return code_map
    
    FLAG_CODE_MAP: Dict[str, str] = _build_flag_code_map.__func__()
    
    # Scheme family short codes
    SCHEME_CODE_MAP: Dict[str, str] = {
        "WEB": "WEB",
        "FILE_TRANSFER": "FT",
        "DATA_EMBEDDED": "DATA",
        "SCRIPT_ACTIVE": "JS",
        "COMMUNICATION": "COMM",
        "SYSTEM_RESOURCE": "SYS",
        "IDENTIFIER_OTHER": "ID",
    }
    
    # Invisible characters to remove (homograph attack vectors)
    # Extended set: original 7 + Soft Hyphen, Word Joiner, Mongolian Vowel Sep, 
    # # Line/Paragraph Separators, BOM variant, Variation Selectors (U+FE00-FE0F)
    INVISIBLE_CHARS: str = (
        '\u200b\u200c\u200d\u200e\u200f\u061c\ufeff'  # Original: ZWJ, ZWNJ, BIDI, BOM
        '\u00ad'      # Soft Hyphen — invisible in most fonts
        '\u2060'      # Word Joiner — zero-width no-break space
        '\u180e'      # Mongolian Vowel Separator
        '\u2028'      # Line Separator
        '\u2029'      # Paragraph Separator
        '\ufffe'      # BOM variant (non-character)
        '\ufe00\ufe01\ufe02\ufe03\ufe04\ufe05\ufe06\ufe07'  # Variation Selectors 1-8
        '\ufe08\ufe09\ufe0a\ufe0b\ufe0c\ufe0d\ufe0e\ufe0f'  # Variation Selectors 9-16
    )


# ============================================================================
# DATA CLASSES
# ============================================================================
@dataclass
class PreprocessConfig:
    """
    Runtime configuration for the preprocessing pipeline.
    
    Holds all configurable parameters for the pipeline execution,
    including input/output paths, processing options, and split settings.
    
    Attributes:
        input_csv: Path to input CSV file with 'input' and 'label' columns
        output_csv: Path for main preprocessed output
        rejected_csv: Path for rejected/corrupted URLs
        local_private_csv: Path for local/private IP URLs
        duplicate_csv: Path for duplicate URLs
        chunk_size: Number of rows to process per chunk
        drop_local_private: Whether to drop local/private IPs
        deduplicate: Master switch for deduplication
        deduplicate_before_split: Enable pre-split deduplication
        deduplicate_after_split: Enable post-split deduplication
        dedup_cache: Path for deduplication cache persistence
        overwrite_outputs: Whether to overwrite existing outputs
        base64_min_len: Minimum length for base64 blob detection
        hex_min_len: Minimum length for hex blob detection
        max_fragment_preview: Maximum fragment preview length
        tld_stats_path: Optional path to TLD statistics Excel file
        tld_stats_sheet: Sheet name in TLD statistics file
        enable_split: Whether to create train/val/test splits
        train_frac: Training set fraction
        val_frac: Validation set fraction
        random_seed: Random seed for reproducibility
        log_path: Path for log file
        tracking_params: Set of tracking parameters to filter
        use_rule_features: Whether to emit rule-assisted features
        longest_debug_csv: Path for longest input_text debug output
        input_text_version: Version identifier for input text format
        split_source: List of split sources to generate
        resume: Whether to resume from previous run
        progress_path: Path for resume progress metadata
        skip_entropy_calibration: Skip entropy threshold calibration
        calibration_sample_size: Sample size for entropy calibration
        enable_multiprocessing: Enable parallel processing
        num_workers: Number of worker processes
    """
    input_csv: Path = Path("/home/hp/SHINU RATHOD/Data_Preprocessing/data_prep1/data8.csv")
    output_csv: Path = DEFAULT_OUTPUT_DIR / "urls_preprocessed.csv"
    rejected_csv: Path = DEFAULT_OUTPUT_DIR / "urls_rejected_corrupted.csv"
    local_private_csv: Path = DEFAULT_OUTPUT_DIR / "urls_local_private.csv"
    duplicate_csv: Path = DEFAULT_OUTPUT_DIR / "urls_duplicates.csv"
    chunk_size: int = 900_000
    drop_local_private: bool = True
    deduplicate_before_split: bool = False
    deduplicate_after_split: bool = True
    deduplicate: bool = True
    dedup_cache: Optional[Path] = DEFAULT_OUTPUT_DIR / "dedup_cache.pkl"
    overwrite_outputs: bool = True
    base64_min_len: int = 20
    hex_min_len: int = 32
    max_fragment_preview: int = 120
    tld_stats_path: Optional[Path] = None
    tld_stats_sheet: str = "Benign_&_Malecious_TLD"
    enable_split: bool = True
    train_frac: float = 0.8
    val_frac: float = 0.1
    random_seed: int = 42
    log_path: Path = DEFAULT_OUTPUT_DIR / "preprocess.log"
    tracking_params: Set[str] = field(default_factory=lambda: set(Constants.TRACKING_PARAMS))
    use_rule_features: bool = True
    longest_debug_csv: Path = DEFAULT_OUTPUT_DIR / "input_text_longest_debug.csv"
    input_text_version: str = Constants.INPUT_TEXT_VERSION
    split_source: List[str] = field(default_factory=lambda: ["preprocessed"])
    resume: bool = True
    progress_path: Optional[Path] = DEFAULT_OUTPUT_DIR / ".preprocess_resume.json"
    skip_entropy_calibration: bool = False
    calibration_sample_size: int = 5000
    enable_multiprocessing: bool = True
    num_workers: int = 8
    model_input_format: str = "canonical"
    output_format: str = "full"
    
    # Redirect resolution settings (DISABLED by default for backward compatibility)
    enable_redirect_resolution: bool = False
    redirect_only_shorteners: bool = True  # Only resolve known URL shorteners
    redirect_max_hops: int = 5
    redirect_timeout_sec: float = 5.0
    redirect_include_features: bool = False  # Add REDIR_* to model input
    
    # IP-to-Domain Resolution settings
    enable_ip_domain_resolution: bool = True
    dns_timeout: float = 5.0


@dataclass
class ProcessedURL:
    """
    Structured data for a successfully processed URL.
    
    This dataclass represents the output format for each URL
    that passes preprocessing validation.
    
    Attributes:
        input_text: The text to be fed into BERT/transformer models
        label: Binary label (0=benign, 1=malicious)
        primary_category: Primary threat category classification
        tld: Top-level domain
        tld_risk: TLD risk level (HIGH, CONTEXTUAL, NORMAL)
        canonical_url: Normalized canonical URL (heavy cleaning for dedup + Preprocessed mode)
        model_url: Minimally-cleaned URL for model training (OFP mode input)
                   Only: invisible strip + NFKC + Punycode applied.
                   Preserves: path traversal, encoding layers, tracking params,
                   ports, blob patterns, IP obfuscation formats.
        raw_url: Original raw URL before processing
        flags_count: Number of active detection flags
        severity_score: Weighted severity score
        flags_bitmask: Bitmask of active flags
    """
    input_text: str
    label: int
    primary_category: str
    tld: str
    tld_risk: str
    canonical_url: str = ""
    model_url: str = ""
    raw_url: str = ""
    flags_count: int = 0
    severity_score: float = 0.0
    flags_bitmask: int = 0
    # Hybrid mode extras (filled when output_format == "hybrid")
    flags_active: List[str] = field(default_factory=list)
    entropy_url: float = 0.0
    entropy_path: float = 0.0
    entropy_query: float = 0.0
    digit_ratio: float = 0.0
    path_depth: int = 0
    url_length: int = 0
    is_ip_host: bool = False
    has_https: bool = False
    query_param_count: int = 0
    has_fragment: bool = False
    # --- NEW V9 features (targeted at failing categories) ---
    # Domain features (TypoSquatting, WebAppPath)
    domain_length: int = 0
    subdomain_count: int = 0
    # Punycode / Unicode features (Punycode_URL, Unicode_URL)
    has_punycode: bool = False
    punycode_char_count: int = 0
    has_unicode: bool = False
    unicode_char_ratio: float = 0.0
    mixed_script_detected: bool = False
    # Tracking features (HasExcessiveParams, IsDynamicQuery)
    has_tracking_params: bool = False
    tracking_param_count: int = 0
    # Path structure features (IsSuspiciousFileType, IsWebAppPath)
    has_double_extension: bool = False
    path_token_count: int = 0
    # Redirect / obfuscation features
    has_redirect_param: bool = False
    redirect_count_in_url: int = 0
    has_at_sign: bool = False


@dataclass
class QueryInfo:
    """
    Metadata extracted from URL query string.
    
    Attributes:
        canonical_query: Normalized query string
        total_params: Total number of query parameters
        tracking_param_count: Number of tracking parameters
        other_param_count: Number of non-tracking parameters
        has_tracking_params: Whether tracking params are present
        has_base64_blob: Whether base64-encoded blob detected
        has_hex_blob: Whether hex-encoded blob detected
        avg_param_entropy: Average entropy of parameter values
    """
    canonical_query: str
    total_params: int
    tracking_param_count: int
    other_param_count: int
    has_tracking_params: bool
    has_base64_blob: bool
    has_hex_blob: bool
    avg_param_entropy: float


@dataclass
class RedirectInfo:
    """
    Metadata about URL redirect resolution.
    
    Attributes:
        resolved: Whether resolution was attempted and successful
        final_url: Final destination URL after following redirects
        redirect_depth: Number of redirect hops followed
        redirect_cross_domain: Whether redirects crossed registrable domains
        error: Error message if resolution failed
        hops: First 3 redirect domains for auditing
    """
    resolved: bool = False
    final_url: str = ""
    redirect_depth: int = 0
    redirect_cross_domain: bool = False
    error: str = ""
    hops: List[str] = field(default_factory=list)


@dataclass
class FragmentInfo:
    """
    Metadata extracted from URL fragment (Refinement v6.3).
    
    Attributes:
        length: Length of the raw fragment
        has_url: Whether fragment appears to contain an embedded URL
        has_token_keyword: Whether security/auth keywords present (e.g., atok, session)
        has_hex_blob: Whether high-entropy hex blob detected
        has_base64_blob: Whether high-entropy base64 blob detected
        decoded_fragment: 10-pass decoded fragment content
        truncated: Whether fragment exceeded processing limit
    """
    length: int = 0
    has_url: bool = False
    has_token_keyword: bool = False
    has_hex_blob: bool = False
    has_base64_blob: bool = False
    decoded_fragment: str = ""
    truncated: bool = False


# ============================================================================
# UNICODE NORMALIZER - WORLD-CLASS URL SECURITY
# ============================================================================
@dataclass
class NormalizationResult:
    """
    Result of Unicode normalization with security metadata.
    
    Captures the normalized output along with security indicators
    that can be used as ML features for phishing detection.
    
    Attributes:
        normalized: The fully normalized string (NFKC + lowercase)
        original: Original input for comparison
        punycode: IDNA/Punycode encoded version (for hostnames)
        had_unicode: Whether input contained non-ASCII characters
        had_invisible: Whether invisible/zero-width chars were stripped
        had_confusables: Whether confusable characters were detected
        mixed_scripts: Whether multiple Unicode scripts were mixed
        scripts_found: Set of Unicode script names detected
        homograph_score: 0.0-1.0 risk score for homograph attacks
        decode_passes: Number of percent-decode iterations needed
        is_suspicious: Composite flag for any security concern
    """
    normalized: str
    original: str
    punycode: str = ""
    had_unicode: bool = False
    had_invisible: bool = False
    had_confusables: bool = False
    mixed_scripts: bool = False
    scripts_found: Set[str] = field(default_factory=set)
    homograph_score: float = 0.0
    decode_passes: int = 1
    is_suspicious: bool = False


class UnicodeNormalizer:
    """
    World-class Unicode normalization and homograph attack detection.
    
    Implements state-of-the-art URL security normalization following:
    - RFC 3987 (IRIs - Internationalized Resource Identifiers)
    - RFC 5892 (IDNA 2008)
    - Unicode Technical Report #36 (Security Considerations)
    - Unicode Technical Report #39 (Security Mechanisms)
    
    The normalization chain is:
        Percent-decode (multi-pass) → Strip invisible → NFKC → Lowercase → Punycode
    
    Security features:
        - Multi-pass percent decoding (handles double/triple encoding)
        - Invisible character stripping (zero-width, RTL markers)
        - NFKC compatibility normalization (handles confusables like ℂ→C)
        - Mixed-script detection (Cyrillic + Latin = suspicious)
        - Skeleton-based confusable detection (а→a detection)
        - Homograph attack scoring for ML features
    
    Attributes:
        max_decode_passes: Maximum percent-decode iterations (default 10)
    """
    
    # Zero-width and invisible characters (homograph attack vectors)
    INVISIBLE_CHARS: Set[str] = {
        '\u200b',  # Zero-width space
        '\u200c',  # Zero-width non-joiner
        '\u200d',  # Zero-width joiner
        '\u200e',  # Left-to-right mark
        '\u200f',  # Right-to-left mark
        '\u061c',  # Arabic letter mark
        '\ufeff',  # Byte order mark / Zero-width no-break space
        '\u00ad',  # Soft hyphen
        '\u034f',  # Combining grapheme joiner
        '\u2060',  # Word joiner
        '\u2061',  # Function application
        '\u2062',  # Invisible times
        '\u2063',  # Invisible separator
        '\u2064',  # Invisible plus
        '\u206a',  # Inhibit symmetric swapping
        '\u206b',  # Activate symmetric swapping
        '\u206c',  # Inhibit Arabic form shaping
        '\u206d',  # Activate Arabic form shaping
        '\u206e',  # National digit shapes
        '\u206f',  # Nominal digit shapes
        '\uffa0',  # Halfwidth hangul filler
    }
    
    # Common confusable character mappings (subset of Unicode confusables.txt)
    # Maps visually similar characters to their ASCII skeleton
    CONFUSABLES: Dict[str, str] = {
        # Cyrillic lookalikes
        'а': 'a', 'е': 'e', 'о': 'o', 'р': 'p', 'с': 'c', 'у': 'y',
        'х': 'x', 'ѕ': 's', 'і': 'i', 'ј': 'j', 'ԁ': 'd', 'ɡ': 'g',
        'А': 'A', 'В': 'B', 'С': 'C', 'Е': 'E', 'Н': 'H', 'К': 'K',
        'М': 'M', 'О': 'O', 'Р': 'P', 'Т': 'T', 'Х': 'X',
        # Greek lookalikes
        'α': 'a', 'ο': 'o', 'ν': 'v', 'τ': 't', 'ρ': 'p', 'ι': 'i',
        'Α': 'A', 'Β': 'B', 'Ε': 'E', 'Ζ': 'Z', 'Η': 'H', 'Ι': 'I',
        'Κ': 'K', 'Μ': 'M', 'Ν': 'N', 'Ο': 'O', 'Ρ': 'P', 'Τ': 'T',
        'Χ': 'X', 'Υ': 'Y',
        # Mathematical/styled variants
        'ℂ': 'C', 'ℍ': 'H', 'ℕ': 'N', 'ℙ': 'P', 'ℚ': 'Q', 'ℝ': 'R',
        'ℤ': 'Z', 'ℯ': 'e', 'ⅈ': 'i', 'ⅉ': 'j',
        # Fullwidth characters
        'Ａ': 'A', 'Ｂ': 'B', 'Ｃ': 'C', 'Ｄ': 'D', 'Ｅ': 'E', 'Ｆ': 'F',
        'Ｇ': 'G', 'Ｈ': 'H', 'Ｉ': 'I', 'Ｊ': 'J', 'Ｋ': 'K', 'Ｌ': 'L',
        'Ｍ': 'M', 'Ｎ': 'N', 'Ｏ': 'O', 'Ｐ': 'P', 'Ｑ': 'Q', 'Ｒ': 'R',
        'Ｓ': 'S', 'Ｔ': 'T', 'Ｕ': 'U', 'Ｖ': 'V', 'Ｗ': 'W', 'Ｘ': 'X',
        'Ｙ': 'Y', 'Ｚ': 'Z',
        'ａ': 'a', 'ｂ': 'b', 'ｃ': 'c', 'ｄ': 'd', 'ｅ': 'e', 'ｆ': 'f',
        'ｇ': 'g', 'ｈ': 'h', 'ｉ': 'i', 'ｊ': 'j', 'ｋ': 'k', 'ｌ': 'l',
        'ｍ': 'm', 'ｎ': 'n', 'ｏ': 'o', 'ｐ': 'p', 'ｑ': 'q', 'ｒ': 'r',
        'ｓ': 's', 'ｔ': 't', 'ｕ': 'u', 'ｖ': 'v', 'ｗ': 'w', 'ｘ': 'x',
        'ｙ': 'y', 'ｚ': 'z',
        # Common substitutions used in phishing
        '0': 'o', '1': 'l', '!': 'i', '$': 's', '@': 'a', '3': 'e',
        # Subscript/superscript
        '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
        '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
    }
    
    # Scripts commonly used in homograph attacks
    SUSPICIOUS_SCRIPT_COMBOS: Set[frozenset] = {
        frozenset({'Latin', 'Cyrillic'}),
        frozenset({'Latin', 'Greek'}),
        frozenset({'Latin', 'Armenian'}),
        frozenset({'Latin', 'Hebrew'}),
    }
    
    def __init__(self, max_decode_passes: int = 10, cache_size: int = 100_000) -> None:
        """
        Initialize Unicode normalizer with LRU cache for high-volume processing.
        
        For 50M+ URLs, many hostnames repeat (google.com, facebook.com, etc).
        Caching avoids redundant normalization of common hostnames.
        
        Args:
            max_decode_passes: Maximum percent-decode iterations (default 10)
            cache_size: Maximum cached hostname results (default 100K)
        """
        self.max_decode_passes = max_decode_passes
        self.cache_size = cache_size
        # LRU cache for hostname normalization results
        # Using OrderedDict for O(1) LRU eviction
        from collections import OrderedDict
        self._host_cache: OrderedDict[str, NormalizationResult] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
    
    def normalize_text(self, value: str) -> NormalizationResult:
        """
        Perform full normalization chain on arbitrary text.
        
        Chain: Percent-decode → Strip invisible → NFKC → Lowercase
        
        Args:
            value: Input string to normalize
            
        Returns:
            NormalizationResult with normalized text and security metadata
        """
        if not value:
            return NormalizationResult(normalized="", original="")
        
        original = value
        current = value
        
        # Step 1: Multi-pass percent decoding
        decode_passes = 0
        for _ in range(self.max_decode_passes):
            try:
                decoded = unquote(current)
                decode_passes += 1
                if decoded == current:
                    break
                current = decoded
            except Exception:
                break
        
        # Step 2: Strip invisible characters
        had_invisible = any(ch in self.INVISIBLE_CHARS for ch in current)
        current = ''.join(ch for ch in current if ch not in self.INVISIBLE_CHARS)
        
        # Step 3: Check for non-ASCII (Unicode) content
        had_unicode = False
        try:
            current.encode('ascii')
        except UnicodeEncodeError:
            had_unicode = True
        
        # Step 4: NFKC normalization (compatibility decomposition + canonical composition)
        try:
            current = unicodedata.normalize('NFKC', current)
        except Exception:
            pass
        
        # Step 5: Detect confusables
        had_confusables = any(ch in self.CONFUSABLES for ch in current)
        
        # Step 6: Script analysis (for mixed-script detection)
        scripts_found = self._detect_scripts(current)
        mixed_scripts = len(scripts_found - {'Common', 'Inherited'}) > 1
        
        # Step 7: Calculate homograph risk score
        homograph_score = self._calculate_homograph_score(
            current, had_unicode, had_confusables, mixed_scripts, scripts_found
        )
        
        # Step 8: Lowercase for final normalization
        normalized = current.lower()
        
        # Step 9: Elite-Tier Script Optimization (Refinement v6.4)
        # To guarantee zero [UNK] tokens for basic LLM tokenizers (like MiniLM)
        # while preserving semantic signals, we hex-encode remaining non-ASCII chars.
        if had_unicode:
            hex_safe = []
            for ch in normalized:
                # If character is non-ASCII (above U+007F)
                if ord(ch) > 127:
                    # Use lowercased hex escape for stability
                    # We use %xx for consistency with URL standards
                    if ord(ch) < 256:
                        hex_safe.append(f"%{ord(ch):02x}")
                    else:
                        # For multi-byte characters, UTF-8 encode then hex
                        for byte in ch.encode('utf-8'):
                            hex_safe.append(f"%{byte:02x}")
                else:
                    hex_safe.append(ch)
            normalized = "".join(hex_safe)

        # Composite suspicious flag
        is_suspicious = (
            had_invisible or
            had_confusables or
            mixed_scripts or
            homograph_score > 0.5
        )
        
        return NormalizationResult(
            normalized=normalized,
            original=original,
            had_unicode=had_unicode,
            had_invisible=had_invisible,
            had_confusables=had_confusables,
            mixed_scripts=mixed_scripts,
            scripts_found=scripts_found,
            homograph_score=homograph_score,
            decode_passes=decode_passes,
            is_suspicious=is_suspicious,
        )
    
    def normalize_host(self, host: str) -> NormalizationResult:
        """
        Normalize hostname with IDNA/Punycode encoding and LRU caching.
        
        For 50M+ URL processing, uses LRU cache to avoid redundant
        normalization of common hostnames (google.com, facebook.com, etc).
        
        Extended chain: Percent-decode → Invisible → NFKC → IDNA → Punycode
        
        Args:
            host: Hostname to normalize
            
        Returns:
            NormalizationResult with punycode and security metadata
        """
        if not host:
            return NormalizationResult(normalized="", original="")
        
        # Strip leading/trailing dots (malformed domain artifacts like .example.com)
        host = host.strip().strip('.')
        
        # LRU cache lookup - O(1)
        if host in self._host_cache:
            self._cache_hits += 1
            # Move to end for LRU ordering
            self._host_cache.move_to_end(host)
            return self._host_cache[host]
        
        self._cache_misses += 1
        
        # First do standard text normalization
        result = self.normalize_text(host)
        
        if not result.normalized:
            return result
        
        # Apply IDNA/Punycode encoding
        try:
            # Check if already ASCII
            result.normalized.encode('ascii')
            result.punycode = result.normalized
        except UnicodeEncodeError:
            try:
                # NFKC + IDNA encode
                nfkc_host = unicodedata.normalize('NFKC', result.normalized)
                punycode_bytes = idna.encode(nfkc_host)
                result.punycode = punycode_bytes.decode('ascii')
                
                # Round-trip verification for integrity
                try:
                    roundtrip = idna.decode(punycode_bytes).lower()
                    if roundtrip != nfkc_host:
                        logging.warning(
                            "IDNA round-trip integrity failed: '%s' != '%s'",
                            nfkc_host, roundtrip
                        )
                        result.is_suspicious = True
                except Exception as rt_exc:
                    logging.debug("IDNA round-trip decode failed: %s", rt_exc)
                    result.is_suspicious = True
                    
            except idna.core.InvalidCodepoint as exc:
                logging.warning("IDNA invalid codepoint in '%s': %s", host, exc)
                result.punycode = result.normalized
                result.is_suspicious = True
            except Exception as exc:
                logging.warning("IDNA encoding failed for '%s': %s", host, exc)
                result.punycode = result.normalized
        
        # Store in LRU cache
        self._host_cache[host] = result
        # Evict oldest if cache full
        if len(self._host_cache) > self.cache_size:
            self._host_cache.popitem(last=False)
        
        return result
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.
        
        Useful for monitoring cache efficiency at scale.
        
        Returns:
            Dict with hits, misses, hit_rate, and current_size
        """
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0.0
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": f"{hit_rate:.2%}",
            "current_size": len(self._host_cache),
            "max_size": self.cache_size,
        }
    
    def compute_skeleton(self, text: str) -> str:
        """
        Compute skeleton form of text for confusable detection.
        
        The skeleton is derived by:
        1. NFKC normalization
        2. Applying confusable mappings
        3. Lowercasing
        
        Two strings with the same skeleton are visually confusable.
        
        Args:
            text: Input string
            
        Returns:
            Skeleton string for comparison
        """
        if not text:
            return ""
        
        # NFKC first
        try:
            normalized = unicodedata.normalize('NFKC', text)
        except Exception:
            normalized = text
        
        # Apply confusable mappings
        skeleton = []
        for ch in normalized.lower():
            skeleton.append(self.CONFUSABLES.get(ch, ch))
        
        return ''.join(skeleton)
    
    def is_confusable_with(self, text1: str, text2: str) -> bool:
        """
        Check if two strings are visually confusable (homograph candidates).
        
        Args:
            text1: First string
            text2: Second string
            
        Returns:
            True if strings have the same skeleton (visually similar)
        """
        return self.compute_skeleton(text1) == self.compute_skeleton(text2)
    
    def _detect_scripts(self, text: str) -> Set[str]:
        """
        Detect Unicode scripts present in text.
        
        Args:
            text: Input string
            
        Returns:
            Set of script names found
        """
        scripts = set()
        for ch in text:
            try:
                script = unicodedata.name(ch, '').split()[0]
                # Map common script identifiers
                if script in {'LATIN', 'DIGIT'}:
                    scripts.add('Latin')
                elif script == 'CYRILLIC':
                    scripts.add('Cyrillic')
                elif script == 'GREEK':
                    scripts.add('Greek')
                elif script == 'ARABIC':
                    scripts.add('Arabic')
                elif script == 'HEBREW':
                    scripts.add('Hebrew')
                elif script == 'CJK':
                    scripts.add('CJK')
                elif script in {'SPACE', 'HYPHEN', 'FULL', 'SOLIDUS'}:
                    scripts.add('Common')
                else:
                    scripts.add('Other')
            except Exception:
                continue
        return scripts
    
    def _calculate_homograph_score(
        self,
        text: str,
        had_unicode: bool,
        had_confusables: bool,
        mixed_scripts: bool,
        scripts_found: Set[str]
    ) -> float:
        """
        Calculate homograph attack risk score (0.0 to 1.0).
        
        Higher scores indicate higher likelihood of homograph attack.
        
        Args:
            text: Normalized text
            had_unicode: Whether non-ASCII was present
            had_confusables: Whether confusables were detected
            mixed_scripts: Whether multiple scripts were mixed
            scripts_found: Set of scripts detected
            
        Returns:
            Risk score from 0.0 (safe) to 1.0 (highly suspicious)
        """
        score = 0.0
        
        # Base scores for indicators
        if had_unicode:
            score += 0.1
        if had_confusables:
            score += 0.3
        if mixed_scripts:
            score += 0.4
        
        # Extra penalty for suspicious script combinations
        scripts_clean = scripts_found - {'Common', 'Inherited', 'Other'}
        for combo in self.SUSPICIOUS_SCRIPT_COMBOS:
            if combo.issubset(scripts_clean):
                score += 0.3
                break
        
        # Check for Cyrillic/Greek mixed with Latin (classic homograph)
        if 'Latin' in scripts_clean and ('Cyrillic' in scripts_clean or 'Greek' in scripts_clean):
            score += 0.2
        
        return min(score, 1.0)  # Cap at 1.0


# ============================================================================
# HELPER UTILITIES CLASS
# ============================================================================
class HelperUtilities:
    """
    Collection of static utility methods for URL preprocessing.
    
    This class provides reusable helper functions for text cleaning,
    entropy calculation, blob detection, and CSV field sanitization.
    All methods are static and do not require instance state.
    """
    
    @staticmethod
    def clean_text(value: str) -> str:
        """
        Clean and sanitize text input.
        
        Removes non-printable characters, invisible Unicode characters,
        and normalizes using NFKC normalization.
        
        Args:
            value: Input string to clean
            
        Returns:
            Cleaned and normalized string
        """
        if value is None:
            return ""
        # Remove non-printable characters
        cleaned = "".join(ch for ch in value if ch.isprintable())
        # Remove zero-width and invisible characters (homograph attack vectors)
        cleaned = ''.join(ch for ch in cleaned if ch not in Constants.INVISIBLE_CHARS)
        # Normalize using NFKC and strictly enforce lowercase for uncased LLMs
        return unicodedata.normalize("NFKC", cleaned.strip()).lower()
    
    @staticmethod
    def safe_unquote(value: str) -> str:
        """
        Safely decode percent-encoded strings with multi-pass decoding.
        
        Handles up to 5 layers of URL encoding (triple+ encoded payloads)
        and applies NFKC normalization. Multi-pass prevents bypass via
        double/triple encoding (e.g., %252e%252e%252f → ../).
        
        Args:
            value: Percent-encoded string
            
        Returns:
            Decoded and normalized string
        """
        current = value
        for _ in range(20):  # Increased to 20 to catch elite-tier nested payloads (e.g., 11+ pass)
            try:
                decoded = unquote(current)
                if decoded == current:
                    break  # Stable — no more encoding layers
                current = decoded
            except Exception:
                break
        try:
            return unicodedata.normalize("NFKC", current)
        except Exception:
            return current
    
    @staticmethod
    def shannon_entropy(value: str) -> float:
        """
        Calculate Shannon entropy of a string.
        
        Measures the randomness/unpredictability of the input,
        useful for detecting obfuscated or encoded content.
        
        Args:
            value: Input string
            
        Returns:
            Entropy value in bits (0.0 for empty string)
        """
        if not value:
            return 0.0
        counts = Counter(value)
        length = float(len(value))
        entropy = -sum((count / length) * math.log2(count / length)for count in counts.values())
        return entropy
    
    @staticmethod
    def digit_ratio(value: str) -> float:
        """
        Calculate the ratio of digits in a string.
        
        Args:
            value: Input string
            
        Returns:
            Ratio of digits (0.0 to 1.0)
        """
        if not value:
            return 0.0
        digits = sum(ch.isdigit() for ch in value)
        return digits / len(value)
    
    @staticmethod
    def looks_like_base64(segment: str, min_len: int) -> bool:
        """
        Check if a string segment looks like base64-encoded data.
        
        Args:
            segment: String segment to check
            min_len: Minimum length threshold
            
        Returns:
            True if segment appears to be base64-encoded
        """
        if len(segment) < min_len:
            return False
        return all(ch in Constants.BASE64_ALLOWED for ch in segment)
    
    @staticmethod
    def looks_like_hex(segment: str, min_len: int) -> bool:
        """
        Check if a string segment looks like hex-encoded data.
        
        Args:
            segment: String segment to check
            min_len: Minimum length threshold
            
        Returns:
            True if segment appears to be hex-encoded
        """
        if len(segment) < min_len:
            return False
        return all(ch in Constants.HEX_ALLOWED for ch in segment)
    
    @staticmethod
    def blob_placeholder(tag: str, value: str) -> str:
        """
        Create a placeholder for blob content.
        
        Args:
            tag: Type tag (e.g., "HEX_BLOB", "BASE64_BLOB")
            value: Original blob value
            
        Returns:
            Placeholder string with prefix
        """
        return f"<{tag}>{(value or '')[:Constants.BLOB_PREFIX_LEN]}"
    
    @staticmethod
    def sanitize_csv_field(value: Any) -> str:
        """
        Sanitize field values for CSV output.
        
        Prevents CSV injection, removes control characters,
        and ensures safe output formatting.
        
        Args:
            value: Field value to sanitize
            
        Returns:
            Sanitized string safe for CSV output
        """
        if value is None:
            return ""
        text = str(value)
        
        # Block CSV formula injection
        if text and text[0] in '=+@-':
            text = ' ' + text
        
        # Remove control characters (except tab)
        text = ''.join( ch for ch in text if ord(ch) >= 32 or ch == '\t')
        
        # Normalize line endings
        text = text.replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ')
        
        # Remove null bytes
        text = text.replace('\x00', '')
        
        return text


# ============================================================================
# AUDIT LOGGER CLASS
# ============================================================================
class AuditLogger:
    """
    Centralized logging with compliance-ready formatting.
    
    Provides structured logging with audit trails, drop reason tracking,
    and performance metrics. Supports both file and console output.
    
    Attributes:
        log_path: Path to the log file
        drop_counts: Counter for drop reasons
        duplicate_count: Count of duplicate URLs
        total_processed: Total URLs processed
        total_kept: Total URLs kept after filtering
    """
    
    def __init__(self, log_path: Path) -> None:
        """
        Initialize the audit logger.
        
        Args:
            log_path: Path to the log file
        """
        self.log_path = log_path
        self.drop_counts: Counter[str] = Counter()
        self.duplicate_count: int = 0
        self.total_processed: int = 0
        self.total_kept: int = 0
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Configure logging handlers for file and console output."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        # Clear existing handlers to avoid duplicates/conflicts
        for handler in root.handlers[:]:
            root.removeHandler(handler)
        
        # File handler (mode='a' to support resume, but we truncate manually if needed)
        # Actually mode='w' since AuditLogger is initialized once per run
        try:
            file_handler = logging.FileHandler(self.log_path, encoding="utf-8", mode="a")
            file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            root.addHandler(file_handler)
        except Exception as e:
            print(f"FAILED TO OPEN LOG FILE: {e}")
            
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        root.addHandler(console_handler)
    
    def log_drop(self, reason: str, count: int = 1) -> None:
        """
        Log a drop event with reason tracking.
        
        Args:
            reason: Reason code for the drop
            count: Number of items dropped
        """
        self.drop_counts[reason] += count
    
    def log_duplicate(self, count: int = 1) -> None:
        """
        Log duplicate detection.
        
        Args:
            count: Number of duplicates found
        """
        self.duplicate_count += count
    
    def increment_processed(self, count: int = 1) -> None:
        """Increment total processed count."""
        self.total_processed += count
    
    def increment_kept(self, count: int = 1) -> None:
        """Increment total kept count."""
        self.total_kept += count
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics.
        
        Returns:
            Dictionary with processing statistics
        """
        return {
            "total_processed": self.total_processed,
            "total_kept": self.total_kept,
            "duplicates": self.duplicate_count,
            "drop_counts": dict(self.drop_counts),
        }
    
    def log_chunk_stats(
        self,
        chunk_idx: int,
        kept: int,
        rejected: int,
        local_private: int
    ) -> None:
        """
        Log statistics for a processed chunk.
        
        Args:
            chunk_idx: Chunk index number
            kept: Number of URLs kept
            rejected: Number of URLs rejected
            local_private: Number of local/private IPs
        """
        logging.info(
            "Chunk %s -> kept=%s rejected=%s local_private=%s total_seen=%s",
            chunk_idx, kept, rejected, local_private, self.total_processed
        )


# ============================================================================
# URL REDIRECT RESOLVER CLASS
# ============================================================================
class URLRedirectResolver:
    """
    Safe URL redirect resolution with security controls.
    
    Follows HTTP redirects to expose final destination URLs while
    enforcing strict safety limits to prevent SSRF, hangs, and abuse.
    
    Safety Features:
        - HEAD-only requests (no content download)
        - Configurable max hops and timeout per hop
        - Private IP blocking (localhost, 10.x, 172.16.x, 192.168.x)
        - Known shortener detection for selective resolution
        - Graceful fallback to original URL on any error
    
    Attributes:
        config: PreprocessConfig instance
        tld_extractor: TLD extraction function for domain comparison
        shorteners: Set of known URL shortener domains (from urls_cate_V6.py)
    """
    
    # Private IP ranges to block (SSRF prevention)
    PRIVATE_RANGES: List[str] = [
        "127.0.0.0/8",      # Localhost
        "10.0.0.0/8",       # Private Class A
        "172.16.0.0/12",    # Private Class B
        "192.168.0.0/16",   # Private Class C
        "169.254.0.0/16",   # Link-local
        "::1/128",          # IPv6 localhost
        "fc00::/7",         # IPv6 unique local
        "fe80::/10",        # IPv6 link-local
    ]
    
    def __init__(
        self, 
        config: PreprocessConfig, 
        tld_extractor,
        shorteners: Optional[Set[str]] = None
    ) -> None:
        """
        Initialize redirect resolver.
        
        Args:
            config: Preprocessing configuration
            tld_extractor: TLD extraction function for domain comparison
            shorteners: Optional custom shortener set (defaults to DEFAULT_URL_SHORTENERS)
        """
        self.config = config
        self.tld_extractor = tld_extractor
        # Use shorteners from urls_cate_V6.py (100+ domains) or custom set
        self.shorteners = shorteners if shorteners is not None else DEFAULT_URL_SHORTENERS
        self._private_networks = [
            ipaddress.ip_network(cidr) for cidr in self.PRIVATE_RANGES
        ]
        # Session for connection pooling
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; PhishURLBot/1.0; +security-research)"
        })
    
    def _is_known_shortener(self, host: str) -> bool:
        """
        Check if host is a known URL shortener.
        
        Args:
            host: Hostname to check
            
        Returns:
            True if host matches a known shortener domain
        """
        host_lower = host.lower()
        # Direct match against urls_cate_V6.DEFAULT_URL_SHORTENERS
        if host_lower in self.shorteners:
            return True
        # Check subdomains (e.g., www.bit.ly -> bit.ly)
        if "." in host_lower:
            parent = host_lower.split(".", 1)[1]
            if parent in self.shorteners:
                return True
        return False
    
    def _is_safe_target(self, url: str) -> Tuple[bool, str]:
        """
        Check if redirect target is safe (not private/localhost).
        
        Args:
            url: URL to check
            
        Returns:
            Tuple of (is_safe, reason_if_unsafe)
        """
        try:
            parsed = urlparse(url)
            host = parsed.hostname or ""
            
            if not host:
                return False, "empty_host"
            
            # Check for localhost by name
            if host.lower() in ("localhost", "localhost.localdomain"):
                return False, "localhost"
            
            # Try to parse as IP and check private ranges
            try:
                ip = ipaddress.ip_address(host)
                for network in self._private_networks:
                    if ip in network:
                        return False, f"private_ip_{network}"
                return True, ""
            except ValueError:
                # Not an IP, assume DNS name is safe
                return True, ""
                
        except Exception as e:
            return False, f"parse_error_{e}"
    
    def _extract_registrable_domain(self, url: str) -> str:
        """
        Extract registrable domain from URL.
        
        Args:
            url: URL to extract domain from
            
        Returns:
            Registrable domain or empty string
        """
        try:
            tld_info = self.tld_extractor(url)
            return tld_info.registered_domain or ""
        except Exception:
            return ""
    
    def resolve(self, url: str, parsed: Optional[Any] = None) -> RedirectInfo:
        """
        Resolve URL redirects with safety controls.
        
        Follows HTTP redirects using HEAD requests, stopping at
        the configured max hops or when a non-redirect response is returned.
        
        Args:
            url: Original URL to resolve
            parsed: Optional parsed URL components (unused, for future API)
            
        Returns:
            RedirectInfo with resolution results
        """
        # Check if resolution is enabled
        if not self.config.enable_redirect_resolution:
            return RedirectInfo()
        
        # Check if we should only resolve known shorteners
        try:
            original_host = urlparse(url).hostname or ""
        except Exception:
            return RedirectInfo(error="url_parse_failed")
        
        if self.config.redirect_only_shorteners:
            if not self._is_known_shortener(original_host):
                return RedirectInfo()  # Skip non-shorteners
        
        # Track redirect chain
        current_url = url
        hops: List[str] = []
        original_domain = self._extract_registrable_domain(url)
        cross_domain = False
        
        try:
            for hop in range(self.config.redirect_max_hops):
                # Safety check before each hop
                is_safe, reason = self._is_safe_target(current_url)
                if not is_safe:
                    return RedirectInfo(
                        resolved=True,
                        final_url=url,  # Fall back to original
                        redirect_depth=hop,
                        redirect_cross_domain=cross_domain,
                        error=f"blocked_{reason}",
                        hops=hops[:3]
                    )
                
                # Make HEAD request (no body download)
                try:
                    response = self._session.head(
                        current_url,
                        allow_redirects=False,
                        timeout=self.config.redirect_timeout_sec,
                        verify=True  # SSL verification
                    )
                except requests.exceptions.SSLError:
                    # Retry with GET for servers that don't support HEAD
                    try:
                        response = self._session.get(
                            current_url,
                            allow_redirects=False,
                            timeout=self.config.redirect_timeout_sec,
                            stream=True,  # Don't download body
                            verify=True
                        )
                        response.close()  # Immediately close to prevent download
                    except Exception as e:
                        return RedirectInfo(
                            resolved=True,
                            final_url=current_url,
                            redirect_depth=hop,
                            redirect_cross_domain=cross_domain,
                            error=f"request_failed_{type(e).__name__}",
                            hops=hops[:3]
                        )
                
                # Check for redirect status codes
                if response.status_code in (301, 302, 303, 307, 308):
                    location = response.headers.get("Location", "")
                    if not location:
                        break  # No redirect target
                    
                    # Handle relative redirects
                    if not location.startswith(("http://", "https://")):
                        from urllib.parse import urljoin
                        location = urljoin(current_url, location)
                    
                    # Track hop domain
                    hop_domain = self._extract_registrable_domain(location)
                    if len(hops) < 3:
                        hops.append(hop_domain or urlparse(location).hostname or "")
                    
                    # Check for cross-domain redirect
                    if hop_domain and original_domain and hop_domain != original_domain:
                        cross_domain = True
                    
                    current_url = location
                else:
                    # Non-redirect response, we've reached the final URL
                    break
            
            return RedirectInfo(
                resolved=True,
                final_url=current_url,
                redirect_depth=len(hops),
                redirect_cross_domain=cross_domain,
                error="",
                hops=hops[:3]
            )
            
        except requests.exceptions.Timeout:
            return RedirectInfo(
                resolved=True,
                final_url=url,
                redirect_depth=len(hops),
                redirect_cross_domain=cross_domain,
                error="timeout",
                hops=hops[:3]
            )
        except requests.exceptions.ConnectionError:
            return RedirectInfo(
                resolved=True,
                final_url=url,
                redirect_depth=len(hops),
                redirect_cross_domain=cross_domain,
                error="connection_error",
                hops=hops[:3]
            )
        except Exception as e:
            return RedirectInfo(
                resolved=True,
                final_url=url,
                redirect_depth=len(hops),
                redirect_cross_domain=cross_domain,
                error=f"unexpected_{type(e).__name__}",
                hops=hops[:3]
            )


# ============================================================================
# URL PARSER CLASS
# ============================================================================
class URLParser:
    """
    URL parsing, normalization, and canonicalization.
    
    Handles all aspects of URL parsing including scheme classification,
    host normalization (IDNA/punycode), path normalization, and
    canonical URL construction.
    
    Integrates UnicodeNormalizer for world-class security:
    - Multi-pass percent decoding
    - NFKC normalization before punycode
    - Homograph attack detection
    - Mixed-script analysis
    
    Attributes:
        config: PreprocessConfig instance
        tld_extractor: TLD extraction function from CategoryConfig
        unicode_normalizer: UnicodeNormalizer for security-aware normalization
    """
    
    def __init__(self, config: PreprocessConfig, tld_extractor) -> None:
        """
        Initialize URL parser with security-focused normalizer.
        
        Args:
            config: Preprocessing configuration
            tld_extractor: TLD extraction function
        """
        self.config = config
        self.tld_extractor = tld_extractor
        self.unicode_normalizer = UnicodeNormalizer(max_decode_passes=10)
    
    def parse_url(self, raw_url: str) -> Optional[Tuple]:
        """
        Parse a raw URL into components.
        
        Args:
            raw_url: Raw URL string
            
        Returns:
            Parsed URL components tuple or None if parsing fails
        """
        try:
            parsed = urlparse(raw_url if "://" in raw_url else f"http://{raw_url}")
            return parsed
        except Exception:
            return None
    
    def normalize_host(self, host: str) -> str:
        """
        Normalize hostname to punycode (IDNA encoding).
        
        Uses UnicodeNormalizer for comprehensive security:
        - Multi-pass percent decoding
        - Invisible character removal
        - NFKC normalization
        - IDNA 2008 punycode encoding
        - Round-trip verification
        
        Args:
            host: Hostname to normalize
            
        Returns:
            Normalized hostname in punycode format
        """
        if not host:
            return ""
        
        # Use world-class UnicodeNormalizer
        result = self.unicode_normalizer.normalize_host(host)
        
        # Log security warnings
        if result.is_suspicious:
            logging.warning(
                "Suspicious host normalization: '%s' -> '%s' "
                "(homograph_score=%.2f, mixed_scripts=%s, confusables=%s)",
                host, result.punycode or result.normalized,
                result.homograph_score, result.mixed_scripts, result.had_confusables
            )
        
        return result.punycode or result.normalized
    
    def normalize_host_extended(self, host: str) -> Tuple[str, NormalizationResult]:
        """
        Normalize hostname with full security metadata for ML features.
        
        Extended version that returns both the normalized host and
        comprehensive security analysis for feature extraction.
        
        Args:
            host: Hostname to normalize
            
        Returns:
            Tuple of (normalized_host, NormalizationResult)
        """
        if not host:
            return "", NormalizationResult(normalized="", original="")
        
        result = self.unicode_normalizer.normalize_host(host)
        return result.punycode or result.normalized, result
    
    @staticmethod
    def normalize_path(path: str) -> str:
        """
        Normalize URL path with multi-pass decoding.
        
        Handles multi-layer percent decoding (up to 5 passes),
        backslash conversion, redundant slashes, and dot segments.
        Multi-pass decoding prevents triple-encoded path traversal
        bypass (e.g., %252e%252e%252f → ../).
        
        Args:
            path: URL path to normalize
            
        Returns:
            Normalized path string
        """
        decoded = path or "/"
        # Multi-pass percent decoding (consistent with host normalization)
        for _ in range(10):
            try:
                new_decoded = unquote(decoded)
                if new_decoded == decoded:
                    break  # Stable — no more encoding layers
                decoded = new_decoded
            except Exception:
                break
        decoded = decoded.replace("\\", "/")
        decoded = re.sub(r"/+", "/", decoded)
        normalized = posixpath.normpath(decoded)
        
        if path.endswith("/") and not normalized.endswith("/"):
            normalized += "/"
        if not normalized.startswith("/"):
            normalized = "/" + normalized
        if normalized == "//":
            normalized = "/"
        return normalized
    
    def classify_scheme(self, scheme: str, path: str) -> Tuple[str, bool]:
        """
        Classify URL scheme into family categories.
        
        Args:
            scheme: URL scheme
            path: URL path (for about:blank detection)
            
        Returns:
            Tuple of (scheme_family, should_keep)
        """
        scheme_lower = (scheme or "").lower()
        for family, members in Constants.SCHEME_FAMILIES.items():
            if scheme_lower in members:
                return family, True
        if scheme_lower == "about" and path.lower() == "blank":
            return "SYSTEM_RESOURCE", False
        return "IDENTIFIER_OTHER", True
    
    @staticmethod
    def _canonical_ip(host: str) -> Optional[str]:
        """
        Attempt to interpret host as an IP in any obfuscated form.
        
        Handles: standard IPv4/v6, decimal Dword (2130706433),
        hex (0x7f000001), octal (0177.0.0.01), and mixed-base
        dotted formats that APT groups use to bypass filters.
        
        Args:
            host: Hostname to check
            
        Returns:
            Canonical dotted-decimal IPv4 or standard IPv6, or None
        """
        h = host.strip().lower().rstrip('.')
        if not h:
            return None
        
        # 1. Standard parse (fast path — covers 99% of IPs)
        try:
            return str(ipaddress.ip_address(h))
        except ValueError:
            pass
        
        # 2. Single-integer (Dword) format: 2130706433 or 0x7f000001
        try:
            # Handle leading 0 as octal for elite-tier consistency
            if h.startswith('0x'):
                dword = int(h, 16)
            elif h.startswith('0') and len(h) > 1:
                try:
                    dword = int(h, 8)
                except ValueError:
                    dword = int(h, 10)
            else:
                dword = int(h, 10)
                
            if 0 <= dword <= 0xFFFFFFFF:
                return str(ipaddress.IPv4Address(dword))
        except (ValueError, OverflowError):
            pass
        
        # 3. Dotted with mixed bases and variable lengths (1-4 parts): 0x7f.1, 0177.0.0.01
        parts = h.split('.')
        if 1 <= len(parts) <= 4:
            octets: List[int] = []
            valid = True
            for i, p in enumerate(parts):
                if not p:
                    valid = False
                    break
                try:
                    if p.startswith('0x'):
                        val = int(p, 16)
                    elif p.startswith('0') and len(p) > 1:
                        try:
                            val = int(p, 8)
                        except ValueError:
                            val = int(p, 10)
                    else:
                        val = int(p, 10)
                    
                    if i < len(parts) - 1:
                        if 0 <= val <= 255:
                            octets.append(val)
                        else:
                            valid = False
                            break
                    else:
                        max_val = 0xFFFFFFFF >> (8 * (len(parts) - 1))
                        if 0 <= val <= max_val:
                            octets.append(val)
                        else:
                            valid = False
                            break
                except ValueError:
                    valid = False
                    break
                    
            if valid and octets:
                try:
                    if len(octets) == 4:
                        res = (octets[0] << 24) | (octets[1] << 16) | (octets[2] << 8) | octets[3]
                    elif len(octets) == 3:
                        res = (octets[0] << 24) | (octets[1] << 16) | octets[2]
                    elif len(octets) == 2:
                        res = (octets[0] << 24) | octets[1]
                    else:
                        res = octets[0]
                    return str(ipaddress.IPv4Address(res))
                except Exception:
                    pass
        
        # 4. Hyphenated IP format (common in cloud infra and phishing): 127-0-0-1 or 127-0-0-1.example.com
        # We check the whole host AND the first part (most common for IP-subdomains)
        candidates = [h, h.split('.')[0]]
        for cand in candidates:
            if '-' in cand and not cand.startswith('-') and not cand.endswith('-'):
                hyphen_parts = cand.split('-')
                if len(hyphen_parts) == 4 and all(p.isdigit() for p in hyphen_parts):
                    potential_dotted = ".".join(hyphen_parts)
                    try:
                        # Validate it's a real IP
                        return str(ipaddress.ip_address(potential_dotted))
                    except ValueError:
                        pass
            
        return None
    
    def is_ip_host(self, host: str) -> Tuple[bool, bool, Optional[str]]:
        """
        Check if host is an IP address, including obfuscated forms.
        
        Handles standard IPs, decimal Dword, hex, octal, and
        mixed-base formats used for evasion.
        
        Args:
            host: Hostname to check
            
        Returns:
            Tuple of (is_ip, is_private, canonical_ip)
        """
        if not host:
            return False, False, None
        if host.lower() in {"localhost", "local"}:
            return True, True, "127.0.0.1"
        canon = self._canonical_ip(host)
        if canon is not None:
            try:
                ip_obj = ipaddress.ip_address(canon)
                return True, bool(ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_reserved), canon
            except Exception:
                return True, False, canon
        return False, False, None
    def resolve_ip_to_domain(self, ip_address: str) -> Optional[str]:
        """
        Perform reverse DNS lookup to resolve an IP to a domain name.
        
        Args:
            ip_address: IP address string
            
        Returns:
            Resolved domain name or None if resolution fails/times out
        """
        if not ip_address:
            return None
            
        try:
            # Set global socket timeout for this operation
            # Note: This affects the current thread
            original_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(self.config.dns_timeout)
            
            try:
                # gethostbyaddr returns (hostname, aliaslist, ipaddrlist)
                resolved_name, _, _ = socket.gethostbyaddr(ip_address)
                return resolved_name.lower()
            finally:
                socket.setdefaulttimeout(original_timeout)
                
        except (socket.herror, socket.gaierror, socket.timeout):
            # DNS resolution failed or timed out
            return None
        except Exception as exc:
            logging.debug("Unexpected error during DNS resolution of %s: %s", ip_address, exc)
            return None

    @staticmethod
    def replace_host_in_url(raw_url: str, new_host: str) -> str:
        """
        Replace the host portion of a URL while preserving all other parts.
        
        Args:
            raw_url: Original URL
            new_host: New hostname/domain to inject
            
        Returns:
            Updated URL string
        """
        try:
            # Use urlsplit for more precise component handling than urlparse
            parts = list(urlsplit(raw_url))
            # parts[1] is netloc (host:port)
            # We want to replace just the host part if there's a port
            netloc = parts[1]
            if ":" in netloc:
                parts_netloc = netloc.rsplit(":", 1)
                new_netloc = f"{new_host}:{parts_netloc[1]}"
                parts[1] = new_netloc
            else:
                parts[1] = new_host
                
            return urlunsplit(parts)
        except Exception:
            return raw_url

    def extract_tld(self, raw_url: str) -> Optional[Any]:
        """
        Extract TLD information from URL.
        
        Args:
            raw_url: Raw URL string
            
        Returns:
            TLD info object or None if extraction fails
        """
        try:
            return self.tld_extractor(raw_url)
        except Exception:
            # Fallback: try punycode conversion
            try:
                punycode_url = raw_url.encode('idna').decode('ascii')
                return self.tld_extractor(punycode_url)
            except Exception:
                # Fallback: try hostname only
                try:
                    parsed = urlparse(raw_url if "://" in raw_url else f"http://{raw_url}")
                    if parsed.hostname:
                        return self.tld_extractor(f"http://{parsed.hostname}/")
                except Exception:
                    pass
        return None
    
    def mask_blobs(self, text: str) -> Tuple[str, bool, bool]:
        """
        Detect and mask hex/base64 blobs in text.
        
        Args:
            text: Text to process
            
        Returns:
            Tuple of (masked_text, has_hex, has_base64)
        """
        has_hex = False
        has_b64 = False
        parts = text.split("/")
        masked_parts: List[str] = []
        
        for part in parts:
            if HelperUtilities.looks_like_hex(part, self.config.hex_min_len):
                masked_parts.append(HelperUtilities.blob_placeholder("HEX_BLOB", part))
                has_hex = True
            elif HelperUtilities.looks_like_base64(part, self.config.base64_min_len):
                masked_parts.append(HelperUtilities.blob_placeholder("BASE64_BLOB", part))
                has_b64 = True
            else:
                masked_parts.append(part)
        
        return "/".join(masked_parts), has_hex, has_b64
    
    def mask_value_blob(self, value: str) -> Tuple[str, bool, bool]:
        """
        Mask a single value if it looks like a blob.
        
        Args:
            value: Value to check and mask
            
        Returns:
            Tuple of (masked_value, is_hex, is_base64)
        """
        if HelperUtilities.looks_like_hex(value, self.config.hex_min_len):
            return HelperUtilities.blob_placeholder("HEX_BLOB", value), True, False
        if HelperUtilities.looks_like_base64(value, self.config.base64_min_len):
            return HelperUtilities.blob_placeholder("BASE64_BLOB", value), False, True
        return value, False, False
    
    def build_fragment_info(self, fragment: str) -> FragmentInfo:
        """
        Build fragment metadata with enhanced analysis (Refinement v6.3).
        
        Performs 10-pass decoding to reveal hidden payloads, masks blobs,
        and identifies high-interest keywords/embedded URLs.
        
        Args:
            fragment: URL fragment string
            
        Returns:
            FragmentInfo dataclass instance
        """
        frag = fragment or ""
        # Multi-pass decode to expose nested payloads (Refinement v6.3)
        decoded_frag = HelperUtilities.safe_unquote(frag)
        
        # Mask blobs for features (but keep original structure in decoded_fragment)
        masked_frag, has_hex, has_b64 = self.mask_blobs(decoded_frag)
        
        preview = masked_frag[:self.config.max_fragment_preview]
        has_url = bool(re.search(r"https?://|[a-zA-Z0-9-]+\\.[a-z]{2,}", preview))
        has_token_keyword = any(token in preview.lower() for token in Constants.TOKEN_KEYWORDS)
        truncated = len(frag) > self.config.max_fragment_preview
        
        return FragmentInfo(
            length=len(frag),
            has_url=has_url,
            has_token_keyword=has_token_keyword,
            has_hex_blob=has_hex,
            has_base64_blob=has_b64,
            decoded_fragment=decoded_frag,
            truncated=truncated
        )
    
    def build_query_info(self, raw_query: str) -> QueryInfo:
        """
        Build query string metadata.
        
        Args:
            raw_query: Raw query string
            
        Returns:
            QueryInfo dataclass instance
        """
        # MUST fully decode query first so obfuscated '&' and '=' delimiters are exposed
        decoded_query = HelperUtilities.safe_unquote(raw_query)
        params = parse_qs(decoded_query, keep_blank_values=True)
        tracking_param_count = sum( 1 for key in params if key.lower() in self.config.tracking_params)
        other_params = { k: v for k, v in params.items() if k.lower() not in self.config.tracking_params}
        
        masked_params: Dict[str, List[str]] = {}
        has_hex = False
        has_b64 = False
        entropies: List[float] = []
        
        for key, values in other_params.items():
            masked_values: List[str] = []
            for value in values:
                value_norm = HelperUtilities.safe_unquote(value)
                entropy_val = HelperUtilities.shannon_entropy(value_norm)
                entropies.append(entropy_val)
                masked_value, hex_flag, b64_flag = self.mask_value_blob(value_norm)
                has_hex = has_hex or hex_flag
                has_b64 = has_b64 or b64_flag
                masked_values.append(masked_value)
            masked_params[key] = masked_values
        
        canonical_query = urlencode(sorted(masked_params.items()), doseq=True, safe="/").lower()
        avg_entropy = float(sum(entropies) / len(entropies)) if entropies else 0.0
        
        return QueryInfo(
            canonical_query=canonical_query,
            total_params=len(params),
            tracking_param_count=tracking_param_count,
            other_param_count=len(other_params),
            has_tracking_params=tracking_param_count > 0,
            has_base64_blob=has_b64,
            has_hex_blob=has_hex,
            avg_param_entropy=avg_entropy,
        )
    
    def build_canonical_url(
        self,
        scheme: str,
        host: str,
        port: Optional[int],
        path: str,
        canonical_query: str
    ) -> str:
        """
        Build canonical URL from components.
        
        Args:
            scheme: URL scheme
            host: Normalized hostname
            port: Port number (optional)
            path: Normalized path
            canonical_query: Canonical query string
            
        Returns:
            Canonical URL string
        """
        netloc = host
        if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
            netloc = f"{host}:{port}"
        
        # Force percent-encode non-ASCII characters and force lowercase hex 
        # so uncased WordPiece tokenizers don't drop them as [UNK]
        clean_path = quote(path, safe="/%+").lower()
        if not clean_path.startswith("/"):
            clean_path = "/" + clean_path
            
        return urlunparse((scheme, netloc, clean_path, "", canonical_query, ""))
    
    def build_model_url(
        self,
        scheme: str,
        normalized_host: str,
        port: Optional[int],
        raw_path: str,
        raw_query: str,
        fragment: str
    ) -> str:
        """
        Build minimally-cleaned URL for model training (OFP mode).
        
        This URL preserves maximum signal for the model by ONLY applying:
          - scheme lowercase
          - host NFKC + Punycode (via normalize_host, already done by caller)
          - invisible character stripping on path (no normpath, no decode, no blob mask)
        
        Deliberately SKIPS:
          - Multi-pass percent decoding (encoding layers = signal)
          - Path normalization / posixpath.normpath (../ traversal = signal)
          - Blob masking (hex/base64 blobs = signal)
          - Query sorting / tracking param removal (all params = context)
          - Default port stripping (port presence = signal)
          - IP canonicalization (obfuscated IP format = signal)
        
        Args:
            scheme: URL scheme (lowercase)
            normalized_host: Host after NFKC + Punycode only
            port: Port number (kept as-is, no default stripping)
            raw_path: Original path (only invisible chars stripped)
            raw_query: Original query string (no sorting/filtering)
            fragment: Original fragment
        
        Returns:
            Minimally-cleaned URL string
        """
        # Strip invisible chars from path (prevents tokenizer [UNK] issues)
        clean_path = ''.join(
            ch for ch in raw_path if ch not in Constants.INVISIBLE_CHARS
        ) if raw_path else "/"
        # Force percent-encode non-ASCII characters and force lowercase hex 
        # so uncased WordPiece tokenizers don't drop them as [UNK]
        clean_path = quote(clean_path, safe="/%+").lower()
        
        if not clean_path.startswith("/"):
            clean_path = "/" + clean_path
        
        # Strip invisible chars from query
        clean_query = ''.join(
            ch for ch in raw_query if ch not in Constants.INVISIBLE_CHARS
        ) if raw_query else ""
        clean_query = quote(clean_query, safe="=&%+").lower()
        
        # Build netloc with port PRESERVED (no default-port stripping)
        netloc = normalized_host
        if port is not None:
            netloc = f"{normalized_host}:{port}"
        
        return urlunparse((scheme, netloc, clean_path, "", clean_query, fragment))
    
    def dedup_key(
        self,
        scheme: str,
        host: str,
        path: str,
        canonical_query: str
    ) -> Tuple[str, str, str, str]:
        """
        Generate deduplication key for a URL.
        
        Args:
            scheme: URL scheme
            host: Hostname
            path: URL path
            canonical_query: Query string
            
        Returns:
            Tuple key for deduplication
        """
        return scheme.lower(), host.lower(), path, canonical_query


# ============================================================================
# FEATURE EXTRACTOR CLASS
# ============================================================================
class FeatureExtractor:
    """
    Feature extraction for ML pipeline.
    
    Extracts and computes features from parsed URLs including TLD risk,
    host type classification, entropy metrics, severity scoring, and
    constructs the input text for transformer models.
    
    Attributes:
        config: PreprocessConfig instance
        analyzer: URLAnalyzer instance
        high_risk_tlds: Set of high-risk TLDs
        contextual_tlds: Set of contextual TLDs
        tld_risk_from_excel: TLD risk mappings from Excel
        entropy_q33: 33rd percentile entropy threshold
        entropy_q67: 67th percentile entropy threshold
    """
    
    def __init__(
        self,
        config: PreprocessConfig,
        analyzer: URLAnalyzer,
        tld_risk_from_excel: Dict[str, str]
    ) -> None:
        """
        Initialize feature extractor.
        
        Args:
            config: Preprocessing configuration
            analyzer: URLAnalyzer instance
            tld_risk_from_excel: TLD risk mappings from Excel
        """
        self.config = config
        self.analyzer = analyzer
        self.high_risk_tlds = set(analyzer.config.SUSPICIOUS_TLDS)
        self.contextual_tlds = set(analyzer.config.CONTEXTUAL_TLDS)
        self.tld_risk_from_excel = tld_risk_from_excel
        self.entropy_q33: float = 3.0
        self.entropy_q67: float = 4.2
    
    def calibrate_entropy_thresholds(self, sample_urls: List[str]) -> Tuple[float, float]:
        """
        Calibrate entropy thresholds from sample data.
        
        Args:
            sample_urls: List of sample URLs for calibration
            
        Returns:
            Tuple of (q33_threshold, q67_threshold)
        """
        if not sample_urls or len(sample_urls) < 100:
            logging.warning(
                "Insufficient samples (%d) for entropy calibration; using defaults.",
                len(sample_urls)
            )
            return 3.0, 4.2
        
        try:
            entropies = [HelperUtilities.shannon_entropy(url) for url in sample_urls]
            entropies = [e for e in entropies if e is not None and e > 0]
            
            if len(entropies) < 50:
                logging.warning("Too few valid entropy values (%d); using defaults.", len(entropies))
                return 3.0, 4.2
            
            import statistics
            q33 = statistics.quantiles(entropies, n=3)[0]
            q67 = statistics.quantiles(entropies, n=3)[1]
            
            self.entropy_q33 = q33
            self.entropy_q67 = q67
            
            logging.info(
                "Entropy thresholds calibrated: L<%.2f, M<%.2f, H>=%.2f (n=%d)",
                q33, q67, q67, len(entropies)
            )
            return q33, q67
        except Exception as exc:
            logging.warning("Entropy calibration failed: %s; using defaults.", exc)
            return 3.0, 4.2
    
    def tld_risk(self, tld: str) -> str:
        """
        Determine TLD risk level.
        
        Args:
            tld: Top-level domain
            
        Returns:
            Risk level: "HIGH", "CONTEXTUAL", or "NORMAL"
        """
        tld_lower = tld.lower()
        if tld_lower in self.tld_risk_from_excel:
            return self.tld_risk_from_excel[tld_lower]
        if tld_lower in self.high_risk_tlds:
            return "HIGH"
        if tld_lower in self.contextual_tlds:
            return "CONTEXTUAL"
        return "NORMAL"
    
    def host_type(self, host: str, is_ip: bool, is_private: bool) -> str:
        """
        Classify host into categorical type.
        
        Args:
            host: Hostname
            is_ip: Whether host is an IP address
            is_private: Whether host is private/reserved
            
        Returns:
            Host type: DNS, IP4P, IP4R, IP6, LOC, or OBF
        """
        if not host:
            return "UNKNOWN"
        host_lower = host.lower()
        
        if host_lower in {"localhost", "local", "127.0.0.1", "::1"}:
            return "LOC"
        
        if is_ip:
            try:
                ip_obj = ipaddress.ip_address(host)
                if isinstance(ip_obj, ipaddress.IPv6Address):
                    return "IP6"
                elif is_private:
                    return "IP4R"
                else:
                    return "IP4P"
            except Exception:
                pass
        
        # Check for obfuscated IP patterns
        if re.match(r"^0x[0-9a-fA-F]{1,8}$", host_lower):
            return "OBF"
        if re.match(r"^\d{1,10}$", host_lower):
            try:
                num = int(host_lower)
                if 0 <= num <= 0xFFFFFFFF:
                    return "OBF"
            except Exception:
                pass
        
        return "DNS"
    
    def idn_flag(self, host: str) -> int:
        """Check if host contains non-ASCII characters."""
        if not host:
            return 0
        return 1 if any(ord(ch) > 127 for ch in host) else 0
    
    def entropy_bucket(self, value: float) -> str:
        """Map entropy to bucket: L (low), M (medium), H (high)."""
        if value < self.entropy_q33:
            return "L"
        elif value < self.entropy_q67:
            return "M"
        return "H"
    
    def digit_ratio_bucket(self, value: float) -> str:
        """Map digit ratio to bucket: L (<0.2), M (0.2-0.6), H (>0.6)."""
        if value < 0.2:
            return "L"
        elif value < 0.6:
            return "M"
        return "H"
    
    def depth_bucket(self, depth: int) -> str:
        """Map path depth to bucket: S (≤2), M (3-5), D (>5)."""
        if depth <= 2:
            return "S"
        elif depth <= 5:
            return "M"
        return "D"
    
    def severity_from_flags(self, flags: List[str]) -> Tuple[int, float]:
        """
        Compute flag count and severity score.
        
        Args:
            flags: List of active flag names
            
        Returns:
            Tuple of (flag_count, severity_score)
        """
        flags_set = set(flags)
        flag_count = len(flags_set)
        severity_score = 0.0
        for flag in flags_set:
            severity_score += float(
                self.analyzer.config.DETECTION_WEIGHTS.get(flag, 1.0)
            )
        return flag_count, severity_score
    
    def severity_bucket(self, severity_score: float, flag_count: int) -> str:
        """Map severity to bucket: L, M, or H."""
        if flag_count == 0:
            return "L"
        if severity_score < 2.0:
            return "L"
        if severity_score >= 5.0 or flag_count >= 4:
            return "H"
        return "M"
    
    def short_flag_codes(self, flags: List[str], max_codes: int = 16) -> Tuple[List[str], bool]:
        """
        Map flag names to short codes.
        
        Args:
            flags: List of flag names
            max_codes: Maximum number of codes to return
            
        Returns:
            Tuple of (codes_list, was_truncated)
        """
        codes = []
        seen = set()
        for flag in flags:
            # Strictly map semantic tokens. Fallback strictly to first 4 chars, abandoning algorithmic dynamic generation.
            code = Constants.FLAG_CODE_MAP.get(flag, flag[:4].upper())
            
            if code not in seen:
                codes.append(code)
                seen.add(code)
            
            if len(codes) >= max_codes:
                break
        
        return sorted(codes), len(flags) > max_codes
    
    def risk_label_short(self, tld_risk: str) -> str:
        """Convert TLD risk to single character: H/C/N/U."""
        risk_upper = (tld_risk or "UNKNOWN").upper()
        if "HIGH" in risk_upper:
            return "H"
        elif "CONTEXTUAL" in risk_upper:
            return "C"
        elif "NORMAL" in risk_upper:
            return "N"
        return "U"
    
    def compress_scheme_family(self, scheme_family: str) -> str:
        """Compress scheme family to short code."""
        return Constants.SCHEME_CODE_MAP.get(
            scheme_family, scheme_family[:4].upper()
        )
    
    def primary_category(self, active_categories: List[str]) -> str:
        """Determine primary category from active categories."""
        if not active_categories:
            return "None"
        for category in Constants.PRIMARY_CATEGORY_PRIORITY:
            if category in active_categories:
                return category
        return active_categories[0] if active_categories else "None"
    
    def rule_feature_values(self, flags_active: List[str]) -> Tuple[int, float, int]:
        """
        Compute rule-assisted feature values.
        
        Args:
            flags_active: List of active flags
            
        Returns:
            Tuple of (flags_count, severity_score, flags_bitmask)
        """
        flags_set = set(flags_active)
        flags_count = len(flags_set)
        severity_score = 0.0
        bitmask = 0
        
        for idx, flag in enumerate(Constants.FLAG_ORDER):
            if flag in flags_set:
                bitmask |= 1 << idx
                severity_score += float(
                    self.analyzer.config.DETECTION_WEIGHTS.get(flag, 1.0)
                )
        
        return flags_count, severity_score, bitmask
    
    def shorten_canonical_url(self, canonical_url: str, max_tail_len: int = 400) -> str:
        """
        Shorten canonical URL while preserving key information.
        
        Args:
            canonical_url: Full canonical URL
            max_tail_len: Maximum length
            
        Returns:
            Shortened URL string
        """
        if len(canonical_url) <= max_tail_len:
            return canonical_url
        
        try:
            parsed = urlparse(canonical_url)
            scheme = parsed.scheme or "http"
            host = parsed.hostname or ""
            port = parsed.port
            path = parsed.path or "/"
            query = parsed.query or ""
            
            if port and not ((scheme == "http" and port == 80) or 
                           (scheme == "https" and port == 443)):
                netloc = f"{host}:{port}"
            else:
                netloc = host
            
            base = f"{scheme}://{netloc}"
            remaining_budget = max_tail_len - len(base) - 10
            
            if remaining_budget < 20:
                return base + path[:30] + "..." if len(path) > 30 else base + path
            
            combined_tail = path + ("?" + query if query else "")
            if len(combined_tail) <= remaining_budget:
                return base + combined_tail
            
            front_len = max(15, remaining_budget // 3)
            back_len = max(10, remaining_budget // 4)
            abbreviated = combined_tail[:front_len] + "..." + combined_tail[-back_len:]
            return base + abbreviated
        except Exception:
            return canonical_url[:max_tail_len] + ("..." if len(canonical_url) > max_tail_len else "")
    
    def build_input_text(
        self,
        tld: str,
        tld_risk: str,
        scheme: str,
        scheme_family: str,
        host: str,
        registrable_domain: str,
        path: str,
        has_base64_blob: bool,
        has_hex_blob: bool,
        is_ip_host: bool,
        is_private_ip: bool,
        query_info: QueryInfo,
        fragment_info: FragmentInfo,
        flags: List[str],
        entropy_url: float,
        entropy_path: float,
        entropy_query: float,
        digit_ratio: float,
        path_depth: int,
        canonical_url: str,
        canonical_ip: Optional[str] = None,
        redirect_info: Optional[RedirectInfo] = None,
    ) -> str:
        """
        Build compact input text for transformer models.
        
        Constructs a fixed-order categorical header with key features
        followed by a shortened canonical URL. Optionally includes
        redirect resolution features when enabled.
        
        Args:
            tld: Top-level domain
            tld_risk: TLD risk level
            scheme: URL scheme
            scheme_family: Scheme family classification
            host: Normalized hostname
            registrable_domain: Registrable domain
            path: Normalized path
            has_base64_blob: Whether base64 blob detected
            has_hex_blob: Whether hex blob detected
            is_ip_host: Whether host is IP address
            is_private_ip: Whether IP is private
            query_info: Query string metadata
            fragment_info: Fragment metadata
            flags: List of active flags
            entropy_url: URL entropy
            entropy_path: Path entropy
            entropy_query: Query entropy
            digit_ratio: Digit ratio
            path_depth: Path depth
            canonical_url: Canonical URL
            canonical_ip: Canonicalized IP string if host is IP
            redirect_info: Optional redirect resolution metadata
            
        Returns:
            Formatted input text string
        """
        primary_cat = self.primary_category(flags)
        host_t = self.host_type(canonical_ip or host, is_ip_host, is_private_ip)
        idn = self.idn_flag(host)
        
        entropy_url_bucket = self.entropy_bucket(entropy_url)
        entropy_path_bucket = self.entropy_bucket(entropy_path)
        entropy_query_bucket = self.entropy_bucket(entropy_query)
        
        digit_ratio_b = self.digit_ratio_bucket(digit_ratio)
        depth_b = self.depth_bucket(path_depth)
        
        flag_count, severity_score = self.severity_from_flags(flags)
        severity_b = self.severity_bucket(severity_score, flag_count)
        
        flag_codes, flags_truncated = self.short_flag_codes(flags, max_codes=8)
        flags_str = ",".join(flag_codes) if flag_codes else "NONE"
        
        risk_short = self.risk_label_short(tld_risk)
        tld_safe = tld or "NONE"
        scheme_code = self.compress_scheme_family(scheme_family)
        
        fru = 1 if fragment_info.has_url else 0
        frtk = 1 if fragment_info.has_token_keyword else 0
        # Refinement v6.3: Multi-layer blob detection (Path/Query/Fragment)
        frb = 1 if (has_hex_blob or has_base64_blob or 
                   query_info.has_hex_blob or query_info.has_base64_blob or
                   fragment_info.has_hex_blob or fragment_info.has_base64_blob) else 0
        frt = 1 if fragment_info.truncated else 0
        
        header_parts = [
            f"PC:{primary_cat}",
            f"SF:{scheme_code}",
            f"TLD:{tld_safe} RSK:{risk_short}",
            f"HT:{host_t}",
            f"IDN:{idn}",
            f"EU:{entropy_url_bucket} EP:{entropy_path_bucket} EQ:{entropy_query_bucket}",
            f"DRB:{digit_ratio_b}",
            f"DPB:{depth_b}",
            f"RSF:{flag_count} RSB:{severity_b}",
            f"FLG:{flags_str}",
            f"FLG_TR:{1 if flags_truncated else 0}",
            f"FRU:{fru} FRTK:{frtk} FRB:{frb} FRT:{frt}",
        ]
        
        # Add redirect features if enabled and present
        if self.config.redirect_include_features and redirect_info is not None:
            header_parts.append(
                f"REDIR_DEPTH:{redirect_info.redirect_depth} "
                f"REDIR_XDOMAIN:{1 if redirect_info.redirect_cross_domain else 0}"
            )
        
        header = " ".join(header_parts)
        shortened_url = self.shorten_canonical_url(canonical_url, max_tail_len=400)
        
        return f"{header} URL:{shortened_url}"


# ============================================================================
# DEDUPLICATION MANAGER CLASS
# ============================================================================
class DeduplicationManager:
    """
    Deduplication with caching and resume support.
    
    Manages URL deduplication using composite keys, versioned cache
    persistence, and supports both pre-split and post-split dedup modes.
    
    Attributes:
        config: PreprocessConfig instance
        seen_keys: Set of deduplication keys
        dedup_stats: Statistics per split
        analyzer_config: URLAnalyzer config for rule version
    """
    
    def __init__(self, config: PreprocessConfig, analyzer_config) -> None:
        """
        Initialize deduplication manager.
        
        Args:
            config: Preprocessing configuration
            analyzer_config: URLAnalyzer configuration for versioning
        """
        self.config = config
        self.analyzer_config = analyzer_config
        self.seen_keys: Set[Tuple[str, str, str, str]] = set()
        self.dedup_stats = {
            'train_duplicates': 0,
            'val_duplicates': 0,
            'test_duplicates': 0,
            'global_duplicates': 0
        }
        
        if config.dedup_cache and Path(config.dedup_cache).exists():
            self._load_cache()
    
    def _compute_rule_version(self) -> int:
        """Compute version hash from detection weights."""
        return hash(str(sorted(self.analyzer_config.DETECTION_WEIGHTS.items())))
    
    def _load_cache(self) -> None:
        """Load deduplication cache with version validation."""
        if not self.config.dedup_cache or not Path(self.config.dedup_cache).exists():
            return
        
        try:
            with open(self.config.dedup_cache, "rb") as handle:
                cache_data = pickle.load(handle)
            
            if isinstance(cache_data, set):
                self.seen_keys = cache_data
                logging.info("Loaded %s dedup keys from legacy cache.", len(cache_data))
            elif isinstance(cache_data, dict) and "keys" in cache_data:
                stored_version = cache_data.get("rule_version")
                current_version = self._compute_rule_version()
                
                if stored_version != current_version:
                    logging.warning(
                        "Cache version mismatch. Discarding stale cache."
                    )
                    self.seen_keys = set()
                else:
                    self.seen_keys = cache_data["keys"]
                    logging.info(
                        "Loaded %s dedup keys from versioned cache.",
                        len(self.seen_keys)
                    )
            else:
                logging.warning("Cache format unrecognized; starting fresh.")
                self.seen_keys = set()
        except Exception as exc:
            logging.warning("Failed to load dedup cache: %s", exc)
            self.seen_keys = set()
    
    def save_cache(self) -> None:
        """Save deduplication cache with version metadata."""
        if not self.config.dedup_cache:
            return
        
        try:
            cache_data = {
                "keys": self.seen_keys,
                "rule_version": self._compute_rule_version(),
                "timestamp": datetime.now().isoformat(),
                "count": len(self.seen_keys)
            }
            with open(self.config.dedup_cache, "wb") as handle:
                pickle.dump(cache_data, handle)
            logging.info("Persisted %s dedup keys to cache.", len(self.seen_keys))
        except Exception as exc:
            logging.warning("Failed to save dedup cache: %s", exc)
    
    def is_duplicate(self, key: Tuple[str, str, str, str]) -> bool:
        """Check if key is a duplicate."""
        return key in self.seen_keys
    
    def add_key(self, key: Tuple[str, str, str, str]) -> None:
        """Add key to seen set."""
        self.seen_keys.add(key)
    
    def load_seen_from_outputs(self, output_csv: Path) -> int:
        """
        Rebuild seen keys from existing output files for resume.
        
        Args:
            output_csv: Path to main output file
            
        Returns:
            Number of keys restored
        """
        candidates: List[Path] = []
        if output_csv.exists():
            candidates.append(output_csv)
        for suffix in ("_train.csv", "_val.csv", "_test.csv"):
            p = Path(str(output_csv).replace('.csv', suffix))
            if p.exists():
                candidates.append(p)
        
        restored = 0
        for p in candidates:
            try:
                with p.open('r', encoding='utf-8', errors='ignore') as fh:
                    header = fh.readline()
                cols = [c.strip() for c in header.split(',')]
                col_lower = [c.lower() for c in cols]
                
                if 'canonical_url' in col_lower:
                    usecols = [cols[col_lower.index('canonical_url')]]
                elif 'raw_url' in col_lower:
                    usecols = [cols[col_lower.index('raw_url')]]
                else:
                    continue
                
                with pd.read_csv(
                    p, usecols=usecols, chunksize=100_000,
                    encoding='utf-8', on_bad_lines='skip'
                ) as reader:
                    for chunk in reader:
                        colname = chunk.columns[0]
                        for val in chunk[colname].fillna(''):
                            try:
                                parsed = urlparse(val if '://' in val else f'http://{val}')
                                key = (
                                    parsed.scheme.lower(),
                                    (parsed.hostname or '').lower(),
                                    parsed.path or '/',
                                    parsed.query or ''
                                )
                                self.seen_keys.add(key)
                                restored += 1
                            except Exception:
                                continue
            except Exception as exc:
                logging.warning("Failed to restore keys from %s: %s", p, exc)
        
        logging.info("Restored %d dedup keys from existing outputs.", restored)
        return restored
    
    def deduplicate_split(self, split_path: Path, split_name: str) -> int:
        """
        Apply deduplication to a single split file.
        
        Args:
            split_path: Path to split CSV
            split_name: Name of split (train/val/test)
            
        Returns:
            Number of duplicates removed
        """
        if not split_path.exists():
            logging.warning("Split file %s does not exist.", split_path)
            return 0
        
        try:
            df = pd.read_csv(
                split_path, dtype=str, keep_default_na=False,
                na_filter=False, on_bad_lines="skip"
            )
        except Exception as exc:
            logging.error("Failed to read %s: %s", split_path, exc)
            return 0
        
        initial_count = len(df)
        col_lower = [c.lower() for c in df.columns]
        
        # Determine URL column
        if 'canonical_url' in col_lower:
            src_col = df.columns[col_lower.index('canonical_url')]
        elif 'raw_url' in col_lower:
            src_col = df.columns[col_lower.index('raw_url')]
        elif 'input' in col_lower:
            src_col = df.columns[col_lower.index('input')]
        else:
            logging.warning("No URL column found in %s", split_path)
            return 0
        
        seen_local: Set[Tuple[str, str, str, str]] = set()
        keep_indices: List[int] = []
        removed = 0
        
        for idx, val in enumerate(df[src_col].fillna('')):
            try:
                val = HelperUtilities.clean_text(val)
                parsed = urlparse(val if '://' in val else f'http://{val}')
                key = (
                    parsed.scheme.lower() or 'http',
                    (parsed.hostname or '').lower(),
                    parsed.path or '/',
                    parsed.query or ''
                )
            except Exception:
                keep_indices.append(idx)
                continue
            
            if key in seen_local:
                removed += 1
                continue
            
            seen_local.add(key)
            keep_indices.append(idx)
        
        if removed > 0:
            try:
                df.iloc[keep_indices].to_csv(split_path, index=False, encoding='utf-8')
                logging.info(
                    "Split '%s' deduplicated: %d -> %d (removed %d)",
                    split_name, initial_count, len(keep_indices), removed
                )
            except Exception as exc:
                logging.warning("Failed to write deduplicated split: %s", exc)
        
        return removed


# ============================================================================
# URL SPLITTER AND REPORTER CLASS
# ============================================================================
class URLSplitterAndReporter:
    """
    Stratified train/val/test splitting and reporting for multiple modes.
    
    Synchronizes splitting across all requested modes (preprocessed, raw_orig, 
    OFP_Minimal, canonical, hybrid) in a single pass over the output data. 
    Guarantees that a specific URL will be in the same split across all modes.
    """
    
    def __init__(
        self,
        config: PreprocessConfig,
        dedup_manager: DeduplicationManager
    ) -> None:
        self.config = config
        self.dedup_manager = dedup_manager
        self.assigned_train = Counter()
        self.assigned_val = Counter()
        self.assigned_test = Counter()

    def _get_group_from_row(self, row_dict: Dict) -> Tuple[int, str, str]:
        """Extract stratification group (label, pc, tr) consistently from a row."""
        label = int(pd.to_numeric(row_dict.get("label", 0), errors="coerce") or 0)
        
        # Primary Category: Prefer h_primary_category if primary_category is missing or empty
        pc = str(row_dict.get("primary_category") or row_dict.get("h_primary_category", "UNKNOWN"))
        if pc == "nan" or not pc or pc == "None":
            pc = "UNKNOWN"
            
        # TLD Risk: Prefer tld_risk if present
        tr = str(row_dict.get("tld_risk") or "UNKNOWN")
        if tr == "UNKNOWN" or tr == "nan" or not tr:
            # Fallback to heuristic risk flags if tld_risk is missing
            if int(row_dict.get("h_tld_risk_critical", 0)): tr = "CRITICAL"
            elif int(row_dict.get("h_tld_risk_high", 0)): tr = "HIGH"
            elif int(row_dict.get("h_tld_risk_normal", 0)): tr = "NORMAL"
            else: tr = "UNKNOWN"
            
        return (label, pc, tr)

    def _count_groups(self, path: Path) -> Dict[Tuple[int, str, str], int]:
        """Count unique groups (label, primary_category, tld_risk) in CSV."""
        if not path.exists():
            logging.warning("Splitter: Path does not exist for counting: %s", path)
            return {}
            
        group_counts: Dict[Tuple[int, str, str], int] = {}
        total_rows_read = 0
        
        try:
            with pd.read_csv(path, dtype=str, chunksize=self.config.chunk_size, 
                            keep_default_na=False, na_filter=False, on_bad_lines="skip") as reader:
                for chunk in reader:
                    if chunk.empty: continue
                    total_rows_read += len(chunk)
                    
                    for row_tuple in chunk.itertuples(index=False):
                        row_dict = row_tuple._asdict()
                        group = self._get_group_from_row(row_dict)
                        group_counts[group] = group_counts.get(group, 0) + 1
                        
            logging.info("Splitter: Counted %d rows across %d groups in %s", 
                         total_rows_read, len(group_counts), path.name)
                        
        except Exception as exc:
            logging.error("Failed to count groups in %s: %s", path, exc)
            
        return group_counts

    def _calculate_group_split_targets(self, count: int) -> Tuple[int, int, int]:
        """Scientific-Grade Balanced Stratification Targets."""
        if count == 0: return (0, 0, 0)
        if count == 1: return (1, 0, 0)
        if count == 2: return (1, 1, 0)
        
        tr, vl, ts = 1, 1, 1 # Min 1 each
        rem = count - 3
        t = int(rem * self.config.train_frac + 0.5)
        v = int(rem * self.config.val_frac + 0.5)
        if t + v > rem:
            if (t - rem * self.config.train_frac) > (v - rem * self.config.val_frac):
                t = max(0, rem - v)
            else:
                v = max(0, rem - t)
        s = rem - t - v
        return (tr + t, vl + v, ts + s)

    def _prepare_mode_row(self, row_dict: Dict, mode: str) -> Dict:
        """Format a row according to the target split mode."""
        label = row_dict.get("label", 0)
        
        if mode == "raw_orig":
            return {
                "input": row_dict.get("raw_url", ""),
                "label": label,
                "primary_category": row_dict.get("primary_category") or row_dict.get("h_primary_category", "UNKNOWN"),
                "tld_risk": row_dict.get("tld_risk", "UNKNOWN"),
            }
            
        elif mode == "OFP_Minimal":
            return {
                "input": row_dict.get("model_url", ""),
                "label": label,
                "primary_category": row_dict.get("primary_category", "UNKNOWN"),
                "tld_risk": row_dict.get("tld_risk", "UNKNOWN")
            }
            
        elif mode == "canonical":
            return {
                "input": row_dict.get("canonical_url", ""),
                "label": label,
                "primary_category": row_dict.get("primary_category", "UNKNOWN"),
                "tld_risk": row_dict.get("tld_risk", "UNKNOWN")
            }
            
        elif mode == "hybrid":
            # Extract all h_ and hF_ columns
            hybrid_row = {"input": row_dict.get("canonical_url", ""), "label": label}
            for k, v in row_dict.items():
                if k.startswith("h_") or k.startswith("hF_"):
                    hybrid_row[k] = v
            return hybrid_row
            
        else: # preprocessed (Legacy/Standard)
            return {"input": row_dict.get("input", ""), "label": label}

    def create_splits(self) -> None:
        """Create ALL configured splits in a single synchronized pass."""
        sources = self.config.split_source
        active_modes = ["preprocessed", "raw_orig", "OFP_Minimal", "canonical", "hybrid"] \
                       if "all" in sources else [s for s in sources if s in 
                       ["preprocessed", "raw_orig", "OFP_Minimal", "canonical", "hybrid"]]
        
        if not active_modes:
            return

        logging.info("Starting Synchronized Stratified Splitting for modes: %s", active_modes)
        
        # 1. Calculate Targets
        group_counts = self._count_groups(self.config.output_csv)
        if not group_counts:
            logging.warning("No data found in output_csv for splitting.")
            return
            
        targets = {group: self._calculate_group_split_targets(count) for group, count in group_counts.items()}
        target_train = {g: t[0] for g, t in targets.items()}
        target_val = {g: t[1] for g, t in targets.items()}
        
        logging.info("Splitter Targets: Total Train=%d, Val=%d, Samples=%d", 
                     sum(target_train.values()), sum(target_val.values()), sum(group_counts.values()))
        
        # 2. Setup File Paths and Cleanup
        mode_files = {}
        for mode in active_modes:
            prefix = "urls_ofp" if mode == "OFP_Minimal" else \
                     ("urls_canonical" if mode == "canonical" else \
                     ("urls_hybrid" if mode == "hybrid" else \
                     ("raw_orig" if mode == "raw_orig" else "urls_preprocessed")))
            
            paths = {
                'train': self.config.output_csv.with_name(f"{prefix}_train.csv"),
                'val': self.config.output_csv.with_name(f"{prefix}_val.csv"),
                'test': self.config.output_csv.with_name(f"{prefix}_test.csv")
            }
            for p in paths.values(): self._safe_unlink(p)
            mode_files[mode] = paths

        # 3. Synchronized Processing Loop
        self.assigned_train = Counter(); self.assigned_val = Counter(); self.assigned_test = Counter()
        
        with pd.read_csv(self.config.output_csv, dtype=str, chunksize=self.config.chunk_size, 
                        keep_default_na=False, na_filter=False, on_bad_lines="skip") as reader:
            for chunk in tqdm(reader, desc="Synchronizing Splits"):
                chunk["label"] = pd.to_numeric(chunk["label"], errors="coerce").fillna(0).astype(int)
                
                # Internal shuffling per chunk to break sequential bias
                shuffled = chunk.sample(frac=1.0, random_state=self.config.random_seed)
                
                # Split buffers per mode
                buffers = {mode: {'train': [], 'val': [], 'test': []} for mode in active_modes}
                i = 0
                
                for row_tuple in shuffled.itertuples(index=False):
                    row_dict = row_tuple._asdict()
                    group = self._get_group_from_row(row_dict)
                    
                    # Determine Split (Once per row, applied to all modes)
                    if self.assigned_train[group] < target_train.get(group, 0):
                        split = 'train'; self.assigned_train[group] += 1
                    elif self.assigned_val[group] < target_val.get(group, 0):
                        split = 'val'; self.assigned_val[group] += 1
                    else:
                        split = 'test'; self.assigned_test[group] += 1
                        
                    # Add to all mode buffers
                    for mode in active_modes:
                        buffers[mode][split].append(self._prepare_mode_row(row_dict, mode))
                
                # Write buffers to disk
                for mode in active_modes:
                    for split in ['train', 'val', 'test']:
                        if buffers[mode][split]:
                            self._append_df(pd.DataFrame(buffers[mode][split]), mode_files[mode][split])

        logging.info("Synchronized splitting complete. Train=%d Val=%d Test=%d",
                    sum(self.assigned_train.values()), sum(self.assigned_val.values()), sum(self.assigned_test.values()))

    def deduplicate_post_split(self) -> None:
        """Apply deduplication to all splits generated in create_splits."""
        sources = self.config.split_source
        active_modes = ["preprocessed", "raw_orig", "OFP_Minimal", "canonical", "hybrid"] \
                       if "all" in sources else [s for s in sources if s in 
                       ["preprocessed", "raw_orig", "OFP_Minimal", "canonical", "hybrid"]]
        
        for mode in active_modes:
            prefix = "urls_ofp" if mode == "OFP_Minimal" else \
                     ("urls_canonical" if mode == "canonical" else \
                     ("urls_hybrid" if mode == "hybrid" else \
                     ("raw_orig" if mode == "raw_orig" else "urls_preprocessed")))
            
            splits = {
                'train': self.config.output_csv.with_name(f"{prefix}_train.csv"),
                'val': self.config.output_csv.with_name(f"{prefix}_val.csv"),
                'test': self.config.output_csv.with_name(f"{prefix}_test.csv")
            }
            
            for split_name, split_path in splits.items():
                if split_path.exists():
                    removed = self.dedup_manager.deduplicate_split(split_path, split_name)
                    self.dedup_manager.dedup_stats[f'{split_name}_duplicates'] = removed

    @staticmethod
    def _safe_unlink(path: Path) -> None:
        """Safely unlink a file with retries for Windows."""
        if not path.exists(): return
        for _ in range(3):
            try:
                path.unlink()
                return
            except PermissionError:
                import time; time.sleep(0.5)
        logging.warning("Failed to unlink %s due to lock", path)

    def _append_df(self, df: pd.DataFrame, path: Path) -> None:
        """Append DataFrame to CSV with proper header handling."""
        path.parent.mkdir(parents=True, exist_ok=True)
        if df.empty: return
        file_exists = path.exists()
        df_clean = df.copy()
        for col in df_clean.select_dtypes(include=['object']).columns:
            df_clean[col] = df_clean[col].apply(HelperUtilities.sanitize_csv_field)
        
        write_mode = "w" if not file_exists else "a"
        write_header = not file_exists
        try:
            df_clean.to_csv(path, mode=write_mode, index=False, header=write_header,
                          encoding="utf-8", quoting=csv.QUOTE_ALL, escapechar='\\', lineterminator='\n')
        except Exception as exc:
            logging.error("Failed to write to %s: %s", path, exc)

    def check_splits_exist(self) -> bool:
        """Check if split files exist for all requested modes."""
        sources = self.config.split_source
        active_modes = ["preprocessed", "raw_orig", "OFP_Minimal", "canonical", "hybrid"] \
                       if "all" in sources else sources
        
        for mode in active_modes:
            prefix = "urls_ofp" if mode == "OFP_Minimal" else \
                     ("urls_canonical" if mode == "canonical" else \
                     ("urls_hybrid" if mode == "hybrid" else \
                     ("raw_orig" if mode == "raw_orig" else "urls_preprocessed")))
            
            files = [
                self.config.output_csv.with_name(f"{prefix}_train.csv"),
                self.config.output_csv.with_name(f"{prefix}_val.csv"),
                self.config.output_csv.with_name(f"{prefix}_test.csv"),
            ]
            if not any(p.exists() for p in files): return False
        return True

    def _audit_physical_files(self, mode: str) -> None:
        """Audit physical CSV files to get exact group counts for reporting."""
        prefix = "urls_ofp" if mode == "OFP_Minimal" else \
                 ("urls_canonical" if mode == "canonical" else \
                 ("urls_hybrid" if mode == "hybrid" else \
                 ("raw_orig" if mode == "raw_orig" else "urls_preprocessed")))
        
        paths = {
            "train": self.config.output_csv.with_name(f"{prefix}_train.csv"),
            "val": self.config.output_csv.with_name(f"{prefix}_val.csv"),
            "test": self.config.output_csv.with_name(f"{prefix}_test.csv")
        }
            
        self.assigned_train = Counter(self._count_groups(paths["train"]))
        self.assigned_val = Counter(self._count_groups(paths["val"]))
        self.assigned_test = Counter(self._count_groups(paths["test"]))

    def generate_report(self, mode: str) -> None:
        """Generate split distribution report using physical file audit."""
        self._audit_physical_files(mode)
        suffixes = {"raw_orig": "raw_orig", "OFP_Minimal": "ofp", "canonical": "canonical", "hybrid": "hybrid"}
        report_name = f"report_splits_{suffixes.get(mode, 'preprocessed')}.txt"
        
        report_path = self.config.output_csv.parent / report_name
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        total = sum(self.assigned_train.values()) + sum(self.assigned_val.values()) + sum(self.assigned_test.values())
        tr_total = sum(self.assigned_train.values())
        vl_total = sum(self.assigned_val.values())
        ts_total = sum(self.assigned_test.values())
        
        tr_pct = tr_total / total if total > 0 else 0
        vl_pct = vl_total / total if total > 0 else 0
        ts_pct = ts_total / total if total > 0 else 0
        
        lines = [
            "=" * 80,
            f"URL Split Report ({mode} mode)",
            "=" * 80,
            f"Generated: {datetime.now(timezone.utc).isoformat()} UTC",
            f"Target Ratios: Train: {self.config.train_frac:.1%}, Val: {self.config.val_frac:.1%}, "
            f"Test: {1 - self.config.train_frac - self.config.val_frac:.1%}",
            "",
            "1. OVERALL SPLIT STATISTICS",
            "--------------------------------------------------------------------------------",
            f"Total Samples: {total}",
            f"Train Set:     {tr_total:>8} ({tr_pct:.2%})",
            f"Val Set:       {vl_total:>8} ({vl_pct:.2%})",
            f"Test Set:      {ts_total:>8} ({ts_pct:.2%})",
            "",
            "2. LABEL (CLASS) DISTRIBUTION",
            "--------------------------------------------------------------------------------",
            f"{'Label':<10} | {'Train':>10} | {'Val':>10} | {'Test':>10} | {'Total':>10}",
            "-" * 80,
        ]
        
        lab_train, lab_val, lab_test = Counter(), Counter(), Counter()
        for (g_lab, g_pc, g_tr), count in self.assigned_train.items(): lab_train[g_lab] += count
        for (g_lab, g_pc, g_tr), count in self.assigned_val.items(): lab_val[g_lab] += count
        for (g_lab, g_pc, g_tr), count in self.assigned_test.items(): lab_test[g_lab] += count
        
        all_labels = sorted(set(lab_train.keys()) | set(lab_val.keys()) | set(lab_test.keys()))
        for lab in all_labels:
            tr_l, vl_l, ts_l = lab_train[lab], lab_val[lab], lab_test[lab]
            lines.append(f"{str(lab):<10} | {tr_l:>10} | {vl_l:>10} | {ts_l:>10} | {tr_l+vl_l+ts_l:>10}")
            
        lines.extend(["", "3. UNIVERSAL CATEGORY COVERAGE DASHBOARD", "-" * 80, f"{'Category':<32} | {'Train':>8} | {'Val':>8} | {'Test':>8} | {'Status'}", "-" * 80])
        
        cat_train, cat_val, cat_test = Counter(), Counter(), Counter()
        for (g_lab, g_pc, g_tr), count in self.assigned_train.items(): cat_train[g_pc] += count
        for (g_lab, g_pc, g_tr), count in self.assigned_val.items(): cat_val[g_pc] += count
        for (g_lab, g_pc, g_tr), count in self.assigned_test.items(): cat_test[g_pc] += count
        
        all_categories = sorted(URLCategory.categories())
        for cat in all_categories:
            tr_c, vl_c, ts_c = cat_train[cat], cat_val[cat], cat_test[cat]
            status = "✅ OK"
            if tr_c == 0 or vl_c == 0 or ts_c == 0:
                missing = [s for s, c in [("Train", tr_c), ("Val", vl_c), ("Test", ts_c)] if c == 0]
                status = f"⚠️ MISSING: {','.join(missing)}"
            lines.append(f"{cat:<32} | {tr_c:>8} | {vl_c:>8} | {ts_c:>8} | {status}")
        
        lines.extend(["", "4. TLD RISK DISTRIBUTION PER SPLIT", "-" * 80, f"{'Risk Level':<12} | {'Train':>10} | {'Val':>10} | {'Test':>10} | {'Total':>10}", "-" * 80])
        
        risk_train, risk_val, risk_test = Counter(), Counter(), Counter()
        for (g_lab, g_pc, g_tr), count in self.assigned_train.items(): risk_train[g_tr] += count
        for (g_lab, g_pc, g_tr), count in self.assigned_val.items(): risk_val[g_tr] += count
        for (g_lab, g_pc, g_tr), count in self.assigned_test.items(): risk_test[g_tr] += count
        
        for risk in ["NORMAL", "MEDIUM", "HIGH", "CRITICAL", "UNKNOWN"]:
            tr_r, vl_r, ts_r = risk_train[risk], risk_val[risk], risk_test[risk]
            if tr_r + vl_r + ts_r > 0:
                lines.append(f"{risk:<12} | {tr_r:>10} | {vl_r:>10} | {ts_r:>10} | {tr_r+vl_r+ts_r:>10}")
            
        with report_path.open("w", encoding="utf-8") as f: f.write("\n".join(lines))
        logging.info("Report generated: %s", report_path)

    def _calculate_group_split_targets(self, count: int) -> Tuple[int, int, int]:
        """
        Scientific-Grade Balanced Stratification.
        Guarantees 1 sample per split if N >= 3, and uses balanced rounding 
        to maintain perfect 80/10/10 distribution across the full dataset.
        """
        if count == 0:
            return (0, 0, 0)
        if count == 1:
            return (1, 0, 0)
        if count == 2:
            return (1, 0, 1)
        
        # Minimum 1 each per split
        tr, vl, ts = 1, 1, 1
        rem = count - 3
        
        # Calculate ideal targets for the remainder
        t_target = rem * self.config.train_frac
        v_target = rem * self.config.val_frac
        
        # Nearest-integer rounding (Balanced Rounding)
        t = int(t_target + 0.5)
        v = int(v_target + 0.5)
        
        # Safety: clamp if rounding pushed us over
        if t + v > rem:
            # Drop the one with the smallest fractional surplus
            if (t - t_target) > (v - v_target):
                t = max(0, rem - v)
            else:
                v = max(0, rem - t)
        
        s = rem - t - v
        return (tr + t, vl + v, ts + s)

    def _create_mode_splits(self, mode: str) -> None:
        """Create splits for preprocessed or original mode."""
        group_counts = self._count_groups(self.config.output_csv)
        if not group_counts:
            logging.warning("No data for splitting.")
            return
        
        if mode == "raw_orig":
            train_path = self.config.output_csv.with_name("raw_orig_train.csv")
            val_path = self.config.output_csv.with_name("raw_orig_val.csv")
            test_path = self.config.output_csv.with_name("raw_orig_test.csv")
        else:
            train_path = self.config.output_csv.with_name("urls_preprocessed_train.csv")
            val_path = self.config.output_csv.with_name("urls_preprocessed_val.csv")
            test_path = self.config.output_csv.with_name("urls_preprocessed_test.csv")
        
        for p in [train_path, val_path, test_path]:
            self._safe_unlink(p)
        
        # Centralized Stratification: Calculate targets using Balanced Rounding
        target_train, target_val, target_test = {}, {}, {}
        for group, count in group_counts.items():
            tr, vl, ts = self._calculate_group_split_targets(count)
            target_train[group] = tr
            target_val[group] = vl
            target_test[group] = ts
        
        self.assigned_train = Counter(); self.assigned_val = Counter(); self.assigned_test = Counter()
        
        if mode == "preprocessed":
            with pd.read_csv(
                self.config.output_csv, dtype={"label": str},
                chunksize=self.config.chunk_size, keep_default_na=False,
                na_filter=False, on_bad_lines="skip"
            ) as reader:
                for chunk in tqdm(reader, desc="Splitting"):
                    chunk["label"] = pd.to_numeric(chunk["label"], errors="coerce").astype("Int64")
                    chunk = chunk.loc[~chunk["label"].isna()]
                    if chunk.empty: continue
                    
                    shuffled = chunk.sample(frac=1.0, random_state=self.config.random_seed)
                    train_rows, val_rows, test_rows = [], [], []
                    
                    for row in shuffled.itertuples(index=False):
                        label = int(getattr(row, "label"))
                        pc = str(getattr(row, "primary_category", "") or "").strip() or "UNKNOWN"
                        tr = str(getattr(row, "tld_risk", "") or "").strip() or "UNKNOWN"
                        group = (label, pc, tr)
                        
                        if self.assigned_train[group] < target_train.get(group, 0):
                            train_rows.append(row); self.assigned_train[group] += 1
                        elif self.assigned_val[group] < target_val.get(group, 0):
                            val_rows.append(row); self.assigned_val[group] += 1
                        else:
                            test_rows.append(row); self.assigned_test[group] += 1
                    
                    self._append_df(pd.DataFrame(train_rows), train_path)
                    self._append_df(pd.DataFrame(val_rows), val_path)
                    self._append_df(pd.DataFrame(test_rows), test_path)
        else:
            mapping = self._build_mapping()
            with pd.read_csv(
                self.config.input_csv, dtype=str,
                chunksize=self.config.chunk_size, keep_default_na=False,
                na_filter=False, on_bad_lines="skip"
            ) as input_reader:
                for chunk in tqdm(input_reader, desc="Splitting raw_orig"):
                    shuffled = chunk.sample(frac=1.0, random_state=self.config.random_seed)
                    train_rows, val_rows, test_rows = [], [], []
                    for row in shuffled.itertuples(index=False):
                        row_dict = row._asdict()
                        raw_string = row_dict.get("input") or row_dict.get("url", "")
                        raw_input = HelperUtilities.clean_text(raw_string)
                        dq = mapping.get(raw_input)
                        if dq and len(dq) > 0:
                            label, pc, tr = dq.popleft()
                            group = (label, pc, tr)
                            if self.assigned_train[group] < target_train.get(group, 0):
                                train_rows.append(row_dict); self.assigned_train[group] += 1
                            elif self.assigned_val[group] < target_val.get(group, 0):
                                val_rows.append(row_dict); self.assigned_val[group] += 1
                            else:
                                test_rows.append(row_dict); self.assigned_test[group] += 1
                    self._append_df(pd.DataFrame(train_rows), train_path)
                    self._append_df(pd.DataFrame(val_rows), val_path)
                    self._append_df(pd.DataFrame(test_rows), test_path)
        
        logging.info(
            "Splits created: Train=%d Val=%d Test=%d",
            sum(self.assigned_train.values()),
            sum(self.assigned_val.values()),
            sum(self.assigned_test.values())
        )
    
    def _create_ofp_splits(self) -> None:
        """Create OFP_Minimal splits."""
        group_counts = self._count_groups(self.config.output_csv)
        if not group_counts:
            return
        
        train_path = self.config.output_csv.with_name("urls_ofp_train.csv")
        val_path = self.config.output_csv.with_name("urls_ofp_val.csv")
        test_path = self.config.output_csv.with_name("urls_ofp_test.csv")
        
        for p in [train_path, val_path, test_path]:
            self._safe_unlink(p)
        
        

    def _build_mapping_ofp(self) -> Dict[str, deque]:
        """Build mapping for OFP with model_url (minimally-cleaned) using chunking."""
        mapping: Dict[str, deque] = {}
        with pd.read_csv(
            self.config.output_csv, dtype=str,
            chunksize=self.config.chunk_size, keep_default_na=False,
            na_filter=False, on_bad_lines="skip"
        ) as reader:
            for chunk in reader:
                chunk["label"] = pd.to_numeric(chunk["label"], errors="coerce").astype("Int64")
                chunk = chunk.loc[~chunk["label"].isna()]
                for r in chunk.itertuples(index=False):
                    raw = str(getattr(r, "raw_url", "") or "")
                    # v5: Use model_url (minimally-cleaned) for OFP input
                    model = str(getattr(r, "model_url", "") or "")
                    if not model:
                        continue  # Skip rows without model_url
                    lab = int(getattr(r, "label"))
                    pc = str(getattr(r, "primary_category", "") or "") or "UNKNOWN"
                    tr = str(getattr(r, "tld_risk", "") or "") or "UNKNOWN"
                    clean_raw = HelperUtilities.clean_text(raw)
                    mapping.setdefault(clean_raw, deque()).append((model, lab, pc, tr))
        return mapping

    def _create_ofp_splits(self) -> None:
        """Create OFP_Minimal splits."""
        group_counts = self._count_groups(self.config.output_csv)
        if not group_counts: return
        
        train_path = self.config.output_csv.with_name("urls_ofp_train.csv")
        val_path = self.config.output_csv.with_name("urls_ofp_val.csv")
        test_path = self.config.output_csv.with_name("urls_ofp_test.csv")
        
        for p in [train_path, val_path, test_path]: self._safe_unlink(p)
        
        # Centralized Stratification: Calculate targets using Balanced Rounding
        target_train, target_val, target_test = {}, {}, {}
        for group, count in group_counts.items():
            tr, vl, ts = self._calculate_group_split_targets(count)
            target_train[group] = tr
            target_val[group] = vl
            target_test[group] = ts
        
        self.assigned_train = Counter(); self.assigned_val = Counter(); self.assigned_test = Counter()
        mapping = self._build_mapping_ofp()
        
        with pd.read_csv(
            self.config.input_csv, dtype=str,
            chunksize=self.config.chunk_size, keep_default_na=False,
            na_filter=False, on_bad_lines="skip"
        ) as input_reader:
            for chunk in tqdm(input_reader, desc="Splitting OFP"):
                shuffled = chunk.sample(frac=1.0, random_state=self.config.random_seed)
                train_rows, val_rows, test_rows = [], [], []
                for row in shuffled.itertuples(index=False):
                    row_dict = row._asdict()
                    raw_string = row_dict.get("input") or row_dict.get("url", "")
                    raw_input = HelperUtilities.clean_text(raw_string)
                    dq = mapping.get(raw_input)
                    if dq and len(dq) > 0:
                        can, lab, pc, tr = dq.popleft()
                        group = (lab, pc, tr)
                        row_dict["input"] = can  # model_url from mapping
                        row_dict["label"] = lab  # Use verified label from preprocessed data
                        row_dict["primary_category"] = pc
                        row_dict["tld_risk"] = tr
                        if self.assigned_train[group] < target_train.get(group, 0):
                            train_rows.append(row_dict); self.assigned_train[group] += 1
                        elif self.assigned_val[group] < target_val.get(group, 0):
                            val_rows.append(row_dict); self.assigned_val[group] += 1
                        else:
                            test_rows.append(row_dict); self.assigned_test[group] += 1
                self._append_df(pd.DataFrame(train_rows), train_path)
                self._append_df(pd.DataFrame(val_rows), val_path)
                self._append_df(pd.DataFrame(test_rows), test_path)
        logging.info(
            "OFP splits created: Train=%d Val=%d Test=%d",
            sum(self.assigned_train.values()),
            sum(self.assigned_val.values()),
            sum(self.assigned_test.values())
        )

    def _build_mapping_canonical(self) -> Dict[str, deque]:
        """Build mapping for canonical with canonical_url using chunking."""
        mapping: Dict[str, deque] = {}
        with pd.read_csv(
            self.config.output_csv, dtype=str,
            chunksize=self.config.chunk_size, keep_default_na=False,
            na_filter=False, on_bad_lines="skip"
        ) as reader:
            for chunk in reader:
                chunk["label"] = pd.to_numeric(chunk["label"], errors="coerce").astype("Int64")
                chunk = chunk.loc[~chunk["label"].isna()]
                for r in chunk.itertuples(index=False):
                    raw = str(getattr(r, "raw_url", "") or "")
                    can = str(getattr(r, "canonical_url", "") or "")
                    if not can:
                        continue
                    lab = int(getattr(r, "label"))
                    pc = str(getattr(r, "primary_category", "") or "") or "UNKNOWN"
                    tr = str(getattr(r, "tld_risk", "") or "") or "UNKNOWN"
                    clean_raw = HelperUtilities.clean_text(raw)
                    mapping.setdefault(clean_raw, deque()).append((can, lab, pc, tr))
        return mapping

    def _create_canonical_splits(self) -> None:
        """Create canonical splits using canonical_url."""
        group_counts = self._count_groups(self.config.output_csv)
        if not group_counts: return
        
        train_path = self.config.output_csv.with_name("urls_canonical_train.csv")
        val_path = self.config.output_csv.with_name("urls_canonical_val.csv")
        test_path = self.config.output_csv.with_name("urls_canonical_test.csv")
        
        for p in [train_path, val_path, test_path]: self._safe_unlink(p)
        
        # Centralized Stratification: Calculate targets using Balanced Rounding
        target_train, target_val, target_test = {}, {}, {}
        for group, count in group_counts.items():
            tr, vl, ts = self._calculate_group_split_targets(count)
            target_train[group] = tr
            target_val[group] = vl
            target_test[group] = ts
        
        self.assigned_train = Counter(); self.assigned_val = Counter(); self.assigned_test = Counter()
        mapping = self._build_mapping_canonical()
        
        with pd.read_csv(
            self.config.input_csv, dtype=str,
            chunksize=self.config.chunk_size, keep_default_na=False,
            na_filter=False, on_bad_lines="skip"
        ) as input_reader:
            for chunk in tqdm(input_reader, desc="Splitting Canonical"):
                shuffled = chunk.sample(frac=1.0, random_state=self.config.random_seed)
                train_rows, val_rows, test_rows = [], [], []
                for row in shuffled.itertuples(index=False):
                    row_dict = row._asdict()
                    raw_string = row_dict.get("input") or row_dict.get("url", "")
                    raw_input = HelperUtilities.clean_text(raw_string)
                    dq = mapping.get(raw_input)
                    if dq and len(dq) > 0:
                        can, lab, pc, tr = dq.popleft()
                        group = (lab, pc, tr)
                        row_dict["input"] = can  # map to canonical
                        row_dict["label"] = lab
                        row_dict["primary_category"] = pc
                        row_dict["tld_risk"] = tr
                        if self.assigned_train[group] < target_train.get(group, 0):
                            train_rows.append(row_dict); self.assigned_train[group] += 1
                        elif self.assigned_val[group] < target_val.get(group, 0):
                            val_rows.append(row_dict); self.assigned_val[group] += 1
                        else:
                            test_rows.append(row_dict); self.assigned_test[group] += 1
                self._append_df(pd.DataFrame(train_rows), train_path)
                self._append_df(pd.DataFrame(val_rows), val_path)
                self._append_df(pd.DataFrame(test_rows), test_path)
        logging.info(
            "Canonical splits created: Train=%d Val=%d Test=%d",
            sum(self.assigned_train.values()),
            sum(self.assigned_val.values()),
            sum(self.assigned_test.values())
        )
    
    def _create_hybrid_splits(self) -> None:
        """Create hybrid (GLU fusion) splits directly from the output CSV.
        
        The output CSV must already be in hybrid format (79 columns:
        input, label, h_* heuristics, hF_* flag booleans).
        Stratifies by (label, h_primary_category, h_tld_risk_high) groups.
        """
        # Count groups from the hybrid output CSV
        group_counts: Dict[tuple, int] = {}
        with pd.read_csv(
            self.config.output_csv, dtype=str,
            chunksize=self.config.chunk_size, keep_default_na=False,
            na_filter=False, on_bad_lines="skip"
        ) as reader:
            for chunk in reader:
                chunk["label"] = pd.to_numeric(chunk["label"], errors="coerce").astype("Int64")
                chunk = chunk.loc[~chunk["label"].isna()]
                for r in chunk.itertuples(index=False):
                    lab = int(getattr(r, "label"))
                    pc = str(getattr(r, "h_primary_category", "") or "") or "UNKNOWN"
                    # Use h_tld_risk_high as a proxy for risk stratification
                    tr_high = int(getattr(r, "h_tld_risk_high", 0))
                    tr_crit = int(getattr(r, "h_tld_risk_critical", 0))
                    tr = "CRITICAL" if tr_crit else ("HIGH" if tr_high else "NORMAL")
                    group = (lab, pc, tr)
                    group_counts[group] = group_counts.get(group, 0) + 1
        
        if not group_counts:
            logging.warning("No valid rows found in hybrid output for splitting.")
            return
        
        train_path = self.config.output_csv.with_name("urls_hybrid_train.csv")
        val_path = self.config.output_csv.with_name("urls_hybrid_val.csv")
        test_path = self.config.output_csv.with_name("urls_hybrid_test.csv")
        
        for p in [train_path, val_path, test_path]: self._safe_unlink(p)
        
        # Centralized Stratification: Calculate targets using Balanced Rounding
        target_train, target_val, target_test = {}, {}, {}
        for group, count in group_counts.items():
            tr, vl, ts = self._calculate_group_split_targets(count)
            target_train[group] = tr
            target_val[group] = vl
            target_test[group] = ts
        
        self.assigned_train = Counter(); self.assigned_val = Counter(); self.assigned_test = Counter()
        
        # Read and split the hybrid output CSV directly
        with pd.read_csv(
            self.config.output_csv, dtype=str,
            chunksize=self.config.chunk_size, keep_default_na=False,
            na_filter=False, on_bad_lines="skip"
        ) as reader:
            for chunk in tqdm(reader, desc="Splitting Hybrid"):
                chunk["label"] = pd.to_numeric(chunk["label"], errors="coerce").astype("Int64")
                chunk = chunk.loc[~chunk["label"].isna()]
                shuffled = chunk.sample(frac=1.0, random_state=self.config.random_seed)
                train_rows, val_rows, test_rows = [], [], []
                for _, row in shuffled.iterrows():
                    lab = int(row["label"])
                    pc = str(row.get("h_primary_category", "") or "") or "UNKNOWN"
                    tr_high = int(row.get("h_tld_risk_high", 0))
                    tr_crit = int(row.get("h_tld_risk_critical", 0))
                    tr = "CRITICAL" if tr_crit else ("HIGH" if tr_high else "NORMAL")
                    group = (lab, pc, tr)
                    row_dict = row.to_dict()
                    if self.assigned_train[group] < target_train.get(group, 0):
                        train_rows.append(row_dict); self.assigned_train[group] += 1
                    elif self.assigned_val[group] < target_val.get(group, 0):
                        val_rows.append(row_dict); self.assigned_val[group] += 1
                    else:
                        test_rows.append(row_dict); self.assigned_test[group] += 1
                self._append_df(pd.DataFrame(train_rows), train_path)
                self._append_df(pd.DataFrame(val_rows), val_path)
                self._append_df(pd.DataFrame(test_rows), test_path)
        logging.info(
            "Hybrid splits created: Train=%d Val=%d Test=%d",
            sum(self.assigned_train.values()),
            sum(self.assigned_val.values()),
            sum(self.assigned_test.values())
        )
    
    def deduplicate_post_split(self) -> None:
        """Apply deduplication to all splits."""
        sources = self.config.split_source
        modes = []
        
        if "all" in sources:
            modes = ["preprocessed", "raw_orig", "OFP_Minimal", "canonical", "hybrid"]
        else:
            modes = [s for s in sources if s in 
                    ["preprocessed", "raw_orig", "OFP_Minimal", "canonical", "hybrid"]]
        
        for mode in modes:
            if mode == "raw_orig":
                splits = {
                    'train': self.config.output_csv.with_name("raw_orig_train.csv"),
                    'val': self.config.output_csv.with_name("raw_orig_val.csv"),
                    'test': self.config.output_csv.with_name("raw_orig_test.csv"),
                }
            elif mode == "OFP_Minimal":
                splits = {
                    'train': self.config.output_csv.with_name("urls_ofp_train.csv"),
                    'val': self.config.output_csv.with_name("urls_ofp_val.csv"),
                    'test': self.config.output_csv.with_name("urls_ofp_test.csv"),
                }
            elif mode == "canonical":
                splits = {
                    'train': self.config.output_csv.with_name("urls_canonical_train.csv"),
                    'val': self.config.output_csv.with_name("urls_canonical_val.csv"),
                    'test': self.config.output_csv.with_name("urls_canonical_test.csv"),
                }
            elif mode == "hybrid":
                splits = {
                    'train': self.config.output_csv.with_name("urls_hybrid_train.csv"),
                    'val': self.config.output_csv.with_name("urls_hybrid_val.csv"),
                    'test': self.config.output_csv.with_name("urls_hybrid_test.csv"),
                }
            else:
                splits = {
                    'train': self.config.output_csv.with_name("urls_preprocessed_train.csv"),
                    'val': self.config.output_csv.with_name("urls_preprocessed_val.csv"),
                    'test': self.config.output_csv.with_name("urls_preprocessed_test.csv"),
                }
            
            for split_name, split_path in splits.items():
                if split_path.exists():
                    removed = self.dedup_manager.deduplicate_split(split_path, split_name)
                    self.dedup_manager.dedup_stats[f'{split_name}_duplicates'] = removed
    @staticmethod
    def _safe_unlink(path: Path) -> None:
        """Safely unlink a file with retries for Windows."""
        if not path.exists():
            return
        for _ in range(3):
            try:
                path.unlink()
                return
            except PermissionError:
                import time
                time.sleep(0.5)
        logging.warning("Failed to unlink %s due to lock", path)
    
    def check_splits_exist(self) -> bool:
        """Check if split files exist."""
        sources = self.config.split_source
        modes = ["preprocessed", "raw_orig", "OFP_Minimal", "canonical", "hybrid"] if "all" in sources else sources
        
        for mode in modes:
            if mode == "preprocessed":
                files = [
                    self.config.output_csv.with_name("urls_preprocessed_train.csv"),
                    self.config.output_csv.with_name("urls_preprocessed_val.csv"),
                    self.config.output_csv.with_name("urls_preprocessed_test.csv"),
                ]
            elif mode == "raw_orig":
                files = [
                    self.config.output_csv.with_name("raw_orig_train.csv"),
                    self.config.output_csv.with_name("raw_orig_val.csv"),
                    self.config.output_csv.with_name("raw_orig_test.csv"),
                ]
            elif mode == "OFP_Minimal":
                files = [
                    self.config.output_csv.with_name("urls_ofp_train.csv"),
                    self.config.output_csv.with_name("urls_ofp_val.csv"),
                    self.config.output_csv.with_name("urls_ofp_test.csv"),
                ]
            elif mode == "canonical":
                files = [
                    self.config.output_csv.with_name("urls_canonical_train.csv"),
                    self.config.output_csv.with_name("urls_canonical_val.csv"),
                    self.config.output_csv.with_name("urls_canonical_test.csv"),
                ]
            elif mode == "hybrid":
                files = [
                    self.config.output_csv.with_name("urls_hybrid_train.csv"),
                    self.config.output_csv.with_name("urls_hybrid_val.csv"),
                    self.config.output_csv.with_name("urls_hybrid_test.csv"),
                ]
            else:
                continue
            
            if not any(p.exists() for p in files):
                return False
        return True
    
    def _audit_physical_files(self, mode: str) -> None:
        """Audit physical CSV files to get exact group counts."""
        if mode == "raw_orig":
            paths = {
                "train": self.config.output_csv.with_name("raw_orig_train.csv"),
                "val": self.config.output_csv.with_name("raw_orig_val.csv"),
                "test": self.config.output_csv.with_name("raw_orig_test.csv")
            }
        elif mode == "OFP_Minimal":
            paths = {
                "train": self.config.output_csv.with_name("urls_ofp_train.csv"),
                "val": self.config.output_csv.with_name("urls_ofp_val.csv"),
                "test": self.config.output_csv.with_name("urls_ofp_test.csv")
            }
        elif mode == "canonical":
            paths = {
                "train": self.config.output_csv.with_name("urls_canonical_train.csv"),
                "val": self.config.output_csv.with_name("urls_canonical_val.csv"),
                "test": self.config.output_csv.with_name("urls_canonical_test.csv")
            }
        elif mode == "hybrid":
            paths = {
                "train": self.config.output_csv.with_name("urls_hybrid_train.csv"),
                "val": self.config.output_csv.with_name("urls_hybrid_val.csv"),
                "test": self.config.output_csv.with_name("urls_hybrid_test.csv")
            }
        else:
            paths = {
                "train": self.config.output_csv.with_name("urls_preprocessed_train.csv"),
                "val": self.config.output_csv.with_name("urls_preprocessed_val.csv"),
                "test": self.config.output_csv.with_name("urls_preprocessed_test.csv")
            }
            
        self.assigned_train = Counter(self._count_groups(paths["train"]))
        self.assigned_val = Counter(self._count_groups(paths["val"]))
        self.assigned_test = Counter(self._count_groups(paths["test"]))

    def generate_report(self, mode: str) -> None:
        """Generate split distribution report using physical file audit."""
        self._audit_physical_files(mode)
        if mode == "raw_orig":
            report_name = "report_splits_raw_orig.txt"
        elif mode == "OFP_Minimal":
            report_name = "report_splits_ofp.txt"
        elif mode == "canonical":
            report_name = "report_splits_canonical.txt"
        elif mode == "hybrid":
            report_name = "report_splits_hybrid.txt"
        else:
            report_name = "report_splits_preprocessed.txt"
        
        report_path = self.config.output_csv.parent / report_name
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        total = sum(self.assigned_train.values()) + sum(self.assigned_val.values()) + sum(self.assigned_test.values())
        tr_total = sum(self.assigned_train.values())
        vl_total = sum(self.assigned_val.values())
        ts_total = sum(self.assigned_test.values())
        
        tr_pct = tr_total / total if total > 0 else 0
        vl_pct = vl_total / total if total > 0 else 0
        ts_pct = ts_total / total if total > 0 else 0
        
        lines = [
            "=" * 80,
            f"URL Split Report ({mode} mode)",
            "=" * 80,
            f"Generated: {datetime.now(timezone.utc).isoformat()} UTC",
            f"Target Ratios: Train: {self.config.train_frac:.1%}, Val: {self.config.val_frac:.1%}, "
            f"Test: {1 - self.config.train_frac - self.config.val_frac:.1%}",
            "",
            "1. OVERALL SPLIT STATISTICS",
            "--------------------------------------------------------------------------------",
            f"Total Samples: {total}",
            f"Train Set:     {tr_total:>8} ({tr_pct:.2%})",
            f"Val Set:       {vl_total:>8} ({vl_pct:.2%})",
            f"Test Set:      {ts_total:>8} ({ts_pct:.2%})",
            "",
            "2. LABEL (CLASS) DISTRIBUTION",
            "--------------------------------------------------------------------------------",
            f"{'Label':<10} | {'Train':>10} | {'Val':>10} | {'Test':>10} | {'Total':>10}",
            "-" * 80,
        ]
        
        # Aggregate assigned_* by label
        lab_train = Counter()
        lab_val = Counter()
        lab_test = Counter()
        
        for (lab, pc, tr), count in self.assigned_train.items(): lab_train[lab] += count
        for (lab, pc, tr), count in self.assigned_val.items(): lab_val[lab] += count
        for (lab, pc, tr), count in self.assigned_test.items(): lab_test[lab] += count
        
        all_labels = sorted(set(lab_train.keys()) | set(lab_val.keys()) | set(lab_test.keys()))
        for lab in all_labels:
            tr_l = lab_train[lab]
            vl_l = lab_val[lab]
            ts_l = lab_test[lab]
            total_l = tr_l + vl_l + ts_l
            lines.append(f"{str(lab):<10} | {tr_l:>10} | {vl_l:>10} | {ts_l:>10} | {total_l:>10}")
            
        lines.extend([
            "3. UNIVERSAL CATEGORY COVERAGE DASHBOARD",
            "--------------------------------------------------------------------------------",
            f"{'Category':<32} | {'Train':>8} | {'Val':>8} | {'Test':>8} | {'Status'}",
            "-" * 80,
        ])
        
        # Aggregate assigned_* by primary_category
        cat_train = Counter()
        cat_val = Counter()
        cat_test = Counter()
        
        for (lab, pc, tr), count in self.assigned_train.items(): cat_train[pc] += count
        for (lab, pc, tr), count in self.assigned_val.items(): cat_val[pc] += count
        for (lab, pc, tr), count in self.assigned_test.items(): cat_test[pc] += count
        
        all_categories = sorted(URLCategory.categories())
        for cat in all_categories:
            tr_c = cat_train[cat]
            vl_c = cat_val[cat]
            ts_c = cat_test[cat]
            
            status = "✅ OK"
            if tr_c == 0 or vl_c == 0 or ts_c == 0:
                missing = []
                if tr_c == 0: missing.append("Train")
                if vl_c == 0: missing.append("Val")
                if ts_c == 0: missing.append("Test")
                status = f"⚠️ MISSING: {','.join(missing)}"
            
            lines.append(f"{cat:<32} | {tr_c:>8} | {vl_c:>8} | {ts_c:>8} | {status}")
        
        lines.extend([
            "",
            "4. TLD RISK DISTRIBUTION PER SPLIT",
            "--------------------------------------------------------------------------------",
            f"{'Risk Level':<12} | {'Train':>10} | {'Val':>10} | {'Test':>10} | {'Total':>10}",
            "-" * 80,
        ])
        
        # Aggregate assigned_* by TLD risk
        risk_train = Counter()
        risk_val = Counter()
        risk_test = Counter()
        
        for (lab, pc, tr), count in self.assigned_train.items(): risk_train[tr] += count
        for (lab, pc, tr), count in self.assigned_val.items(): risk_val[tr] += count
        for (lab, pc, tr), count in self.assigned_test.items(): risk_test[tr] += count
        
        all_risks = ["NORMAL", "MEDIUM", "HIGH", "CRITICAL", "UNKNOWN"]
        for risk in all_risks:
            tr_r = risk_train[risk]
            vl_r = risk_val[risk]
            ts_r = risk_test[risk]
            total_r = tr_r + vl_r + ts_r
            if total_r > 0:
                lines.append(f"{risk:<12} | {tr_r:>10} | {vl_r:>10} | {ts_r:>10} | {total_r:>10}")
            
        lines.extend([
            "",
            "5. GROUP DISTRIBUTION PER SPLIT (LABEL, CATEGORY, TLD_RISK)",
            "--------------------------------------------------------------------------------",
        ])
        
        # Consolidated group view
        all_groups = sorted(set(self.assigned_train.keys()) | 
                          set(self.assigned_val.keys()) | 
                          set(self.assigned_test.keys()))
        
        for g in all_groups:
            tr_g = self.assigned_train[g]
            vl_g = self.assigned_val[g]
            ts_g = self.assigned_test[g]
            total_g = tr_g + vl_g + ts_g
            lines.append(f"  {g}: train={tr_g} val={vl_g} test={ts_g} total={total_g}")

        with report_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        
        logging.info("Report generated: %s", report_path)


# ============================================================================
# PIPELINE ORCHESTRATOR CLASS  
# ============================================================================
class PipelineOrchestrator:
    """
    Main workflow coordination and execution.
    
    Orchestrates the complete preprocessing pipeline, coordinating
    all component classes and managing chunk-based streaming.
    
    Attributes:
        config: PreprocessConfig instance
        logger: AuditLogger instance
        url_parser: URLParser instance
        feature_extractor: FeatureExtractor instance
        dedup_manager: DeduplicationManager instance
        splitter: URLSplitterAndReporter instance
        analyzer: URLAnalyzer instance
        longest_rows: List for tracking longest input texts
    """
    
    def __init__(self, config: PreprocessConfig) -> None:
        """
        Initialize pipeline orchestrator.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config
        self.logger = AuditLogger(config.log_path)
        
        # Initialize URL analyzer
        # Route categorization output to preprocessor output folder to avoid stray folders in SRC
        category_config = CategoryConfig(OUTPUT_DIR=config.output_csv.parent)
        self.analyzer = URLAnalyzer(category_config)
        
        # Load TLD risk from Excel if provided
        tld_risk_from_excel = self._load_tld_risk_from_excel()
        
        # Initialize components
        self.url_parser = URLParser(config, category_config.tld_extract)
        self.feature_extractor = FeatureExtractor(
            config, self.analyzer, tld_risk_from_excel
        )
        self.dedup_manager = DeduplicationManager(config, category_config)
        self.splitter = URLSplitterAndReporter(config, self.dedup_manager)
        
        # Initialize redirect resolver (optional, disabled by default)
        self.redirect_resolver = URLRedirectResolver(config, category_config.tld_extract)
        
        self.longest_rows: List[Dict] = []
        self._entropy_calibrated = False

    def _build_row(self, result: ProcessedURL) -> Dict[str, Any]:
        """
        Build a row dictionary for output CSV.
        
        Ensures that if splitting is enabled, a superset of columns is emitted
        so that all split modes (OFP, Canonical, Hybrid, etc.) can be generated.
        """
        is_split_enabled = self.config.enable_split
        out_format = self.config.output_format
        
        # Short flag set for efficient membership checks
        flag_set = set(result.flags_active)
        
        # Base metadata (The "Superset" for Splitting)
        row: Dict[str, Any] = {
            "input": result.canonical_url, # Default
            "label": result.label,
            "raw_url": result.raw_url,
            "model_url": result.model_url,
            "canonical_url": result.canonical_url,
            "primary_category": result.primary_category,
            "tld_risk": result.tld_risk,
            "flags_count": result.flags_count,
            "severity_score": round(result.severity_score, 4),
            "flags_bitmask": result.flags_bitmask,
        }
        
        # Override 'input' if user requested model_url specifically for preprocessed mode
        if self.config.model_input_format == "model":
            row["input"] = result.model_url

        # Add Hybrid heuristics (Required if mode is hybrid OR splitting is on)
        if out_format == "hybrid" or is_split_enabled:
            row.update({
                "h_flags_count": result.flags_count,
                "h_severity_score": round(result.severity_score, 4),
                "h_flags_bitmask": result.flags_bitmask,
                "h_entropy_url": round(result.entropy_url, 4),
                "h_entropy_path": round(result.entropy_path, 4),
                "h_entropy_query": round(result.entropy_query, 4),
                "h_digit_ratio": round(result.digit_ratio, 4),
                "h_path_depth": result.path_depth,
                "h_url_length": result.url_length,
                "h_query_param_count": result.query_param_count,
                "h_is_ip_host": int(result.is_ip_host),
                "h_has_https": int(result.has_https),
                "h_has_fragment": int(result.has_fragment),
                "h_tld_risk_normal": int(result.tld_risk == "NORMAL"),
                "h_tld_risk_high": int(result.tld_risk == "HIGH"),
                "h_tld_risk_critical": int(result.tld_risk == "CRITICAL"),
                "h_primary_category": result.primary_category,
                # --- NEW V9 heuristic features ---
                # Domain features
                "h_domain_length": result.domain_length,
                "h_subdomain_count": result.subdomain_count,
                # Punycode / Unicode features
                "h_has_punycode": int(result.has_punycode),
                "h_punycode_char_count": result.punycode_char_count,
                "h_has_unicode": int(result.has_unicode),
                "h_unicode_char_ratio": round(result.unicode_char_ratio, 4),
                "h_mixed_script": int(result.mixed_script_detected),
                # Tracking features
                "h_has_tracking_params": int(result.has_tracking_params),
                "h_tracking_param_count": result.tracking_param_count,
                # Path structure features
                "h_has_double_extension": int(result.has_double_extension),
                "h_path_token_count": result.path_token_count,
                # Redirect / obfuscation features
                "h_has_redirect_param": int(result.has_redirect_param),
                "h_redirect_count": result.redirect_count_in_url,
                "h_has_at_sign": int(result.has_at_sign),
            })
            # Add per-flag booleans (hF_ prefix)
            for flag in Constants.FLAG_ORDER:
                col = f"hF_{Constants.FLAG_CODE_MAP.get(flag, flag[:6])}"
                row[col] = int(flag in flag_set)
        
        # Add Full format specifics
        if out_format == "full":
            row["input_text"] = result.input_text
            row["tld"] = result.tld
            
        # If user ONLY wants clean output and NO splitting, strip extra columns
        if out_format == "clean" and not is_split_enabled:
            return {"input": row["input"], "label": row["label"]}
            
        return row
    
    def _load_tld_risk_from_excel(self) -> Dict[str, str]:
        """Load TLD risk mappings from Excel file."""
        if not self.config.tld_stats_path:
            return {}
        
        try:
            df = pd.read_excel(
                self.config.tld_stats_path,
                sheet_name=self.config.tld_stats_sheet
            )
            risk_col = None
            tld_col = None
            for col in df.columns:
                if "risk" in col.lower():
                    risk_col = col
                if "tld" in col.lower():
                    tld_col = col
            
            if risk_col and tld_col:
                return {
                    str(row[tld_col]).lower(): str(row[risk_col])
                    for _, row in df.iterrows()
                }
        except Exception as exc:
            logging.warning("Failed to load TLD stats: %s", exc)
        return {}
    
    def _setup_outputs(self) -> None:
        """Initialize output files."""
        output_dir = self.config.output_csv.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.config.overwrite_outputs and not self.config.resume:
            for p in [
                self.config.output_csv,
                self.config.rejected_csv,
                self.config.local_private_csv,
            ]:
                if p.exists():
                    p.unlink()
    
    def _append_df(self, df: pd.DataFrame, path: Path) -> None:
        """Append DataFrame to CSV."""
        if df.empty:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        
        file_exists = path.exists()
        df_clean = df.copy()
        for col in df_clean.select_dtypes(include=['object']).columns:
            df_clean[col] = df_clean[col].apply(HelperUtilities.sanitize_csv_field)
        
        try:
            df_clean.to_csv(
                path, mode="a" if file_exists else "w",
                index=False, header=not file_exists,
                encoding="utf-8", quoting=csv.QUOTE_ALL,
                escapechar='\\'
            )
        except OSError as exc:
            logging.error("DISK FULL or I/O error writing %s: %s", path, exc)
            raise  # Re-raise to stop pipeline cleanly
        except Exception as exc:
            logging.warning("CSV write failed for %s: %s", path, exc)
    
    def _process_url(self, raw_url: str, label: int) -> Optional[Tuple[ProcessedURL, str]]:
        """
        Process a single URL through the pipeline.
        
        Returns:
            Tuple of (ProcessedURL, None) if success,
            Tuple of (None, drop_reason) if dropped.
        """
        # Clean input
        cleaned = HelperUtilities.clean_text(raw_url)
        if not cleaned:
            return None, "empty_url"
        
        # Parse URL
        parsed = self.url_parser.parse_url(cleaned)
        if not parsed or not parsed.hostname:
            return None, "unparseable"
        
        scheme = (parsed.scheme or "http").lower()
        host = parsed.hostname or ""
        # Port access can raise ValueError for malformed URLs (e.g., port='\')
        try:
            port = parsed.port
        except ValueError:
            return None, "malformed_port"
        path = parsed.path or "/"
        raw_query = parsed.query or ""
        fragment = parsed.fragment or ""
        
        # Classify scheme
        scheme_family, should_keep = self.url_parser.classify_scheme(scheme, path)
        if not should_keep:
            return None, "scheme_rejected"
        
        # Check IP host
        is_ip, is_private, canon_ip = self.url_parser.is_ip_host(host)
        if is_private and self.config.drop_local_private:
            return None, "local_private_ip"
        
        # IP-to-Domain Resolution (Reverse DNS)
        # We use the UNMASKED canon_ip (127.0.0.1) instead of raw host (0x7f.1)
        # to ensure standard DNS resolvers can process the request.
        if is_ip and canon_ip and self.config.enable_ip_domain_resolution:
            resolved_domain = self.url_parser.resolve_ip_to_domain(canon_ip)
            if resolved_domain:
                # Replace host in working URL and re-parse
                new_cleaned = self.url_parser.replace_host_in_url(cleaned, resolved_domain)
                new_parsed = self.url_parser.parse_url(new_cleaned)
                
                if new_parsed and new_parsed.hostname:
                    # Log successful resolution
                    logging.info("IP Resolved: %s -> %s", cleaned, new_cleaned)
                    
                    # Update state
                    cleaned = new_cleaned
                    parsed = new_parsed
                    scheme = (parsed.scheme or "http").lower()
                    host = parsed.hostname or ""
                    try:
                        port = parsed.port
                    except ValueError:
                        return None, "malformed_port"
                    path = parsed.path or "/"
                    raw_query = parsed.query or ""
                    fragment = parsed.fragment or ""
                    
                    # Re-verify IP status (should be DNS now)
                    is_ip, is_private, canon_ip = self.url_parser.is_ip_host(host)
        
        # Normalize
        normalized_host = self.url_parser.normalize_host(host)
        normalized_path = self.url_parser.normalize_path(path)
        
        # Mask blobs in path
        masked_path, path_has_hex, path_has_b64 = self.url_parser.mask_blobs(normalized_path)
        
        # Query info
        query_info = self.url_parser.build_query_info(raw_query)
        
        # Fragment info
        fragment_info = self.url_parser.build_fragment_info(fragment)
        
        # Build canonical URL (heavily normalized — for dedup + Preprocessed mode)
        canonical_url = self.url_parser.build_canonical_url(
            scheme, normalized_host, port, masked_path, query_info.canonical_query
        )
        
        # Build model_url (minimally cleaned — for OFP mode / model training)
        # Only: scheme lowercase + host NFKC+Punycode + invisible strip on path/query
        # Preserves: encoding layers, ../ traversal, blobs, all query params, ports
        model_url = self.url_parser.build_model_url(
            scheme, normalized_host, port, path, raw_query, fragment
        )
        
        # Deduplication key (still uses canonical_url components for accuracy)
        dedup_key = self.url_parser.dedup_key(
            scheme, normalized_host, masked_path, query_info.canonical_query
        )
        
        if self.config.deduplicate_before_split:
            if self.dedup_manager.is_duplicate(dedup_key):
                return None, "duplicate"
            self.dedup_manager.add_key(dedup_key)
        
        # Extract TLD
        tld_info = self.url_parser.extract_tld(cleaned)
        tld = tld_info.suffix if tld_info else ""
        registrable_domain = tld_info.top_domain_under_public_suffix if tld_info else ""
        
        # TLD risk
        tld_risk = self.feature_extractor.tld_risk(tld)
        
        # Run URL analyzer
        analysis = self.analyzer.analyze_url(cleaned)
        flags_active = [cat for cat, triggered in analysis.items() if triggered]
        
        # Compute features
        flags_count, severity_score, flags_bitmask = self.feature_extractor.rule_feature_values(flags_active)
        primary_category = self.feature_extractor.primary_category(flags_active)
        
        # Entropy metrics
        entropy_url = HelperUtilities.shannon_entropy(cleaned)
        entropy_path = HelperUtilities.shannon_entropy(normalized_path)
        entropy_query = query_info.avg_param_entropy
        digit_ratio = HelperUtilities.digit_ratio(cleaned)
        path_depth = len([p for p in normalized_path.split("/") if p])
        
        # Redirect resolution (optional - runs only if enabled)
        redirect_info = self.redirect_resolver.resolve(cleaned)
        
        # If redirect resolved to a different URL, we still use original for analysis
        # but the redirect_info features will be included in the model input
        
        # Build input text (for Preprocessed mode — uses canonical_url)
        input_text = self.feature_extractor.build_input_text(
            tld=tld,
            tld_risk=tld_risk,
            scheme=scheme,
            scheme_family=scheme_family,
            host=normalized_host,
            registrable_domain=registrable_domain,
            path=masked_path,
            has_base64_blob=path_has_b64,
            has_hex_blob=path_has_hex,
            is_ip_host=is_ip,
            is_private_ip=is_private,
            query_info=query_info,
            fragment_info=fragment_info,
            flags=flags_active,
            entropy_url=entropy_url,
            entropy_path=entropy_path,
            entropy_query=entropy_query,
            digit_ratio=digit_ratio,
            path_depth=path_depth,
            canonical_url=canonical_url,
            canonical_ip=canon_ip,
            redirect_info=redirect_info,
        )
        
        # --- NEW V9 features ---
        # Domain features
        domain_length = len(normalized_host)
        subdomain_parts = normalized_host.split('.')
        # subdomain_count = total labels minus registrable domain (e.g. a.b.example.com = 2 subdomains)
        subdomain_count = max(0, len(subdomain_parts) - 2) if len(subdomain_parts) > 2 else 0
        
        # Punycode / Unicode features (use extended normalizer for full metadata)
        _, host_norm_result = self.url_parser.normalize_host_extended(host)
        has_punycode = any(lbl.startswith('xn--') for lbl in (host_norm_result.punycode or '').split('.'))
        punycode_char_count = sum(1 for ch in host if ord(ch) > 127)
        has_unicode = host_norm_result.had_unicode
        total_host_chars = max(len(host), 1)
        unicode_char_ratio = punycode_char_count / total_host_chars
        mixed_script_detected = host_norm_result.mixed_scripts
        
        # Tracking features (already computed in query_info)
        has_tracking_params = query_info.has_tracking_params
        tracking_param_count = query_info.tracking_param_count
        
        # Path structure features
        path_tokens = [p for p in normalized_path.split('/') if p]
        path_token_count = len(path_tokens)
        # Double extension detection (e.g., document.pdf.exe)
        last_segment = path_tokens[-1] if path_tokens else ''
        dot_parts = last_segment.split('.')
        has_double_extension = len(dot_parts) >= 3 and all(len(p) > 0 for p in dot_parts[-2:])
        
        # Redirect / obfuscation features
        redirect_keywords = {'redirect', 'redir', 'url=', 'next=', 'dest=', 'goto=', 'return=', 'returnurl', 'continue='}
        url_lower = cleaned.lower()
        has_redirect_param = any(kw in url_lower for kw in redirect_keywords)
        redirect_count_in_url = sum(1 for kw in redirect_keywords if kw in url_lower)
        has_at_sign = '@' in host
        
        return ProcessedURL(
            input_text=input_text,
            label=label,
            primary_category=primary_category,
            tld=tld,
            tld_risk=tld_risk,
            canonical_url=canonical_url,
            model_url=model_url,
            raw_url=raw_url,
            flags_count=flags_count,
            severity_score=severity_score,
            flags_bitmask=flags_bitmask,
            flags_active=flags_active,
            entropy_url=entropy_url,
            entropy_path=entropy_path,
            entropy_query=entropy_query,
            digit_ratio=digit_ratio,
            path_depth=path_depth,
            url_length=len(canonical_url),
            is_ip_host=is_ip,
            has_https=(scheme == "https"),
            query_param_count=query_info.total_params,
            has_fragment=bool(fragment),
            # New V9 features
            domain_length=domain_length,
            subdomain_count=subdomain_count,
            has_punycode=has_punycode,
            punycode_char_count=punycode_char_count,
            has_unicode=has_unicode,
            unicode_char_ratio=round(unicode_char_ratio, 4),
            mixed_script_detected=mixed_script_detected,
            has_tracking_params=has_tracking_params,
            tracking_param_count=tracking_param_count,
            has_double_extension=has_double_extension,
            path_token_count=path_token_count,
            has_redirect_param=has_redirect_param,
            redirect_count_in_url=redirect_count_in_url,
            has_at_sign=has_at_sign,
        ), None
    
    def _update_longest_rows(self, df: pd.DataFrame) -> None:
        """Track longest input_text rows for debugging."""
        if "input_text" not in df.columns:
            return
        
        for _, row in df.iterrows():
            text_len = len(str(row.get("input_text", "")))
            if len(self.longest_rows) < 100:
                self.longest_rows.append({"length": text_len, **row.to_dict()})
                self.longest_rows.sort(key=lambda x: x["length"], reverse=True)
            elif text_len > self.longest_rows[-1]["length"]:
                self.longest_rows[-1] = {"length": text_len, **row.to_dict()}
                self.longest_rows.sort(key=lambda x: x["length"], reverse=True)
    
    def run(self) -> None:
        """Execute the complete preprocessing pipeline."""
        logging.info("=" * 80)
        logging.info("STARTING URL PREPROCESSING PIPELINE (v5 Refactored — model_url for OFP)")
        logging.info("=" * 80)
        
        self._setup_outputs()
        
        # Resume support: count already-processed rows to skip chunks
        resume_skip_rows = 0
        if self.config.resume:
            self.dedup_manager.load_seen_from_outputs(self.config.output_csv)
            # Count rows already in output to know how many input rows to skip
            if self.config.output_csv.exists():
                try:
                    resume_skip_rows = sum(
                        len(chunk) for chunk in pd.read_csv(
                            self.config.output_csv, usecols=[0],
                            chunksize=100_000, on_bad_lines="skip"
                        )
                    )
                    logging.info(
                        "Resume: found %d rows in output. Will skip equivalent input chunks.",
                        resume_skip_rows
                    )
                except Exception as exc:
                    logging.warning("Resume: failed to count output rows: %s", exc)
        
        # Process chunks
        try:
            reader = pd.read_csv(
                self.config.input_csv,
                chunksize=self.config.chunk_size,
                dtype=str,
                encoding="utf-8",
                on_bad_lines="skip",
                keep_default_na=False,
                na_filter=False,
            )
        except Exception as exc:
            logging.error("Failed to read input: %s", exc)
            return
        
        rows_skipped = 0
        # tqdm can raise OSError([Errno 22] Invalid argument) on Windows when the
        # output stream isn't a real TTY (e.g., IDE consoles, redirected output).
        # We already log chunk-level progress, so it's safe to disable tqdm in
        # non-interactive contexts.
        try:
            _tqdm_disable = not (getattr(sys.stderr, "isatty", lambda: False)())
        except Exception:
            _tqdm_disable = True

        for idx, chunk in enumerate(tqdm(reader, desc="Processing", disable=_tqdm_disable)):
            # Auto-map 'url' to 'input' if 'input' is missing
            if "input" not in chunk.columns and "url" in chunk.columns:
                chunk["input"] = chunk["url"]
            
            # Resume: skip chunks that were already processed in a previous run
            if resume_skip_rows > 0 and rows_skipped < resume_skip_rows:
                rows_skipped += len(chunk)
                if rows_skipped <= resume_skip_rows:
                    continue
                # Partial chunk: only process the unprocessed tail
                overlap = rows_skipped - resume_skip_rows
                chunk = chunk.tail(overlap).copy()
                logging.info("Resume: partially processing chunk %d (%d new rows)", idx, overlap)
            
            # Calibrate entropy on first actually-processed chunk
            if not self._entropy_calibrated and not self.config.skip_entropy_calibration:
                sample = chunk["input"].sample(
                    min(self.config.calibration_sample_size, len(chunk)),
                    random_state=self.config.random_seed
                ).tolist()
                self.feature_extractor.calibrate_entropy_thresholds(sample)
                self._entropy_calibrated = True
            
            processed_rows = []
            rejected_rows = []
            local_private_rows = []
            
            # Drop rows with missing/invalid labels and convert to int
            chunk_label_raw = chunk["label"].copy()
            # Convert string labels to numeric (handles "nan", "", float strings)
            chunk["_label_numeric"] = pd.to_numeric(chunk_label_raw, errors="coerce")
            # Drop rows where label is NaN (missing or unparseable)
            valid_mask = chunk["_label_numeric"].notna()
            dropped_label_count = (~valid_mask).sum()
            if dropped_label_count > 0:
                logging.info(
                    "Chunk %d: dropped %d rows with missing/invalid labels",
                    idx, dropped_label_count
                )
            chunk = chunk.loc[valid_mask].copy()
            chunk["_label_int"] = chunk["_label_numeric"].astype(int)
            if chunk.empty:
                continue
            
            if self.config.enable_multiprocessing:
                # Parallel processing
                row_data_items = [(str(row.get("input", "")), int(row["_label_int"])) for _, row in chunk.iterrows()]
                worker_args = [(self.config, item) for item in row_data_items]
                
                with ProcessPoolExecutor(max_workers=self.config.num_workers) as executor:
                    results = list(executor.map(_multiprocess_worker, worker_args))
                
                for i, (result, drop_reason) in enumerate(results):
                    self.logger.increment_processed()
                    row_dict = chunk.iloc[i].to_dict()
                    if result:
                        processed_rows.append(self._build_row(result))
                        self.logger.increment_kept()
                    elif drop_reason == "local_private_ip":
                        local_private_rows.append(row_dict)
                        self.logger.log_drop(drop_reason)
                    elif drop_reason:
                        rejected_rows.append({**row_dict, "drop_reason": drop_reason})
                        self.logger.log_drop(drop_reason)
            else:
                # Serial processing
                for _, row in chunk.iterrows():
                    raw_url = str(row.get("input", ""))
                    label = int(row["_label_int"])
                    self.logger.increment_processed()
                    result, drop_reason = self._process_url(raw_url, label)
                    
                    if result:
                        processed_rows.append(self._build_row(result))
                        self.logger.increment_kept()
                    elif drop_reason == "local_private_ip":
                        local_private_rows.append(row.to_dict())
                        self.logger.log_drop(drop_reason)
                    elif drop_reason:
                        rejected_rows.append({**row.to_dict(), "drop_reason": drop_reason})
                        self.logger.log_drop(drop_reason)
            
            # Write outputs
            output_df = pd.DataFrame(processed_rows)
            self._update_longest_rows(output_df)
            self._append_df(output_df, self.config.output_csv)
            self._append_df(pd.DataFrame(rejected_rows), self.config.rejected_csv)
            self._append_df(pd.DataFrame(local_private_rows), self.config.local_private_csv)
            
            self.logger.log_chunk_stats(
                idx, len(processed_rows), len(rejected_rows), len(local_private_rows)
            )
        
        # Save dedup cache
        if self.config.deduplicate:
            self.dedup_manager.save_cache()
        
        logging.info("Processing complete. Kept %d of %d rows.",
                    self.logger.total_kept, self.logger.total_processed)
        logging.info("Drop reasons: %s", dict(self.logger.drop_counts))
        
        # Save longest debug
        if self.longest_rows and self.config.longest_debug_csv:
            try:
                pd.DataFrame(self.longest_rows).to_csv(
                    self.config.longest_debug_csv, index=False,
                    quoting=csv.QUOTE_ALL, escapechar='\\'
                )
            except OSError as exc:
                logging.error("DISK FULL writing longest debug CSV: %s", exc)
            except Exception as exc:
                logging.warning("Failed to save longest debug CSV: %s", exc)
        
        # Create splits
        if self.config.enable_split and self.logger.total_kept > 0:
            if self.config.resume and self.splitter.check_splits_exist():
                logging.info("Resume: splits exist; skipping creation.")
            else:
                self.splitter.create_splits()
            
            if self.config.deduplicate_after_split:
                self.splitter.deduplicate_post_split()
            
            # Generate reports AFTER all processing (including deduplication) is complete
            # This ensures reports reflect the actual physical file contents
            report_modes = ["preprocessed", "raw_orig", "OFP_Minimal", "canonical", "hybrid"] if "all" in self.config.split_source else self.config.split_source
            for mode in report_modes:
                self.splitter.generate_report(mode)
        
        logging.info("=" * 80)
        logging.info("PIPELINE COMPLETE")
        logging.info("=" * 80)


# ============================================================================
# CLI ARGUMENT PARSING
# ============================================================================
def parse_args() -> PreprocessConfig:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess phishing URLs for transformer models (v5 Refactored — model_url for OFP)")
    parser.add_argument("--input", type=Path, default=Path("data.csv"),help="Input CSV with columns input,label")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR / "urls_preprocessed.csv",help="Output CSV path")
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--keep-private", action="store_true")
    parser.add_argument("--no-dedup", action="store_true")
    parser.add_argument("--dedup-cache", type=Path, default=DEFAULT_OUTPUT_DIR / "dedup_cache.pkl")
    parser.add_argument("--tld-stats", type=Path)
    parser.add_argument("--tld-stats-sheet", type=str, default="Benign_&_Malecious_TLD")
    parser.add_argument("--no-overwrite", action="store_true")
    parser.add_argument("--enable-split", action="store_true")
    parser.add_argument(
        "--split-source", nargs="+",
        choices=["preprocessed", "raw_orig", "OFP_Minimal", "canonical", "hybrid", "all"],
        default=["canonical"]
    )
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_OUTPUT_DIR / "preprocess.log")
    parser.add_argument("--use-rule-features", action="store_true", default=True)
    parser.add_argument("--no-rule-features", action="store_false", dest="use_rule_features")
    parser.add_argument("--longest-debug-csv", type=Path, default=DEFAULT_OUTPUT_DIR / "input_text_longest_debug.csv")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-path", type=Path, default=Path(".preprocess_resume.json"))
    parser.add_argument("--enable-multiprocessing", action="store_true", default=True)
    parser.add_argument("--disable-multiprocessing", action="store_false", dest="enable_multiprocessing")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--skip-entropy-calibration", action="store_true", default=False)
    
    parser.add_argument(
        "--model-input", type=str, choices=["model", "canonical"], default="canonical",
        help="Determines which processed URL format becomes the main 'input' column"
    )
    parser.add_argument(
        "--output-format", type=str, choices=["full", "clean", "hybrid"], default="clean",
        help="full=all 60+ features, clean=input+label only, hybrid=GLU fusion dataset (canonical input + 68+ heuristic columns)"
    )
    
    # Redirect resolution arguments (disabled by default)
    parser.add_argument("--enable-redirect-resolution", action="store_true", default=False,
                        help="Enable following HTTP redirects to expose final URLs")
    parser.add_argument("--redirect-all", action="store_true", default=False,
                        help="Resolve all URLs, not just known shorteners")
    parser.add_argument("--redirect-max-hops", type=int, default=5,
                        help="Maximum redirect hops to follow (default: 5)")
    parser.add_argument("--redirect-timeout", type=float, default=5.0,
                        help="Timeout per redirect hop in seconds (default: 5.0)")
    parser.add_argument("--redirect-include-features", action="store_true", default=False,
                        help="Include REDIR_DEPTH and REDIR_XDOMAIN in model input")
    
    parser.add_argument("--disable-ip-resolution", action="store_true", default=False,
                        help="Disable reverse DNS resolution for IP-based URLs")
    
    args = parser.parse_args()
    
    max_workers = multiprocessing.cpu_count()
    if args.num_workers > max_workers:
        args.num_workers = max_workers
        
    output_csv = resolve_project_path(args.output)
    output_dir = output_csv.parent
    
    return PreprocessConfig(
        input_csv=args.input,
        output_csv=output_csv,
        rejected_csv=output_dir / "urls_rejected_corrupted.csv",
        local_private_csv=output_dir / "urls_local_private.csv",
        duplicate_csv=output_dir / "urls_duplicates.csv",
        chunk_size=args.chunk_size,
        drop_local_private=not args.keep_private,
        deduplicate=not args.no_dedup,
        deduplicate_before_split=False,
        deduplicate_after_split=True,
        dedup_cache=output_dir / "dedup_cache.pkl" if not args.no_dedup else None,
        overwrite_outputs=not args.no_overwrite,
        tld_stats_path=args.tld_stats,
        tld_stats_sheet=args.tld_stats_sheet,
        enable_split=args.enable_split,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        log_path=output_dir / "preprocess.log",
        use_rule_features=args.use_rule_features,
        longest_debug_csv=output_dir / "input_text_longest_debug.csv",
        split_source=args.split_source,
        resume=args.resume,
        progress_path=output_dir / ".preprocess_resume.json",
        enable_multiprocessing=args.enable_multiprocessing,
        num_workers=args.num_workers,
        skip_entropy_calibration=args.skip_entropy_calibration,
        model_input_format=args.model_input,
        output_format=args.output_format,
        # Redirect resolution settings
        enable_redirect_resolution=args.enable_redirect_resolution,
        redirect_only_shorteners=not args.redirect_all,
        redirect_max_hops=args.redirect_max_hops,
        redirect_timeout_sec=args.redirect_timeout,
        redirect_include_features=args.redirect_include_features,
        enable_ip_domain_resolution=not args.disable_ip_resolution,
    )


# ============================================================================
# MULTIPROCESSING WORKERS
# ============================================================================
_worker_orchestrator = None

def _get_worker_orchestrator(config: PreprocessConfig) -> PipelineOrchestrator:
    """Singleton-like access to orchestrator within worker processes."""
    global _worker_orchestrator
    if _worker_orchestrator is None:
        _worker_orchestrator = PipelineOrchestrator(config)
    return _worker_orchestrator

def _multiprocess_worker(args: Tuple[PreprocessConfig, Tuple[str, int]]) -> Tuple[Optional[ProcessedURL], Optional[str]]:
    """Worker function for parallel processing."""
    config, row_data = args
    try:
        orchestrator = _get_worker_orchestrator(config)
        return orchestrator._process_url(*row_data)
    except Exception:
        return None, "worker_crash"


def main() -> None:
    """Main entry point."""
    try:
        config = parse_args()
        
        # Configure logging early
        config.log_path.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(config.log_path, encoding="utf-8"),
                logging.StreamHandler()
            ],
            force=True
        )
        
        logging.info("=" * 80)
        logging.info("URL PREPROCESSING PIPELINE v8 (Elite Unmasking)")
        logging.info("Input: %s", config.input_csv)
        
        orchestrator = PipelineOrchestrator(config)
        orchestrator.run()
    except Exception as e:
        logging.error("Main execution failed: %s", e, exc_info=True)


if __name__ == "__main__":
    # Windows multiprocessing compatibility
    multiprocessing.freeze_support()
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    main()



#*****************************************************************************************************************
# python 2_preprocess_urls_v8_refactored.py \
#   --input "C:\Users\HP\Desktop\DataPrep8\1_MiniLM_V4_Model_On_Raw_Data_and_OFP_and_Canonical_Inferencing\DATA\3_LNU_Phish1.csv" \
#   --enable-split \
#   --split-source all \
#   --chunk-size 100000 \
#   --use-rule-features \
#   --enable-multiprocessing \
#   --num-workers 8



# python 2_preprocess_urls_v8_refactored.py \
#   --input "/home/hp/SHINU RATHOD/Dataset/dataset10/final_master_dataset102_36302801.csv" \
#   --enable-split \
#   --split-source raw_orig \
#   --chunk-size 100000 \
#   --use-rule-features \
#   --enable-multiprocessing \
#   --num-workers 8
