#!/usr/bin/env python3
"""
============================================================================
 SAMSUNG-GRADE HEURISTIC FEATURE DISTRIBUTION ANALYSIS  v2.0
 ──────────────────────────────────────────────────────────────
 A comprehensive, publication-quality feature distribution analysis toolkit
 for the PhishURLDetect hybrid model's 87 heuristic features.

 Methodology:
   • Kolmogorov-Smirnov (KS) two-sample test for distributional separation
   • Cohen's d effect size for standardised mean differences
   • Mann-Whitney U test for non-parametric group comparison
   • Mutual Information (MI) for non-linear dependency detection
   • Chi-square / Cramér's V for categorical association strength
   • Pearson & Point-biserial correlation
   • Composite Feature Power Score (weighted multi-metric ranking)
   • Feature Tier Classification (Critical / High / Moderate / Low / Noise)

 Output:
   • Structured markdown report  (Abstract → Appendix)
   • 10+ publication-quality visualisations
   • Unified feature ranking with actionable recommendations

 Version: 2.0  (Samsung-Grade Distribution Analysis)
============================================================================
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

# Force UTF-8 output on Windows (avoid cp1252 encoding errors)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import time
import subprocess
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# Optional: scikit-learn for Mutual Information
try:
    from sklearn.feature_selection import mutual_info_classif
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ============================================================================
# CONSTANTS & PREMIUM THEME
# ============================================================================
COLORS = {
    "benign":       "#3B82F6",
    "malicious":    "#EF4444",
    "primary":      "#1E40AF",
    "accent":       "#F59E0B",
    "bg":           "#FFFFFF",
    "panel":        "#F8FAFC",
    "grid":         "#E2E8F0",
    "text":         "#1E293B",
    "text_sec":     "#64748B",
    "tier_critical":"#DC2626",
    "tier_high":    "#EA580C",
    "tier_moderate":"#CA8A04",
    "tier_low":     "#6B7280",
    "tier_noise":   "#D1D5DB",
}

TIER_CFG = [
    ("Critical", 0.65, COLORS["tier_critical"]),
    ("High",     0.40, COLORS["tier_high"]),
    ("Moderate", 0.20, COLORS["tier_moderate"]),
    ("Low",      0.08, COLORS["tier_low"]),
    ("Noise",    0.00, COLORS["tier_noise"]),
]

FEATURE_LABELS: Dict[str, str] = {
    "h_entropy_url":          "URL Entropy",
    "h_entropy_path":         "Path Entropy",
    "h_entropy_query":        "Query Entropy",
    "h_digit_ratio":          "Digit Ratio",
    "h_path_depth":           "Path Depth",
    "h_url_length":           "URL Length",
    "h_query_param_count":    "Query Param Count",
    "h_domain_length":        "Domain Length",
    "h_subdomain_count":      "Subdomain Count",
    "h_punycode_char_count":  "Punycode Char Count",
    "h_unicode_char_ratio":   "Unicode Char Ratio",
    "h_tracking_param_count": "Tracking Param Count",
    "h_path_token_count":     "Path Token Count",
    "h_redirect_count":       "Redirect Count",
    "h_is_ip_host":           "IP-Address Host",
    "h_has_fragment":         "Has Fragment",
    "h_tld_risk_normal":      "TLD Risk Normal",
    "h_tld_risk_high":        "TLD Risk High",
    "h_tld_risk_critical":    "TLD Risk Critical",
    "h_has_punycode":         "Has Punycode",
    "h_has_unicode":          "Has Unicode",
    "h_mixed_script":         "Mixed Script",
    "h_has_tracking_params":  "Has Tracking Params",
    "h_has_double_extension":  "Double Extension",
    "h_has_redirect_param":   "Has Redirect Param",
    "h_has_at_sign":          "Has @ Sign",
    "h_flags_bitmask":        "Flags Bitmask",
}


def _label(name: str) -> str:
    """Return a human-readable label for a feature column name."""
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    if name.startswith("hF_"):
        return f"Flag:{name[3:]}"
    return name.replace("h_", "").replace("_", " ").title()


# ============================================================================
# CLI
# ============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Samsung-Grade Heuristic Feature Distribution Analysis v2.0")
    p.add_argument("--input", type=str, required=True,
                   help="Path to the input CSV (raw or preprocessed).")
    p.add_argument("--output-dir", type=str, default=".",
                   help="Directory for report and plots.")
    p.add_argument("--chunk-size", type=int, default=500_000,
                   help="Chunk size for on-the-fly preprocessing.")
    p.add_argument("--workers", type=int, default=4,
                   help="Workers for on-the-fly preprocessing.")
    p.add_argument("--sample-size", type=int, default=500_000,
                   help="Max rows for MI and heavy computation (0 = all).")
    p.add_argument("--top-n", type=int, default=20,
                   help="Number of top features to highlight in plots.")
    return p.parse_args()


# ============================================================================
# PREMIUM THEME
# ============================================================================
def setup_premium_theme():
    """Configure matplotlib for publication-quality output."""
    plt.rcParams.update({
        "figure.facecolor":  COLORS["bg"],
        "axes.facecolor":    COLORS["panel"],
        "axes.edgecolor":    "#CBD5E1",
        "axes.linewidth":    0.8,
        "axes.grid":         True,
        "grid.color":        COLORS["grid"],
        "grid.linewidth":    0.5,
        "grid.alpha":        0.7,
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Segoe UI", "Arial", "DejaVu Sans"],
        "font.size":         11,
        "axes.titlesize":    14,
        "axes.titleweight":  "bold",
        "axes.labelsize":    12,
        "xtick.labelsize":   10,
        "ytick.labelsize":   10,
        "legend.fontsize":   10,
        "legend.framealpha": 0.9,
        "legend.edgecolor":  "#CBD5E1",
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "savefig.pad_inches":0.15,
    })
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="seaborn")
    warnings.filterwarnings("ignore", category=RuntimeWarning)


# ============================================================================
# STATISTICAL UTILITIES
# ============================================================================
def _cohens_d(x1: np.ndarray, x2: np.ndarray) -> float:
    n1, n2 = len(x1), len(x2)
    if n1 < 2 or n2 < 2:
        return 0.0
    v1, v2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
    pooled = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    return 0.0 if pooled == 0 else (np.mean(x1) - np.mean(x2)) / pooled


def _cramers_v(x: np.ndarray, y: np.ndarray) -> float:
    try:
        ct = pd.crosstab(pd.Series(x), pd.Series(y))
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            return 0.0
        chi2 = stats.chi2_contingency(ct)[0]
        n = len(x)
        k = min(ct.shape) - 1
        return np.sqrt(chi2 / (n * k)) if (n * k) > 0 else 0.0
    except Exception:
        return 0.0


def _effect_label(d: float) -> str:
    d = abs(d)
    if d >= 1.2: return "Very Large"
    if d >= 0.8: return "Large"
    if d >= 0.5: return "Medium"
    if d >= 0.2: return "Small"
    return "Negligible"


def _ks_label(k: float) -> str:
    if k >= 0.8: return "Excellent"
    if k >= 0.5: return "Strong"
    if k >= 0.3: return "Moderate"
    if k >= 0.1: return "Weak"
    return "Negligible"


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    c = a.corr(b)
    return 0.0 if (c is None or np.isnan(c)) else c


def _norm(s: pd.Series) -> pd.Series:
    """Min-max normalise to [0, 1]."""
    lo, hi = s.min(), s.max()
    return pd.Series(0.0, index=s.index) if (hi - lo) < 1e-12 else (s - lo) / (hi - lo)


def _sample(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    return df if len(df) <= n else df.sample(n=n, random_state=seed)


# ============================================================================
# FEATURE IDENTIFICATION
# ============================================================================
KNOWN_BINARY = {
    "h_is_ip_host", "h_has_fragment", "h_has_punycode", "h_has_unicode",
    "h_mixed_script", "h_has_tracking_params", "h_has_double_extension",
    "h_has_redirect_param", "h_has_at_sign",
    "h_tld_risk_normal", "h_tld_risk_high", "h_tld_risk_critical",
}


def identify_features(df: pd.DataFrame):
    """Split columns into continuous, binary, flag lists."""
    exclude = {"h_flags_bitmask"}
    all_h = [c for c in df.columns
             if c.startswith("h_") and c not in exclude
             and df[c].dtype in (np.float64, np.int64, np.float32, np.int32)]
    binary_cols = [c for c in all_h if (df[c].nunique() <= 2 or c in KNOWN_BINARY)]
    numeric_cols = [c for c in all_h if c not in binary_cols]
    flag_cols = [c for c in df.columns if c.startswith("hF_")]
    return numeric_cols, binary_cols, flag_cols


# ============================================================================
# CORE ANALYSIS ENGINE
# ============================================================================
def compute_continuous_stats(df, label_col, cols):
    db = df[df[label_col] == 0]
    dm = df[df[label_col] == 1]
    rows = []
    for c in cols:
        xb = db[c].dropna().values.astype(float)
        xm = dm[c].dropna().values.astype(float)
        mb, sb = (np.mean(xb), np.std(xb)) if len(xb) else (0.0, 0.0)
        mm, sm = (np.mean(xm), np.std(xm)) if len(xm) else (0.0, 0.0)
        cd = _cohens_d(xm, xb)
        ks, ksp = stats.ks_2samp(xb, xm) if (len(xb) and len(xm)) else (0.0, 1.0)
        try:
            mw_u, mw_p = stats.mannwhitneyu(xb, xm, alternative="two-sided")
            mw_r = 1 - (2 * mw_u) / (len(xb) * len(xm)) if (len(xb) * len(xm)) else 0.0
        except Exception:
            mw_p, mw_r = 1.0, 0.0
        corr = _safe_corr(df[c], df[label_col])
        rows.append({
            "Feature": c, "Type": "Continuous",
            "Benign_Mean": mb, "Benign_Std": sb,
            "Malicious_Mean": mm, "Malicious_Std": sm,
            "Cohens_d": cd, "KS_Stat": ks, "KS_pval": ksp,
            "MW_effect": abs(mw_r), "MW_pval": mw_p,
            "Correlation": corr,
            "Separability": ks,
            "Effect_Size": min(abs(cd), 5.0),  # cap extreme outliers
        })
    return pd.DataFrame(rows)


def compute_binary_stats(df, label_col, cols):
    db = df[df[label_col] == 0]
    dm = df[df[label_col] == 1]
    rows = []
    for c in cols:
        rb = db[c].mean() if len(db) else 0.0
        rm = dm[c].mean() if len(dm) else 0.0
        diff = rm - rb
        corr = _safe_corr(df[c], df[label_col])
        cv = _cramers_v(df[c].values, df[label_col].values)
        rows.append({
            "Feature": c, "Type": "Binary",
            "Benign_Rate": rb, "Malicious_Rate": rm,
            "Difference": diff, "Correlation": corr,
            "Cramers_V": cv,
            "Separability": cv,
            "Effect_Size": abs(diff),
        })
    return pd.DataFrame(rows)


def compute_flag_stats(df, label_col, cols):
    db = df[df[label_col] == 0]
    dm = df[df[label_col] == 1]
    rows = []
    for c in cols:
        rb = db[c].mean() if len(db) else 0.0
        rm = dm[c].mean() if len(dm) else 0.0
        diff = rm - rb
        corr = _safe_corr(df[c], df[label_col])
        cv = _cramers_v(df[c].values, df[label_col].values)
        rows.append({
            "Feature": c, "Type": "Flag",
            "Benign_Rate": rb, "Malicious_Rate": rm,
            "Difference": diff, "Correlation": corr,
            "Cramers_V": cv,
            "Separability": cv,
            "Effect_Size": abs(diff),
        })
    return pd.DataFrame(rows)


def compute_mutual_information(df, all_cols, label_col, max_samples):
    if not HAS_SKLEARN:
        return pd.Series(0.0, index=all_cols)
    ds = _sample(df, max_samples) if max_samples > 0 else df
    X = ds[all_cols].fillna(0).values.astype(np.float64)
    y = ds[label_col].values.astype(int)
    discrete = np.array([ds[c].nunique() <= 5 for c in all_cols])
    mi = mutual_info_classif(X, y, discrete_features=discrete, random_state=42, n_neighbors=5)
    return pd.Series(mi, index=all_cols)


def compute_power_scores(df_cont, df_bin, df_flag, mi_scores):
    """Compute unified Feature Power Score across all feature types."""
    parts = []
    for src in (df_cont, df_bin, df_flag):
        if src.empty:
            continue
        sub = src[["Feature", "Type", "Correlation", "Separability", "Effect_Size"]].copy()
        sub["MI"] = sub["Feature"].map(mi_scores).fillna(0.0)
        parts.append(sub)
    if not parts:
        return pd.DataFrame()
    unified = pd.concat(parts, ignore_index=True)
    unified["abs_corr"] = unified["Correlation"].abs()
    # Normalise each metric to [0, 1] across ALL features
    unified["n_MI"]   = _norm(unified["MI"])
    unified["n_corr"] = _norm(unified["abs_corr"])
    unified["n_sep"]  = _norm(unified["Separability"])
    unified["n_eff"]  = _norm(unified["Effect_Size"])
    # Weighted composite
    unified["Power"] = (
        0.35 * unified["n_MI"]
      + 0.25 * unified["n_corr"]
      + 0.20 * unified["n_sep"]
      + 0.20 * unified["n_eff"]
    )
    unified.sort_values("Power", ascending=False, inplace=True)
    unified.reset_index(drop=True, inplace=True)
    unified["Rank"] = unified.index + 1
    return unified


def classify_tiers(df_unified):
    """Assign tier labels based on Power score thresholds."""
    def _tier(p):
        for name, thresh, _ in TIER_CFG:
            if p >= thresh:
                return name
        return "Noise"
    def _tier_color(p):
        for _, thresh, color in TIER_CFG:
            if p >= thresh:
                return color
        return COLORS["tier_noise"]
    df_unified["Tier"] = df_unified["Power"].apply(_tier)
    df_unified["Tier_Color"] = df_unified["Power"].apply(_tier_color)
    return df_unified


# ============================================================================
# VISUALISATION ENGINE
# ============================================================================
def _log(msg):
    print(f"  [{datetime.now().strftime('%H:%M:%S')}] {msg}")


def generate_all_plots(df, df_u, df_cont, df_bin, df_flag, mi_scores,
                       label_col, plots_dir, top_n):
    plots_dir.mkdir(parents=True, exist_ok=True)
    dist_dir = plots_dir / "distributions"
    dist_dir.mkdir(parents=True, exist_ok=True)

    total_plots = 9
    n = 0

    # --- 1. Executive Dashboard ---
    n += 1; _log(f"[{n}/{total_plots}] Executive Dashboard …")
    _plot_executive_dashboard(df, df_u, label_col, plots_dir, top_n)

    # --- 2. Feature Power Ranking ---
    n += 1; _log(f"[{n}/{total_plots}] Feature Power Ranking …")
    _plot_power_ranking(df_u, plots_dir, top_n)

    # --- 3. Feature Tiers ---
    n += 1; _log(f"[{n}/{total_plots}] Feature Tier Summary …")
    _plot_tier_summary(df_u, plots_dir)

    # --- 4. Violin Distributions ---
    n += 1; _log(f"[{n}/{total_plots}] Violin Distributions …")
    _plot_violins(df, df_cont, label_col, plots_dir)

    # --- 5. Binary Diverging Bar ---
    n += 1; _log(f"[{n}/{total_plots}] Binary Diverging Bar …")
    _plot_binary_diverging(df_bin, plots_dir)

    # --- 6. Flag Activation Heatmap ---
    n += 1; _log(f"[{n}/{total_plots}] Flag Activation Heatmap …")
    _plot_flag_heatmap(df_flag, plots_dir)

    # --- 7. Correlation Matrix ---
    n += 1; _log(f"[{n}/{total_plots}] Correlation Matrix …")
    _plot_correlation_matrix(df, df_u, label_col, plots_dir)

    # --- 8. Mutual Information ---
    n += 1; _log(f"[{n}/{total_plots}] Mutual Information Chart …")
    _plot_mutual_information(df_u, plots_dir, top_n)

    # --- 9. Individual Distributions ---
    n += 1; _log(f"[{n}/{total_plots}] Individual Distribution Plots …")
    _plot_individual_distributions(df, df_u, df_cont, label_col, dist_dir)


# ---------- 1. EXECUTIVE DASHBOARD ----------
def _plot_executive_dashboard(df, df_u, label_col, pdir, top_n):
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.35)

    nb = int((df[label_col] == 0).sum())
    nm = int((df[label_col] == 1).sum())
    total = len(df)

    # Panel 1: Top features by Power
    ax1 = fig.add_subplot(gs[0, 0:2])
    top = df_u.head(min(top_n, 15)).copy()
    top = top.iloc[::-1]
    bars = ax1.barh(range(len(top)), top["Power"],
                    color=[c for c in top["Tier_Color"]], edgecolor="white", linewidth=0.5)
    ax1.set_yticks(range(len(top)))
    ax1.set_yticklabels([_label(f) for f in top["Feature"]], fontsize=9)
    ax1.set_xlabel("Feature Power Score")
    ax1.set_title("Top Features by Composite Power Score")
    ax1.set_xlim(0, 1.05)
    for i, (pw, tier) in enumerate(zip(top["Power"], top["Tier"])):
        ax1.text(pw + 0.015, i, f"{pw:.2f} [{tier}]", va="center", fontsize=8, color=COLORS["text_sec"])

    # Panel 2: Class balance donut
    ax2 = fig.add_subplot(gs[0, 2])
    wedges, _ = ax2.pie([nb, nm], colors=[COLORS["benign"], COLORS["malicious"]],
                        startangle=90, wedgeprops=dict(width=0.35, edgecolor="white"))
    ax2.add_artist(plt.Circle((0, 0), 0.55, fc="white"))
    ax2.text(0, 0.08, f"{total:,}", ha="center", va="center", fontsize=16, fontweight="bold", color=COLORS["text"])
    ax2.text(0, -0.15, "Total URLs", ha="center", va="center", fontsize=9, color=COLORS["text_sec"])
    ax2.set_title("Class Balance")
    ax2.legend(wedges, [f"Benign {nb:,} ({nb/total:.1%})", f"Malicious {nm:,} ({nm/total:.1%})"],
               loc="lower center", fontsize=8, frameon=False)

    # Panel 3: Tier distribution donut
    ax3 = fig.add_subplot(gs[1, 2])
    tier_counts = df_u["Tier"].value_counts()
    tier_order = [t[0] for t in TIER_CFG]
    tier_vals = [tier_counts.get(t, 0) for t in tier_order]
    tier_colors = [t[2] for t in TIER_CFG]
    non_zero = [(v, n, c) for v, n, c in zip(tier_vals, tier_order, tier_colors) if v > 0]
    if non_zero:
        vals, names, cols = zip(*non_zero)
        wedges2, _ = ax3.pie(vals, colors=cols, startangle=90,
                             wedgeprops=dict(width=0.35, edgecolor="white"))
        ax3.add_artist(plt.Circle((0, 0), 0.55, fc="white"))
        ax3.text(0, 0, f"{len(df_u)}", ha="center", va="center", fontsize=16, fontweight="bold", color=COLORS["text"])
        ax3.set_title("Feature Tier Distribution")
        ax3.legend(wedges2, [f"{n}: {v}" for n, v in zip(names, vals)],
                   loc="lower center", fontsize=8, frameon=False)

    # Panel 4: Key statistics text
    ax4 = fig.add_subplot(gs[1, 0:2])
    ax4.axis("off")
    n_cont = len(df_u[df_u["Type"] == "Continuous"])
    n_bin = len(df_u[df_u["Type"] == "Binary"])
    n_flag = len(df_u[df_u["Type"] == "Flag"])
    n_crit = tier_counts.get("Critical", 0)
    n_high = tier_counts.get("High", 0)
    top1 = df_u.iloc[0] if len(df_u) else None
    lines = [
        f"ANALYSIS OVERVIEW",
        f"─────────────────────────────────────────────────────",
        f"Dataset Size:      {total:>12,} URLs",
        f"Benign / Malicious: {nb:>10,} / {nm:,}  ({nb/total:.1%} / {nm/total:.1%})",
        f"",
        f"Feature Breakdown:  {n_cont} Continuous  |  {n_bin} Binary  |  {n_flag} Threat Flags",
        f"Total Features:    {len(df_u):>12}",
        f"",
        f"Critical Features: {n_crit:>12}   (Power ≥ 0.65)",
        f"High Features:     {n_high:>12}   (Power ≥ 0.40)",
        f"",
    ]
    if top1 is not None:
        lines.append(f"#1 Most Powerful:   {_label(top1['Feature'])}  (Power = {top1['Power']:.3f})")
    info_text = "\n".join(lines)
    ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment="top", fontfamily="monospace", color=COLORS["text"],
             bbox=dict(boxstyle="round,pad=0.6", facecolor="#F1F5F9", edgecolor="#CBD5E1", alpha=0.9))

    fig.suptitle("PhishURLDetect — Feature Distribution Executive Dashboard",
                 fontsize=16, fontweight="bold", y=0.98, color=COLORS["primary"])
    fig.savefig(pdir / "01_executive_dashboard.png")
    plt.close(fig)


# ---------- 2. FEATURE POWER RANKING ----------
def _plot_power_ranking(df_u, pdir, top_n):
    show = df_u.head(min(top_n, 30)).copy().iloc[::-1]
    fig, ax = plt.subplots(figsize=(12, max(6, len(show) * 0.35)))
    ax.barh(range(len(show)), show["Power"], color=show["Tier_Color"].values,
            edgecolor="white", linewidth=0.5, height=0.7)
    ax.set_yticks(range(len(show)))
    ax.set_yticklabels([_label(f) for f in show["Feature"]], fontsize=9)
    ax.set_xlabel("Feature Power Score (0 – 1)")
    ax.set_title(f"Unified Feature Power Ranking — Top {len(show)}")
    ax.set_xlim(0, 1.08)
    for i, (pw, tier, typ) in enumerate(zip(show["Power"], show["Tier"], show["Type"])):
        ax.text(pw + 0.012, i, f"{pw:.3f}  [{tier}] ({typ})", va="center",
                fontsize=7.5, color=COLORS["text_sec"])
    # Add tier legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, label=f"{n} (≥{t:.2f})") for n, t, c in TIER_CFG]
    ax.legend(handles=handles, loc="lower right", fontsize=8, title="Tiers", title_fontsize=9)
    fig.tight_layout()
    fig.savefig(pdir / "02_feature_power_ranking.png")
    plt.close(fig)


# ---------- 3. TIER SUMMARY ----------
def _plot_tier_summary(df_u, pdir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [2, 1]})
    # Left: Grouped bar by tier
    ax = axes[0]
    for _, t_name, t_color in TIER_CFG:
        sub = df_u[df_u["Tier"] == t_name]
        if sub.empty:
            continue
        sub_sorted = sub.sort_values("Power", ascending=True)
        y_pos = range(len(sub_sorted))
        ax.barh(y_pos, sub_sorted["Power"], color=t_color, edgecolor="white",
                linewidth=0.4, height=0.8, label=t_name)
        for j, (_, row) in enumerate(sub_sorted.iterrows()):
            ax.text(row["Power"] + 0.01, j, _label(row["Feature"]), fontsize=6.5,
                    va="center", color=COLORS["text_sec"])
        ax.set_xlim(0, 1.15)
    ax.set_title("All Features Grouped by Tier")
    ax.set_xlabel("Power Score")
    ax.set_yticks([])

    # Right: Summary counts
    ax2 = axes[1]
    ax2.axis("off")
    tier_counts = df_u["Tier"].value_counts()
    text_lines = ["  TIER CLASSIFICATION SUMMARY", "  " + "─" * 40]
    for t_name, t_thresh, t_col in TIER_CFG:
        cnt = tier_counts.get(t_name, 0)
        text_lines.append(f"  {t_name:<12}  {cnt:>3} features   (Power ≥ {t_thresh:.2f})")
    text_lines += ["", f"  Total:        {len(df_u):>3} features"]
    ax2.text(0.05, 0.95, "\n".join(text_lines), transform=ax2.transAxes,
             fontsize=10, va="top", fontfamily="monospace", color=COLORS["text"],
             bbox=dict(boxstyle="round,pad=0.6", facecolor="#F1F5F9", edgecolor="#CBD5E1"))
    fig.suptitle("Feature Tier Classification", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(pdir / "03_feature_tiers.png")
    plt.close(fig)


# ---------- 4. VIOLIN DISTRIBUTIONS ----------
def _plot_violins(df, df_cont, label_col, pdir):
    if df_cont.empty:
        return
    top_features = df_cont.sort_values("KS_Stat", ascending=False).head(8)["Feature"].tolist()
    if not top_features:
        return
    n_cols = min(4, len(top_features))
    n_rows = (len(top_features) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows))
    axes = np.atleast_2d(axes)
    ds = _sample(df, 100_000)
    for idx, feat in enumerate(top_features):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]
        sns.violinplot(data=ds, x=label_col, y=feat, ax=ax, inner="quartile",
                       palette={0: COLORS["benign"], 1: COLORS["malicious"], "0": COLORS["benign"], "1": COLORS["malicious"]},
                       saturation=0.85, cut=0)
        ax.set_title(_label(feat), fontsize=11)
        ax.set_xlabel("")
        ax.set_xticklabels(["Benign", "Malicious"])
        ax.set_ylabel("")
    # hide unused
    for idx in range(len(top_features), n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].set_visible(False)
    fig.suptitle("Top Continuous Features — Violin Distributions (Benign vs Malicious)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(pdir / "04_violin_distributions.png")
    plt.close(fig)


# ---------- 5. BINARY DIVERGING BAR ----------
def _plot_binary_diverging(df_bin, pdir):
    if df_bin.empty:
        return
    show = df_bin.sort_values("Difference", key=abs, ascending=True).copy()
    # filter out features with no difference at all
    show = show[show["Difference"].abs() > 1e-6]
    if show.empty:
        show = df_bin.sort_values("Difference", key=abs, ascending=True).copy()
    fig, ax = plt.subplots(figsize=(12, max(4, len(show) * 0.4)))
    y = range(len(show))
    ax.barh(y, -show["Benign_Rate"].values * 100, color=COLORS["benign"],
            edgecolor="white", linewidth=0.4, height=0.7, label="Benign")
    ax.barh(y, show["Malicious_Rate"].values * 100, color=COLORS["malicious"],
            edgecolor="white", linewidth=0.4, height=0.7, label="Malicious")
    ax.set_yticks(y)
    ax.set_yticklabels([_label(f) for f in show["Feature"]], fontsize=9)
    ax.axvline(0, color=COLORS["text"], linewidth=0.8)
    ax.set_xlabel("← Benign Activation %          Malicious Activation % →")
    ax.set_title("Binary Feature Activation — Diverging Comparison")
    ax.legend(loc="lower right", fontsize=9)
    # annotate percentages
    for i, (_, row) in enumerate(show.iterrows()):
        if row["Benign_Rate"] > 0.005:
            ax.text(-row["Benign_Rate"] * 100 - 1, i, f"{row['Benign_Rate']*100:.1f}%",
                    ha="right", va="center", fontsize=7, color=COLORS["benign"])
        if row["Malicious_Rate"] > 0.005:
            ax.text(row["Malicious_Rate"] * 100 + 1, i, f"{row['Malicious_Rate']*100:.1f}%",
                    ha="left", va="center", fontsize=7, color=COLORS["malicious"])
    fig.tight_layout()
    fig.savefig(pdir / "05_binary_diverging.png")
    plt.close(fig)


# ---------- 6. FLAG ACTIVATION HEATMAP ----------
def _plot_flag_heatmap(df_flag, pdir):
    if df_flag.empty:
        return
    # Only show flags with at least 0.5% activation in either class
    active = df_flag[(df_flag["Benign_Rate"].abs() > 0.005) | (df_flag["Malicious_Rate"].abs() > 0.005)].copy()
    if active.empty:
        active = df_flag.head(20).copy()
    active = active.sort_values("Difference", ascending=False)
    # Build heatmap data
    hm = pd.DataFrame({
        "Benign %":    (active["Benign_Rate"].values * 100),
        "Malicious %": (active["Malicious_Rate"].values * 100),
    }, index=[_label(f) for f in active["Feature"]])
    fig, ax = plt.subplots(figsize=(8, max(4, len(hm) * 0.35)))
    sns.heatmap(hm, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax,
                linewidths=0.5, linecolor="white", cbar_kws={"label": "Activation %"})
    ax.set_title("Threat Flag Activation Rates (Benign vs Malicious)")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(pdir / "06_flag_heatmap.png")
    plt.close(fig)


# ---------- 7. CORRELATION MATRIX ----------
def _plot_correlation_matrix(df, df_u, label_col, pdir):
    top15 = df_u.head(15)["Feature"].tolist()
    if len(top15) < 3:
        return
    cols = [c for c in top15 if c in df.columns] + [label_col]
    corr = df[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                vmin=-1, vmax=1, center=0, ax=ax, linewidths=0.5, linecolor="white",
                xticklabels=[_label(c) for c in cols],
                yticklabels=[_label(c) for c in cols],
                cbar_kws={"label": "Pearson Correlation"})
    ax.set_title("Correlation Matrix — Top 15 Features + Label")
    fig.tight_layout()
    fig.savefig(pdir / "07_correlation_matrix.png")
    plt.close(fig)


# ---------- 8. MUTUAL INFORMATION ----------
def _plot_mutual_information(df_u, pdir, top_n):
    show = df_u.head(min(top_n, 25)).copy().iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, max(5, len(show) * 0.35)))
    ax.barh(range(len(show)), show["MI"], color=show["Tier_Color"].values,
            edgecolor="white", linewidth=0.4, height=0.7)
    ax.set_yticks(range(len(show)))
    ax.set_yticklabels([_label(f) for f in show["Feature"]], fontsize=9)
    ax.set_xlabel("Mutual Information (bits)")
    ax.set_title(f"Mutual Information with Phishing Label — Top {len(show)}")
    for i, mi_val in enumerate(show["MI"]):
        if mi_val > 0:
            ax.text(mi_val + 0.002, i, f"{mi_val:.4f}", va="center", fontsize=7.5, color=COLORS["text_sec"])
    fig.tight_layout()
    fig.savefig(pdir / "08_mutual_information.png")
    plt.close(fig)


# ---------- 9. INDIVIDUAL DISTRIBUTIONS ----------
def _plot_individual_distributions(df, df_u, df_cont, label_col, dist_dir):
    top_feats = df_u[df_u["Type"] == "Continuous"].head(10)["Feature"].tolist()
    if not top_feats:
        return
    ds = _sample(df, 100_000)
    cont_lookup = {}
    if not df_cont.empty:
        cont_lookup = df_cont.set_index("Feature").to_dict("index")
    power_lookup = df_u.set_index("Feature")

    for feat in top_feats:
        fig, ax = plt.subplots(figsize=(9, 5))
        xb = ds[ds[label_col] == 0][feat].dropna()
        xm = ds[ds[label_col] == 1][feat].dropna()
        # KDE with fill
        if len(xb) > 1:
            sns.kdeplot(xb, ax=ax, fill=True, alpha=0.35, color=COLORS["benign"],
                        label="Benign", linewidth=1.5)
            ax.axvline(xb.mean(), color=COLORS["benign"], linestyle="--", linewidth=1, alpha=0.8)
        if len(xm) > 1:
            sns.kdeplot(xm, ax=ax, fill=True, alpha=0.35, color=COLORS["malicious"],
                        label="Malicious", linewidth=1.5)
            ax.axvline(xm.mean(), color=COLORS["malicious"], linestyle="--", linewidth=1, alpha=0.8)

        ax.set_xlabel(_label(feat))
        ax.set_ylabel("Density")
        ax.set_title(f"Distribution of {_label(feat)} — Benign vs Malicious")
        ax.legend(fontsize=9)

        # Annotation box with statistics
        info = cont_lookup.get(feat, {})
        pw_row = power_lookup.loc[feat] if feat in power_lookup.index else None
        stats_lines = []
        if info:
            cd = info.get("Cohens_d", 0)
            ks = info.get("KS_Stat", 0)
            stats_lines.append(f"Benign μ={info.get('Benign_Mean',0):.3f}  σ={info.get('Benign_Std',0):.3f}")
            stats_lines.append(f"Malicious μ={info.get('Malicious_Mean',0):.3f}  σ={info.get('Malicious_Std',0):.3f}")
            stats_lines.append(f"Cohen's d = {cd:.3f} ({_effect_label(cd)})")
            stats_lines.append(f"KS Stat   = {ks:.3f} ({_ks_label(ks)})")
        if pw_row is not None:
            stats_lines.append(f"MI Score  = {pw_row.get('MI', 0):.4f}")
            stats_lines.append(f"Power     = {pw_row.get('Power', 0):.3f} [{pw_row.get('Tier', '?')}]")
        if stats_lines:
            box_text = "\n".join(stats_lines)
            ax.text(0.97, 0.95, box_text, transform=ax.transAxes, fontsize=8,
                    va="top", ha="right", fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#F1F5F9",
                              edgecolor="#CBD5E1", alpha=0.92))
        fig.tight_layout()
        fig.savefig(dist_dir / f"{feat}.png")
        plt.close(fig)


# ============================================================================
# REPORT GENERATOR
# ============================================================================
def generate_report(output_dir: Path, df, df_u, df_cont, df_bin, df_flag,
                    mi_scores, label_col, n_cont, n_bin, n_flag):
    report_path = output_dir / "feature_distribution_report.md"
    nb = int((df[label_col] == 0).sum())
    nm = int((df[label_col] == 1).sum())
    total = len(df)
    tier_counts = df_u["Tier"].value_counts() if not df_u.empty else pd.Series(dtype=int)

    with open(report_path, "w", encoding="utf-8") as f:
        w = f.write  # shorthand

        # ── Title ──
        w("# Heuristic Feature Distribution Analysis Report\n")
        w("### PhishURLDetect Research — Samsung-Grade Analysis v2.0\n\n")
        w(f"**Generated:** {datetime.now().isoformat()}  \n")
        w(f"**Dataset Size:** {total:,} URLs  |  **Features Analysed:** {len(df_u)}\n\n")
        w("---\n\n")

        # ── Abstract ──
        w("## Abstract\n\n")
        w("This report presents a rigorous, multi-metric statistical analysis of ")
        w(f"{len(df_u)} heuristic features extracted from {total:,} URLs in the PhishURLDetect pipeline. ")
        w("We evaluate each feature's discriminative power between benign and malicious URL classes using ")
        w("six complementary statistical techniques: **Mutual Information**, **Kolmogorov-Smirnov tests**, ")
        w("**Cohen's d effect sizes**, **Mann-Whitney U tests**, **Chi-square / Cramér's V** association, ")
        w("and **Pearson correlation**. ")
        w("These metrics are synthesised into a unified **Feature Power Score** that ranks every feature ")
        w("on a 0–1 scale and assigns it to one of five tiers (Critical → Noise). ")
        if not df_u.empty:
            n_crit = tier_counts.get("Critical", 0)
            n_high = tier_counts.get("High", 0)
            top1 = df_u.iloc[0]
            w(f"Our analysis identifies **{n_crit} Critical** and **{n_high} High** impact features, ")
            w(f"with **{_label(top1['Feature'])}** emerging as the single most powerful discriminator ")
            w(f"(Power = {top1['Power']:.3f}).\n\n")
        w("---\n\n")

        # ── 1. Executive Summary ──
        w("## 1. Executive Summary\n\n")
        w("### Key Findings\n\n")
        w("![Executive Dashboard](plots/01_executive_dashboard.png)\n\n")

        if not df_u.empty:
            w("#### Top 5 Most Powerful Features\n\n")
            w("| Rank | Feature | Type | Power Score | Tier |\n")
            w("|:----:|---------|------|:-----------:|:----:|\n")
            for _, row in df_u.head(5).iterrows():
                w(f"| {int(row['Rank'])} | **{_label(row['Feature'])}** (`{row['Feature']}`) | {row['Type']} | {row['Power']:.3f} | {row['Tier']} |\n")
            w("\n")

        w("### Dataset Overview\n\n")
        w(f"| Metric | Value |\n|--------|-------|\n")
        w(f"| Total URLs Analysed | {total:,} |\n")
        w(f"| Benign (Label 0) | {nb:,} ({nb/total:.2%}) |\n")
        w(f"| Malicious (Label 1) | {nm:,} ({nm/total:.2%}) |\n")
        w(f"| Continuous Features | {n_cont} |\n")
        w(f"| Binary Features | {n_bin} |\n")
        w(f"| Threat Flag Features | {n_flag} |\n")
        w(f"| Total Features | {n_cont + n_bin + n_flag} |\n\n")
        w("---\n\n")

        # ── 2. Methodology ──
        w("## 2. Methodology\n\n")
        w("We employ a **multi-metric ensemble** approach to evaluate each feature's discriminative ")
        w("ability. No single metric captures every dimension of usefulness, so we combine six ")
        w("complementary tests into a unified score.\n\n")

        w("### 2.1 Statistical Tests Applied\n\n")
        w("| Test | Applies To | What It Measures | Scale |\n")
        w("|------|-----------|-----------------|-------|\n")
        w("| **Mutual Information (MI)** | All features | Non-linear statistical dependency with the label | 0 → ∞ (bits) |\n")
        w("| **Pearson / Point-Biserial Correlation** | All features | Linear association strength and direction | −1 → +1 |\n")
        w("| **Kolmogorov-Smirnov (KS) Test** | Continuous | Maximum separation between class CDFs | 0 → 1 |\n")
        w("| **Cohen's d** | Continuous | Standardised mean difference (effect size) | 0 → ∞ |\n")
        w("| **Mann-Whitney U** | Continuous | Non-parametric rank-based group difference | 0 → 1 (effect size) |\n")
        w("| **Chi-Square / Cramér's V** | Binary & Flags | Categorical association strength | 0 → 1 |\n\n")

        w("### 2.2 Feature Power Score Computation\n\n")
        w("Each metric is min-max normalised to [0, 1] across **all** features, then combined:\n\n")
        w("```\n")
        w("Power = 0.35 × norm(MI)  +  0.25 × norm(|Correlation|)\n")
        w("      + 0.20 × norm(Separability)  +  0.20 × norm(Effect Size)\n")
        w("```\n\n")
        w("Where *Separability* = KS stat for continuous features, Cramér's V for binary/flags; ")
        w("and *Effect Size* = |Cohen's d| for continuous features, |proportion difference| for binary/flags.\n\n")

        w("### 2.3 Tier Classification\n\n")
        w("| Tier | Power Score Range | Interpretation |\n")
        w("|------|:-----------------:|----------------|\n")
        w("| **Critical** | ≥ 0.65 | Dominant discriminator — must be preserved |\n")
        w("| **High** | 0.40 – 0.64 | Strong signal — important for model accuracy |\n")
        w("| **Moderate** | 0.20 – 0.39 | Useful supplementary signal |\n")
        w("| **Low** | 0.08 – 0.19 | Marginal contribution — candidate for pruning |\n")
        w("| **Noise** | < 0.08 | No meaningful signal — safe to remove |\n\n")
        w("---\n\n")

        # ── 3. Results ──
        w("## 3. Results\n\n")

        # 3.1 Unified ranking
        w("### 3.1 Unified Feature Power Ranking\n\n")
        w("![Feature Power Ranking](plots/02_feature_power_ranking.png)\n\n")

        w("#### Complete Feature Ranking Table\n\n")
        w("| Rank | Feature | Type | MI | Corr | Sep. | Effect | Power | Tier |\n")
        w("|:----:|---------|:----:|:---:|:----:|:----:|:------:|:-----:|:----:|\n")
        for _, row in df_u.iterrows():
            w(f"| {int(row['Rank'])} | `{row['Feature']}` | {row['Type']} | {row['MI']:.4f} | {row['abs_corr']:.3f} | {row['Separability']:.3f} | {row['Effect_Size']:.3f} | **{row['Power']:.3f}** | {row['Tier']} |\n")
        w("\n")

        # 3.2 Tier classification
        w("### 3.2 Feature Tier Classification\n\n")
        w("![Feature Tiers](plots/03_feature_tiers.png)\n\n")
        for t_name, t_thresh, _ in TIER_CFG:
            tier_feats = df_u[df_u["Tier"] == t_name]
            if tier_feats.empty:
                continue
            w(f"**{t_name} Tier** ({len(tier_feats)} features): ")
            w(", ".join([f"`{r['Feature']}`" for _, r in tier_feats.iterrows()]))
            w("\n\n")

        # 3.3 Continuous
        w("### 3.3 Continuous Feature Distributions\n\n")
        w("![Violin Distributions](plots/04_violin_distributions.png)\n\n")

        if not df_cont.empty:
            w("#### Detailed Continuous Feature Statistics\n\n")
            w("| Feature | Benign μ (±σ) | Malicious μ (±σ) | Cohen's d | KS Stat | MW Effect | Corr |\n")
            w("|---------|:-------------:|:----------------:|:---------:|:-------:|:---------:|:----:|\n")
            df_c_sorted = df_cont.sort_values("KS_Stat", ascending=False)
            for _, r in df_c_sorted.iterrows():
                cd = r["Cohens_d"]
                w(f"| `{r['Feature']}` | {r['Benign_Mean']:.3f} (±{r['Benign_Std']:.2f}) | {r['Malicious_Mean']:.3f} (±{r['Malicious_Std']:.2f}) | {cd:.3f} ({_effect_label(cd)}) | {r['KS_Stat']:.3f} ({_ks_label(r['KS_Stat'])}) | {r['MW_effect']:.3f} | {r['Correlation']:.3f} |\n")
            w("\n")

            # Embed top individual distribution plots
            top_cont = df_c_sorted.head(5)["Feature"].tolist()
            w("#### Top Feature Distribution Plots\n\n")
            for feat in top_cont:
                w(f"![{_label(feat)}](plots/distributions/{feat}.png)\n\n")

        # 3.4 Binary
        w("### 3.4 Binary Feature Analysis\n\n")
        w("![Binary Diverging](plots/05_binary_diverging.png)\n\n")

        if not df_bin.empty:
            w("#### Detailed Binary Feature Statistics\n\n")
            w("| Feature | Benign % | Malicious % | Δ (M−B) | Cramér's V | Corr |\n")
            w("|---------|:--------:|:-----------:|:-------:|:----------:|:----:|\n")
            df_b_sorted = df_bin.sort_values("Difference", key=abs, ascending=False)
            for _, r in df_b_sorted.iterrows():
                w(f"| `{r['Feature']}` | {r['Benign_Rate']*100:.2f}% | {r['Malicious_Rate']*100:.2f}% | {r['Difference']*100:+.2f}% | {r['Cramers_V']:.3f} | {r['Correlation']:.3f} |\n")
            w("\n")

        # 3.5 Flags
        w("### 3.5 Threat Category Flags\n\n")
        w("![Flag Heatmap](plots/06_flag_heatmap.png)\n\n")

        if not df_flag.empty:
            w("#### Detailed Flag Statistics\n\n")
            active_flags = df_flag[(df_flag["Benign_Rate"].abs() > 0.001) | (df_flag["Malicious_Rate"].abs() > 0.001)]
            if active_flags.empty:
                active_flags = df_flag
            active_flags = active_flags.sort_values("Difference", key=abs, ascending=False)
            w("| Flag | Benign % | Malicious % | Δ (M−B) | Cramér's V | Corr |\n")
            w("|------|:--------:|:-----------:|:-------:|:----------:|:----:|\n")
            for _, r in active_flags.iterrows():
                w(f"| `{r['Feature']}` | {r['Benign_Rate']*100:.2f}% | {r['Malicious_Rate']*100:.2f}% | {r['Difference']*100:+.2f}% | {r['Cramers_V']:.3f} | {r['Correlation']:.3f} |\n")
            w("\n")

            # Inactive (zero-signal) flags
            zero_flags = df_flag[(df_flag["Benign_Rate"].abs() < 0.001) & (df_flag["Malicious_Rate"].abs() < 0.001)]
            if not zero_flags.empty:
                w(f"**{len(zero_flags)} threat flags showed zero activation** in both classes ")
                w("and carry no discriminative signal in this dataset: ")
                w(", ".join([f"`{r['Feature']}`" for _, r in zero_flags.iterrows()]))
                w("\n\n")

        # 3.6 Correlation & MI
        w("### 3.6 Feature Correlations & Mutual Information\n\n")
        w("![Correlation Matrix](plots/07_correlation_matrix.png)\n\n")
        w("![Mutual Information](plots/08_mutual_information.png)\n\n")
        w("---\n\n")

        # ── 4. Interpretation ──
        w("## 4. Interpretation & Insights\n\n")

        # Features that drive separation
        w("### 4.1 Features That Drive Class Separation\n\n")
        critical_feats = df_u[df_u["Tier"].isin(["Critical", "High"])]
        if not critical_feats.empty:
            w("The following features exhibit the strongest statistical separation between benign ")
            w("and malicious URL classes across multiple independent metrics:\n\n")
            for _, r in critical_feats.iterrows():
                w(f"- **{_label(r['Feature'])}** (`{r['Feature']}`) — Power {r['Power']:.3f} [{r['Tier']}]: ")
                if r["Type"] == "Continuous":
                    info = df_cont[df_cont["Feature"] == r["Feature"]]
                    if not info.empty:
                        i = info.iloc[0]
                        w(f"KS={i['KS_Stat']:.3f}, Cohen's d={i['Cohens_d']:.3f} ({_effect_label(i['Cohens_d'])})")
                else:
                    info_b = df_bin[df_bin["Feature"] == r["Feature"]]
                    info_f = df_flag[df_flag["Feature"] == r["Feature"]]
                    src = info_b if not info_b.empty else info_f
                    if not src.empty:
                        i = src.iloc[0]
                        w(f"Δ={i['Difference']*100:+.1f}%, V={i['Cramers_V']:.3f}")
                w("\n")
            w("\n")

        # Redundancy analysis
        w("### 4.2 Potential Feature Redundancy\n\n")
        w("Features with high inter-correlation (|r| > 0.8) may carry redundant information. ")
        w("See the correlation matrix above for clusters of co-varying features. ")
        w("Candidates for deduplication should be validated via ablation experiments on the ")
        w("trained model before removal.\n\n")

        # Noise features
        noise_feats = df_u[df_u["Tier"] == "Noise"]
        low_feats = df_u[df_u["Tier"] == "Low"]
        w("### 4.3 Features With Low/No Discriminative Power\n\n")
        if not noise_feats.empty:
            w(f"**{len(noise_feats)} Noise-tier features** (Power < 0.08) show negligible statistical ")
            w("separation between classes in this dataset. These features may be safely removed or ")
            w("deprioritised to simplify the model without expected accuracy loss.\n\n")
        if not low_feats.empty:
            w(f"**{len(low_feats)} Low-tier features** (0.08 ≤ Power < 0.20) provide marginal signal. ")
            w("Consider retaining them only if they capture rare but high-severity threat patterns ")
            w("not covered by higher-tier features.\n\n")
        w("---\n\n")

        # ── 5. Recommendations ──
        w("## 5. Recommendations\n\n")
        w("### For Model Development\n")
        w("1. **Preserve all Critical and High-tier features** as primary model inputs.\n")
        w("2. **Evaluate Moderate-tier features** via ablation study — remove one at a time and measure AUC impact.\n")
        w("3. **Consider dropping Noise-tier features** to reduce input dimensionality and training overhead.\n")
        w("4. **Monitor feature interactions** — some individually weak features may be powerful in combination.\n\n")

        w("### For Feature Engineering\n")
        w("1. **Investigate zero-signal threat flags** — if they are important threat categories but inactive in the dataset, the rules engine may need recalibration.\n")
        w("2. **Create composite features** from correlated groups (e.g., combine entropy measures into a single \"URL complexity\" score).\n")
        w("3. **Add temporal features** if timestamp data becomes available, to capture time-of-day or campaign-burst patterns.\n\n")

        w("### For Monitoring & Drift Detection\n")
        w("1. **Track Critical-tier feature distributions** in production — significant distributional shift (KS > 0.1 vs. training data) signals model staleness.\n")
        w("2. **Log Mutual Information** between features and outcomes periodically to detect concept drift.\n")
        w("3. **Set alerts** for sudden changes in threat flag activation rates.\n\n")
        w("---\n\n")

        # ── 6. Limitations ──
        w("## 6. Limitations\n\n")
        w("1. **Distribution analysis ≠ model importance.** Features that separate classes statistically may not be the features the trained model actually relies on. Model-level analysis (e.g., SHAP on the trained MiniLM Hybrid model) is required for definitive feature importance.\n")
        w("2. **Linear vs. non-linear.** Pearson correlation and Cohen's d only capture linear/monotone relationships. MI partially addresses this, but complex interaction effects may be missed.\n")
        w("3. **Dataset representativeness.** Results reflect the specific distribution of URLs in this dataset. Production traffic may exhibit different patterns.\n")
        w("4. **No causal inference.** All metrics are associational. A feature correlated with phishing may be a side-effect rather than a root cause.\n")
        w(f"5. **Sample size.** This analysis was performed on {total:,} URLs. Statistical significance should be interpreted accordingly.\n\n")
        w("---\n\n")

        # ── Appendix ──
        w("## Appendix\n\n")
        w("### A. Data Provenance\n\n")
        w(f"- **Analysis Date:** {datetime.now().isoformat()}\n")
        w(f"- **Total Rows:** {total:,}\n")
        w(f"- **Column Count:** {len(df.columns)}\n")
        w(f"- **Features Analysed:** {len(df_u)} ({n_cont} continuous, {n_bin} binary, {n_flag} flags)\n")
        w(f"- **Class Balance:** {nb:,} benign / {nm:,} malicious ({nb/total:.2%} / {nm/total:.2%})\n")
        w(f"- **Scikit-learn MI Available:** {'Yes' if HAS_SKLEARN else 'No'}\n\n")

        w("### B. Software Versions\n\n")
        w(f"- Python: {sys.version.split()[0]}\n")
        w(f"- NumPy: {np.__version__}\n")
        w(f"- Pandas: {pd.__version__}\n")
        import scipy
        w(f"- SciPy: {scipy.__version__}\n")
        w(f"- Matplotlib: {matplotlib.__version__}\n")
        w(f"- Seaborn: {sns.__version__}\n")
        if HAS_SKLEARN:
            import sklearn
            w(f"- Scikit-learn: {sklearn.__version__}\n")
        w("\n")

        w("### C. Feature Power Score Weights\n\n")
        w("| Component | Weight | Source Metric |\n")
        w("|-----------|:------:|---------------|\n")
        w("| Mutual Information | 0.35 | `sklearn.feature_selection.mutual_info_classif` |\n")
        w("| Correlation | 0.25 | Pearson / Point-biserial |\n")
        w("| Separability | 0.20 | KS Stat (continuous) / Cramér's V (binary/flags) |\n")
        w("| Effect Size | 0.20 | |Cohen's d| (continuous) / |Δ proportion| (binary/flags) |\n\n")

        w("### D. Visualisation Reference\n\n")
        w("| # | Plot | Description |\n")
        w("|:-:|------|-------------|\n")
        w("| 1 | `01_executive_dashboard.png` | Multi-panel executive overview |\n")
        w("| 2 | `02_feature_power_ranking.png` | Unified feature ranking by Power Score |\n")
        w("| 3 | `03_feature_tiers.png` | Feature tier classification summary |\n")
        w("| 4 | `04_violin_distributions.png` | Violin plots for top continuous features |\n")
        w("| 5 | `05_binary_diverging.png` | Diverging bar chart for binary features |\n")
        w("| 6 | `06_flag_heatmap.png` | Threat flag activation heatmap |\n")
        w("| 7 | `07_correlation_matrix.png` | Correlation matrix (top 15 features) |\n")
        w("| 8 | `08_mutual_information.png` | Mutual Information scores |\n")
        w("| 9 | `distributions/*.png` | Individual annotated distribution plots |\n\n")

        w("---\n")
        w("*Report generated by PhishURLDetect Feature Distribution Analyser v2.0*\n")

    return report_path


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def main():
    args = parse_args()
    start_time = time.time()

    print("=" * 80)
    print("   SAMSUNG-GRADE FEATURE DISTRIBUTION ANALYSIS  v2.0")
    print("=" * 80)

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"[ERROR] Input CSV file does not exist at: {args.input}")
        sys.exit(1)

    # ── Load or preprocess ──
    _log("Loading dataset …")
    first_row = pd.read_csv(input_path, nrows=1)
    has_features = any(c.startswith("h_") for c in first_row.columns)

    if not has_features:
        _log("Input CSV is raw (input, label). Running preprocessing on the fly …")
        temp_out = output_dir / "temp_preprocessed.csv"
        script_dir = Path(__file__).parent
        prep_script = script_dir / "2_preprocess_urls_v8_refactored.py"
        if not prep_script.exists():
            for ps in [script_dir / "../2_preprocess_urls_v8_refactored.py",
                       script_dir / "../../SRC/2_preprocess_urls_v8_refactored.py",
                       script_dir / "../../../2_preprocess_urls_v8_refactored.py"]:
                if ps.exists():
                    prep_script = ps
                    break
        if not prep_script.exists():
            # Fallback to CWD
            prep_script = Path("2_preprocess_urls_v8_refactored.py")
            if not prep_script.exists():
                for ps in [Path("../2_preprocess_urls_v8_refactored.py"),
                           Path("../../SRC/2_preprocess_urls_v8_refactored.py"),
                           Path("../../../2_preprocess_urls_v8_refactored.py")]:
                    if ps.exists():
                        prep_script = ps
                        break
        if not prep_script.exists():
            print("[ERROR] 2_preprocess_urls_v8_refactored.py not found!")
            sys.exit(1)
        _log(f"Preprocessing script: {prep_script.resolve()}")
        cmd = [
            sys.executable, str(prep_script),
            "--input", str(input_path.resolve()),
            "--output", str(temp_out.resolve()),
            "--chunk-size", str(args.chunk_size),
            "--output-format", "hybrid",
            "--use-rule-features",
        ]
        if args.workers:
            cmd.extend(["--enable-multiprocessing", "--num-workers", str(args.workers)])
        subprocess.run(cmd, check=True)
        df = pd.read_csv(temp_out)
        try:
            temp_out.unlink()
            for p in [Path("urls_rejected_corrupted.csv"), Path("urls_local_private.csv"),
                      Path("urls_duplicates.csv"), Path("preprocess.log")]:
                if p.exists():
                    p.unlink()
        except Exception:
            pass
    else:
        _log("Input CSV contains preprocessed features. Loading directly …")
        df = pd.read_csv(input_path)

    label_col = "label"
    if label_col not in df.columns:
        print("[ERROR] 'label' column not found!")
        sys.exit(1)
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)

    nb = int((df[label_col] == 0).sum())
    nm = int((df[label_col] == 1).sum())
    total = len(df)
    _log(f"Dataset: {total:,} rows | Benign: {nb:,} ({nb/total:.1%}) | Malicious: {nm:,} ({nm/total:.1%})")

    # ── Identify features ──
    numeric_cols, binary_cols, flag_cols = identify_features(df)
    all_feature_cols = numeric_cols + binary_cols + flag_cols
    _log(f"Features: {len(numeric_cols)} Continuous | {len(binary_cols)} Binary | {len(flag_cols)} Threat Flags")

    # ── Setup theme ──
    setup_premium_theme()

    # ── Compute statistics ──
    _log("Computing continuous feature statistics …")
    df_cont = compute_continuous_stats(df, label_col, numeric_cols)

    _log("Computing binary feature statistics …")
    df_bin = compute_binary_stats(df, label_col, binary_cols)

    _log("Computing threat flag statistics …")
    df_flag = compute_flag_stats(df, label_col, flag_cols)

    _log("Computing Mutual Information scores …")
    mi_scores = compute_mutual_information(df, all_feature_cols, label_col,
                                           args.sample_size if args.sample_size > 0 else 0)

    _log("Computing Feature Power Scores …")
    df_unified = compute_power_scores(df_cont, df_bin, df_flag, mi_scores)
    df_unified = classify_tiers(df_unified)

    # Print tier summary
    tier_counts = df_unified["Tier"].value_counts()
    for t_name, _, _ in TIER_CFG:
        cnt = tier_counts.get(t_name, 0)
        if cnt > 0:
            _log(f"  > {t_name:<12} {cnt:>3} features")

    # ── Generate visualisations ──
    plots_dir = output_dir / "plots"
    _log(f"Generating visualisations …")
    generate_all_plots(df, df_unified, df_cont, df_bin, df_flag, mi_scores,
                       label_col, plots_dir, args.top_n)

    # ── Generate report ──
    _log("Writing comprehensive report …")
    report_path = generate_report(output_dir, df, df_unified, df_cont, df_bin, df_flag,
                                  mi_scores, label_col,
                                  len(numeric_cols), len(binary_cols), len(flag_cols))

    elapsed = time.time() - start_time
    print(f"  [OK] Analysis complete in {elapsed:.1f} seconds")
    print(f"  [OK] Report:  {report_path.resolve()}")
    print(f"  [OK] Plots:   {plots_dir.resolve()}/")
    print("=" * 80)


if __name__ == "__main__":
    main()





'''
Option 1: Run on Raw 5M Dataset (input, label) Processes the raw URLs on-the-fly via multiprocessing before generating stats & plots.

bash
python analyze_feature_distribution.py \
  --input "../Category_balanced_dataset_for_experiment/final_master_5M_balanced.csv" \
  --output-dir "./OUTPUT_5M_RAW" \
  --workers 16 \
  --sample-size 500000
Option 2: Run directly on Preprocessed Train Dataset (e.g., urls_hybrid_train.csv) Bypasses preprocessing entirely and computes stats/plots in seconds.

bash
python analyze_feature_distribution.py \
  --input "/path/to/RESULTS_&_MODELS/2_preprocess_urls_output/urls_hybrid_train.csv" \
  --output-dir "./OUTPUT_TRAIN" \
  --sample-size 500000
Option 3: Run on Combined Preprocessed 5M Dataset Loads all 87 features directly from the full combined preprocessed output file.

bash
python analyze_feature_distribution.py \
  --input "/path/to/combined_preprocessed_5M.csv" \
  --output-dir "./OUTPUT_5M_PREPROCESSED" \
  --sample-size 500000
'''