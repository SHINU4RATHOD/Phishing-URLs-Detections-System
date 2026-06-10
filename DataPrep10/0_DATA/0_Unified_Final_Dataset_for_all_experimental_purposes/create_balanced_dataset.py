from __future__ import annotations
import argparse
import csv
import logging
import multiprocessing
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# GLOBAL WORKER INITIALIZATION AND RUNTIME
# ---------------------------------------------------------------------------
_analyzer = None
_tld_cache = {}
_priority_list = []

def init_worker(priority_list: List[str]):
    """Initialize the URLAnalyzer and optimize tldextract inside the worker process."""
    global _analyzer, _priority_list
    _priority_list = priority_list
    
    from urls_cate_V7 import URLAnalyzer, CategoryConfig
    config = CategoryConfig()
    
    # Speed optimization: Cache tldextract calls at hostname level
    original_extract = config.tld_extract
    def optimized_tld_extract(url_or_host: str):
        if "://" in url_or_host or url_or_host.startswith("//"):
            host = url_or_host.split("://", 1)[-1].split("/", 1)[0].split(":", 1)[0]
        else:
            host = url_or_host.split("/", 1)[0].split(":", 1)[0]
            
        if host not in _tld_cache:
            _tld_cache[host] = original_extract(host)
        return _tld_cache[host]
        
    config.tld_extract = optimized_tld_extract
    _analyzer = URLAnalyzer(config)

def classify_batch(urls: List[str]) -> List[str]:
    """Classify a batch of URLs and return their primary categories or 'EXCLUDED'."""
    global _analyzer, _priority_list
    import ipaddress
    from urllib.parse import urlparse
    
    categories = []
    for url in urls:
        url_clean = (url or "").strip()
        if not url_clean:
            categories.append("EXCLUDED")
            continue
            
        try:
            flags = _analyzer.analyze_url(url_clean)
        except Exception:
            categories.append("EXCLUDED")
            continue
            
        # 1. Exclude specific categories (Shortened, Chrome internal, IP-based, File/Local, Malformed)
        excluded_categories = {
            "Shortened_URL",
            "Chrome_Internal_URL",
            "Decimal_Hex_IP_URL",
            "IP_Address_Unusual_Port_URL",
            "File_URL",
            "Structural_Malformation_URL"
        }
        is_excluded = False
        for cat in excluded_categories:
            if flags.get(cat, False):
                is_excluded = True
                break
                
        if is_excluded:
            categories.append("EXCLUDED")
            continue
            
        # 2. Hostname level IP-based and local/private checks
        try:
            if "://" not in url_clean and not url_clean.startswith("//"):
                parsed = urlparse("http://" + url_clean)
            else:
                parsed = urlparse(url_clean)
            host = (parsed.hostname or "").strip().lower()
        except Exception:
            categories.append("EXCLUDED")
            continue
            
        if not host:
            categories.append("EXCLUDED")
            continue
            
        if host in {"localhost", "local", "loopback"}:
            categories.append("EXCLUDED")
            continue
            
        is_ip = False
        is_private = False
        ip_str = host.strip("[]")
        
        try:
            ip_obj = ipaddress.ip_address(ip_str)
            is_ip = True
            is_private = ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_reserved
        except ValueError:
            if host.isdigit():
                try:
                    ip_obj = ipaddress.ip_address(int(host))
                    is_ip = True
                    is_private = ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_reserved
                except ValueError:
                    pass
                    
        if is_ip or is_private:
            categories.append("EXCLUDED")
            continue
            
        # 3. Otherwise, assign primary category
        primary = "UNKNOWN"
        for cat in _priority_list:
            if flags.get(cat, False):
                primary = cat
                break
        categories.append(primary)
        
    return categories


# ---------------------------------------------------------------------------
# WATER-FILLING BALANCING ALGORITHM
# ---------------------------------------------------------------------------
def water_fill_balancing(category_counts: Dict[str, int], target_total: int) -> Dict[str, int]:
    """
    Distribute the target total capacity across categories as evenly as possible,
    constrained by the total available samples in each category.
    """
    categories = list(category_counts.keys())
    targets = {cat: 0 for cat in categories}
    remaining_budget = target_total
    
    # Active categories that still have available capacity and budget
    active_cats = [cat for cat in categories if category_counts[cat] > 0]
    
    while remaining_budget > 0 and active_cats:
        fair_share = remaining_budget // len(active_cats)
        if fair_share == 0:
            # Distribute remaining budget to categories with the most remaining capacity
            active_cats.sort(key=lambda cat: category_counts[cat] - targets[cat], reverse=True)
            for i in range(min(remaining_budget, len(active_cats))):
                targets[active_cats[i]] += 1
            break
            
        new_active_cats = []
        budget_used = 0
        for cat in active_cats:
            available = category_counts[cat]
            current_target = targets[cat]
            if current_target + fair_share >= available:
                added = available - current_target
                targets[cat] = available
                budget_used += added
            else:
                targets[cat] += fair_share
                budget_used += fair_share
                new_active_cats.append(cat)
                
        remaining_budget -= budget_used
        active_cats = new_active_cats
        
    return targets


# ---------------------------------------------------------------------------
# MAIN PIPELINE EXECUTION
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Create a unified, balanced 5M dataset from the 38.2M master dataset.")
    parser.add_argument("--input", type=str, default=r"D:\IIT ROPAR\phishing URL Detection\01_Research Tracker\2_Model_Building\PhishURLDetect-with-LLMS\Dataset\Dataset10\final_master_dataset103_38290035.csv",
                        help="Path to the input master CSV file (input,label columns).")
    parser.add_argument("--output", type=str, default="./final_master_5M_balanced.csv",
                        help="Path to save the output balanced CSV file.")
    parser.add_argument("--target-size", type=int, default=5_000_000,
                        help="Total number of rows for the final balanced dataset.")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of multiprocessing workers to use.")
    parser.add_argument("--chunk-size", type=int, default=500_000,
                        help="Number of rows to read and process per chunk.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    args = parser.parse_args()

    random.seed(args.seed)
    start_time = time.time()

    # 1. Resolve category priorities
    from urls_cate_V7 import URLCategory
    severity_rank = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    priority_list = sorted(
        URLCategory.CATEGORIES.keys(),
        key=lambda cat: (
            severity_rank.get(URLCategory.CATEGORIES[cat].get("severity", "LOW"), 99),
            cat
        )
    )

    print("=" * 70)
    print("      UNIFIED BALANCED 5M DATASET GENERATION PIPELINE")
    print("=" * 70)
    print(f"Input Master CSV: {args.input}")
    print(f"Output Path:      {args.output}")
    print(f"Target Size:      {args.target_size}")
    print(f"Workers Count:    {args.workers}")
    print(f"Chunk Size:       {args.chunk_size}")
    print("-" * 70)

    if not os.path.exists(args.input):
        print(f"[ERROR] Input master CSV file does not exist at: {args.input}")
        sys.exit(1)

    # 2. Scanning Phase (Multiprocessing)
    category_indices = defaultdict(list)
    total_rows = 0

    # Initialize multiprocessing pool
    pool = multiprocessing.Pool(processes=args.workers, initializer=init_worker, initargs=(priority_list,))

    print(f"[{datetime.now()}] Starting scanning phase...")
    
    # Read the master CSV in chunks
    chunk_iterator = pd.read_csv(args.input, chunksize=args.chunk_size, usecols=["input"])
    
    pending_tasks = []
    
    for chunk_idx, chunk_df in enumerate(chunk_iterator):
        urls = chunk_df["input"].astype(str).tolist()
        task = pool.apply_async(classify_batch, args=(urls,))
        pending_tasks.append((chunk_idx, len(urls), task))
        
    print(f"[{datetime.now()}] Submitted all chunks to worker pool. Waiting for results...")
    
    for chunk_idx, num_urls, task in pending_tasks:
        categories = task.get()
        
        # Aggregate indices by category
        start_idx = total_rows
        for i, category in enumerate(categories):
            category_indices[category].append(start_idx + i)
            
        total_rows += num_urls
        print(f"[{datetime.now()}] Chunk {chunk_idx + 1} completed. Scanned: {total_rows} URLs.")

    pool.close()
    pool.join()
    
    print("-" * 70)
    print(f"[{datetime.now()}] Scanning complete! Total scanned rows: {total_rows}")
    print("Raw category counts:")
    for cat in sorted(category_indices.keys()):
        print(f"  - {cat}: {len(category_indices[cat])}")

    # 3. Balancing Phase (Water-Filling)
    category_counts = {cat: len(indices) for cat, indices in category_indices.items() if cat != "EXCLUDED"}
    # Include all 60 categories plus UNKNOWN
    all_categories = priority_list + ["UNKNOWN"]
    for cat in all_categories:
        if cat not in category_counts:
            category_counts[cat] = 0

    print(f"[{datetime.now()}] Running water-filling balancing algorithm...")
    balanced_targets = water_fill_balancing(category_counts, args.target_size)

    print("\nBalanced targets vs Available:")
    for cat in sorted(balanced_targets.keys()):
        print(f"  - {cat}: Target={balanced_targets[cat]} (Available={category_counts[cat]})")

    # Sample indices based on targets
    selected_indices = set()
    for cat, target in balanced_targets.items():
        if target > 0:
            indices = category_indices[cat]
            sampled = random.sample(indices, target)
            selected_indices.update(sampled)

    print(f"\nTotal selected indices: {len(selected_indices)}")
    sorted_selected_indices = sorted(selected_indices)

    # 4. Stream Writing Phase
    print(f"\n[{datetime.now()}] Writing balanced dataset to: {args.output}...")
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.input, "r", encoding="utf-8", newline="") as infile, \
         open(args.output, "w", encoding="utf-8", newline="") as outfile:
        
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        # Write header
        header = next(reader)
        writer.writerow(header)
        
        selected_idx_set = set(sorted_selected_indices)
        
        row_idx = 0
        written_count = 0
        
        for row in reader:
            if row_idx in selected_idx_set:
                writer.writerow(row)
                written_count += 1
            row_idx += 1
            if row_idx % 1_000_000 == 0:
                print(f"[{datetime.now()}] Streamed {row_idx} rows... Written: {written_count}")

    # 5. Write Split Summary Report
    report_path = output_dir / "final_master_5M_balanced_report.txt"
    print(f"[{datetime.now()}] Writing summary report to: {report_path}")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("BALANCED 5M DATASET SUMMARY REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated:       {datetime.now().isoformat()}\n")
        f.write(f"Source Dataset:  {args.input} ({total_rows} rows)\n")
        f.write(f"Output Dataset:  {args.output} ({written_count} rows)\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Category Name':<35} | {'Target (Balanced)':<18} | {'Available (Source)':<18}\n")
        f.write("-" * 80 + "\n")
        for cat in sorted(balanced_targets.keys()):
            f.write(f"{cat:<35} | {balanced_targets[cat]:<18} | {category_counts[cat]:<18}\n")
        f.write("-" * 80 + "\n")
        excluded_count = len(category_indices.get("EXCLUDED", []))
        f.write(f"Total Excluded URLs: {excluded_count}\n")
        f.write("=" * 80 + "\n")

    elapsed_time = time.time() - start_time
    print(f"[{datetime.now()}] Complete! Elapsed time: {elapsed_time/3600:.2f} hours ({elapsed_time:.1f} seconds).")


if __name__ == "__main__":
    # Fix for multiprocessing on Windows
    multiprocessing.freeze_support()
    main()
