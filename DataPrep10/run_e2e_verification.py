# run_e2e_verification.py - Automated End-to-End Pipeline Verification
# Aligned with the 7-Phase verification plan.

import os
import sys
import shutil
import subprocess
from pathlib import Path

# Config
PYTHON_EXE = r"D:\IIT ROPAR\phishing URL Detection\01_Research Tracker\.venv\Scripts\python.exe"
WORKSPACE_ROOT = Path(__file__).resolve().parent

subprojects = [
    {
        "name": "Subproject 1 (MiniLM V4 Raw & Canonical)",
        "path": "1_MiniLM_V4_Model_On_Raw_Data_and_OFP_and_Canonical_Inferencing",
        "preprocess_source": "raw_orig",
        "output_format": "clean"
    },
    {
        "name": "Subproject 2 (MiniLM Hybrid FF V4)",
        "path": "2_MiniLM_Hybrid_FF_V4",
        "preprocess_source": "hybrid",
        "output_format": "hybrid"
    }
]

def clean_historical(sub_path: Path):
    print(f"Cleaning historical outputs for {sub_path.name}...")
    res_models = sub_path / "RESULTS_&_MODELS"
    mlruns = sub_path / "mlruns"
    mlflow_db = sub_path / "mlflow.db"
    stray_preprocess = sub_path / "preprocess_urls_output"
    stray_url_cate = sub_path / "SRC" / "url_categories"
    stray_fold1 = sub_path / "fold_1"

    for path in [res_models, mlruns, stray_preprocess, stray_url_cate, stray_fold1]:
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
    if mlflow_db.exists():
        try:
            mlflow_db.unlink()
        except Exception:
            pass

def run_command(args, cwd):
    print(f"\nRunning command: {' '.join(args)} in {cwd}", flush=True)
    res = subprocess.run(args, cwd=str(cwd), stdout=sys.stdout, stderr=sys.stderr)
    if res.returncode != 0:
        print(f"[FAIL] Command failed with return code {res.returncode}", flush=True)
        sys.exit(res.returncode)
    else:
        print("[OK] Command succeeded.", flush=True)

def main():
    print("======================================================================")
    print(" STARTING E2E VERIFICATION PIPELINE FOR BOTH SUBPROJECTS (PYTHON)")
    print("======================================================================")
    print(f"Python:    {PYTHON_EXE}")
    print(f"Workspace: {WORKSPACE_ROOT}")
    print("======================================================================\n")

    # Clean
    for sub in subprojects:
        clean_historical(WORKSPACE_ROOT / sub["path"])

    # Loop through subprojects
    for sub in subprojects:
        sub_dir = WORKSPACE_ROOT / sub["path"]
        print("\n" + "="*80)
        print(f" EXECUTING PIPELINE FOR: {sub['name']}")
        print("="*80)

        # Phase 1: URL Categorization
        print("\n[PHASE 1] URL Categorization...")
        run_command([
            PYTHON_EXE, "SRC/urls_cate_V7.py",
            "--input", "DATA/3_LNU_Phish1.csv",
            "--output", "RESULTS_&_MODELS/1_url_cate_data10_output"
        ], sub_dir)

        # Phase 2: Preprocessing + Stratified Splitting
        print("\n[PHASE 2] Preprocessing & Stratified Splitting...")
        run_command([
            PYTHON_EXE, "SRC/2_preprocess_urls_v8_refactored.py",
            "--input", "RESULTS_&_MODELS/1_url_cate_data10_output/data_cleaned.csv",
            "--output", "RESULTS_&_MODELS/2_preprocess_urls_output/urls_preprocessed.csv",
            "--enable-split",
            "--split-source", sub["preprocess_source"],
            "--output-format", sub["output_format"],
            "--disable-multiprocessing",
            "--num-workers", "0"
        ], sub_dir)

        # Phase 3: GPU/Auto Training
        print("\n[PHASE 3] Model Training...")
        run_command([
            PYTHON_EXE, "SRC/5_train.py",
            "--config", "SRC/test_config.yaml"
        ], sub_dir)

        # Phase 4: 3-Mode Inference Audit
        print("\n[PHASE 4] 3-Mode Inference Audit...")
        # Mode 1: PyTorch Merged FP32
        print("  -> PyTorch Merged FP32 Inference...")
        run_command([
            PYTHON_EXE, "SRC/6_inference.py",
            "--mode", "inference",
            "--config", "SRC/test_config.yaml"
        ], sub_dir)

        # Mode 2: ONNX FP32
        print("  -> ONNX FP32 Inference...")
        run_command([
            PYTHON_EXE, "SRC/6_inference.py",
            "--mode", "onnx-inference",
            "--onnx-model", "fp32",
            "--config", "SRC/test_config.yaml"
        ], sub_dir)

        # Mode 3: ONNX INT8 Quantized
        print("  -> ONNX INT8 Quantized Inference...")
        run_command([
            PYTHON_EXE, "SRC/6_inference.py",
            "--mode", "onnx-inference",
            "--onnx-model", "int8",
            "--config", "SRC/test_config.yaml"
        ], sub_dir)

        # Phase 5: Threshold Re-Evaluation Sweeps
        print("\n[PHASE 5] Threshold Re-Evaluation Sweeps...")
        run_command([
            PYTHON_EXE, "SRC/7_re_evaluate_thresholds.py",
            "--config", "SRC/test_config.yaml"
        ], sub_dir)

    # Phase 6: MLflow Health Audit
    print("\n" + "="*80)
    print(" [PHASE 6] MLFLOW HEALTH AUDIT")
    print("="*80)
    run_command([PYTHON_EXE, "mlflow_audit.py"], WORKSPACE_ROOT)

    # Phase 7: Pristine Output Tree Audit
    print("\n" + "="*80)
    print(" [PHASE 7] PRISTINE OUTPUT TREE AUDIT")
    print("="*80)
    stray_dirs_found = 0
    for sub in subprojects:
        sub_dir = WORKSPACE_ROOT / sub["path"]
        stray_paths = [
            sub_dir / "preprocess_urls_output",
            sub_dir / "fold_1",
            sub_dir / "SRC" / "url_categories",
            sub_dir / "SRC" / "preprocess_urls_output",
            sub_dir / "SRC" / "fold_1"
        ]
        for path in stray_paths:
            if path.exists():
                print(f"[STRAY DETECTED] STRAY FOLDER DETECTED: {path}")
                stray_dirs_found += 1
    
    if stray_dirs_found == 0:
        print("[PASS] PASS: 0 stray directories found in the workspace roots or SRC folders!")
        print("[PASS] All artifacts are perfectly placed under RESULTS_&_MODELS/")
    else:
        print(f"[FAIL] FAIL: {stray_dirs_found} stray directories found!")
        sys.exit(1)

    print("\n" + "="*80)
    print(" [SUCCESS] ALL PHASES COMPLETED AND VERIFIED SUCCESSFULLY!")
    print("="*80)

if __name__ == "__main__":
    main()
