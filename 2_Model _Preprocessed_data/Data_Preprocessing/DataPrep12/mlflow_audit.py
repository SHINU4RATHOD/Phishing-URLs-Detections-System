import os
import sys
from pathlib import Path

# Ensure we import mlflow
try:
    import mlflow
except ImportError:
    print("[ERROR] mlflow is not installed in the virtualenv. Please check the virtualenv.")
    sys.exit(1)

def audit_subproject(name: str, db_path: Path):
    print(f"\n============================================================")
    print(f" AUDITING MLFLOW DATABASE: {name}")
    print(f"============================================================")
    
    if not db_path.exists():
        print(f"[WARN] Database not found at: {db_path}")
        return
        
    db_uri = f"sqlite:///{db_path.resolve()}"
    mlflow.set_tracking_uri(db_uri)
    
    client = mlflow.tracking.MlflowClient()
    try:
        experiments = client.search_experiments()
        print(f"Found {len(experiments)} Experiments:")
        for exp in experiments:
            print(f"\n  Experiment ID: {exp.experiment_id} | Name: {exp.name}")
            runs = client.search_runs(exp.experiment_id)
            print(f"  Total Runs: {len(runs)}")
            for run in runs:
                print(f"    - Run ID: {run.info.run_id}")
                print(f"      Run Name: {run.info.run_name}")
                print(f"      Status:   {run.info.status}")
                
                # Print params
                params = run.data.params
                print(f"      Logged Params (subset):")
                important_params = ['lr', 'batch_size', 'lora_r', 'epochs', 'device', 'num_epochs', 'focal_gamma_pos', 'focal_gamma_neg', 'focal_alpha']
                for k in important_params:
                    if k in params:
                        print(f"        {k}: {params[k]}")
                        
                # Print metrics
                metrics = run.data.metrics
                print(f"      Logged Metrics (final values):")
                for k, v in sorted(metrics.items()):
                    print(f"        {k}: {v:.6f}")
    except Exception as e:
        print(f"[FAIL] Failed to audit database: {e}")

def main():
    root = Path(__file__).resolve().parent
    sub1 = root / "1_MiniLM_V4_Model_On_Raw_Data_and_OFP_and_Canonical" / "mlflow.db"
    sub2 = root / "2_MiniLM_Hybrid_FF_V4" / "mlflow.db"
    
    audit_subproject("Subproject 1 (MiniLM V4 Raw & Canonical)", sub1)
    audit_subproject("Subproject 2 (MiniLM Hybrid FF V4)", sub2)

if __name__ == "__main__":
    main()
