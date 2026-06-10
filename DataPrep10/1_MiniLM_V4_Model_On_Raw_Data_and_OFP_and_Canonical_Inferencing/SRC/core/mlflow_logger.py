import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from core.config import Config

logger = logging.getLogger("MLflowManager")


class MLflowManager:
    """
    Production-grade MLOps Lifecycle Manager for MLflow tracking.
    
    Provides total exception safety, run resumption continuity via checkpoint run IDs,
    epoch-level batched metric tracing, and structured artifact/model registry logging.
    """
    
    def __init__(self):
        self.enabled = Config.MLFLOW_ENABLED
        self.mlflow = None
        self.active_run = None
        
        if not self.enabled:
            logger.info("MLflow logging is disabled in configuration.")
            return

        try:
            import mlflow
            self.mlflow = mlflow
            logger.info("[MLflow] Library imported successfully.")
        except ImportError:
            logger.warning("[MLflow] mlflow library is not installed. Run: pip install mlflow")
            logger.warning("[MLflow] Falling back to silent local-only logging.")
            self.enabled = False

    def start_run(self, run_id: Optional[str] = None) -> Optional[str]:
        """
        Starts or resumes an MLflow tracking run.
        
        Args:
            run_id: Optional run ID to resume (vital for 50M URLs training continuity).
            
        Returns:
            The active MLflow run_id or None if disabled/failed.
        """
        if not self.enabled or self.mlflow is None:
            return None
            
        try:
            # Configure tracking backend
            self.mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
            
            # Resolve or create experiment
            self.mlflow.set_experiment(Config.MLFLOW_EXPERIMENT_NAME)
            
            # Setup run
            if run_id:
                logger.info(f"[MLflow] Attempting to resume run continuity with ID: {run_id}")
                self.active_run = self.mlflow.start_run(run_id=run_id, run_name=Config.MLFLOW_RUN_NAME)
            else:
                logger.info("[MLflow] Launching a fresh pipeline tracking run.")
                self.active_run = self.mlflow.start_run(run_name=Config.MLFLOW_RUN_NAME)
                
            active_run_id = self.active_run.info.run_id
            logger.info(f"[MLflow] Active tracking Run ID: {active_run_id}")
            
            # Log initial static settings if fresh run
            if not run_id:
                self._log_initial_configurations()
                
            return active_run_id
            
        except Exception as e:
            logger.error(f"[MLflow] Failed to initialize tracking session: {e}")
            logger.warning("[MLflow] Suppressing failure to safeguard execution stability.")
            return None

    def _log_initial_configurations(self):
        """Helper to log hyperparameters and metadata tags."""
        if not self.enabled or self.mlflow is None:
            return
            
        try:
            # 1. Define structural parameters
            params = {
                "SEED": Config.SEED,
                "MODEL_NAME": Config.MODEL_NAME,
                "MAX_LEN": Config.MAX_LEN,
                "NUM_CLASSES": Config.NUM_CLASSES,
                "DROPOUT": Config.DROPOUT,
                "CLASSIFIER_DIMS": str(Config.CLASSIFIER_DIMS),
                "BATCH_SIZE": Config.BATCH_SIZE,
                "NUM_EPOCHS": Config.NUM_EPOCHS,
                "WEIGHT_DECAY": Config.WEIGHT_DECAY,
                "PATIENCE": Config.PATIENCE,
                "GRAD_ACCUM_STEPS": Config.GRAD_ACCUM_STEPS,
                "GRAD_CLIP_NORM": Config.GRAD_CLIP_NORM,
                "LEARNING_RATE": Config.LR,
                "LR_WARMUP_RATIO": Config.LR_WARMUP_RATIO,
                "LR_MIN_RATIO": Config.LR_MIN_RATIO,
                "LORA_R": Config.LORA_R,
                "LORA_ALPHA": Config.LORA_ALPHA,
                "LORA_DROPOUT": Config.LORA_DROPOUT,
                "LORA_TARGET_MODULES": str(Config.LORA_TARGET_MODULES),
                "FOCAL_GAMMA_POS": Config.FOCAL_GAMMA_POS,
                "FOCAL_GAMMA_NEG": Config.FOCAL_GAMMA_NEG,
                "FOCAL_ALPHA": str(Config.FOCAL_ALPHA),
                "LABEL_SMOOTHING": Config.LABEL_SMOOTHING,
                "USE_STRATIFIED_KFOLD": Config.USE_STRATIFIED_KFOLD,
                "KFOLD_SPLITS": Config.KFOLD_SPLITS,
                "USE_WEIGHTED_SAMPLING": Config.USE_WEIGHTED_SAMPLING,
                "PRUNING_RATIO": Config.PRUNING_RATIO,
                "USE_AMP": Config.USE_AMP,
                "ONNX_OPSET": Config.ONNX_OPSET,
                "DEVICE": str(Config.DEVICE),
                "NUM_WORKERS": Config.NUM_WORKERS
            }
            
            # 2. Log parameters in batches
            self.mlflow.log_params(params)
            
            # 3. Log descriptive enterprise tags
            tags = {
                "project": "PhishURL-Detection",
                "architecture": "MiniLM-L12-H384 + LoRA + Focal Loss",
                "framework": "PyTorch 2.x",
                "scale": "Production 50M URL ready",
                "mlops_tool": "MLflow",
                "environment": "Sanity test" if "sanity" in Config.MLFLOW_EXPERIMENT_NAME.lower() else "Production"
            }
            self.mlflow.set_tags(tags)
            logger.info("[MLflow] Initial hyperparameters and metadata tags logged successfully.")
            
        except Exception as e:
            logger.warning(f"[MLflow] Failed to log static configurations: {e}")

    def log_epoch_metrics(self, epoch: int, train_loss: float, train_acc: float, 
                          val_loss: float, val_metrics: Dict[str, Any], threshold: float):
        """Logs metrics at the end of each training epoch."""
        if not self.enabled or self.mlflow is None:
            return
            
        try:
            metrics = {
                "epoch/train_loss": train_loss,
                "epoch/train_accuracy": train_acc,
                "epoch/val_loss": val_loss,
                "epoch/val_accuracy": val_metrics.get("accuracy", 0.0),
                "epoch/val_precision": val_metrics.get("precision", 0.0),
                "epoch/val_recall": val_metrics.get("recall", 0.0),
                "epoch/val_f1": val_metrics.get("f1", 0.0),
                "epoch/val_auc": val_metrics.get("auc", 0.0),
                "epoch/val_fnr": val_metrics.get("fnr", 0.0),
                "epoch/val_fpr": val_metrics.get("fpr", 0.0),
                "epoch/val_tpr": val_metrics.get("tpr", 0.0),
                "epoch/val_tnr": val_metrics.get("tnr", 0.0),
                "epoch/val_fdr": val_metrics.get("fdr", 0.0),
                "epoch/val_balanced_accuracy": val_metrics.get("balanced_accuracy", 0.0),
                "epoch/val_kpi_score": val_metrics.get("kpi_score", 0.0),
                "epoch/val_threshold": threshold
            }
            self.mlflow.log_metrics(metrics, step=epoch)
            logger.info(f"[MLflow] Epoch {epoch} summary metrics tracked successfully.")
            
        except Exception as e:
            logger.warning(f"[MLflow] Failed to track epoch metrics: {e}")

    def log_test_evaluation(self, test_metrics: Dict[str, Any], best_epoch: int, threshold: float):
        """Logs final test dataset performance and metrics."""
        if not self.enabled or self.mlflow is None:
            return
            
        try:
            metrics = {
                "test/accuracy": test_metrics.get("accuracy", 0.0),
                "test/precision": test_metrics.get("precision", 0.0),
                "test/recall": test_metrics.get("recall", 0.0),
                "test/f1_score": test_metrics.get("f1", 0.0),
                "test/auc_roc": test_metrics.get("auc", 0.0),
                "test/fnr": test_metrics.get("fnr", 0.0),
                "test/fpr": test_metrics.get("fpr", 0.0),
                "test/tpr": test_metrics.get("tpr", 0.0),
                "test/tnr": test_metrics.get("tnr", 0.0),
                "test/fdr": test_metrics.get("fdr", 0.0),
                "test/balanced_accuracy": test_metrics.get("balanced_accuracy", 0.0),
                "test/tn": test_metrics.get("tn", 0),
                "test/fp": test_metrics.get("fp", 0),
                "test/fn": test_metrics.get("fn", 0),
                "test/tp": test_metrics.get("tp", 0),
                "test/loss": test_metrics.get("test_loss", 0.0),
                "best_epoch": best_epoch,
                "optimal_threshold": threshold
            }
            self.mlflow.log_metrics(metrics)
            
            # Log structural KPIs as tags for quick dashboard filtering
            self.mlflow.set_tag("kpi_compliance", "PASS" if test_metrics.get("kpi_compliance", False) else "FAIL")
            self.mlflow.set_tag("best_epoch", str(best_epoch))
            logger.info("[MLflow] Final test evaluation dataset metrics logged successfully.")
            
        except Exception as e:
            logger.warning(f"[MLflow] Failed to log final test metrics: {e}")

    def log_artifacts(self, artifact_dir: Path, artifact_path: str = "model_output_artifacts"):
        """Recursively uploads all locally generated artifacts and plots to MLflow."""
        if not self.enabled or self.mlflow is None or not artifact_dir.exists():
            return
            
        try:
            logger.info(f"[MLflow] Archiving all artifacts in: {artifact_dir.name}")
            # Upload files under local directories dynamically to MLflow run artifacts
            self.mlflow.log_artifacts(str(artifact_dir), artifact_path=artifact_path)
            logger.info("[MLflow] Artifacts successfully archived.")
        except Exception as e:
            logger.warning(f"[MLflow] Failed to archive run artifacts: {e}")

    def register_model_version(self, model_name: str, model_uri: str, description: str = ""):
        """Registers a trained model to the central MLflow Model Registry."""
        if not self.enabled or self.mlflow is None:
            return
            
        try:
            logger.info(f"[MLflow] Registering model to Registry: '{model_name}'")
            model_details = self.mlflow.register_model(model_uri, model_name)
            logger.info(f"[MLflow] Model registered successfully! Version: {model_details.version}")
            
            # Optionally add descriptions/metadata to the version
            if description:
                client = self.mlflow.tracking.MlflowClient()
                client.update_model_version(
                    name=model_name,
                    version=model_details.version,
                    description=description
                )
        except Exception as e:
            logger.warning(f"[MLflow] Model Registry registration skipped/failed: {e}")

    def end_run(self):
        """Ends the active MLflow tracking run cleanly."""
        if not self.enabled or self.mlflow is None or self.active_run is None:
            return
            
        try:
            self.mlflow.end_run()
            logger.info("[MLflow] Tracking session closed cleanly.")
            self.active_run = None
        except Exception as e:
            logger.warning(f"[MLflow] Failed to terminate tracking session cleanly: {e}")
