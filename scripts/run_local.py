"""

You can run in ur laptop:
    python scripts/run_local.py
    python scripts/run_local.py --rules-data ./data/rules.csv

Python 3.11 Compatible
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PROJECT_ROOT)

from endpoints.training.train import run_training, generate_synthetic_data
from endpoints.inference.inference import ODLDInferenceEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("odld-local")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="ODLD Local Runner")
    parser.add_argument("--rules-data", type=str, default=None)
    parser.add_argument("--policy-data", type=str, default=None)
    parser.add_argument("--claims-data", type=str, default=None)
    parser.add_argument(
        "--model-dir", type=str, default="./artifacts/models"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./artifacts/output"
    )
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # ---------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1: TRAINING")
    logger.info("=" * 70)

    # Build training args namespace
    class TrainArgs:
        rules_data = args.rules_data
        policy_data = args.policy_data
        claims_data = args.claims_data
        model_dir = args.model_dir
        output_dir = args.output_dir
        train = None
        use_synthetic = args.rules_data is None
        synthetic_rules = 10000
        synthetic_policy = 50000
        synthetic_claims = 5000

    metrics = run_training(TrainArgs())

    # ---------------------------------------------------------------
    # Inference
    # ---------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2: INFERENCE")
    logger.info("=" * 70)

    engine = ODLDInferenceEngine(args.model_dir)

    # Single prediction examples
    test_cases = [
        {
            "make": "MARUTI", "model": "SWIFT", "fuel_type": "PETROL",
            "vehicle_age": 2, "rto": "MH01", "ncb_pct": 25, "zero_dep_flag": 0,
        },
        {
            "make": "TATA", "model": "NEXON", "fuel_type": "DIESEL",
            "vehicle_age": 9, "rto": "DL05", "ncb_pct": 0, "zero_dep_flag": 1,
        },
        {
            "make": "HYUNDAI", "model": "CRETA", "fuel_type": "PETROL",
            "vehicle_age": 1, "rto": "KA03", "ncb_pct": 50, "zero_dep_flag": 0,
        },
        {
            "make": "MAHINDRA", "model": "THAR", "fuel_type": "DIESEL",
            "vehicle_age": 12, "rto": "UP16", "ncb_pct": 0, "zero_dep_flag": 0,
        },
        {
            "make": "HONDA", "model": "CITY", "fuel_type": "PETROL",
            "vehicle_age": 5, "rto": "TN07", "ncb_pct": 35, "zero_dep_flag": 1,
        },
    ]

    logger.info("\n--- Single Prediction Results ---")
    for i, tc in enumerate(test_cases, 1):
        result = engine.predict_single(tc)
        logger.info(
            f"  [{i}] {tc['make']} {tc['model']} | "
            f"Age={tc['vehicle_age']}yr | RTO={tc['rto']} | "
            f"NCB={tc['ncb_pct']}% | "
            f"→ ODLD={result['ODLD_Pct']}% ({result['Direction']}) "
            f"[{result['Confidence']}]"
        )

    # Bulk prediction
    logger.info("\n--- Bulk Prediction ---")
    bulk_df = pd.DataFrame(test_cases)
    output = engine.predict_dataframe(bulk_df)

    # Save Jarvis-compatible CSV
    jarvis_csv_path = os.path.join(args.output_dir, "jarvis_odld_output.csv")
    output.to_csv(jarvis_csv_path, index=False)
    logger.info(f"Jarvis CSV saved to: {jarvis_csv_path}")

    # Display output
    logger.info("\n--- Output for Jarvis ---")
    logger.info(f"\n{output.to_string(index=False)}")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    # Key metrics
    if "test_classifier" in metrics:
        tc = metrics["test_classifier"]
        logger.info(f"Test Classification:")
        logger.info(f"  Loading Recall:    {tc.get('loading_recall', 'N/A')}")
        logger.info(f"  Loading Precision: {tc.get('loading_precision', 'N/A')}")
        logger.info(f"  Loading F1:        {tc.get('loading_f1', 'N/A')}")
        logger.info(f"  PR-AUC:            {tc.get('pr_auc', 'N/A')}")

    if "test_regression" in metrics:
        tr = metrics["test_regression"]
        logger.info(f"Test Regression:")
        logger.info(f"  MAE:  {tr.get('test_mae', 'N/A')}")
        logger.info(f"  RMSE: {tr.get('test_rmse', 'N/A')}")
        logger.info(f"  R²:   {tr.get('test_r2', 'N/A')}")

    logger.info(f"\nArtifacts saved to: {args.model_dir}")
    logger.info(f"Output saved to: {args.output_dir}")
    logger.info("\nDone!")


if __name__ == "__main__":
    main()
