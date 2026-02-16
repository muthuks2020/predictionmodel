"""
Usage:
  Local:
    python endpoints/training/train.py \
      --rules-data ./data/rules.csv \
      --policy-data ./data/policy.csv \
      --claims-data ./data/claims.csv \
      --model-dir ./artifacts/models \
      --output-dir ./artifacts/output

  SageMaker:
    invoke with /opt/ml/ paths

Python 3.11 Compatible
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add project root to path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import ODLDDataPreprocessor
from src.model import TwoStageODLDModel
from config.model_config import (
    FeatureConfig,
    ClassifierConfig,
    RegressorConfig,
    TrainingConfig,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("odld-training")


# =====================================================================
# Synthetic Data Generator (for testing when real data is unavailable)
# =====================================================================
def generate_synthetic_data(
    n_rules: int = 10000,
    n_policy: int = 50000,
    n_claims: int = 5000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic data matching Zuno's schema for development/testing.
    Replace with actual data loading in production.
    """
    rng = np.random.RandomState(seed)

    makes = ["MARUTI", "HYUNDAI", "TATA", "MAHINDRA", "HONDA", "TOYOTA",
             "KIA", "MG", "SKODA", "VOLKSWAGEN", "FORD", "RENAULT"]
    models_map = {
        "MARUTI": ["SWIFT", "BALENO", "BREZZA", "ALTO", "WAGON_R", "ERTIGA", "DZIRE"],
        "HYUNDAI": ["CRETA", "VENUE", "I20", "I10", "VERNA", "TUCSON"],
        "TATA": ["NEXON", "PUNCH", "HARRIER", "SAFARI", "ALTROZ", "TIAGO"],
        "MAHINDRA": ["XUV700", "THAR", "SCORPIO", "BOLERO", "XUV300"],
        "HONDA": ["CITY", "AMAZE", "WRV", "JAZZ"],
        "TOYOTA": ["FORTUNER", "INNOVA", "GLANZA", "URBAN_CRUISER"],
        "KIA": ["SELTOS", "SONET", "CARENS", "EV6"],
        "MG": ["HECTOR", "ASTOR", "ZS_EV", "GLOSTER"],
        "SKODA": ["KUSHAQ", "SLAVIA", "OCTAVIA", "SUPERB"],
        "VOLKSWAGEN": ["TAIGUN", "VIRTUS", "POLO"],
        "FORD": ["ECOSPORT", "ENDEAVOUR", "FIGO"],
        "RENAULT": ["KIGER", "TRIBER", "KWID"],
    }
    fuel_types = ["PETROL", "DIESEL", "CNG", "ELECTRIC"]
    rto_prefixes = ["MH", "DL", "KA", "TN", "UP", "GJ", "RJ", "WB", "AP", "TS",
                    "KL", "HR", "PB", "MP", "CG", "BR", "OR", "GA"]
    rtos = [f"{p}{rng.randint(1, 50):02d}" for p in rto_prefixes for _ in range(5)]

    def _make_records(n: int, source: str, loading_ratio: float) -> pd.DataFrame:
        chosen_makes = rng.choice(makes, n)
        chosen_models = [rng.choice(models_map[m]) for m in chosen_makes]
        chosen_fuel = rng.choice(fuel_types, n, p=[0.5, 0.3, 0.15, 0.05])
        chosen_rto = rng.choice(rtos, n)
        vehicle_age = rng.exponential(3, n).clip(0, 20).round(0).astype(int)
        ncb_pct = rng.choice([0, 20, 25, 35, 45, 50], n, p=[0.3, 0.15, 0.15, 0.15, 0.15, 0.1])
        zero_dep = rng.choice([0, 1], n, p=[0.65, 0.35])

        # Generate ODLD with controlled loading ratio
        is_loading = rng.random(n) < loading_ratio
        odld = np.where(
            is_loading,
            rng.uniform(1, 40, n).round(1),     # loading: positive
            -rng.uniform(1, 45, n).round(1),     # discount: negative
        )

        # Add some risk-based correlation
        # Older vehicles + certain RTOs more likely to get loading
        risk_boost = (vehicle_age > 7).astype(float) * 0.3
        high_risk_rto = np.isin([r[:2] for r in chosen_rto], ["DL", "MH", "UP"])
        risk_boost += high_risk_rto.astype(float) * 0.2

        # Adjust ODLD based on risk
        odld = np.where(
            risk_boost > 0.3,
            odld + risk_boost * 15,
            odld,
        )

        return pd.DataFrame({
            "make": chosen_makes,
            "model": chosen_models,
            "fuel_type": chosen_fuel,
            "rto": chosen_rto,
            "vehicle_age": vehicle_age,
            "ncb_pct": ncb_pct,
            "zero_dep_flag": zero_dep,
            "odld_pct": odld.round(2),
        })

    # Rules data: healthier ratio (~26% loading, matching Jarvis)
    rules_df = _make_records(n_rules, "rules", loading_ratio=0.26)

    # Policy data: skewed (~2% loading, matching EDA findings)
    policy_df = _make_records(n_policy, "policy", loading_ratio=0.02)

    # Claims data
    claims_makes = rng.choice(makes, n_claims)
    claims_models = [rng.choice(models_map[m]) for m in claims_makes]
    claims_df = pd.DataFrame({
        "make": claims_makes,
        "model": claims_models,
        "fuel_type": rng.choice(fuel_types, n_claims, p=[0.5, 0.3, 0.15, 0.05]),
        "rto": rng.choice(rtos, n_claims),
        "vehicle_age": rng.exponential(4, n_claims).clip(0, 20).round(0).astype(int),
        "ncb_pct": rng.choice([0, 20, 25, 35, 45, 50], n_claims),
        "zero_dep_flag": rng.choice([0, 1], n_claims),
        "claim_amount": rng.exponential(25000, n_claims).round(2),
        "claim_frequency": rng.poisson(1.2, n_claims),
    })

    return rules_df, policy_df, claims_df


# =====================================================================
# Training Pipeline
# =====================================================================
def run_training(args: argparse.Namespace) -> dict:
    """Execute the full training pipeline."""
    logger.info("=" * 70)
    logger.info("ODLD RECOMMENDATION ENGINE - TRAINING PIPELINE")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("=" * 70)

    # Initialize configs
    feature_config = FeatureConfig()
    training_config = TrainingConfig()

    # -------------------------------------------------------------------
    # Load Data
    # -------------------------------------------------------------------
    logger.info("\n--- Step 1: Loading Data ---")
    preprocessor = ODLDDataPreprocessor()

    if args.use_synthetic:
        logger.info("Using SYNTHETIC data for development/testing")
        rules_df, policy_df, claims_df = generate_synthetic_data(
            n_rules=args.synthetic_rules,
            n_policy=args.synthetic_policy,
            n_claims=args.synthetic_claims,
        )
    else:
        rules_df = preprocessor.load_rules_data(args.rules_data)
        policy_df = (
            preprocessor.load_policy_data(args.policy_data)
            if args.policy_data
            else None
        )
        claims_df = (
            preprocessor.load_claims_data(args.claims_data)
            if args.claims_data
            else None
        )

    # -------------------------------------------------------------------
    # Clean Data
    # -------------------------------------------------------------------
    logger.info("\n--- Step 2: Cleaning Data ---")
    rules_df = preprocessor.clean_data(rules_df)
    if policy_df is not None:
        policy_df = preprocessor.clean_data(policy_df)

    # -------------------------------------------------------------------
    # Compute Claims Aggregates
    # -------------------------------------------------------------------
    logger.info("\n--- Step 3: Computing Claims Aggregates ---")
    if claims_df is not None:
        preprocessor.compute_claims_aggregates(claims_df)
    else:
        logger.info("No claims data provided. Skipping aggregation.")

    # -------------------------------------------------------------------
    # Feature Engineering
    # -------------------------------------------------------------------
    logger.info("\n--- Step 4: Feature Engineering ---")
    rules_df = preprocessor.engineer_features(rules_df)
    if policy_df is not None:
        policy_df = preprocessor.engineer_features(policy_df)

    # -------------------------------------------------------------------
    # Data Blending
    # -------------------------------------------------------------------
    logger.info("\n--- Step 5: Blending Training Data ---")
    blended_df = preprocessor.blend_training_data(rules_df, policy_df)

    # Log class distribution
    loading_count = (blended_df["is_loading"] == 1).sum()
    discount_count = (blended_df["is_loading"] == 0).sum()
    logger.info(f"Final class distribution:")
    logger.info(f"  Discount: {discount_count} ({discount_count/len(blended_df):.1%})")
    logger.info(f"  Loading:  {loading_count} ({loading_count/len(blended_df):.1%})")

    # -------------------------------------------------------------------
    # Encode Features
    # -------------------------------------------------------------------
    logger.info("\n--- Step 6: Encoding Features ---")
    all_categorical = [
        col for col in (
            feature_config.CATEGORICAL_FEATURES
            + feature_config.ENGINEERED_FEATURES
        )
        if col in blended_df.columns and blended_df[col].dtype == "object"
    ]
    preprocessor.fit_encoders(blended_df, all_categorical)
    blended_df = preprocessor.transform_features(blended_df, all_categorical)

    # -------------------------------------------------------------------
    # Prepare Feature Matrix
    # -------------------------------------------------------------------
    logger.info("\n--- Step 7: Preparing Feature Matrix ---")
    available_features = [
        f for f in feature_config.all_input_features if f in blended_df.columns
    ]
    preprocessor.feature_names = available_features

    X = blended_df[available_features].copy()
    y_class = blended_df["is_loading"].copy()
    y_reg = blended_df["odld_pct"].copy()

    logger.info(f"Feature matrix: {X.shape}")
    logger.info(f"Features used: {available_features}")

    # -------------------------------------------------------------------
    # Train-Validation-Test Split
    # -------------------------------------------------------------------
    logger.info("\n--- Step 8: Splitting Data ---")

    # First split: train+val vs test
    X_trainval, X_test, y_class_trainval, y_class_test, y_reg_trainval, y_reg_test = (
        train_test_split(
            X, y_class, y_reg,
            test_size=training_config.test_size,
            stratify=y_class,
            random_state=training_config.random_state,
        )
    )

    # Second split: train vs val
    X_train, X_val, y_class_train, y_class_val, y_reg_train, y_reg_val = (
        train_test_split(
            X_trainval, y_class_trainval, y_reg_trainval,
            test_size=training_config.validation_size,
            stratify=y_class_trainval,
            random_state=training_config.random_state,
        )
    )

    logger.info(f"Train:      {X_train.shape[0]} samples")
    logger.info(f"Validation: {X_val.shape[0]} samples")
    logger.info(f"Test:       {X_test.shape[0]} samples")
    logger.info(
        f"Train loading ratio: "
        f"{(y_class_train == 1).mean():.1%}"
    )

    # -------------------------------------------------------------------
    # SMOTE (Optional, training set only)
    # -------------------------------------------------------------------
    if training_config.apply_smote:
        logger.info("\n--- Step 9: Applying SMOTE ---")
        try:
            from imblearn.over_sampling import SMOTE

            smote = SMOTE(
                sampling_strategy=training_config.smote_sampling_strategy,
                random_state=training_config.random_state,
            )
            X_train_res, y_class_train_res = smote.fit_resample(X_train, y_class_train)

            # Align regression target with resampled indices
            # For SMOTE-generated samples, use mean ODLD of the class
            n_original = len(X_train)
            n_synthetic = len(X_train_res) - n_original

            if n_synthetic > 0:
                loading_mean_odld = y_reg_train[y_class_train == 1].mean()
                synthetic_y_reg = pd.Series(
                    [loading_mean_odld] * n_synthetic, dtype=float
                )
                y_reg_train_res = pd.concat(
                    [y_reg_train.reset_index(drop=True), synthetic_y_reg],
                    ignore_index=True,
                )
                X_train = pd.DataFrame(X_train_res, columns=X_train.columns)
                y_class_train = pd.Series(y_class_train_res)
                y_reg_train = y_reg_train_res

                logger.info(
                    f"SMOTE applied: {n_original} -> {len(X_train)} samples "
                    f"({n_synthetic} synthetic loading samples added)"
                )
            else:
                logger.info("SMOTE: no synthetic samples needed")
        except ImportError:
            logger.warning(
                "imbalanced-learn not installed. Skipping SMOTE. "
                "Install with: pip install imbalanced-learn"
            )

    # -------------------------------------------------------------------
    # Train Model
    # -------------------------------------------------------------------
    logger.info("\n--- Step 10: Training Two-Stage XGBoost ---")

    model = TwoStageODLDModel()
    metrics = model.train(
        X_train=X_train,
        y_class_train=y_class_train,
        y_reg_train=y_reg_train,
        X_val=X_val,
        y_class_val=y_class_val,
        y_reg_val=y_reg_val,
    )

    # -------------------------------------------------------------------
    # Evaluate on Test Set
    # -------------------------------------------------------------------
    logger.info("\n--- Step 11: Test Set Evaluation ---")

    test_predictions = model.predict(X_test, return_details=True)

    # Classification metrics on test
    y_test_pred_class = (test_predictions["direction"] == "LOADING").astype(int)
    test_clf_report = model._compute_classifier_metrics(
        y_class_test,
        y_test_pred_class.values,
        test_predictions["loading_probability"].values,
    )
    metrics["test_classifier"] = test_clf_report

    # Regression metrics on test
    test_reg_metrics = model._compute_regressor_metrics(
        y_reg_test, test_predictions["odld_pct"].values, "test"
    )
    metrics["test_regression"] = test_reg_metrics

    logger.info("Test Set Classification Metrics:")
    for k, v in test_clf_report.items():
        if k != "confusion_matrix":
            logger.info(f"  {k}: {v}")

    logger.info("Test Set Regression Metrics:")
    for k, v in test_reg_metrics.items():
        logger.info(f"  {k}: {v}")

    # -------------------------------------------------------------------
    # Feature Importance
    # -------------------------------------------------------------------
    logger.info("\n--- Step 12: Feature Importance ---")
    importance = model.get_feature_importance()
    logger.info("Top 5 features per model:")
    for model_name, imp_dict in importance.items():
        top5 = list(imp_dict.items())[:5]
        logger.info(f"  {model_name}:")
        for feat, score in top5:
            logger.info(f"    {feat}: {score:.4f}")

    # -------------------------------------------------------------------
    # Save Artifacts
    # -------------------------------------------------------------------
    logger.info("\n--- Step 13: Saving Artifacts ---")

    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)

    # Save model
    model.save(model_dir)

    # Save preprocessor
    preprocessor.save(os.path.join(model_dir, "preprocessor.pkl"))

    # Save training metrics
    metrics_path = os.path.join(model_dir, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    # Save feature importance
    importance_path = os.path.join(model_dir, "feature_importance.json")
    with open(importance_path, "w") as f:
        json.dump(importance, f, indent=2)

    # Save sample predictions for validation
    sample_output = pd.concat(
        [X_test.reset_index(drop=True), test_predictions.reset_index(drop=True)],
        axis=1,
    ).head(100)
    sample_path = os.path.join(model_dir, "sample_predictions.csv")
    sample_output.to_csv(sample_path, index=False)

    logger.info(f"All artifacts saved to {model_dir}")
    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)

    return metrics


# =====================================================================
# CLI Entry Point
# =====================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ODLD Recommendation Engine - Training Pipeline"
    )

    # Data paths
    parser.add_argument(
        "--rules-data", type=str, default=None,
        help="Path to Rules Data file (CSV/XLSX)",
    )
    parser.add_argument(
        "--policy-data", type=str, default=None,
        help="Path to Policy Data file (CSV/XLSX)",
    )
    parser.add_argument(
        "--claims-data", type=str, default=None,
        help="Path to Claims Data file (CSV/XLSX)",
    )

    # Output paths (SageMaker compatible)
    parser.add_argument(
        "--model-dir", type=str,
        default=os.environ.get("SM_MODEL_DIR", "./artifacts/models"),
        help="Directory to save model artifacts",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=os.environ.get("SM_OUTPUT_DIR", "./artifacts/output"),
        help="Directory for training output",
    )

    # SageMaker channel paths
    parser.add_argument(
        "--train", type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN", None),
        help="SageMaker training data channel",
    )

    # Synthetic data options (for development)
    parser.add_argument(
        "--use-synthetic", action="store_true", default=False,
        help="Use synthetic data for testing",
    )
    parser.add_argument("--synthetic-rules", type=int, default=10000)
    parser.add_argument("--synthetic-policy", type=int, default=50000)
    parser.add_argument("--synthetic-claims", type=int, default=5000)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # SageMaker compatibility: check channel paths
    if args.train and not args.rules_data:
        # SageMaker mode: look for data in channel directory
        channel_dir = args.train
        for f in os.listdir(channel_dir):
            if "rules" in f.lower():
                args.rules_data = os.path.join(channel_dir, f)
            elif "policy" in f.lower():
                args.policy_data = os.path.join(channel_dir, f)
            elif "claims" in f.lower():
                args.claims_data = os.path.join(channel_dir, f)

    # If no data provided, use synthetic
    if not args.rules_data:
        logger.info("No data files provided. Using synthetic data.")
        args.use_synthetic = True

    metrics = run_training(args)
