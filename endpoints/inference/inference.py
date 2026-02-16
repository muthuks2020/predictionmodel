
import os
import sys
import io
import json
import logging
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import ODLDDataPreprocessor
from src.model import TwoStageODLDModel
from config.model_config import FeatureConfig, InferenceConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("odld-inference")


# =====================================================================
# Global Model Store (loaded once, reused across requests)
# =====================================================================
_model: Optional[TwoStageODLDModel] = None
_preprocessor: Optional[ODLDDataPreprocessor] = None
_feature_config: Optional[FeatureConfig] = None
_inference_config: Optional[InferenceConfig] = None


# =====================================================================
# SageMaker Interface Functions
# =====================================================================
def model_fn(model_dir: str) -> dict:
    """
    SageMaker model loading function.
    Called once when the endpoint starts up.
    """
    global _model, _preprocessor, _feature_config, _inference_config

    logger.info(f"Loading model from {model_dir}")

    _feature_config = FeatureConfig()
    _inference_config = InferenceConfig()

    # Load model
    _model = TwoStageODLDModel()
    _model.load(model_dir)

    # Load preprocessor
    preprocessor_path = os.path.join(model_dir, "preprocessor.pkl")
    _preprocessor = ODLDDataPreprocessor()
    _preprocessor.load(preprocessor_path)

    logger.info("Model and preprocessor loaded successfully")
    logger.info(f"Feature names: {_model.feature_names}")

    return {
        "model": _model,
        "preprocessor": _preprocessor,
        "feature_config": _feature_config,
        "inference_config": _inference_config,
    }


def input_fn(request_body: str, request_content_type: str) -> pd.DataFrame:
    """
    SageMaker input deserialization.
    Supports JSON and CSV input formats.
    """
    logger.info(f"Parsing input. Content-Type: {request_content_type}")

    if request_content_type == "application/json":
        return _parse_json_input(request_body)
    elif request_content_type in ("text/csv", "application/csv"):
        return _parse_csv_input(request_body)
    else:
        raise ValueError(
            f"Unsupported content type: {request_content_type}. "
            f"Use 'application/json' or 'text/csv'."
        )


def predict_fn(input_data: pd.DataFrame, model_dict: dict) -> pd.DataFrame:
    """
    SageMaker prediction function.
    Runs the two-stage model pipeline.
    """
    model = model_dict["model"]
    preprocessor = model_dict["preprocessor"]
    feature_config = model_dict["feature_config"]
    inference_config = model_dict["inference_config"]

    logger.info(f"Predicting for {len(input_data)} records")

    # Preserve original input for output
    original_input = input_data.copy()

    # Preprocess
    X = preprocessor.prepare_inference_data(input_data, feature_config)

    # Predict
    predictions = model.predict(X, return_details=True)

    # Apply ODLD bounds (safety guardrail)
    predictions["odld_pct"] = predictions["odld_pct"].clip(
        inference_config.min_odld_pct,
        inference_config.max_odld_pct,
    )

    # Round to slabs if configured
    if inference_config.odld_slabs:
        predictions["odld_pct"] = predictions["odld_pct"].apply(
            lambda x: _snap_to_nearest_slab(x, inference_config.odld_slabs)
        )

    # Build output DataFrame matching Ankit's spec
    output = _build_output(original_input, predictions)

    logger.info(f"Prediction complete. Output shape: {output.shape}")
    return output


def output_fn(prediction: pd.DataFrame, accept: str) -> str:
    """
    SageMaker output serialization.
    Supports JSON and CSV output formats.
    """
    if accept == "application/json":
        return prediction.to_json(orient="records", indent=2)
    elif accept in ("text/csv", "application/csv", "*/*"):
        return prediction.to_csv(index=False)
    else:
        raise ValueError(f"Unsupported accept type: {accept}")


# =====================================================================
# Input Parsers
# =====================================================================
def _parse_json_input(body: str) -> pd.DataFrame:
    """
    Parse JSON input.

    Supports:
    - Single record: {"make": "MARUTI", "model": "SWIFT", ...}
    - Multiple records: [{"make": "MARUTI", ...}, {"make": "HYUNDAI", ...}]
    """
    data = json.loads(body)

    if isinstance(data, dict):
        # Single record
        return pd.DataFrame([data])
    elif isinstance(data, list):
        # Multiple records
        return pd.DataFrame(data)
    else:
        raise ValueError("JSON input must be a dict (single) or list (bulk).")


def _parse_csv_input(body: str) -> pd.DataFrame:
    """Parse CSV input."""
    return pd.read_csv(io.StringIO(body))


# =====================================================================
# Output Builder
# =====================================================================
def _build_output(
    original_input: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
 
    col_map = {
        "make": "Make",
        "model": "Model",
        "fuel_type": "Fuel_Type",
        "vehicle_age": "Vehicle_Age",
        "rto": "RTO",
        "ncb_pct": "NCB_Pct",
        "zero_dep_flag": "Zero_Dep_Flag",
    }

    output = pd.DataFrame()

    # Echo input features with proper names
    for raw_col, display_col in col_map.items():
        if raw_col in original_input.columns:
            output[display_col] = original_input[raw_col].values
        else:
            # Try case-insensitive match
            matched = [
                c for c in original_input.columns if c.lower() == raw_col.lower()
            ]
            if matched:
                output[display_col] = original_input[matched[0]].values
            else:
                output[display_col] = "N/A"

    # Add predictions
    output["ODLD_Pct"] = predictions["odld_pct"].values
    output["Direction"] = predictions["direction"].values
    output["Confidence"] = predictions["confidence"].values
    output["Loading_Probability"] = predictions["loading_probability"].values

    return output


# =====================================================================
# Helper Functions
# =====================================================================
def _snap_to_nearest_slab(value: float, slabs: list[float]) -> float:
    """Snap ODLD value to nearest valid slab."""
    slabs_arr = np.array(slabs)
    idx = np.argmin(np.abs(slabs_arr - value))
    return float(slabs_arr[idx])


# =====================================================================
# Standalone Inference Runner (for local testing)
# =====================================================================
class ODLDInferenceEngine:
    """
    Standalone inference engine for local usage and testing.
    Wraps the SageMaker interface functions.
    """

    def __init__(self, model_dir: str):
        """Load model from directory."""
        self.model_dict = model_fn(model_dir)
        logger.info("Inference engine initialized")

    def predict_single(self, features: dict) -> dict:
       
        input_df = pd.DataFrame([features])
        output = predict_fn(input_df, self.model_dict)
        return output.iloc[0].to_dict()

    def predict_bulk(self, input_path: str, output_path: str) -> pd.DataFrame:
        
        logger.info(f"Bulk prediction from {input_path}")
        input_df = pd.read_csv(input_path)
        output = predict_fn(input_df, self.model_dict)
        output.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")
        return output

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict ODLD for a DataFrame."""
        return predict_fn(df, self.model_dict)


# =====================================================================
# CLI for Local Testing
# =====================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="ODLD Model - Inference Runner"
    )
    parser.add_argument(
        "--model-dir", type=str, required=True,
        help="Path to model artifacts directory",
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Input CSV file for bulk prediction",
    )
    parser.add_argument(
        "--output", type=str, default="./predictions.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--single", action="store_true",
        help="Run single prediction interactively",
    )

    args = parser.parse_args()

    engine = ODLDInferenceEngine(args.model_dir)

    if args.single:
        # Interactive single prediction
        print("\n--- ODLD Single Prediction ---")
        features = {
            "make": input("Make (e.g., MARUTI): ").strip().upper(),
            "model": input("Model (e.g., SWIFT): ").strip().upper(),
            "fuel_type": input("Fuel Type (PETROL/DIESEL/CNG/ELECTRIC): ").strip().upper(),
            "vehicle_age": int(input("Vehicle Age (years): ")),
            "rto": input("RTO (e.g., MH01): ").strip().upper(),
            "ncb_pct": float(input("NCB % (0/20/25/35/45/50): ")),
            "zero_dep_flag": int(input("Zero Dep (0/1): ")),
        }
        result = engine.predict_single(features)
        print("\n--- Prediction Result ---")
        for k, v in result.items():
            print(f"  {k}: {v}")

    elif args.input:
        # Bulk prediction
        engine.predict_bulk(args.input, args.output)
        print(f"Predictions saved to {args.output}")

    else:
        # Demo prediction
        print("\n--- Demo Predictions ---")
        demo_data = pd.DataFrame([
            {"make": "MARUTI", "model": "SWIFT", "fuel_type": "PETROL",
             "vehicle_age": 3, "rto": "MH01", "ncb_pct": 25, "zero_dep_flag": 0},
            {"make": "TATA", "model": "NEXON", "fuel_type": "DIESEL",
             "vehicle_age": 8, "rto": "DL05", "ncb_pct": 0, "zero_dep_flag": 1},
            {"make": "HYUNDAI", "model": "CRETA", "fuel_type": "PETROL",
             "vehicle_age": 1, "rto": "KA03", "ncb_pct": 50, "zero_dep_flag": 0},
            {"make": "MAHINDRA", "model": "THAR", "fuel_type": "DIESEL",
             "vehicle_age": 12, "rto": "UP16", "ncb_pct": 0, "zero_dep_flag": 0},
        ])

        output = engine.predict_dataframe(demo_data)
        print(output.to_string(index=False))
