"""
Data Preprocessing & Feature Engineering


Python 3.11 Compatible
"""

import os
import logging
import json
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

logger = logging.getLogger(__name__)



# Vehicle Age Bucketing

def bucket_vehicle_age(age: float) -> str:
    """Bucket vehicle age into underwriter-friendly categories."""
    if age <= 0:
        return "NEW"
    elif age <= 3:
        return "1-3YR"
    elif age <= 7:
        return "4-7YR"
    elif age <= 10:
        return "8-10YR"
    else:
        return "10+YR"



# NCB Tier Mapping (Standard Indian Motor Insurance NCB Slabs)

def map_ncb_tier(ncb_pct: float) -> str:
    """Map NCB percentage to standard tier."""
    if ncb_pct <= 0:
        return "ZERO"
    elif ncb_pct <= 20:
        return "T1_20"
    elif ncb_pct <= 25:
        return "T2_25"
    elif ncb_pct <= 35:
        return "T3_35"
    elif ncb_pct <= 45:
        return "T4_45"
    else:
        return "T5_50"



# RTO Zone Clustering

def extract_rto_state(rto_code: str) -> str:
    """Extract state prefix from RTO code (e.g., MH01 -> MH)."""
    if not isinstance(rto_code, str) or len(rto_code) < 2:
        return "UNKNOWN"
    return rto_code[:2].upper()


# High-risk RTO zones based on typical Indian motor insurance claim patterns
RTO_RISK_ZONES = {
    "MH": "WEST_HIGH",    # Maharashtra - high traffic density
    "DL": "NORTH_HIGH",   # Delhi - high traffic density
    "KA": "SOUTH_MED",    # Karnataka
    "TN": "SOUTH_HIGH",   # Tamil Nadu
    "UP": "NORTH_HIGH",   # Uttar Pradesh
    "GJ": "WEST_MED",     # Gujarat
    "RJ": "NORTH_MED",    # Rajasthan
    "WB": "EAST_MED",     # West Bengal
    "AP": "SOUTH_MED",    # Andhra Pradesh
    "TS": "SOUTH_MED",    # Telangana
    "KL": "SOUTH_MED",    # Kerala
    "HR": "NORTH_MED",    # Haryana
    "PB": "NORTH_MED",    # Punjab
    "MP": "CENTRAL_LOW",  # Madhya Pradesh
    "CG": "CENTRAL_LOW",  # Chhattisgarh
    "BR": "EAST_LOW",     # Bihar
    "OR": "EAST_LOW",     # Odisha
    "JH": "EAST_LOW",     # Jharkhand
    "GA": "WEST_LOW",     # Goa
    "AS": "EAST_LOW",     # Assam
}


def map_rto_zone(rto_code: str) -> str:
    """Map RTO code to risk zone."""
    state = extract_rto_state(rto_code)
    return RTO_RISK_ZONES.get(state, "OTHER")



# Main Preprocessing Class

class ODLDDataPreprocessor:
    """
    Preprocesses raw data for the ODLD two-stage XGBoost model.

    Handles:
    1. Loading and merging Rules, Policy, and Claims data
    2. Feature engineering
    3. Encoding categorical features
    4. Claims-derived feature aggregation
    5. Training data blending
    """

    def __init__(self):
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.ordinal_encoder: Optional[OrdinalEncoder] = None
        self.claims_aggregates: Optional[pd.DataFrame] = None
        self.feature_names: list[str] = []
        self._fitted = False

   
    # 1. Data Loading
   
    def load_rules_data(self, filepath: str) -> pd.DataFrame:
        """Load Rules Data from Jarvis export."""
        logger.info(f"Loading Rules Data from {filepath}")
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".csv":
            df = pd.read_csv(filepath)
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(filepath)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        df = self._standardize_columns(df)
        df["data_source"] = "rules"
        logger.info(f"Rules Data loaded: {len(df)} records")
        return df

    def load_policy_data(self, filepath: str) -> pd.DataFrame:
        """Load Policy Data."""
        logger.info(f"Loading Policy Data from {filepath}")
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".csv":
            df = pd.read_csv(filepath)
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(filepath)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        df = self._standardize_columns(df)
        df["data_source"] = "policy"
        logger.info(f"Policy Data loaded: {len(df)} records")
        return df

    def load_claims_data(self, filepath: str) -> pd.DataFrame:
        """Load Claims Data for deriving segment-level features."""
        logger.info(f"Loading Claims Data from {filepath}")
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".csv":
            df = pd.read_csv(filepath)
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(filepath)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        df = self._standardize_columns(df)
        logger.info(f"Claims Data loaded: {len(df)} records")
        return df

   
    # 2. Column Standardization
   
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to match our feature spec."""
        # Normalize column names
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(r"[^a-z0-9]+", "_", regex=True)
            .str.strip("_")
        )

        # Common column mappings (adapt these to actual Zuno schema)
        column_map = {
            "vehicle_make": "make",
            "car_make": "make",
            "manufacturer": "make",
            "vehicle_model": "model",
            "car_model": "model",
            "fuel": "fuel_type",
            "req_fuel_type": "fuel_type",
            "fueltype": "fuel_type",
            "age_of_vehicle": "vehicle_age",
            "req_vehicle_age": "vehicle_age",
            "vehicleage": "vehicle_age",
            "vehicle_age_years": "vehicle_age",
            "rto_code": "rto",
            "rto_location": "rto",
            "ncb": "ncb_pct",
            "ncb_percentage": "ncb_pct",
            "ncb_percent": "ncb_pct",
            "zero_dep": "zero_dep_flag",
            "zero_dep_addon": "zero_dep_flag",
            "is_zero_dep": "zero_dep_flag",
            "flag_of_zero_dep_add_on": "zero_dep_flag",
            "odld": "odld_pct",
            "odld_percentage": "odld_pct",
            "odld_percent": "odld_pct",
            "discount_loading": "odld_pct",
            "claims_amount": "claim_amount",
            "total_claim_amount": "claim_amount",
            "claim_amt": "claim_amount",
            "claims_frequency": "claim_frequency",
            "claim_count": "claim_frequency",
            "claim_freq": "claim_frequency",
        }

        df = df.rename(columns=column_map)
        return df

   
    # 3. Claims Aggregation (Segment-Level Features)
   
    def compute_claims_aggregates(self, claims_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute segment-level claims features.
        Aggregated at Make-Model-FuelType-RTO level for use as enrichment features.
        """
        logger.info("Computing claims aggregates at segment level...")

        # Ensure required columns exist
        segment_cols = ["make", "model", "fuel_type", "rto"]
        available_cols = [c for c in segment_cols if c in claims_df.columns]

        if not available_cols:
            logger.warning("No segment columns found in claims data. Skipping aggregation.")
            return pd.DataFrame()

        # Aggregate at available segment level
        agg_dict = {}
        if "claim_amount" in claims_df.columns:
            agg_dict["claim_amount"] = ["mean", "sum", "count"]
        if "claim_frequency" in claims_df.columns:
            agg_dict["claim_frequency"] = ["mean", "sum"]

        if not agg_dict:
            logger.warning("No claims metrics found for aggregation.")
            return pd.DataFrame()

        agg_df = claims_df.groupby(available_cols, observed=True).agg(agg_dict)
        agg_df.columns = ["_".join(col).strip("_") for col in agg_df.columns]
        agg_df = agg_df.reset_index()

        # Rename to standardized names
        rename_map = {}
        for col in agg_df.columns:
            if "claim_amount_mean" in col:
                rename_map[col] = "segment_avg_claim_amount"
            elif "claim_amount_count" in col:
                rename_map[col] = "segment_claim_count"
            elif "claim_frequency_mean" in col:
                rename_map[col] = "segment_claim_frequency"

        agg_df = agg_df.rename(columns=rename_map)
        self.claims_aggregates = agg_df
        logger.info(f"Claims aggregates computed for {len(agg_df)} segments")
        return agg_df

   
    # 4. Feature Engineering
   
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering transformations."""
        logger.info("Engineering features...")

        # Make-Model interaction
        if "make" in df.columns and "model" in df.columns:
            df["make_model"] = (
                df["make"].astype(str).str.upper().str.strip()
                + "_"
                + df["model"].astype(str).str.upper().str.strip()
            )

        # Vehicle age bucketing
        if "vehicle_age" in df.columns:
            df["vehicle_age_bucket"] = df["vehicle_age"].apply(bucket_vehicle_age)

        # NCB tier mapping
        if "ncb_pct" in df.columns:
            df["ncb_tier"] = df["ncb_pct"].apply(map_ncb_tier)

        # RTO zone mapping
        if "rto" in df.columns:
            df["rto_zone"] = df["rto"].astype(str).apply(map_rto_zone)

        # Merge claims aggregates if available
        if self.claims_aggregates is not None and len(self.claims_aggregates) > 0:
            segment_cols = ["make", "model", "fuel_type", "rto"]
            available_cols = [c for c in segment_cols if c in df.columns and c in self.claims_aggregates.columns]
            if available_cols:
                merge_cols = available_cols + [
                    c for c in self.claims_aggregates.columns
                    if c.startswith("segment_")
                ]
                df = df.merge(
                    self.claims_aggregates[merge_cols].drop_duplicates(),
                    on=available_cols,
                    how="left",
                )

        # Fill missing claims features with 0 (no claims signal)
        for col in ["segment_claim_frequency", "segment_avg_claim_amount"]:
            if col in df.columns:
                df[col] = df[col].fillna(0)
            else:
                df[col] = 0

        # Create classification target
        if "odld_pct" in df.columns:
            df["is_loading"] = (df["odld_pct"] > 0).astype(int)

        logger.info(f"Feature engineering complete. Shape: {df.shape}")
        return df

   
    # 5. Data Cleaning
   
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate raw data."""
        logger.info(f"Cleaning data. Initial shape: {df.shape}")

        # Standardize string columns
        str_cols = ["make", "model", "fuel_type", "rto"]
        for col in str_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.upper().str.strip()
                df[col] = df[col].replace({"NAN": "UNKNOWN", "": "UNKNOWN", "NONE": "UNKNOWN"})

        # Ensure numeric columns are numeric
        num_cols = ["vehicle_age", "ncb_pct", "zero_dep_flag", "odld_pct"]
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Handle zero_dep_flag as binary
        if "zero_dep_flag" in df.columns:
            df["zero_dep_flag"] = df["zero_dep_flag"].fillna(0).astype(int).clip(0, 1)

        # Handle missing values
        if "vehicle_age" in df.columns:
            df["vehicle_age"] = df["vehicle_age"].fillna(df["vehicle_age"].median())

        if "ncb_pct" in df.columns:
            df["ncb_pct"] = df["ncb_pct"].fillna(0)

        # Drop rows with missing target (only for training data)
        if "odld_pct" in df.columns:
            initial_len = len(df)
            df = df.dropna(subset=["odld_pct"])
            dropped = initial_len - len(df)
            if dropped > 0:
                logger.warning(f"Dropped {dropped} rows with missing ODLD target")

        logger.info(f"Cleaning complete. Final shape: {df.shape}")
        return df

   
    # 6. Data Blending (Rules + Policy with imbalance-aware sampling)
   
    def blend_training_data(
        self,
        rules_df: pd.DataFrame,
        policy_df: Optional[pd.DataFrame] = None,
        target_loading_ratio: float = 0.20,
    ) -> pd.DataFrame:
        """
        Blend Rules and Policy data with deliberate overweighting of loading cases.

        Strategy:
        - Use Rules Data as primary source (healthier 3:1 ratio)
        - Downsample Policy Data discount class to match Rules distribution
        - Target ~20% loading representation in final training set
        """
        logger.info("Blending training data...")

        # Start with all rules data
        blended = rules_df.copy()
        logger.info(
            f"Rules Data: {len(rules_df)} records, "
            f"Loading: {(rules_df['is_loading'] == 1).sum()} "
            f"({(rules_df['is_loading'] == 1).mean():.1%})"
        )

        if policy_df is not None and len(policy_df) > 0:
            # Separate policy loading and discount
            policy_loading = policy_df[policy_df["is_loading"] == 1]
            policy_discount = policy_df[policy_df["is_loading"] == 0]

            # Include ALL loading cases from policy (we need every one)
            blended = pd.concat([blended, policy_loading], ignore_index=True)

            # Downsample policy discounts to maintain target ratio
            total_loading = (blended["is_loading"] == 1).sum()
            target_total = int(total_loading / target_loading_ratio)
            target_discount = target_total - total_loading
            current_discount = (blended["is_loading"] == 0).sum()

            if current_discount < target_discount and len(policy_discount) > 0:
                additional_needed = min(
                    target_discount - current_discount, len(policy_discount)
                )
                sampled_discount = policy_discount.sample(
                    n=additional_needed, random_state=42
                )
                blended = pd.concat([blended, sampled_discount], ignore_index=True)

            logger.info(
                f"After blending with Policy Data: "
                f"Policy Loading added: {len(policy_loading)}, "
                f"Policy Discount sampled: {len(blended) - len(rules_df) - len(policy_loading)}"
            )

        # Final stats
        loading_ratio = (blended["is_loading"] == 1).mean()
        logger.info(
            f"Blended dataset: {len(blended)} records, "
            f"Loading ratio: {loading_ratio:.1%}"
        )

        return blended.reset_index(drop=True)

   
    # 7. Encoding
   
    def fit_encoders(self, df: pd.DataFrame, categorical_features: list[str]) -> None:
        """Fit label encoders on training data."""
        logger.info("Fitting encoders...")
        self.label_encoders = {}

        for col in categorical_features:
            if col in df.columns:
                le = LabelEncoder()
                # Add UNKNOWN for handling unseen categories at inference
                unique_vals = list(df[col].astype(str).unique()) + ["UNKNOWN"]
                le.fit(unique_vals)
                self.label_encoders[col] = le
                logger.info(f"  {col}: {len(le.classes_)} categories")

        self._fitted = True

    def transform_features(
        self, df: pd.DataFrame, categorical_features: list[str]
    ) -> pd.DataFrame:
        """Transform categorical features using fitted encoders."""
        if not self._fitted:
            raise RuntimeError("Encoders not fitted. Call fit_encoders() first.")

        df = df.copy()
        for col in categorical_features:
            if col in df.columns and col in self.label_encoders:
                le = self.label_encoders[col]
                # Handle unseen categories
                df[col] = df[col].astype(str).apply(
                    lambda x: x if x in le.classes_ else "UNKNOWN"
                )
                df[col] = le.transform(df[col])

        return df

   
    # 8. Full Pipeline
   
    def prepare_training_data(
        self,
        rules_path: str,
        policy_path: Optional[str] = None,
        claims_path: Optional[str] = None,
        feature_config=None,
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Full preprocessing pipeline for training.

        Returns:
            X: Feature matrix
            y_class: Classification labels (is_loading)
            y_reg: Regression target (odld_pct)
        """
        from config.model_config import FeatureConfig

        if feature_config is None:
            feature_config = FeatureConfig()

        # Load data
        rules_df = self.load_rules_data(rules_path)

        policy_df = None
        if policy_path:
            policy_df = self.load_policy_data(policy_path)

        claims_df = None
        if claims_path:
            claims_df = self.load_claims_data(claims_path)
            self.compute_claims_aggregates(claims_df)

        # Clean
        rules_df = self.clean_data(rules_df)
        if policy_df is not None:
            policy_df = self.clean_data(policy_df)

        # Engineer features
        rules_df = self.engineer_features(rules_df)
        if policy_df is not None:
            policy_df = self.engineer_features(policy_df)

        # Blend
        blended = self.blend_training_data(rules_df, policy_df)

        # Identify all categorical features for encoding
        all_categorical = [
            col for col in (
                feature_config.CATEGORICAL_FEATURES
                + feature_config.ENGINEERED_FEATURES
            )
            if col in blended.columns and blended[col].dtype == "object"
        ]

        # Fit and transform
        self.fit_encoders(blended, all_categorical)
        blended = self.transform_features(blended, all_categorical)

        # Select features
        available_features = [
            f for f in feature_config.all_input_features if f in blended.columns
        ]
        self.feature_names = available_features

        X = blended[available_features].copy()
        y_class = blended["is_loading"].copy()
        y_reg = blended["odld_pct"].copy()

        logger.info(f"Training data prepared: X={X.shape}, Features={available_features}")

        return X, y_class, y_reg

   
    # 9. Inference Preprocessing
   
    def prepare_inference_data(
        self, df: pd.DataFrame, feature_config=None
    ) -> pd.DataFrame:
        """
        Preprocess input data for inference.
        Expects the 7 raw input features.
        """
        from config.model_config import FeatureConfig

        if feature_config is None:
            feature_config = FeatureConfig()

        if not self._fitted:
            raise RuntimeError(
                "Preprocessor not fitted. Load a saved preprocessor or fit first."
            )

        df = self._standardize_columns(df.copy())

        # Clean
        str_cols = ["make", "model", "fuel_type", "rto"]
        for col in str_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.upper().str.strip()
                df[col] = df[col].replace({"NAN": "UNKNOWN", "": "UNKNOWN", "NONE": "UNKNOWN"})

        num_cols = ["vehicle_age", "ncb_pct", "zero_dep_flag"]
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "zero_dep_flag" in df.columns:
            df["zero_dep_flag"] = df["zero_dep_flag"].fillna(0).astype(int).clip(0, 1)

        # Engineer features
        df = self.engineer_features(df)

        # Transform with fitted encoders
        all_categorical = [
            col for col in (
                feature_config.CATEGORICAL_FEATURES
                + feature_config.ENGINEERED_FEATURES
            )
            if col in df.columns and col in self.label_encoders
        ]
        df = self.transform_features(df, all_categorical)

        # Select features in correct order
        available_features = [f for f in self.feature_names if f in df.columns]

        # Add missing features as 0
        for f in self.feature_names:
            if f not in df.columns:
                df[f] = 0

        return df[self.feature_names]

   
    # 10. Serialization
   
    def save(self, filepath: str) -> None:
        """Save preprocessor state."""
        import pickle

        state = {
            "label_encoders": self.label_encoders,
            "claims_aggregates": self.claims_aggregates,
            "feature_names": self.feature_names,
            "fitted": self._fitted,
        }
        with open(filepath, "wb") as f:
            pickle.dump(state, f)
        logger.info(f"Preprocessor saved to {filepath}")

    def load(self, filepath: str) -> None:
        """Load preprocessor state."""
        import pickle

        with open(filepath, "rb") as f:
            state = pickle.load(f)
        self.label_encoders = state["label_encoders"]
        self.claims_aggregates = state["claims_aggregates"]
        self.feature_names = state["feature_names"]
        self._fitted = state["fitted"]
        logger.info(f"Preprocessor loaded from {filepath}")
