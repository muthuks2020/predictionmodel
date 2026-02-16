"""

Shellkode <> Zuno General Insurance
Recommendation Engine - Two-Stage XGBoost

Python 3.11 Compatible
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FeatureConfig:
    """Feature definitions matching Ankit's 7-input specification."""

    # 7 Input Features as per Ankit's Jan 13 email
    CATEGORICAL_FEATURES: list = field(default_factory=lambda: [
        "make",
        "model",
        "fuel_type",
        "rto",
    ])

    NUMERIC_FEATURES: list = field(default_factory=lambda: [
        "vehicle_age",
        "ncb_pct",
        "zero_dep_flag",
    ])

    # Derived features from claims data (aggregated at segment level)
    CLAIMS_DERIVED_FEATURES: list = field(default_factory=lambda: [
        "segment_claim_frequency",
        "segment_avg_claim_amount",
    ])

    # Engineered features
    ENGINEERED_FEATURES: list = field(default_factory=lambda: [
        "make_model",           # interaction: make + model combined
        "vehicle_age_bucket",   # bucketed: new/1-3yr/4-7yr/8+
        "ncb_tier",            # standard NCB slabs
        "rto_zone",            # clustered RTO zones by risk
    ])

    # Target variable
    TARGET: str = "odld_pct"

    # Classification target (derived from odld_pct)
    CLASSIFICATION_TARGET: str = "is_loading"  # 1 if odld_pct > 0, else 0

    @property
    def all_input_features(self) -> list:
        return (
            self.CATEGORICAL_FEATURES
            + self.NUMERIC_FEATURES
            + self.CLAIMS_DERIVED_FEATURES
            + self.ENGINEERED_FEATURES
        )

    @property
    def raw_input_features(self) -> list:
        """The 7 original features from Ankit's spec."""
        return self.CATEGORICAL_FEATURES + self.NUMERIC_FEATURES


@dataclass
class ClassifierConfig:
    """XGBoost Classifier config - Stage 1: Discount vs Loading."""

    objective: str = "binary:logistic"
    eval_metric: str = "aucpr"  # PR-AUC, NOT accuracy (imbalance!)
    max_depth: int = 6
    learning_rate: float = 0.05
    n_estimators: int = 600
    scale_pos_weight: float = 17.0  # ~17:1 imbalance ratio
    min_child_weight: int = 10
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    random_state: int = 42
    early_stopping_rounds: int = 50
    tree_method: str = "hist"  # fast histogram-based method
    enable_categorical: bool = True
    verbosity: int = 1

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class RegressorConfig:
    """XGBoost Regressor config - Stage 2: ODLD magnitude."""

    objective: str = "reg:squarederror"
    eval_metric: str = "mae"
    max_depth: int = 5
    learning_rate: float = 0.05
    n_estimators: int = 500
    min_child_weight: int = 8
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    random_state: int = 42
    early_stopping_rounds: int = 50
    tree_method: str = "hist"
    enable_categorical: bool = True
    verbosity: int = 1

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class TrainingConfig:
    """Training pipeline configuration."""

    test_size: float = 0.2
    validation_size: float = 0.15
    stratify_column: str = "is_loading"
    random_state: int = 42

    # Confidence thresholds for classification
    high_confidence_threshold: float = 0.8
    low_confidence_threshold: float = 0.4

    # SMOTE configuration for training set
    apply_smote: bool = True
    smote_sampling_strategy: float = 0.3  # target 30% minority

    # Model artifact paths (SageMaker compatible)
    model_dir: str = "/opt/ml/model"
    output_dir: str = "/opt/ml/output"
    input_dir: str = "/opt/ml/input/data"

    # Local development paths
    local_model_dir: str = "./artifacts/models"
    local_data_dir: str = "./data"


@dataclass
class InferenceConfig:
    """Inference endpoint configuration."""

    model_dir: str = "/opt/ml/model"

    # Output ODLD bounds (safety guardrail)
    min_odld_pct: float = -50.0  # max 50% discount
    max_odld_pct: float = 50.0   # max 50% loading

    # Confidence levels
    high_confidence: float = 0.8
    medium_confidence: float = 0.6
    low_confidence: float = 0.4

    # Batch processing
    batch_size: int = 1000

    # ODLD slabs (if Zuno uses fixed slabs - confirm with Ankit)
    # Set to None if continuous values are acceptable
    odld_slabs: Optional[list] = None
    # Example: [-50, -40, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 40, 50]


@dataclass
class SageMakerConfig:
    """SageMaker deployment configuration."""

    # Training
    training_instance_type: str = "ml.m5.xlarge"
    training_instance_count: int = 1
    training_max_runtime: int = 7200  # 2 hours

    # Real-time endpoint (single prediction)
    endpoint_instance_type: str = "ml.m5.large"
    endpoint_instance_count: int = 1

    # Batch transform (bulk upload)
    batch_instance_type: str = "ml.m5.xlarge"
    batch_instance_count: int = 1

    # Model monitoring
    monitor_schedule_cron: str = "cron(0 * ? * * *)"  # hourly

    # S3 paths
    s3_bucket: str = "zuno-odld-model"
    s3_prefix: str = "recommendation-engine"

    @property
    def s3_model_path(self) -> str:
        return f"s3://{self.s3_bucket}/{self.s3_prefix}/models"

    @property
    def s3_data_path(self) -> str:
        return f"s3://{self.s3_bucket}/{self.s3_prefix}/data"

    @property
    def s3_output_path(self) -> str:
        return f"s3://{self.s3_bucket}/{self.s3_prefix}/output"
