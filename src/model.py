"""
Two-Stage XGBoost ODLD Model
==============================
Stage 1: XGBClassifier  - Discount vs Loading classification
Stage 2: XGBRegressor   - ODLD % magnitude prediction (separate for each class)


Python 3.11 Compatible
"""

import os
import json
import logging
import pickle
from typing import Optional

import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix,
)

logger = logging.getLogger(__name__)


class TwoStageODLDModel:
    """
    Two-Stage XGBoost model for ODLD prediction.

    Stage 1 (Classifier):
    
    Stage 2 (Regressors):
    
    """

    def __init__(
        self,
        classifier_params: Optional[dict] = None,
        regressor_params: Optional[dict] = None,
    ):
        from config.model_config import ClassifierConfig, RegressorConfig

        # Stage 1: Classifier
        clf_config = ClassifierConfig()
        clf_params = clf_config.to_dict() if classifier_params is None else classifier_params
        # Remove non-XGBoost params
        clf_params.pop("early_stopping_rounds", None)
        self._early_stopping_clf = (
            classifier_params.get("early_stopping_rounds", 50) if classifier_params
            else clf_config.early_stopping_rounds
        )
        self.classifier = XGBClassifier(**clf_params)

        # Stage 2: Regressors
        reg_config = RegressorConfig()
        reg_params = reg_config.to_dict() if regressor_params is None else regressor_params
        reg_params.pop("early_stopping_rounds", None)
        self._early_stopping_reg = (
            regressor_params.get("early_stopping_rounds", 50) if regressor_params
            else reg_config.early_stopping_rounds
        )
        self.discount_regressor = XGBRegressor(**reg_params)
        self.loading_regressor = XGBRegressor(**reg_params)

        # State
        self._trained = False
        self.training_metrics: dict = {}
        self.feature_names: list[str] = []

    
    # TRAINING
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_class_train: pd.Series,
        y_reg_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_class_val: Optional[pd.Series] = None,
        y_reg_val: Optional[pd.Series] = None,
    ) -> dict:
        """
        Train the two-stage model.

        Args:
            X_train: Training features
            y_class_train: Binary labels (0=discount, 1=loading)
            y_reg_train: ODLD percentage values
            X_val: Validation features (optional but recommended)
            y_class_val: Validation binary labels
            y_reg_val: Validation ODLD values

        Returns:
            Dictionary of training metrics
        """
        self.feature_names = list(X_train.columns)
        metrics = {}

        # ----- Stage 1: Train Classifier -----
        logger.info("=" * 60)
        logger.info("STAGE 1: Training Classifier (Discount vs Loading)")
        logger.info("=" * 60)

        clf_eval_set = [(X_train, y_class_train)]
        if X_val is not None and y_class_val is not None:
            clf_eval_set.append((X_val, y_class_val))

        self.classifier.fit(
            X_train,
            y_class_train,
            eval_set=clf_eval_set,
            verbose=50,
        )

        # Classifier metrics
        if X_val is not None and y_class_val is not None:
            y_pred_proba = self.classifier.predict_proba(X_val)[:, 1]
            y_pred_class = (y_pred_proba >= 0.5).astype(int)

            # Use 0.5 threshold initially, will optimize later
            clf_metrics = self._compute_classifier_metrics(
                y_class_val, y_pred_class, y_pred_proba
            )
            metrics["classifier"] = clf_metrics
            logger.info(f"Classifier Validation Metrics:")
            for k, v in clf_metrics.items():
                logger.info(f"  {k}: {v}")

        # ----- Stage 2A: Train Discount Regressor -----
        logger.info("=" * 60)
        logger.info("STAGE 2A: Training Discount Regressor (ODLD <= 0)")
        logger.info("=" * 60)

        discount_mask_train = y_class_train == 0
        X_discount_train = X_train[discount_mask_train]
        y_discount_train = y_reg_train[discount_mask_train]

        logger.info(f"Discount training samples: {len(X_discount_train)}")

        disc_eval_set = [(X_discount_train, y_discount_train)]
        if X_val is not None and y_reg_val is not None:
            discount_mask_val = y_class_val == 0
            if discount_mask_val.sum() > 0:
                X_discount_val = X_val[discount_mask_val]
                y_discount_val = y_reg_val[discount_mask_val]
                disc_eval_set.append((X_discount_val, y_discount_val))

        self.discount_regressor.fit(
            X_discount_train,
            y_discount_train,
            eval_set=disc_eval_set,
            verbose=50,
        )

        # Discount regressor metrics
        if X_val is not None and y_reg_val is not None and discount_mask_val.sum() > 0:
            y_disc_pred = self.discount_regressor.predict(X_discount_val)
            disc_metrics = self._compute_regressor_metrics(
                y_discount_val, y_disc_pred, "discount"
            )
            metrics["discount_regressor"] = disc_metrics
            logger.info(f"Discount Regressor Validation Metrics:")
            for k, v in disc_metrics.items():
                logger.info(f"  {k}: {v}")

        # ----- Stage 2B: Train Loading Regressor -----
        logger.info("=" * 60)
        logger.info("STAGE 2B: Training Loading Regressor (ODLD > 0)")
        logger.info("=" * 60)

        loading_mask_train = y_class_train == 1
        X_loading_train = X_train[loading_mask_train]
        y_loading_train = y_reg_train[loading_mask_train]

        logger.info(f"Loading training samples: {len(X_loading_train)}")

        if len(X_loading_train) < 10:
            logger.warning(
                "Very few loading training samples! "
                "Loading regressor may not perform well."
            )

        load_eval_set = [(X_loading_train, y_loading_train)]
        if X_val is not None and y_reg_val is not None:
            loading_mask_val = y_class_val == 1
            if loading_mask_val.sum() > 0:
                X_loading_val = X_val[loading_mask_val]
                y_loading_val = y_reg_val[loading_mask_val]
                load_eval_set.append((X_loading_val, y_loading_val))

        self.loading_regressor.fit(
            X_loading_train,
            y_loading_train,
            eval_set=load_eval_set,
            verbose=50,
        )

        # Loading regressor metrics
        if X_val is not None and y_reg_val is not None and loading_mask_val.sum() > 0:
            y_load_pred = self.loading_regressor.predict(X_loading_val)
            load_metrics = self._compute_regressor_metrics(
                y_loading_val, y_load_pred, "loading"
            )
            metrics["loading_regressor"] = load_metrics
            logger.info(f"Loading Regressor Validation Metrics:")
            for k, v in load_metrics.items():
                logger.info(f"  {k}: {v}")

        # ----- Combined End-to-End Metrics -----
        if X_val is not None and y_reg_val is not None:
            y_combined_pred = self._predict_combined(X_val, y_pred_proba)
            combined_metrics = self._compute_regressor_metrics(
                y_reg_val, y_combined_pred, "combined"
            )
            metrics["combined"] = combined_metrics
            logger.info(f"Combined End-to-End Validation Metrics:")
            for k, v in combined_metrics.items():
                logger.info(f"  {k}: {v}")

        self._trained = True
        self.training_metrics = metrics
        return metrics

    
    # PREDICTION
    
    def predict(
        self,
        X: pd.DataFrame,
        return_details: bool = True,
    ) -> pd.DataFrame:
        """
        Generate ODLD predictions.

        Args:
            X: Feature matrix (preprocessed)
            return_details: If True, include confidence and direction

        Returns:
            DataFrame with predictions
        """
        if not self._trained:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        # Stage 1: Classification probability
        proba = self.classifier.predict_proba(X)[:, 1]

        # Stage 2: Predict magnitudes from BOTH regressors
        discount_pred = self.discount_regressor.predict(X)
        loading_pred = self.loading_regressor.predict(X)

        # Combine: use classifier decision to select regressor output
        is_loading = proba >= 0.5
        odld_pred = np.where(is_loading, loading_pred, discount_pred)

        # Safety: ensure discount predictions are <= 0 and loading > 0
        odld_pred = np.where(
            is_loading,
            np.maximum(odld_pred, 0.01),  # loading must be positive
            np.minimum(odld_pred, 0.0),   # discount must be non-positive
        )

        result = pd.DataFrame({"odld_pct": np.round(odld_pred, 2)})

        if return_details:
            result["loading_probability"] = np.round(proba, 4)
            result["direction"] = np.where(is_loading, "LOADING", "DISCOUNT")
            result["confidence"] = self._compute_confidence(proba)

        return result

    def predict_single(self, features: dict) -> dict:
        """
        Predict ODLD for a single input.

        Args:
            features: Dictionary with 7 input features

        Returns:
            Dictionary with prediction results
        """
        df = pd.DataFrame([features])
        # Note: caller must preprocess the df before calling this
        result = self.predict(df, return_details=True)
        return result.iloc[0].to_dict()

    
    # HELPERS
    
    def _predict_combined(
        self, X: pd.DataFrame, proba: np.ndarray
    ) -> np.ndarray:
        """Combine classifier + regressors for end-to-end prediction."""
        is_loading = proba >= 0.5
        discount_pred = self.discount_regressor.predict(X)
        loading_pred = self.loading_regressor.predict(X)
        return np.where(is_loading, loading_pred, discount_pred)

    def _compute_confidence(self, proba: np.ndarray) -> list[str]:
        """Map probability to confidence level."""
        confidence = []
        for p in proba:
            dist_from_boundary = abs(p - 0.5)
            if dist_from_boundary >= 0.3:  # p < 0.2 or p > 0.8
                confidence.append("HIGH")
            elif dist_from_boundary >= 0.1:  # p < 0.4 or p > 0.6
                confidence.append("MEDIUM")
            else:
                confidence.append("LOW")
        return confidence

    def _compute_classifier_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
    ) -> dict:
        """Compute classification metrics with focus on loading recall."""
        report = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred)

        metrics = {
            "accuracy": round(float(report["accuracy"]), 4),
            "loading_precision": round(float(report.get("1", {}).get("precision", 0)), 4),
            "loading_recall": round(float(report.get("1", {}).get("recall", 0)), 4),
            "loading_f1": round(float(report.get("1", {}).get("f1-score", 0)), 4),
            "discount_precision": round(float(report.get("0", {}).get("precision", 0)), 4),
            "discount_recall": round(float(report.get("0", {}).get("recall", 0)), 4),
            "discount_f1": round(float(report.get("0", {}).get("f1-score", 0)), 4),
            "pr_auc": round(float(average_precision_score(y_true, y_proba)), 4),
            "confusion_matrix": cm.tolist(),
        }
        return metrics

    def _compute_regressor_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        stage: str,
    ) -> dict:
        """Compute regression metrics."""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = r2_score(y_true, y_pred)

        return {
            f"{stage}_mae": round(mae, 4),
            f"{stage}_rmse": round(rmse, 4),
            f"{stage}_r2": round(r2, 4),
            f"{stage}_mean_actual": round(float(y_true.mean()), 4),
            f"{stage}_mean_predicted": round(float(y_pred.mean()), 4),
        }

    
    # FEATURE IMPORTANCE
    
    def get_feature_importance(self) -> dict:
        """Get feature importance from all three models."""
        if not self._trained:
            raise RuntimeError("Model not trained.")

        importance = {}

        # Classifier importance
        clf_imp = self.classifier.feature_importances_
        importance["classifier"] = dict(
            sorted(
                zip(self.feature_names, clf_imp.tolist()),
                key=lambda x: x[1],
                reverse=True,
            )
        )

        # Discount regressor importance
        disc_imp = self.discount_regressor.feature_importances_
        importance["discount_regressor"] = dict(
            sorted(
                zip(self.feature_names, disc_imp.tolist()),
                key=lambda x: x[1],
                reverse=True,
            )
        )

        # Loading regressor importance
        load_imp = self.loading_regressor.feature_importances_
        importance["loading_regressor"] = dict(
            sorted(
                zip(self.feature_names, load_imp.tolist()),
                key=lambda x: x[1],
                reverse=True,
            )
        )

        return importance

    
    # SERIALIZATION
    
    def save(self, model_dir: str) -> None:
        """Save all model artifacts."""
        os.makedirs(model_dir, exist_ok=True)

        # Save XGBoost models in native format
        self.classifier.save_model(os.path.join(model_dir, "classifier.json"))
        self.discount_regressor.save_model(os.path.join(model_dir, "discount_regressor.json"))
        self.loading_regressor.save_model(os.path.join(model_dir, "loading_regressor.json"))

        # Save metadata
        metadata = {
            "feature_names": self.feature_names,
            "training_metrics": self.training_metrics,
            "trained": self._trained,
        }
        with open(os.path.join(model_dir, "model_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Model saved to {model_dir}")

    def load(self, model_dir: str) -> None:
        """Load model artifacts."""
        self.classifier = XGBClassifier()
        self.classifier.load_model(os.path.join(model_dir, "classifier.json"))

        self.discount_regressor = XGBRegressor()
        self.discount_regressor.load_model(
            os.path.join(model_dir, "discount_regressor.json")
        )

        self.loading_regressor = XGBRegressor()
        self.loading_regressor.load_model(
            os.path.join(model_dir, "loading_regressor.json")
        )

        # Load metadata
        with open(os.path.join(model_dir, "model_metadata.json"), "r") as f:
            metadata = json.load(f)

        self.feature_names = metadata["feature_names"]
        self.training_metrics = metadata.get("training_metrics", {})
        self._trained = True

        logger.info(f"Model loaded from {model_dir}")
