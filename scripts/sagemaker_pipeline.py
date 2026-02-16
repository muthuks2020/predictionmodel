"""
SageMaker Pipeline Orchestration

- Training job submission to SageMaker
- Model registration
- Endpoint deployment (real-time)
- Batch transform job creation
- Model monitoring setup

Python 3.11 Compatible
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path

import boto3
import sagemaker
from sagemaker import Session
from sagemaker.estimator import Estimator
from sagemaker.model import Model
from sagemaker.transformer import Transformer
from sagemaker.model_monitor import DefaultModelMonitor
from sagemaker.inputs import TrainingInput

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PROJECT_ROOT)

from config.model_config import SageMakerConfig

logger = logging.getLogger(__name__)


class ODLDSageMakerPipeline:
    """Orchestrates SageMaker training, deployment, and monitoring."""

    def __init__(self, config: SageMakerConfig | None = None):
        self.config = config or SageMakerConfig()
        self.session = Session()
        self.role = sagemaker.get_execution_role()
        self.region = self.session.boto_region_name
        self.account_id = boto3.client("sts").get_caller_identity()["Account"]
        self.sm_client = boto3.client("sagemaker")

        # ECR image URI for custom container
        self.image_uri = (
            f"{self.account_id}.dkr.ecr.{self.region}.amazonaws.com/"
            f"odld-recommendation-engine:latest"
        )

        # Timestamps for unique naming
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    
    # TRAINING
    
    def submit_training_job(
        self,
        rules_data_s3: str,
        policy_data_s3: str | None = None,
        claims_data_s3: str | None = None,
    ) -> str:
        """
        Submit training job to SageMaker.

        Args:
            rules_data_s3: S3 URI for rules data
            policy_data_s3: S3 URI for policy data (optional)
            claims_data_s3: S3 URI for claims data (optional)

        Returns:
            Training job name
        """
        job_name = f"odld-training-{self.timestamp}"

        estimator = Estimator(
            image_uri=self.image_uri,
            role=self.role,
            instance_count=self.config.training_instance_count,
            instance_type=self.config.training_instance_type,
            output_path=self.config.s3_model_path,
            max_run=self.config.training_max_runtime,
            sagemaker_session=self.session,
            base_job_name="odld-rec-engine",
            hyperparameters={
                "max_depth": "6",
                "learning_rate": "0.05",
                "n_estimators": "600",
                "scale_pos_weight": "17",
                "subsample": "0.8",
                "colsample_bytree": "0.8",
                "eval_metric": "aucpr",
            },
        )

        # Prepare input channels
        inputs = {
            "train": TrainingInput(
                s3_data=rules_data_s3,
                content_type="text/csv",
            ),
        }

        if policy_data_s3:
            inputs["policy"] = TrainingInput(
                s3_data=policy_data_s3,
                content_type="text/csv",
            )

        if claims_data_s3:
            inputs["claims"] = TrainingInput(
                s3_data=claims_data_s3,
                content_type="text/csv",
            )

        logger.info(f"Submitting training job: {job_name}")
        estimator.fit(inputs, job_name=job_name, wait=False)

        return job_name

    
    # DEPLOYMENT - Real-Time Endpoint
    
    def deploy_realtime_endpoint(
        self,
        model_data_s3: str,
        endpoint_name: str | None = None,
    ) -> str:
        """
        Deploy model as a real-time SageMaker endpoint.

        Args:
            model_data_s3: S3 URI of model.tar.gz
            endpoint_name: Custom endpoint name (optional)

        Returns:
            Endpoint name
        """
        endpoint_name = endpoint_name or f"odld-endpoint-{self.timestamp}"

        model = Model(
            image_uri=self.image_uri,
            model_data=model_data_s3,
            role=self.role,
            sagemaker_session=self.session,
            name=f"odld-model-{self.timestamp}",
        )

        logger.info(f"Deploying real-time endpoint: {endpoint_name}")

        predictor = model.deploy(
            initial_instance_count=self.config.endpoint_instance_count,
            instance_type=self.config.endpoint_instance_type,
            endpoint_name=endpoint_name,
            serializer=sagemaker.serializers.JSONSerializer(),
            deserializer=sagemaker.deserializers.JSONDeserializer(),
        )

        logger.info(f"Endpoint deployed: {endpoint_name}")
        return endpoint_name

    
    # BATCH TRANSFORM
    
    def run_batch_transform(
        self,
        model_data_s3: str,
        input_s3: str,
        output_s3: str | None = None,
    ) -> str:
        """
        Run batch transform for bulk predictions.

        Args:
            model_data_s3: S3 URI of model.tar.gz
            input_s3: S3 URI of input CSV file
            output_s3: S3 URI for output (optional)

        Returns:
            Transform job name
        """
        job_name = f"odld-batch-{self.timestamp}"
        output_s3 = output_s3 or f"{self.config.s3_output_path}/batch-{self.timestamp}"

        model = Model(
            image_uri=self.image_uri,
            model_data=model_data_s3,
            role=self.role,
            sagemaker_session=self.session,
        )

        transformer = model.transformer(
            instance_count=self.config.batch_instance_count,
            instance_type=self.config.batch_instance_type,
            output_path=output_s3,
            accept="text/csv",
            strategy="MultiRecord",
            max_payload=6,  # MB
        )

        logger.info(f"Starting batch transform: {job_name}")
        transformer.transform(
            data=input_s3,
            content_type="text/csv",
            split_type="Line",
            job_name=job_name,
            wait=False,
        )

        return job_name

    
    # MODEL MONITORING
    
    def setup_model_monitor(self, endpoint_name: str) -> None:
        """
        Set up SageMaker Model Monitor for drift detection.
        """
        logger.info(f"Setting up model monitor for endpoint: {endpoint_name}")

        monitor = DefaultModelMonitor(
            role=self.role,
            instance_count=1,
            instance_type="ml.m5.large",
            volume_size_in_gb=20,
            max_runtime_in_seconds=3600,
        )

        # Create baseline from training data
        monitor_s3_uri = f"{self.config.s3_output_path}/monitor"

        monitor.create_monitoring_schedule(
            monitor_schedule_name=f"odld-monitor-{self.timestamp}",
            endpoint_input=endpoint_name,
            output_s3_uri=monitor_s3_uri,
            schedule_cron_expression=self.config.monitor_schedule_cron,
        )

        logger.info("Model monitor schedule created")

    
    # ENDPOINT INVOCATION (Client-side)
    
    @staticmethod
    def invoke_endpoint(
        endpoint_name: str,
        payload: dict | list,
        region: str = "ap-south-1",
    ) -> dict | list:
        """
        Invoke a deployed SageMaker endpoint.

        Args:
            endpoint_name: Name of the deployed endpoint
            payload: Single dict or list of dicts

        Returns:
            Prediction result(s)

        Example:
            result = ODLDSageMakerPipeline.invoke_endpoint(
                "odld-endpoint-20260208",
                {"make": "MARUTI", "model": "SWIFT", "fuel_type": "PETROL",
                 "vehicle_age": 3, "rto": "MH01", "ncb_pct": 25,
                 "zero_dep_flag": 0}
            )
        """
        runtime = boto3.client("sagemaker-runtime", region_name=region)

        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Accept="application/json",
            Body=json.dumps(payload),
        )

        result = json.loads(response["Body"].read().decode())
        return result

    
    # CLEANUP
    
    def delete_endpoint(self, endpoint_name: str) -> None:
        """Delete a deployed endpoint."""
        logger.info(f"Deleting endpoint: {endpoint_name}")
        self.sm_client.delete_endpoint(EndpointName=endpoint_name)
        logger.info("Endpoint deleted")


==
# CLI
==
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SageMaker Pipeline Manager")
    parser.add_argument(
        "action",
        choices=["train", "deploy", "batch", "monitor", "delete"],
        help="Pipeline action to perform",
    )
    parser.add_argument("--rules-s3", type=str, help="S3 URI for rules data")
    parser.add_argument("--policy-s3", type=str, help="S3 URI for policy data")
    parser.add_argument("--claims-s3", type=str, help="S3 URI for claims data")
    parser.add_argument("--model-s3", type=str, help="S3 URI for model artifacts")
    parser.add_argument("--input-s3", type=str, help="S3 URI for batch input")
    parser.add_argument("--endpoint", type=str, help="Endpoint name")

    args = parser.parse_args()
    pipeline = ODLDSageMakerPipeline()

    if args.action == "train":
        job = pipeline.submit_training_job(
            args.rules_s3, args.policy_s3, args.claims_s3
        )
        print(f"Training job submitted: {job}")

    elif args.action == "deploy":
        ep = pipeline.deploy_realtime_endpoint(args.model_s3, args.endpoint)
        print(f"Endpoint deployed: {ep}")

    elif args.action == "batch":
        job = pipeline.run_batch_transform(args.model_s3, args.input_s3)
        print(f"Batch transform started: {job}")

    elif args.action == "monitor":
        pipeline.setup_model_monitor(args.endpoint)

    elif args.action == "delete":
        pipeline.delete_endpoint(args.endpoint)
