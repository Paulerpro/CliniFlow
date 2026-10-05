"""Run the dataset validator as a SageMaker Processing job (CPU, data-only).

Usage (after `aws configure` works for cliniflow-dev):
    python registry/launch_processing_job.py

This is the Pillar 2 data job from docs/design: it reads the dataset splits
from S3, validates them in a SageMaker container, and writes a report back to S3.
It does not train, host, or merge anything.
"""
import os

import boto3
import sagemaker
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.image_uris import retrieve

REGION = "eu-north-1"
BUCKET = "cliniflow-adapters-paul-2026"
ACCOUNT_ID = os.environ["AWS_ACCOUNT_ID"]  # set locally, never committed
ROLE_ARN = f"arn:aws:iam::{ACCOUNT_ID}:role/cliniflow-sagemaker-exec"
INSTANCE_TYPE = "ml.t3.medium"  # smallest CPU option; the job is seconds of work


def main():
    boto_session = boto3.Session(region_name=REGION)
    # Pin the bucket: the SDK would otherwise try to create a default bucket,
    # and cliniflow-dev intentionally has no CreateBucket permission.
    session = sagemaker.Session(boto_session=boto_session, default_bucket=BUCKET)

    image_uri = retrieve(framework="sklearn", region=REGION, version="1.2-1", py_version="py3",
                         instance_type=INSTANCE_TYPE)

    processor = ScriptProcessor(
        role=ROLE_ARN,
        image_uri=image_uri,
        command=["python3"],
        instance_type=INSTANCE_TYPE,
        instance_count=1,
        sagemaker_session=session,
        base_job_name="cliniflow-validate-dataset",
    )

    processor.run(
        code="registry/validate_dataset.py",
        inputs=[
            ProcessingInput(
                source=f"s3://{BUCKET}/dataset/",
                destination="/opt/ml/processing/input",
            )
        ],
        outputs=[
            ProcessingOutput(
                source="/opt/ml/processing/output",
                destination=f"s3://{BUCKET}/reports/validation/",
            )
        ],
        wait=True,
        logs=True,
    )

    print(f"Report written to s3://{BUCKET}/reports/validation/")


if __name__ == "__main__":
    main()
