"""Stop or terminate the cliniflow-vllm instance. Run this when you're done demoing.

Usage:
    python serving/stop_ec2.py            # stop (keeps the instance, can restart later)
    python serving/stop_ec2.py --terminate  # delete it entirely (frees the Spot quota slot)
"""
import argparse
import boto3

REGION = "eu-north-1"
TAG_NAME = "cliniflow-vllm"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--terminate", action="store_true", help="delete the instance instead of stopping it")
    args = parser.parse_args()

    ec2 = boto3.client("ec2", region_name=REGION)
    found = ec2.describe_instances(Filters=[
        {"Name": "tag:Name", "Values": [TAG_NAME]},
        {"Name": "instance-state-name", "Values": ["pending", "running", "stopping", "stopped"]},
    ])
    ids = [i["InstanceId"] for r in found["Reservations"] for i in r["Instances"]]
    if not ids:
        print("no cliniflow-vllm instance found (already stopped/terminated, or never launched)")
        return

    if args.terminate:
        ec2.terminate_instances(InstanceIds=ids)
        print(f"terminating: {ids}")
    else:
        ec2.stop_instances(InstanceIds=ids)
        print(f"stopping: {ids}")


if __name__ == "__main__":
    main()
