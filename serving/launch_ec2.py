"""Launch (or reuse) the g4dn.xlarge Spot instance that serves vLLM.

Prerequisites, done once by hand in the console (see serving/README.md):
  - cliniflow-dev has the EC2 permissions in registry/iam/cliniflow-dev-ec2-policy.json
  - an instance profile named cliniflow-ec2-serve exists, built from the role+policy in
    registry/iam/cliniflow-ec2-serve-policy.json (S3 read-only, so the instance can pull
    the adapter without copying your access key onto it)
  - the EC2 GPU quota (G and VT family) is approved in eu-north-1

Usage:
    export EC2_AMI_ID=ami-xxxxxxxx   # see "finding an AMI" in serving/README.md
    python serving/launch_ec2.py

This does not install anything on the box or start vLLM. It only gets you a running
GPU machine reachable on port 22 (SSH) and 8000 (the API) from your own IP. The rest
of serving/README.md's "Running the server on the instance" section takes it from there.
"""
import json
import os
import sys
import time
import urllib.request

import boto3

REGION = "eu-north-1"
INSTANCE_TYPE = "g4dn.xlarge"
INSTANCE_PROFILE_NAME = "cliniflow-ec2-serve"
KEY_NAME = "cliniflow-ec2-key"
SECURITY_GROUP_NAME = "cliniflow-vllm-sg"
TAG_NAME = "cliniflow-vllm"
KEY_FILE = os.path.expanduser(f"~/.ssh/{KEY_NAME}.pem")


def my_public_ip():
    """So the security group opens only to this machine, never 0.0.0.0/0."""
    with urllib.request.urlopen("http://checkip.amazonaws.com", timeout=5) as resp:
        return resp.read().decode().strip()


def get_or_create_security_group(ec2, vpc_id, my_ip):
    existing = ec2.describe_security_groups(Filters=[
        {"Name": "group-name", "Values": [SECURITY_GROUP_NAME]},
        {"Name": "vpc-id", "Values": [vpc_id]},
    ])["SecurityGroups"]
    if existing:
        sg_id = existing[0]["GroupId"]
        print(f"reusing security group {sg_id}")
    else:
        sg_id = ec2.create_security_group(
            GroupName=SECURITY_GROUP_NAME,
            Description="CliniFlow vLLM demo: SSH + API, caller IP only",
            VpcId=vpc_id,
        )["GroupId"]
        print(f"created security group {sg_id}")

    cidr = f"{my_ip}/32"
    for port, label in [(22, "SSH"), (8000, "vLLM API")]:
        try:
            ec2.authorize_security_group_ingress(
                GroupId=sg_id,
                IpPermissions=[{
                    "IpProtocol": "tcp", "FromPort": port, "ToPort": port,
                    "IpRanges": [{"CidrIp": cidr, "Description": f"{label} from launcher's IP"}],
                }],
            )
            print(f"opened {label} ({port}) to {cidr}")
        except ec2.exceptions.ClientError as e:
            if "InvalidPermission.Duplicate" in str(e):
                print(f"{label} ({port}) already open to {cidr}")
            else:
                raise
    return sg_id


def get_or_create_key_pair(ec2):
    existing = ec2.describe_key_pairs(Filters=[{"Name": "key-name", "Values": [KEY_NAME]}])["KeyPairs"]
    if existing:
        if not os.path.exists(KEY_FILE):
            print(f"WARNING: key pair {KEY_NAME} exists in AWS but {KEY_FILE} is not on this machine.\n"
                  f"You will not be able to SSH in with it. Delete the AWS key pair and rerun to make a new one.")
        else:
            print(f"reusing key pair {KEY_NAME}")
        return
    key = ec2.create_key_pair(KeyName=KEY_NAME, KeyType="rsa", KeyFormat="pem")
    os.makedirs(os.path.dirname(KEY_FILE), exist_ok=True)
    with open(KEY_FILE, "w", encoding="utf-8") as f:
        f.write(key["KeyMaterial"])
    os.chmod(KEY_FILE, 0o400)
    print(f"created key pair {KEY_NAME}, saved to {KEY_FILE} (keep this local, never commit it)")


def main():
    ami_id = os.environ.get("EC2_AMI_ID")
    if not ami_id:
        sys.exit("Set EC2_AMI_ID first. See 'Finding an AMI' in serving/README.md.")

    ec2 = boto3.client("ec2", region_name=REGION)

    running = ec2.describe_instances(Filters=[
        {"Name": "tag:Name", "Values": [TAG_NAME]},
        {"Name": "instance-state-name", "Values": ["pending", "running"]},
    ])
    for r in running["Reservations"]:
        for inst in r["Instances"]:
            print(f"already {inst['State']['Name']}: {inst['InstanceId']} "
                  f"public_ip={inst.get('PublicIpAddress')}")
            return

    vpc_id = ec2.describe_vpcs(Filters=[{"Name": "is-default", "Values": ["true"]}])["Vpcs"][0]["VpcId"]
    subnet_id = ec2.describe_subnets(Filters=[{"Name": "vpc-id", "Values": [vpc_id]}])["Subnets"][0]["SubnetId"]
    my_ip = my_public_ip()
    sg_id = get_or_create_security_group(ec2, vpc_id, my_ip)
    get_or_create_key_pair(ec2)

    print(f"requesting {INSTANCE_TYPE} Spot instance in {REGION}...")
    resp = ec2.run_instances(
        ImageId=ami_id,
        InstanceType=INSTANCE_TYPE,
        KeyName=KEY_NAME,
        MinCount=1, MaxCount=1,
        NetworkInterfaces=[{
            "DeviceIndex": 0, "SubnetId": subnet_id, "Groups": [sg_id],
            "AssociatePublicIpAddress": True,
        }],
        IamInstanceProfile={"Name": INSTANCE_PROFILE_NAME},
        InstanceMarketOptions={
            "MarketType": "spot",
            "SpotOptions": {"InstanceInterruptionBehavior": "terminate"},
        },
        TagSpecifications=[{"ResourceType": "instance", "Tags": [{"Key": "Name", "Value": TAG_NAME}]}],
    )
    instance_id = resp["Instances"][0]["InstanceId"]
    print(f"launched {instance_id}, waiting for it to start running...")

    waiter = ec2.get_waiter("instance_running")
    waiter.wait(InstanceIds=[instance_id])
    desc = ec2.describe_instances(InstanceIds=[instance_id])
    ip = desc["Reservations"][0]["Instances"][0]["PublicIpAddress"]

    print(f"\nrunning: {instance_id}")
    print(f"public IP: {ip}")
    print(f"SSH:  ssh -i {KEY_FILE} ubuntu@{ip}")
    print(f"client: export SERVER_URL=http://{ip}:8000")
    print("\nStop it when done: see 'Shutting down' in serving/README.md")


if __name__ == "__main__":
    main()
