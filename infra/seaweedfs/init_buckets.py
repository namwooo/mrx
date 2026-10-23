import argparse
import getpass
import os
import sys

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

BUCKETS = ["mrx-models", "mrx-datasets", "mrx-outputs"]
VERSIONED = ["mrx-models"]  # a model re-uploaded under the same key keeps its previous version


def make_client(endpoint, access_key, secret_key):
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="us-east-1",
        config=Config(
            s3={"addressing_style": "path"},
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
            retries={"max_attempts": 2},
        ),
    )


def key_from_env(name, secret=False):
    value = os.environ.get(name)
    if value:
        return value
    return getpass.getpass(f"{name}: ") if secret else input(f"{name}: ")


def create_buckets(s3):
    existing = {b["Name"] for b in s3.list_buckets().get("Buckets", [])}
    for bucket in BUCKETS:
        if bucket in existing:
            print(f"exists   {bucket}")
        else:
            s3.create_bucket(Bucket=bucket)
            print(f"created  {bucket}")

    for bucket in VERSIONED:
        s3.put_bucket_versioning(Bucket=bucket, VersioningConfiguration={"Status": "Enabled"})
        status = s3.get_bucket_versioning(Bucket=bucket).get("Status")
        print(f"versioning {bucket}: {status}")


def check_app_account(s3):
    """The app account must read/write the mrx-* buckets and must not create buckets."""
    ok = True
    key = "_init_check/hello.txt"
    for bucket in BUCKETS:
        try:
            s3.put_object(Bucket=bucket, Key=key, Body=b"hello")
            body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            s3.delete_object(Bucket=bucket, Key=key)
            print(f"app ok   {bucket} (write/read/delete)" if body == b"hello" else f"app FAIL {bucket}: wrong content")
            ok = ok and body == b"hello"
        except ClientError as exc:
            print(f"app FAIL {bucket}: {exc.response['Error']['Code']}")
            ok = False

    try:
        s3.create_bucket(Bucket="mrx-should-not-exist")
        print("app FAIL could create a bucket: the app account has too many permissions")
        ok = False
    except ClientError as exc:
        print(f"app ok   cannot create buckets ({exc.response['Error']['Code']})")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("endpoint", help="S3 address, e.g. http://10.0.0.5:8333")
    args = parser.parse_args()

    admin = make_client(
        args.endpoint,
        key_from_env("S3_ADMIN_ACCESS_KEY"),
        key_from_env("S3_ADMIN_SECRET_KEY", secret=True),
    )
    create_buckets(admin)

    if os.environ.get("S3_ACCESS_KEY") and os.environ.get("S3_SECRET_KEY"):
        app = make_client(args.endpoint, os.environ["S3_ACCESS_KEY"], os.environ["S3_SECRET_KEY"])
        if not check_app_account(app):
            sys.exit(1)
    else:
        print("app account not checked (set S3_ACCESS_KEY and S3_SECRET_KEY to check it)")


if __name__ == "__main__":
    main()