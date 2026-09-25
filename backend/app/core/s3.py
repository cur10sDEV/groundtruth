from uuid import uuid4

import boto3
from botocore.exceptions import ClientError

from app.core.config import get_settings
from app.core.errors import StorageError


def _client():
    s = get_settings()
    return boto3.client(
        "s3",
        endpoint_url=s.s3_endpoint,
        aws_access_key_id=s.s3_access_key,
        aws_secret_access_key=s.s3_secret_key,
        region_name=s.s3_region,
    )


def _ensure_bucket(session, bucket: str) -> None:
    try:
        session.head_bucket(Bucket=bucket)
    except ClientError:
        session.create_bucket(Bucket=bucket)
        session.put_bucket_versioning(Bucket=bucket, VersioningConfiguration={"Status": "Enabled"})


def build_key(org_id: str, user_id: str, doc_id: str, uuid_ext: str) -> str:
    return f"documents/{org_id}/{user_id}/{doc_id}/{uuid_ext}"


def key_to_parts(s3_key: str) -> tuple[str, str, str, str]:
    _bucket, org_id, user_id, doc_id, name = s3_key.split("/", 4)
    return org_id, user_id, doc_id, name


def put_object(org_id: str, user_id: str, doc_id: str, filename: str, data: bytes) -> str:
    session = _client()
    s = get_settings()
    _ensure_bucket(session, s.s3_bucket)
    key = build_key(org_id, user_id, doc_id, f"{uuid4()}.{filename.split('.')[-1]}")
    try:
        session.put_object(
            Bucket=s.s3_bucket,
            Key=key,
            Body=data,
            Metadata={"original_filename": filename},
        )
    except ClientError as exc:
        raise StorageError(detail=f"s3 put failed: {exc}") from exc
    return key


def get_object(s3_key: str) -> bytes:
    session = _client()
    s = get_settings()
    try:
        resp = session.get_object(Bucket=s.s3_bucket, Key=s3_key)
        return resp["Body"].read()
    except ClientError as exc:
        raise StorageError(detail=f"s3 get failed: {exc}") from exc


def delete_object(s3_key: str) -> None:
    session = _client()
    s = get_settings()
    try:
        session.delete_object(Bucket=s.s3_bucket, Key=s3_key)
    except ClientError as exc:
        raise StorageError(detail=f"s3 delete failed: {exc}") from exc


def presign_upload(org_id: str, user_id: str, doc_id: str, ext: str) -> dict:
    s = get_settings()
    client = _client()
    key = build_key(org_id, user_id, doc_id, f"{uuid4()}.{ext}")
    post = client.generate_presigned_post(
        Bucket=s.s3_bucket,
        Key=key,
        Conditions=[["content-length-range", 0, s.upload_max_bytes]],
        ExpiresIn=s.presign_expiry_seconds,
    )
    return {"url": post["url"], "fields": post["fields"], "key": key}


def delete_prefix(org_id: str, user_id: str, doc_id: str) -> int:
    s = get_settings()
    client = _client()
    prefix = f"documents/{org_id}/{user_id}/{doc_id}/"
    deleted = 0
    keys: list[str] = []
    resp = client.list_objects_v2(Bucket=s.s3_bucket, Prefix=prefix)
    keys = [o["Key"] for o in resp.get("Contents", [])]
    for i in range(0, len(keys), 1000):
        batch = keys[i : i + 1000]
        client.delete_objects(Bucket=s.s3_bucket, Delete={"Objects": [{"Key": k} for k in batch]})
        deleted += len(batch)
    return deleted
