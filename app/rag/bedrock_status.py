import os
import json
import boto3
from botocore.exceptions import BotoCoreError, ClientError


def bedrock_status() -> dict:
    """
    Returns a structured status report for AWS creds + Bedrock reachability + model access.
    Safe to call during demos.
    """
    region = os.getenv("AWS_REGION", "us-east-1")
    use_bedrock = os.getenv("USE_BEDROCK", "false").lower() == "true"

    embedding_model_id = os.getenv(
        "BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0"
    )
    chat_model_id = os.getenv(
        "BEDROCK_CHAT_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0"
    )

    report = {
        "use_bedrock": use_bedrock,
        "region": region,
        "embedding_model_id": embedding_model_id,
        "chat_model_id": chat_model_id,
        "aws_identity": None,
        "bedrock_control_plane_ok": False,
        "bedrock_runtime_ok": False,
        "embedding_invoke_ok": False,
        "embedding_dim": None,
        "chat_invoke_ok": False,
        "errors": [],
    }

    # 1) AWS identity
    try:
        sts = boto3.client("sts", region_name=region)
        ident = sts.get_caller_identity()
        report["aws_identity"] = {
            "account": ident.get("Account"),
            "arn": ident.get("Arn"),
            "user_id": ident.get("UserId"),
        }
    except (BotoCoreError, ClientError) as e:
        report["errors"].append(f"STS get_caller_identity failed: {e}")
        return report  # no point continuing if creds aren't valid

    # 2) Bedrock control plane (list models)
    try:
        bedrock = boto3.client("bedrock", region_name=region)
        _ = bedrock.list_foundation_models()  # permission + reachability check
        report["bedrock_control_plane_ok"] = True
    except (BotoCoreError, ClientError) as e:
        report["errors"].append(f"Bedrock control plane check failed: {e}")

    # 3) Bedrock runtime checks (invoke models)
    try:
        runtime = boto3.client("bedrock-runtime", region_name=region)
        report["bedrock_runtime_ok"] = True
    except (BotoCoreError, ClientError) as e:
        report["errors"].append(f"Bedrock runtime client init failed: {e}")
        return report

    # 3a) Embeddings invoke (cheap)
    try:
        resp = runtime.invoke_model(
            modelId=embedding_model_id,
            body=json.dumps({"inputText": "ping"}),
            accept="application/json",
            contentType="application/json",
        )
        payload = json.loads(resp["body"].read().decode("utf-8"))
        emb = payload.get("embedding")
        if isinstance(emb, list):
            report["embedding_invoke_ok"] = True
            report["embedding_dim"] = len(emb)
        else:
            report["errors"].append("Embeddings invoke succeeded but no 'embedding' returned.")
    except (BotoCoreError, ClientError, ValueError, KeyError) as e:
        report["errors"].append(f"Embeddings invoke failed: {e}")

    # 3b) Chat invoke (tiny)
    try:
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 10,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": "Reply with: OK"}],
        }
        resp = runtime.invoke_model(
            modelId=chat_model_id,
            body=json.dumps(body),
            accept="application/json",
            contentType="application/json",
        )
        payload = json.loads(resp["body"].read().decode("utf-8"))
        # If we got a JSON payload back, the model is callable
        if "content" in payload:
            report["chat_invoke_ok"] = True
        else:
            report["errors"].append("Chat invoke returned unexpected payload (no 'content').")
    except (BotoCoreError, ClientError, ValueError, KeyError) as e:
        report["errors"].append(f"Chat invoke failed: {e}")

    return report
