import os
import json
import boto3
from botocore.exceptions import BotoCoreError, ClientError


def generate_answer(prompt: str) -> str:
    use_bedrock = os.getenv("USE_BEDROCK", "false").lower() == "true"
    if not use_bedrock:
        return (
            "Bedrock generation disabled. "
            "Retrieval completed successfully; enable USE_BEDROCK to generate answers."
        )

    region = os.getenv("AWS_REGION", "us-east-1")
    model_id = os.getenv(
        "BEDROCK_CHAT_MODEL_ID",
        "anthropic.claude-3-5-sonnet-20240620-v1:0",
    )

    client = boto3.client("bedrock-runtime", region_name=region)

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 600,
        "temperature": 0.2,
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps(body),
            accept="application/json",
            contentType="application/json",
        )
        payload = json.loads(response["body"].read().decode("utf-8"))
        return "".join(block.get("text", "") for block in payload.get("content", []))
    except (BotoCoreError, ClientError, ValueError, KeyError):
        return "Bedrock invocation failed. Check credentials, region, and model access."
