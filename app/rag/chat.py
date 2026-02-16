import os
import json
import boto3
from botocore.exceptions import BotoCoreError, ClientError

def generate_answer(prompt: str) -> str:
    if os.getenv("USE_BEDROCK", "true").lower() not in ("1", "true", "yes"):
        return "Bedrock disabled (USE_BEDROCK=false)."

    region = os.getenv("AWS_REGION", "us-east-1")
    model_id = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")

    try:
        client = boto3.client("bedrock-runtime", region_name=region)

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 500,
            "temperature": 0.2,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ],
        }

        resp = client.invoke_model(
            modelId=model_id,
            body=json.dumps(body),
            accept="application/json",
            contentType="application/json",
        )

        data = json.loads(resp["body"].read())
        # Claude returns content list
        return data["content"][0]["text"]

    except (ClientError, BotoCoreError, Exception) as e:
        return f"Bedrock invocation failed: {type(e).__name__}: {e}"

