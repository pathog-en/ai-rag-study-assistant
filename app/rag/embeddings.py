import os
import hashlib
from typing import List

import boto3
from botocore.exceptions import BotoCoreError, ClientError


def _mock_embedding(text: str, dim: int = 1024) -> List[float]:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    out = []
    for i in range(dim):
        b = h[i % len(h)]
        out.append(((b / 255.0) * 2.0) - 1.0)
    return out


def embed_texts(texts: List[str]) -> List[List[float]]:
    use_bedrock = os.getenv("USE_BEDROCK", "false").lower() == "true"
    if not use_bedrock:
        return [_mock_embedding(t) for t in texts]

    region = os.getenv("AWS_REGION", "us-east-1")
    model_id = os.getenv("BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
    client = boto3.client("bedrock-runtime", region_name=region)

    vectors: List[List[float]] = []
    for t in texts:
        try:
            resp = client.invoke_model(
                modelId=model_id,
                body=f'{{"inputText": {t!r}}}',
                accept="application/json",
                contentType="application/json",
            )
            body = resp["body"].read().decode("utf-8")
            import json
            emb = json.loads(body)["embedding"]
            vectors.append(emb)
        except (BotoCoreError, ClientError, KeyError, ValueError):
            vectors.append(_mock_embedding(t))

    return vectors
