"""Settings and model factory. Values come from exported env vars or the .env file."""

import os
from functools import lru_cache
from pathlib import Path

from botocore.config import Config
from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse

ROOT_DIR = Path(__file__).resolve().parent.parent
SKILLS_DIR = ROOT_DIR / "skills"
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"

load_dotenv(ROOT_DIR / ".env")  # exported variables take precedence

AWS_PROFILE = os.getenv("AWS_PROFILE", "mugambo")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AGENT_MODEL_ID = os.getenv("AGENT_MODEL_ID", "us.amazon.nova-pro-v1:0")
JUDGE_MODEL_ID = os.getenv("JUDGE_MODEL_ID", "us.amazon.nova-lite-v1:0")
SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_md")


@lru_cache
def get_llm(model_id: str = AGENT_MODEL_ID, max_tokens: int = 4096) -> ChatBedrockConverse:
    """Bedrock chat model authenticated with the named AWS CLI profile."""
    return ChatBedrockConverse(
        model=model_id,
        region_name=AWS_REGION,
        credentials_profile_name=AWS_PROFILE,
        temperature=0,
        max_tokens=max_tokens,
        config=Config(retries={"max_attempts": 8, "mode": "adaptive"}),  # back off on throttling
    )
