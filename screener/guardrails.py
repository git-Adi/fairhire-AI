"""The two LLM guardrails. Both check what the model WROTE, not what it was given.

1. Bias guardrail: a small judge model checks scoring reasoning for protected attributes.
2. PII guardrail: Presidio scans final outputs before anything is shown or downloaded.
"""

from dataclasses import dataclass, field
from typing import Literal, TypeVar

from pydantic import BaseModel, Field

from screener import pii
from screener.config import JUDGE_MODEL_ID, get_llm
from screener.schemas import CandidateScore

# ---------------------------------------------------------------- 1. Bias guardrail

PROTECTED_ATTRIBUTES = (
    "age or generation, gender, pregnancy or parental leave, marital or family status, "
    "nationality, ethnicity, caste, religion, disability or health, appearance, "
    "or proxies for these such as graduation year used to imply age, or 'culture fit'"
)

BIAS_JUDGE_PROMPT = f"""You are a fairness auditor for a hiring system.
Read the reasoning a screening model wrote about a candidate. Decide whether any part of it
uses or refers to a protected attribute: {PROTECTED_ATTRIBUTES}.

Neutral mentions of job-relevant facts are fine (skills, projects, employers, years of experience
used to judge scope). Mentioning a career gap neutrally is fine; treating it as a weakness or
speculating about its reason is NOT.

Reasoning to audit:
---
{{reasoning}}
---"""


class BiasVerdict(BaseModel):
    biased: bool
    attributes: list[str] = Field(default_factory=list, description="Protected attributes referenced")
    explanation: str = Field(description="One sentence, quoting the problem text if biased")


def _reasoning_text(score: CandidateScore) -> str:
    lines = [f"Summary: {score.summary}"]
    lines += [f"{s.competency} ({s.score}/10): {s.evidence}" for s in score.scores]
    lines += [f"Strength: {s}" for s in score.strengths]
    lines += [f"Gap: {g}" for g in score.gaps]
    lines += [f"Red flag: {r}" for r in score.red_flags]
    return "\n".join(lines)


def check_bias(text_or_score: str | CandidateScore) -> BiasVerdict:
    reasoning = text_or_score if isinstance(text_or_score, str) else _reasoning_text(text_or_score)
    judge = get_llm(JUDGE_MODEL_ID, max_tokens=512).with_structured_output(BiasVerdict)
    return judge.invoke(BIAS_JUDGE_PROMPT.format(reasoning=reasoning))


# ---------------------------------------------------------------- 2. PII output guardrail

PiiAction = Literal["PASS", "MASK", "BLOCK"]
Model = TypeVar("Model", bound=BaseModel)


@dataclass
class PiiResult:
    action: PiiAction
    cleaned: object  # same type as the input (str or pydantic model)
    findings: list[dict] = field(default_factory=list)


def _scrub(value, findings: list[dict]):
    """Redact every string inside nested dicts/lists, collecting what was found."""
    if isinstance(value, str):
        hits = pii.find_pii(value, pii.OUTPUT_ENTITIES)
        findings.extend(pii.describe(value, hits))
        return pii.redact(value, hits, placeholder="[REDACTED]")
    if isinstance(value, dict):
        return {k: (v if k == "candidate_id" else _scrub(v, findings)) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub(v, findings) for v in value]
    return value


def check_pii(output: str | Model) -> PiiResult:
    """PASS: nothing found. MASK: names/URLs redacted. BLOCK: contact details leaked (regenerate)."""
    findings: list[dict] = []
    if isinstance(output, BaseModel):
        cleaned = type(output).model_validate(_scrub(output.model_dump(), findings))
    else:
        cleaned = _scrub(output, findings)

    if not findings:
        action: PiiAction = "PASS"
    elif any(f["type"] in pii.CONTACT_ENTITIES for f in findings):
        action = "BLOCK"
    else:
        action = "MASK"
    return PiiResult(action=action, cleaned=cleaned, findings=findings)
