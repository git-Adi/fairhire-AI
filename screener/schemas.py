"""Data models shared by the agents, graph and UI.

No numeric/length limits in these schemas: Bedrock structured output rejects them.
Ranges are stated in descriptions and enforced in code.
"""

from pydantic import BaseModel, Field


class Competency(BaseModel):
    name: str = Field(description="Competency name, exactly as in the competency framework")
    weight: float = Field(description="Importance for this role, 0-1; all weights sum to 1")
    required_level: int = Field(description="Level the role needs: 1, 2, 3 or 4")
    reason: str = Field(description="One line from the JD that justifies this competency")


class JobProfile(BaseModel):
    title: str
    seniority: str = Field(description="e.g. Junior, Mid, Senior, Lead")
    must_haves: list[str]
    nice_to_haves: list[str]
    competencies: list[Competency] = Field(description="4-6 competencies")

    @property
    def weights(self) -> dict[str, float]:
        return {c.name: c.weight for c in self.competencies}


class CompetencyScore(BaseModel):
    competency: str
    score: float = Field(description="0-10")
    evidence: str = Field(description="Quote or close paraphrase from the resume")


class CandidateScore(BaseModel):
    candidate_id: str
    scores: list[CompetencyScore]
    overall: float
    summary: str
    strengths: list[str]
    gaps: list[str]
    red_flags: list[str] = []
    bias_flagged: bool = False
    bias_note: str = ""


class InterviewQuestion(BaseModel):
    competency: str
    question: str
    why_ask: str = Field(description="Which strength or gap of this candidate the question probes")
    good_answer: str = Field(description="What a strong answer would include")


class InterviewKit(BaseModel):
    candidate_id: str
    focus_areas: list[str]
    questions: list[InterviewQuestion] = Field(description="8 questions")


class AuditEvent(BaseModel):
    ts: str
    kind: str  # "hook" or "guardrail"
    name: str
    action: str  # e.g. PASS, MASK, BLOCK, REDACT, SAVED
    candidate_id: str = ""
    detail: str = ""
