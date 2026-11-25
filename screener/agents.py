"""The three agents. Each is a LangChain tool-calling agent running on Amazon Bedrock."""

from langchain.agents import create_agent

from screener.config import get_llm
from screener.context import ScreeningRun
from screener.hooks import hooks_middleware
from screener.schemas import CandidateScore, InterviewKit, JobProfile
from screener.skills import skills_catalogue
from screener.tools import matcher_tools, skill_tools

# ---------------------------------------------------------------- Agent 1: JD Analyzer

JD_ANALYZER_PROMPT = f"""You are a senior technical recruiter who turns job descriptions into a precise job profile.

Available skills (call load_skill to read one before you start):
{skills_catalogue(["competency-framework"])}

Steps:
1. Load the competency-framework skill.
2. Read the job description and choose 4-6 competencies from the framework, with weights summing to 1.0.
3. Return the job profile."""


def analyze_jd(run: ScreeningRun, jd_text: str) -> JobProfile:
    agent = create_agent(
        get_llm(),
        tools=skill_tools(),
        system_prompt=JD_ANALYZER_PROMPT,
        middleware=[hooks_middleware(run)],
        response_format=JobProfile,
    )
    result = agent.invoke({"messages": [{"role": "user", "content": f"Job description:\n\n{jd_text}"}]})
    profile: JobProfile = result["structured_response"]

    total = sum(max(c.weight, 0) for c in profile.competencies) or 1.0
    for competency in profile.competencies:
        competency.weight = round(max(competency.weight, 0) / total, 3)
        competency.required_level = min(max(competency.required_level, 1), 4)
    return profile


# ---------------------------------------------------------------- Agent 2: Resume Matcher

MATCHER_PROMPT = f"""You are a fair, evidence-driven resume screener. You never see candidate identities.

Available skills (call load_skill to read one before you start):
{skills_catalogue(["scoring-rubric"])}

Steps:
1. Load the scoring-rubric skill and follow it strictly.
2. Read the resume of your assigned candidate with read_resume.
3. For each competency, find a job or project bullet that shows it being USED. A skill that appears only
   in a SKILLS/KEYWORDS list, a summary, or "attended training" is NOT evidence: score it 4 or lower.
4. Call submit_score once, scoring EVERY competency in the job profile using the exact competency names.
   The evidence must quote the bullet you relied on.
If a tool call is blocked, read the reason, fix it and call the tool again."""


def score_candidate(run: ScreeningRun, candidate_id: str, feedback: str = "") -> CandidateScore:
    agent = create_agent(
        get_llm(),
        tools=matcher_tools(run),
        system_prompt=MATCHER_PROMPT,
        middleware=[hooks_middleware(run, scope_id=candidate_id)],
    )
    task = (
        f"Assigned candidate: {candidate_id}\n\n"
        f"Job profile:\n{run.job_profile.model_dump_json(indent=2)}"
    )
    if feedback:
        task += f"\n\nIMPORTANT: your previous scorecard was rejected by the fairness guardrail: {feedback}\n" \
                "Score again using only job-relevant evidence."
    run.scores.pop(candidate_id, None)
    agent.invoke({"messages": [{"role": "user", "content": task}]})

    if candidate_id not in run.scores:
        raise RuntimeError(f"Matcher agent finished without submitting a score for {candidate_id}")
    return run.scores[candidate_id]


# ---------------------------------------------------------------- Agent 3: Interview Designer

DESIGNER_PROMPT = f"""You are an experienced hiring manager who designs structured interviews.

Available skills (call load_skill to read one before you start):
{skills_catalogue(["competency-framework"])}

Steps:
1. Load the competency-framework skill to calibrate questions to the required level.
2. Write 8 questions for THIS candidate, based on their scorecard:
   - at least 3 questions dig into specific projects or claims from the evidence
     (e.g. "You reduced model incidents by 60%. Walk me through how you detected drift."),
   - at least 3 questions probe their listed gaps directly,
   - no generic questions that could be asked to anyone.
3. Return the interview kit. focus_areas = 2-4 short phrases naming what this interview must verify.

Rules: never use names or contact details of anyone. Refer to the candidate only as {{{{candidate}}}}.
Never ask about age, family, health, religion, nationality or career-gap reasons."""


def design_interview(run: ScreeningRun, candidate_id: str, feedback: str = "") -> InterviewKit:
    agent = create_agent(
        get_llm(),
        tools=skill_tools(),
        system_prompt=DESIGNER_PROMPT,
        middleware=[hooks_middleware(run, scope_id=candidate_id)],
        response_format=InterviewKit,
    )
    task = (
        f"Job profile:\n{run.job_profile.model_dump_json(indent=2)}\n\n"
        f"Candidate scorecard:\n{run.scores[candidate_id].model_dump_json(indent=2)}"
    )
    if feedback:
        task += f"\n\nIMPORTANT: your previous kit was blocked by the PII guardrail: {feedback}"
    result = agent.invoke({"messages": [{"role": "user", "content": task}]})
    kit: InterviewKit = result["structured_response"]
    kit.candidate_id = candidate_id
    return kit
