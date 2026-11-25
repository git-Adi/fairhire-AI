"""Tools the agents can call. Built per run so they share that run's state."""

from langchain_core.tools import BaseTool, tool

from screener import skills
from screener.context import ScreeningRun
from screener.schemas import CandidateScore, CompetencyScore


def skill_tools() -> list[BaseTool]:
    @tool("load_skill")
    def load_skill(name: str) -> str:
        """Load the full instructions and resources of a skill by name."""
        try:
            return skills.load_skill(name)
        except FileNotFoundError as error:
            return str(error)

    return [load_skill]


def matcher_tools(run: ScreeningRun) -> list[BaseTool]:
    @tool("read_resume")
    def read_resume(candidate_id: str) -> str:
        """Read a candidate's resume. Identity and personal details are removed before you see it."""
        return run.candidates[candidate_id].raw_text

    @tool("submit_score")
    def submit_score(
        candidate_id: str,
        scores: list[CompetencyScore],
        summary: str,
        strengths: list[str],
        gaps: list[str],
        red_flags: list[str],
    ) -> str:
        """Submit the final scorecard: one entry per job-profile competency, each with resume evidence."""
        for s in scores:
            s.score = min(max(s.score, 0.0), 10.0)
        compute = skills.load_skill_script("scoring-rubric", "compute_score")
        overall = compute.weighted_score({s.competency: s.score for s in scores}, run.job_profile.weights)
        run.scores[candidate_id] = CandidateScore(
            candidate_id=candidate_id,
            scores=scores,
            overall=overall,
            summary=summary,
            strengths=strengths,
            gaps=gaps,
            red_flags=red_flags,
        )
        return f"Scorecard saved. Overall {overall}/10 (computed by scoring-rubric/compute_score.py)."

    return [*skill_tools(), read_resume, submit_score]
