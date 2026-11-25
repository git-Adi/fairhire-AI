"""LangGraph workflow.

  analyze_jd -> score_candidate (x N, parallel) -> bias_guardrail
             -> design_interview (x shortlist, parallel) -> pii_guardrail -> END
"""

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from screener import agents
from screener.context import ScreeningRun
from screener.guardrails import check_bias, check_pii
from screener.schemas import CandidateScore, InterviewKit, JobProfile

MAX_PARALLEL_AGENTS = 3  # keeps Bedrock request rates under the default account quota


def merge(left: dict, right: dict) -> dict:
    return {**left, **right}


class ScreeningState(TypedDict, total=False):
    jd_text: str
    shortlist_size: int
    job_profile: JobProfile
    scores: Annotated[dict[str, CandidateScore], merge]
    shortlist: list[str]
    kits: Annotated[dict[str, InterviewKit], merge]


class CandidateTask(TypedDict):
    candidate_id: str


def build_graph(run: ScreeningRun):
    # ------------------------------------------------------------ nodes

    def analyze_jd(state: ScreeningState) -> dict:
        run.job_profile = agents.analyze_jd(run, state["jd_text"])
        return {"job_profile": run.job_profile}

    def score_candidate(task: CandidateTask) -> dict:
        cid = task["candidate_id"]
        return {"scores": {cid: agents.score_candidate(run, cid)}}

    def bias_guardrail(state: ScreeningState) -> dict:
        def audit(score: CandidateScore) -> CandidateScore:
            cid = score.candidate_id
            verdict = check_bias(score)
            if not verdict.biased:
                run.audit.guardrail("bias_check", "PASS", cid)
                return score

            run.audit.guardrail("bias_check", "BLOCK", cid, f"{verdict.attributes}: {verdict.explanation}")
            rescored = agents.score_candidate(run, cid, feedback=verdict.explanation)
            second = check_bias(rescored)
            if second.biased:
                rescored.bias_flagged, rescored.bias_note = True, second.explanation
                run.audit.guardrail("bias_check", "HUMAN_REVIEW", cid, second.explanation)
            else:
                run.audit.guardrail("bias_check", "PASS", cid, "passed after re-scoring")
            return rescored

        with ThreadPoolExecutor(MAX_PARALLEL_AGENTS) as pool:
            checked = list(pool.map(audit, state["scores"].values()))
        ranked = sorted(checked, key=lambda s: s.overall, reverse=True)
        return {
            "scores": {s.candidate_id: s for s in checked},
            "shortlist": [s.candidate_id for s in ranked[: state["shortlist_size"]]],
        }

    def design_interview(task: CandidateTask) -> dict:
        cid = task["candidate_id"]
        return {"kits": {cid: agents.design_interview(run, cid)}}

    def pii_guardrail(state: ScreeningState) -> dict:
        scores = {}
        for cid, score in state["scores"].items():
            result = check_pii(score)
            run.audit.guardrail("pii_check", result.action, cid, _summary(result.findings, "scorecard"))
            scores[cid] = result.cleaned

        def clean_kit(kit: InterviewKit) -> InterviewKit:
            cid = kit.candidate_id
            result = check_pii(kit)
            if result.action == "BLOCK":
                run.audit.guardrail("pii_check", "BLOCK", cid, _summary(result.findings, "interview kit"))
                retry = agents.design_interview(run, cid, feedback="it contained contact details; remove them")
                result = check_pii(retry)
            run.audit.guardrail("pii_check", result.action, cid, _summary(result.findings, "interview kit"))
            return result.cleaned

        with ThreadPoolExecutor(MAX_PARALLEL_AGENTS) as pool:
            kits = list(pool.map(clean_kit, state["kits"].values()))
        return {"scores": scores, "kits": {k.candidate_id: k for k in kits}}

    # ------------------------------------------------------------ fan-out edges

    def fan_out_scoring(state: ScreeningState) -> list[Send]:
        return [Send("score_candidate", {"candidate_id": cid}) for cid in run.candidates]

    def fan_out_interviews(state: ScreeningState) -> list[Send]:
        return [Send("design_interview", {"candidate_id": cid}) for cid in state["shortlist"]]

    graph = StateGraph(ScreeningState)
    graph.add_node("analyze_jd", analyze_jd)
    graph.add_node("score_candidate", score_candidate)
    graph.add_node("bias_guardrail", bias_guardrail)
    graph.add_node("design_interview", design_interview)
    graph.add_node("pii_guardrail", pii_guardrail)

    graph.add_edge(START, "analyze_jd")
    graph.add_conditional_edges("analyze_jd", fan_out_scoring, ["score_candidate"])
    graph.add_edge("score_candidate", "bias_guardrail")
    graph.add_conditional_edges("bias_guardrail", fan_out_interviews, ["design_interview"])
    graph.add_edge("design_interview", "pii_guardrail")
    graph.add_edge("pii_guardrail", END)
    return graph.compile()


def _summary(findings: list[dict], where: str) -> str:
    if not findings:
        return f"{where}: clean"
    return f"{where}: " + ", ".join(f"{f['type']} '{f['text']}'" for f in findings)


def stream_screening(run: ScreeningRun, jd_text: str, shortlist_size: int) -> Iterator[tuple[str, dict]]:
    """Yield (node_name, node_output) as each step finishes, then ("done", final_state)."""
    graph = build_graph(run)
    final_state: dict = {}
    for mode, chunk in graph.stream(
        {"jd_text": jd_text, "shortlist_size": shortlist_size},
        config={"max_concurrency": MAX_PARALLEL_AGENTS},
        stream_mode=["updates", "values"],
    ):
        if mode == "values":
            final_state = chunk
        else:
            for node, output in chunk.items():
                yield node, output or {}
    yield "done", final_state
