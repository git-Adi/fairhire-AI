"""Hooks: plain, deterministic code that runs around EVERY tool call an agent makes.

PreToolUse  hooks run before a tool executes and can BLOCK the call (the agent sees why and retries).
PostToolUse hooks run after a tool executes and can rewrite its output before the model sees it.

They are attached to each agent as LangChain middleware (see `hooks_middleware`).
"""

import csv
from collections.abc import Callable

from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage

from screener import pii
from screener.context import SCORES_FILE, ScreeningRun
from screener.documents import EMAIL_RE, PHONE_RE, Candidate

PreHook = Callable[[ScreeningRun, str, str, dict], str | None]  # returns a block reason, or None
PostHook = Callable[[ScreeningRun, str, str, dict, str], str]  # returns the (possibly new) output


# ---------------------------------------------------------------- PreToolUse hooks

def enforce_candidate_scope(run: ScreeningRun, scope_id: str, tool: str, args: dict) -> str | None:
    """Each matcher agent may only touch the one candidate it was assigned."""
    requested = args.get("candidate_id")
    if scope_id and requested and requested != scope_id:
        return f"You are assigned to {scope_id} only; access to {requested} is not allowed."
    return None


def validate_score_submission(run: ScreeningRun, scope_id: str, tool: str, args: dict) -> str | None:
    """Reject incomplete scorecards so every candidate is scored on exactly the same competencies."""
    if tool != "submit_score":
        return None
    expected = set(run.job_profile.weights)
    submitted = [s.get("competency") for s in args.get("scores", [])]
    missing = expected - set(submitted)
    unknown = set(submitted) - expected
    no_evidence = [s.get("competency") for s in args.get("scores", []) if not str(s.get("evidence", "")).strip()]
    problems = []
    if missing:
        problems.append(f"missing competencies: {sorted(missing)}")
    if unknown:
        problems.append(f"unknown competencies (use exact names): {sorted(unknown)}")
    if len(submitted) != len(set(submitted)):
        problems.append("a competency was scored more than once")
    if no_evidence:
        problems.append(f"no evidence given for: {no_evidence}")
    return "; ".join(problems) or None


# ---------------------------------------------------------------- PostToolUse hooks

def anonymized_resume(candidate: Candidate) -> tuple[str, list[dict]]:
    """The resume exactly as the agents see it, plus what was removed."""
    # Contact lines usually also hold the address/city, so drop them whole.
    removed, lines = [], []
    for line in candidate.raw_text.splitlines():
        if EMAIL_RE.search(line) or PHONE_RE.search(line):
            removed.append({"type": "CONTACT_LINE", "text": line.strip(), "score": 1.0})
            lines.append("<CONTACT_DETAILS>")
        else:
            lines.append(line)
    text = "\n".join(lines)

    known = [n for n in [candidate.name, *candidate.name.split()] if len(n) > 2]
    findings = pii.find_pii(text, pii.RESUME_ENTITIES, known_names=known, threshold=0.5)
    removed += pii.describe(text, findings)
    return f"Candidate {candidate.candidate_id}\n" + pii.redact(text, findings), removed


def anonymize_resume(run: ScreeningRun, scope_id: str, tool: str, args: dict, output: str) -> str:
    """Blind screening: strip identity and protected attributes before the model reads a resume."""
    if tool != "read_resume":
        return output
    candidate = run.candidates[args["candidate_id"]]
    text, removed = anonymized_resume(candidate)
    types = sorted({r["type"] for r in removed})
    run.audit.hook(
        "anonymize_resume", "REDACT", candidate.candidate_id,
        f"{len(removed)} fields removed: {', '.join(types) or 'none'}",
    )
    return text


def record_score(run: ScreeningRun, scope_id: str, tool: str, args: dict, output: str) -> str:
    """Write every accepted scorecard to outputs/scores.csv for the audit trail."""
    if tool != "submit_score":
        return output
    score = run.scores[args["candidate_id"]]
    is_new_file = not SCORES_FILE.exists()
    with SCORES_FILE.open("a", newline="") as f:
        writer = csv.writer(f)
        if is_new_file:
            writer.writerow(["candidate_id", "overall", *[s.competency for s in score.scores]])
        writer.writerow([score.candidate_id, score.overall, *[s.score for s in score.scores]])
    run.audit.hook("record_score", "SAVED", score.candidate_id, f"overall {score.overall}/10 -> scores.csv")
    return output


PRE_TOOL_USE: list[PreHook] = [enforce_candidate_scope, validate_score_submission]
POST_TOOL_USE: list[PostHook] = [anonymize_resume, record_score]


# ---------------------------------------------------------------- wiring into LangChain

def hooks_middleware(run: ScreeningRun, scope_id: str = ""):
    """LangChain middleware that runs the hook lists around each tool call."""

    @wrap_tool_call
    def run_hooks(request, handler):
        call = request.tool_call
        for hook in PRE_TOOL_USE:
            reason = hook(run, scope_id, call["name"], call["args"])
            if reason:
                run.audit.hook(hook.__name__, "BLOCK", scope_id, reason)
                return ToolMessage(
                    content=f"Blocked by hook '{hook.__name__}': {reason}. Fix this and call the tool again.",
                    tool_call_id=call["id"],
                    name=call["name"],
                    status="error",
                )

        result = handler(request)
        if isinstance(result, ToolMessage) and result.status != "error":
            output = str(result.content)
            for hook in POST_TOOL_USE:
                output = hook(run, scope_id, call["name"], call["args"], output)
            result.content = output
        return result

    return run_hooks
