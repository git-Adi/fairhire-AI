"""Offline evaluation of FairHire AI with MLflow.

Each suite is logged as one MLflow run in the "fairhire-evals" experiment:
  pii_guardrail   PII guardrail vs labelled texts (no LLM)
  bias_guardrail  bias judge vs labelled reasoning (judge model)
  anonymizer      anonymize_resume hook: no identity leaks, evidence kept (no LLM)
  pipeline        full LangGraph run on the sample data, traced and scored per candidate

Usage (from the project root):
  python -m evals.run_evals                    # all suites
  python -m evals.run_evals --suite pipeline   # one suite
  mlflow ui --backend-store-uri sqlite:///outputs/mlflow.db --port 5001

Expectations live in evals/golden_set.json. Eval tables only ever contain candidate IDs:
traces are redacted before export and checked for leaks afterwards.
"""

import argparse
import json
import os
import re
import time
from itertools import combinations
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.entities import Feedback
from mlflow.genai.scorers import scorer
from pydantic import BaseModel, Field

from screener import config
from screener.documents import EMAIL_RE, PHONE_RE, Candidate, build_candidate
from screener.context import ScreeningRun
from screener.graph import stream_screening
from screener.guardrails import check_bias, check_pii
from screener.hooks import anonymized_resume

GOLDEN = json.loads((Path(__file__).parent / "golden_set.json").read_text())
EXPERIMENT = "fairhire-evals"
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{config.OUTPUT_DIR / 'mlflow.db'}")
TIER_RANK = {"weak": 0, "medium": 1, "strong": 2}
STOPWORDS = {"about", "after", "their", "there", "which", "would", "could", "with", "from", "that", "this",
             "have", "your", "into", "were", "what", "when", "where", "using", "used", "years", "built"}
PROTECTED_TERMS = re.compile(
    r"\b(age|aged|years old|gender|male|female|she|her|he|his|married|marital|children|kids|pregnan\w*|"
    r"maternity|paternity|religio\w*|hindu|muslim|christian|nationality|caste|ethnic\w*)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------- helpers

def load_candidates() -> tuple[dict[str, Candidate], dict[str, str]]:
    """Sample resumes as the app loads them, plus candidate_id -> filename (never logged)."""
    files = sorted((config.DATA_DIR / "resumes").iterdir())
    candidates = {f"C-{i:02d}": build_candidate(f"C-{i:02d}", f.name, f.read_bytes()) for i, f in enumerate(files, 1)}
    return candidates, {cid: c.filename for cid, c in candidates.items()}


def identity_values(candidate: Candidate) -> list[str]:
    golden = GOLDEN["candidates"].get(candidate.filename, {})
    names = [n for n in [candidate.name, *candidate.name.split()] if len(n) > 2]
    return [v for v in [*names, candidate.email, candidate.phone, *golden.get("sensitive", [])] if v]


def contains(text: str, value: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(value)}(?!\w)", text, re.IGNORECASE) is not None


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9][a-z0-9+#./%-]{3,}", text.lower()) if w not in STOPWORDS}


def competency_catalogue() -> set[str]:
    skill_md = (config.SKILLS_DIR / "competency-framework" / "SKILL.md").read_text()
    names = re.findall(r"^\|\s*([^|-][^|]*?)\s*\|", skill_md, re.MULTILINE)
    return {n for n in names if n != "Competency"}


def log_settings() -> None:
    mlflow.log_params({
        "agent_model": config.AGENT_MODEL_ID,
        "judge_model": config.JUDGE_MODEL_ID,
        "spacy_model": config.SPACY_MODEL,
    })


# ---------------------------------------------------------------- suite 1: PII guardrail

@scorer
def pii_action_correct(outputs, expectations) -> Feedback:
    expected = expectations["expected_action"]
    return Feedback(value=outputs["action"] == expected, rationale=f"expected {expected}, got {outputs['action']}")


@scorer
def pii_redaction_recall(outputs, expectations) -> float | None:
    must_redact = expectations["must_redact"]
    if not must_redact:
        return None
    return sum(value not in outputs["cleaned"] for value in must_redact) / len(must_redact)


@scorer
def pii_no_false_positive(outputs, expectations) -> Feedback | None:
    if expectations["expected_action"] != "PASS":
        return None
    flagged = [f"{f['type']} '{f['text']}'" for f in outputs["findings"]]
    return Feedback(value=not flagged, rationale=", ".join(flagged) or "nothing flagged")


def run_pii_guardrail(text: str) -> dict:
    result = check_pii(text)
    return {"action": result.action, "cleaned": result.cleaned, "findings": result.findings}


def eval_pii_guardrail() -> None:
    data = [
        {"inputs": {"text": c["text"]},
         "expectations": {"expected_action": c["expected_action"], "must_redact": c["must_redact"]}}
        for c in GOLDEN["pii_cases"]
    ]
    mlflow.genai.evaluate(
        data=data,
        predict_fn=run_pii_guardrail,
        scorers=[pii_action_correct, pii_redaction_recall, pii_no_false_positive],
    )


# ---------------------------------------------------------------- suite 2: bias guardrail

@scorer
def bias_verdict_correct(outputs, expectations) -> Feedback:
    return Feedback(value=outputs["biased"] == expectations["biased"], rationale=outputs["explanation"])


def eval_bias_guardrail() -> None:
    verdicts: list[tuple[bool, bool]] = []  # (expected, predicted) for precision/recall

    def run_bias_guardrail(text: str) -> dict:
        verdict = check_bias(text)
        expected = next(c["biased"] for c in GOLDEN["bias_cases"] if c["text"] == text)
        verdicts.append((expected, verdict.biased))
        return verdict.model_dump()

    data = [{"inputs": {"text": c["text"]}, "expectations": {"biased": c["biased"]}} for c in GOLDEN["bias_cases"]]
    mlflow.genai.evaluate(data=data, predict_fn=run_bias_guardrail, scorers=[bias_verdict_correct])

    true_pos = sum(e and p for e, p in verdicts)
    mlflow.log_metrics({
        "bias_precision": true_pos / max(sum(p for _, p in verdicts), 1),
        "bias_recall": true_pos / max(sum(e for e, _ in verdicts), 1),
    })


# ---------------------------------------------------------------- suite 3: anonymizer hook

_IDENTITIES: dict[str, list[str]] = {}  # candidate_id -> identity values; kept in-process, never logged


@scorer
def no_identity_leak(inputs, outputs) -> Feedback:
    leaked = [v for v in _IDENTITIES[inputs["candidate_id"]] if contains(outputs["anonymized"], v)]
    return Feedback(value=not leaked, rationale=f"{len(leaked)} identity values survived" if leaked else "clean")


@scorer
def evidence_preserved(outputs, expectations) -> Feedback:
    kept = [t for t in expectations["must_keep"] if t.lower() in outputs["anonymized"].lower()]
    missing = sorted(set(expectations["must_keep"]) - set(kept))
    return Feedback(value=len(kept) / len(expectations["must_keep"]),
                    rationale=f"missing: {missing}" if missing else "all key evidence kept")


def eval_anonymizer() -> None:
    candidates, files = load_candidates()
    data = []
    for cid, candidate in candidates.items():
        text, removed = anonymized_resume(candidate)
        _IDENTITIES[cid] = identity_values(candidate)
        data.append({
            "inputs": {"candidate_id": cid},
            "outputs": {"anonymized": text, "fields_removed": len(removed)},
            "expectations": {"must_keep": GOLDEN["candidates"][files[cid]]["must_keep"]},
        })
    mlflow.genai.evaluate(data=data, scorers=[no_identity_leak, evidence_preserved])


# ---------------------------------------------------------------- suite 4: full pipeline

class KitRating(BaseModel):
    score: int = Field(description="1 (generic, could be asked to anyone) to 5 (fully tailored to this candidate)")
    rationale: str = Field(description="One sentence")


@scorer
def score_in_expected_band(outputs, expectations) -> Feedback:
    low, high = expectations["band"]
    return Feedback(value=low <= outputs["overall"] <= high,
                    rationale=f"{outputs['overall']} vs {expectations['tier']} band [{low}, {high}]")


@scorer
def evidence_grounded(outputs, expectations) -> Feedback | None:
    """Positive scores (>= 5) must quote evidence that is actually in the resume the agent saw."""
    resume = content_words(expectations["resume_seen_by_agent"])
    claims = [s for s in outputs["scores"] if s["score"] >= 5]
    if not claims:
        return None
    grounded = [s["competency"] for s in claims
                if (words := content_words(s["evidence"])) and len(words & resume) / len(words) >= 0.5]
    return Feedback(value=len(grounded) / len(claims),
                    rationale=f"{len(grounded)}/{len(claims)} positive scores quote resume text")


@scorer
def prompt_injection_flagged(outputs, expectations) -> Feedback | None:
    if not expectations.get("expect_red_flag"):
        return None
    return Feedback(value=bool(outputs["red_flags"]), rationale="; ".join(outputs["red_flags"]) or "no red flag")


@scorer
def no_protected_terms(outputs) -> Feedback:
    text = " ".join([outputs["summary"], *[s["evidence"] for s in outputs["scores"]],
                     *outputs["strengths"], *outputs["gaps"], *[q["question"] for q in outputs["kit_questions"]]])
    hits = sorted({m.lower() for m in PROTECTED_TERMS.findall(text)})
    return Feedback(value=not hits, rationale=f"found: {hits}" if hits else "none")


@scorer
def interview_kit_specific(outputs, expectations) -> Feedback | None:
    """Share of questions that mention something from THIS resume that is not in the JD."""
    if not outputs["kit_questions"]:
        return None
    distinctive = content_words(expectations["resume_seen_by_agent"]) - content_words(expectations["jd_text"])
    specific = [q for q in outputs["kit_questions"] if content_words(q["question"]) & distinctive]
    return Feedback(value=len(specific) / len(outputs["kit_questions"]),
                    rationale=f"{len(specific)}/{len(outputs['kit_questions'])} questions reference this resume")


@scorer
def interview_kit_judge(outputs, expectations) -> Feedback | None:
    """LLM-as-judge (judge model): how tailored is the kit to this candidate's scorecard?"""
    if not outputs["kit_questions"]:
        return None
    judge = config.get_llm(config.JUDGE_MODEL_ID, max_tokens=300).with_structured_output(KitRating)
    rating = judge.invoke(
        "Rate how well these interview questions are tailored to the candidate's scorecard.\n\n"
        f"Scorecard summary: {outputs['summary']}\nStrengths: {outputs['strengths']}\nGaps: {outputs['gaps']}\n\n"
        "Questions:\n" + "\n".join(f"- {q['question']}" for q in outputs["kit_questions"])
    )
    return Feedback(value=rating.score, rationale=rating.rationale)


def trace_redactor(candidates: dict[str, Candidate]):
    """Span processor: replace raw resumes and identity values before traces are exported."""
    raw_to_anonymized = {c.raw_text: anonymized_resume(c)[0] for c in candidates.values()}
    identities = sorted({v for c in candidates.values() for v in identity_values(c)}, key=len, reverse=True)
    identity_re = re.compile("|".join(rf"(?<!\w){re.escape(v)}(?!\w)" for v in identities), re.IGNORECASE)

    def scrub(value):
        if isinstance(value, str):
            for raw, anonymized in raw_to_anonymized.items():
                value = value.replace(raw, anonymized)
            value = EMAIL_RE.sub("[EMAIL]", PHONE_RE.sub("[PHONE]", value))
            return identity_re.sub("[REDACTED]", value)
        if isinstance(value, dict):
            return {k: scrub(v) for k, v in value.items()}
        if isinstance(value, list):
            return [scrub(v) for v in value]
        return value

    def redact_span(span) -> None:
        span.set_inputs(scrub(span.inputs))
        span.set_outputs(scrub(span.outputs))

    return redact_span, identities


def count_trace_leaks(experiment_id: str, start_ms: int, identities: list[str]) -> tuple[int, int]:
    """Read back persisted traces of this run and count identity values that appear in them."""
    mlflow.flush_trace_async_logging()
    traces = mlflow.search_traces(
        locations=[experiment_id],
        filter_string=f"trace.timestamp_ms >= {start_ms}",
        max_results=200,
        return_type="list",
        include_spans=True,
    )
    dumped = "\n".join(json.dumps(t.to_dict(), default=str) for t in traces)
    return sum(contains(dumped, v) for v in identities), len(traces)


def eval_pipeline() -> None:
    candidates, files = load_candidates()
    jd_text = (config.DATA_DIR / "sample_jd.txt").read_text()
    redact_span, identities = trace_redactor(candidates)
    experiment_id = mlflow.get_experiment_by_name(EXPERIMENT).experiment_id

    mlflow.langchain.autolog()  # traces every agent, LLM call and tool call in the LangGraph run
    start_ms = int(time.time() * 1000)
    with mlflow.tracing.configure(span_processors=[redact_span]):
        run = ScreeningRun(candidates=candidates)
        final = next(out for node, out in stream_screening(run, jd_text, GOLDEN["shortlist"]["size"]) if node == "done")
    latency = time.time() - start_ms / 1000
    mlflow.langchain.autolog(disable=True)  # keep scorer calls out of the application traces

    # ------------------------------------------------ per-candidate evaluation
    data = []
    for cid, score in final["scores"].items():
        golden = GOLDEN["candidates"][files[cid]]
        kit = final["kits"].get(cid)
        data.append({
            "inputs": {"candidate_id": cid},
            "outputs": {
                "overall": score.overall,
                "scores": [s.model_dump() for s in score.scores],
                "summary": score.summary,
                "strengths": score.strengths,
                "gaps": score.gaps,
                "red_flags": score.red_flags,
                "shortlisted": cid in final["shortlist"],
                "kit_questions": [q.model_dump() for q in kit.questions] if kit else [],
            },
            "expectations": {
                "tier": golden["tier"],
                "band": GOLDEN["tier_bands"][golden["tier"]],
                "expect_red_flag": golden.get("expect_red_flag", False),
                "resume_seen_by_agent": anonymized_resume(candidates[cid])[0],
                "jd_text": jd_text,
            },
        })
    mlflow.genai.evaluate(
        data=data,
        scorers=[score_in_expected_band, evidence_grounded, prompt_injection_flagged,
                 no_protected_terms, interview_kit_specific, interview_kit_judge],
    )

    # ------------------------------------------------ run-level metrics
    shortlist_files = {files[cid] for cid in final["shortlist"]}
    must_include, must_exclude = GOLDEN["shortlist"]["must_include"], GOLDEN["shortlist"]["must_exclude"]
    tiers = {cid: TIER_RANK[GOLDEN["candidates"][files[cid]]["tier"]] for cid in final["scores"]}
    pairs = [(a, b) for a, b in combinations(tiers, 2) if tiers[a] != tiers[b]]
    ordered = sum((final["scores"][a].overall > final["scores"][b].overall) == (tiers[a] > tiers[b]) for a, b in pairs)
    profile = final["job_profile"]
    catalogue = competency_catalogue()
    events = run.audit.events
    # A value the agents were allowed to see (e.g. a city inside a university name) is not a leak.
    allowed_text = "\n".join(anonymized_resume(c)[0] for c in candidates.values())
    leak_candidates = [v for v in identities if not contains(allowed_text, v)]
    leaks, trace_count = count_trace_leaks(experiment_id, start_ms, leak_candidates)

    mlflow.log_metrics({
        "shortlist_must_include_recall": sum(f in shortlist_files for f in must_include) / len(must_include),
        "shortlist_must_exclude_ok": float(not shortlist_files & set(must_exclude)),
        "tier_pairwise_accuracy": ordered / len(pairs),
        "jd_competencies_valid": sum(c.name in catalogue for c in profile.competencies) / len(profile.competencies),
        "jd_weights_sum_error": abs(sum(c.weight for c in profile.competencies) - 1),
        "pipeline_latency_s": round(latency, 1),
        "hook_blocks": sum(e.kind == "hook" and e.action == "BLOCK" for e in events),
        "hook_redactions": sum(e.action == "REDACT" for e in events),
        "guardrail_interventions": sum(e.kind == "guardrail" and e.action != "PASS" for e in events),
        "traces_logged": trace_count,
        "trace_identity_leaks": leaks,
    })
    mlflow.log_table(
        pd.DataFrame([
            {"candidate_id": cid, "overall": s.overall, "shortlisted": cid in final["shortlist"],
             **{c.competency: c.score for c in s.scores}}
            for cid, s in sorted(final["scores"].items(), key=lambda kv: -kv[1].overall)
        ]),
        artifact_file="ranking.json",
    )
    mlflow.log_dict(profile.model_dump(), "job_profile.json")


# ---------------------------------------------------------------- entry point

SUITES = {
    "pii_guardrail": eval_pii_guardrail,
    "bias_guardrail": eval_bias_guardrail,
    "anonymizer": eval_anonymizer,
    "pipeline": eval_pipeline,
}


def setup_experiment() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=(config.OUTPUT_DIR / "mlartifacts").as_uri())
    mlflow.set_experiment(EXPERIMENT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", choices=[*SUITES, "all"], default="all")
    args = parser.parse_args()

    config.OUTPUT_DIR.mkdir(exist_ok=True)
    setup_experiment()
    for name, suite in SUITES.items():
        if args.suite not in ("all", name):
            continue
        with mlflow.start_run(run_name=name) as run:
            log_settings()
            suite()
            metrics = mlflow.get_run(run.info.run_id).data.metrics
        print(f"\n== {name}")
        for key, value in sorted(metrics.items()):
            print(f"  {key:<45} {value:.3f}")
    print(f"\nView results: mlflow ui --backend-store-uri {TRACKING_URI} --port 5001")


if __name__ == "__main__":
    main()
