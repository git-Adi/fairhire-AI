"""FairHire AI: blind resume screening with agents, skills, hooks and guardrails.

Run with:  streamlit run app.py
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from screener import config
from screener.context import AUDIT_FILE, SCORES_FILE, ScreeningRun
from screener.documents import build_candidate, interview_kit_docx
from screener.graph import stream_screening
from screener.guardrails import check_bias, check_pii
from screener.hooks import anonymized_resume

st.set_page_config(page_title="FairHire AI", page_icon="🧑‍💼", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem;}
      .hero h1 {margin-bottom: 0;}
      .hero p {color: #6b7280; margin-top: .25rem;}
      .step {border-radius: 10px; padding: .6rem .8rem; text-align: center; font-size: .85rem;
             border: 1px solid rgba(128,128,128,.25);}
      .step.todo {opacity: .45;}
      .step.running {border-color: #f59e0b; background: rgba(245,158,11,.10);}
      .step.done {border-color: #10b981; background: rgba(16,185,129,.10);}
      .tag {display:inline-block; padding:.1rem .5rem; border-radius:999px; font-size:.75rem; margin-right:.25rem;}
      .PASS, .SAVED {background:#d1fae5; color:#065f46;}
      .MASK, .REDACT, .REVEAL {background:#fef3c7; color:#92400e;}
      .BLOCK, .HUMAN_REVIEW {background:#fee2e2; color:#991b1b;}
    </style>
    """,
    unsafe_allow_html=True,
)

PIPELINE = [
    ("analyze_jd", "🤖 JD Analyzer"),
    ("score_candidate", "🤖 Resume Matcher"),
    ("bias_guardrail", "🛡️ Bias Guardrail"),
    ("design_interview", "🤖 Interview Designer"),
    ("pii_guardrail", "🛡️ PII Guardrail"),
]

state = st.session_state
state.setdefault("jd_text", "")
state.setdefault("run", None)
state.setdefault("result", None)
state.setdefault("revealed_logged", False)


# ---------------------------------------------------------------- helpers

def label(cid: str) -> str:
    if state.get("reveal") and state.run:
        return f"{state.run.candidates[cid].name} ({cid})"
    return cid


def render_pipeline(placeholder, done: set[str], running: str | None, progress: str = "") -> None:
    """Redraw the step chips in place (placeholder is an st.empty)."""
    with placeholder.container():
        for col, (node, title) in zip(st.columns(len(PIPELINE)), PIPELINE):
            status = "done" if node in done else "running" if node == running else "todo"
            icon = {"done": "✅", "running": "⏳", "todo": "▫️"}[status]
            suffix = f" {progress}" if status == "running" and progress else ""
            col.markdown(f'<div class="step {status}">{icon} {title}{suffix}</div>', unsafe_allow_html=True)


def render_activity(container) -> None:
    container.empty()
    if not state.run:
        container.caption("Run a screening to see hooks and guardrails fire here.")
        return
    with container.container():
        for event in reversed(state.run.audit.events[-25:]):
            icon = "🪝" if event.kind == "hook" else "🛡️"
            st.markdown(
                f'{icon} <span class="tag {event.action}">{event.action}</span> '
                f"**{event.name}** {event.candidate_id}<br><small>{event.detail}</small>",
                unsafe_allow_html=True,
            )


def load_candidates(uploads, use_samples: bool) -> dict:
    files = [(f.name, f.getvalue()) for f in uploads]
    if use_samples or not files:
        files = [(p.name, p.read_bytes()) for p in sorted((config.DATA_DIR / "resumes").iterdir())]
    candidates = {}
    for i, (name, data) in enumerate(files, start=1):
        cid = f"C-{i:02d}"
        candidates[cid] = build_candidate(cid, name, data)
    return candidates


def radar(score, profile) -> go.Figure:
    names = [c.name for c in profile.competencies]
    by_name = {s.competency: s.score for s in score.scores}
    required = [c.required_level * 2.5 for c in profile.competencies]  # L1-L4 mapped onto 0-10
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=required + required[:1], theta=names + names[:1],
                                  name="Role requirement", line=dict(dash="dash", color="#9ca3af")))
    values = [by_name.get(n, 0) for n in names]
    fig.add_trace(go.Scatterpolar(r=values + values[:1], theta=names + names[:1], fill="toself",
                                  name=label(score.candidate_id), line=dict(color="#6366f1")))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 10])), height=380,
                      margin=dict(l=40, r=40, t=30, b=30), legend=dict(orientation="h"))
    return fig


# ---------------------------------------------------------------- sidebar

with st.sidebar:
    st.subheader("⚙️ Setup")
    st.caption(f"AWS profile **{config.AWS_PROFILE}** · {config.AWS_REGION}")
    st.caption(f"Agents: `{config.AGENT_MODEL_ID}`")
    st.caption(f"Bias judge: `{config.JUDGE_MODEL_ID}`")
    shortlist_size = st.slider("Shortlist size", 1, 5, 3)
    if state.pop("reset_reveal", False):  # a new run starts blind again
        state.reveal = False
    st.toggle(
        "🔓 Reveal identities",
        key="reveal",
        disabled=state.result is None,
        help="Names come from the identity vault, never from the LLM. Available only after ranking.",
    )
    if state.get("reveal") and state.run and not state.revealed_logged:
        state.run.audit.record("hook", "identity_reveal", "REVEAL", detail="HR revealed identities after ranking")
        state.revealed_logged = True

    st.divider()
    st.subheader("Live activity")
    activity = st.empty()
    render_activity(activity)


# ---------------------------------------------------------------- header

st.markdown(
    '<div class="hero"><h1>🧑‍💼 FairHire AI</h1>'
    "<p>Blind resume screening · 3 agents · 2 skills · 4 hooks · 2 guardrails · "
    "LangGraph on Amazon Bedrock</p></div>",
    unsafe_allow_html=True,
)

tab_screen, tab_board, tab_kits, tab_lab, tab_audit = st.tabs(
    ["1️⃣ Screen", "2️⃣ Leaderboard", "3️⃣ Interview Kits", "🧪 Guardrail Lab", "📜 Audit Log"]
)

# ---------------------------------------------------------------- tab 1: screen

with tab_screen:
    left, right = st.columns([3, 2])
    with left:
        if st.button("Load sample JD"):
            state.jd_text = (config.DATA_DIR / "sample_jd.txt").read_text()
        st.text_area("Job description", key="jd_text", height=300)
    with right:
        uploads = st.file_uploader("Resumes (PDF, DOCX or TXT)", type=["pdf", "docx", "txt"],
                                   accept_multiple_files=True)
        use_samples = st.checkbox("Use the 6 sample resumes", value=not uploads)
        st.info("Each resume is anonymized by a hook **before** any agent reads it. "
                "Names are only shown after ranking, from a server-side vault.", icon="🔒")

    pipeline_box = st.empty()
    render_pipeline(pipeline_box, done=set(), running=None)

    if st.button("🚀 Run screening", type="primary", disabled=not state.jd_text.strip()):
        state.run = ScreeningRun(candidates=load_candidates(uploads or [], use_samples))
        state.result, state.revealed_logged, state.reset_reveal = None, False, True
        total = {"score_candidate": len(state.run.candidates), "design_interview": shortlist_size}
        finished = {"score_candidate": 0, "design_interview": 0}  # fan-out nodes finish one task at a time
        done: set[str] = set()
        render_pipeline(pipeline_box, done, running="analyze_jd")

        with st.status(f"Screening {len(state.run.candidates)} candidates…", expanded=True) as status:
            try:
                for node, output in stream_screening(state.run, state.jd_text, shortlist_size):
                    if node == "done":
                        state.result = output
                        break
                    if node == "analyze_jd":
                        profile = output["job_profile"]
                        st.write(f"🤖 **JD Analyzer** built the profile for *{profile.title}* "
                                 f"with {len(profile.competencies)} weighted competencies")
                    elif node == "score_candidate":
                        for cid, score in output["scores"].items():
                            st.write(f"🤖 **Resume Matcher** scored {cid}: **{score.overall}/10**")
                    elif node == "bias_guardrail":
                        st.write(f"🛡️ **Bias guardrail** audited all scorecards · shortlist: {output['shortlist']}")
                    elif node == "design_interview":
                        st.write(f"🤖 **Interview Designer** wrote a kit for {', '.join(output['kits'])}")
                    elif node == "pii_guardrail":
                        st.write("🛡️ **PII guardrail** scanned every scorecard and interview kit")

                    if node == "bias_guardrail":
                        total["design_interview"] = len(output["shortlist"])
                    if node in finished:
                        finished[node] += 1
                        if finished[node] >= total[node]:
                            done.add(node)
                    else:
                        done.add(node)
                    next_node = next((n for n, _ in PIPELINE if n not in done), None)
                    progress = f"({finished[next_node]}/{total[next_node]})" if next_node in finished else ""
                    render_pipeline(pipeline_box, done, running=next_node, progress=progress)
                    render_activity(activity)
                status.update(label="Screening complete", state="complete", expanded=False)
            except Exception as error:  # surface Bedrock/auth errors in the UI instead of a stack trace
                status.update(label="Screening failed", state="error")
                st.error(f"{type(error).__name__}: {error}")
        st.rerun()

    if state.result:
        render_pipeline(pipeline_box, done={n for n, _ in PIPELINE}, running=None)
        profile = state.result["job_profile"]
        st.success("Screening complete. Open the **Leaderboard** tab.", icon="✅")
        st.markdown(f"#### Job profile: {profile.title} · {profile.seniority}")
        st.dataframe(
            pd.DataFrame([
                {"Competency": c.name, "Weight": c.weight, "Required level": f"L{c.required_level}",
                 "From the JD": c.reason}
                for c in profile.competencies
            ]),
            hide_index=True,
            width="stretch",
            column_config={"Weight": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")},
        )

# ---------------------------------------------------------------- tab 2: leaderboard

with tab_board:
    if not state.result:
        st.info("Run a screening first.")
    else:
        profile = state.result["job_profile"]
        scores = sorted(state.result["scores"].values(), key=lambda s: s.overall, reverse=True)
        shortlist = state.result["shortlist"]
        interventions = sum(e.action in ("BLOCK", "MASK", "HUMAN_REVIEW") for e in state.run.audit.events)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Candidates screened", len(scores))
        m2.metric("Shortlisted", len(shortlist))
        m3.metric("Top score", f"{scores[0].overall}/10")
        m4.metric("Hook & guardrail interventions", interventions)

        rows = []
        for rank, s in enumerate(scores, start=1):
            row = {"Rank": rank, "Candidate": label(s.candidate_id), "Overall": s.overall}
            row.update({c.competency: c.score for c in s.scores})
            row["Status"] = ("⚠️ Human review" if s.bias_flagged
                             else "✅ Shortlisted" if s.candidate_id in shortlist else "—")
            rows.append(row)
        st.dataframe(
            pd.DataFrame(rows), hide_index=True, width="stretch",
            column_config={
                "Candidate": st.column_config.TextColumn(width="medium"),
                "Overall": st.column_config.ProgressColumn(min_value=0, max_value=10, format="%.1f"),
            },
        )

        st.divider()
        pick = st.selectbox("Candidate details", [s.candidate_id for s in scores], format_func=label)
        chosen = state.result["scores"][pick]
        chart_col, detail_col = st.columns([1, 1])
        with chart_col:
            st.plotly_chart(radar(chosen, profile), width="stretch")
        with detail_col:
            st.markdown(f"**Overall: {chosen.overall}/10**")
            st.write(chosen.summary)
            st.markdown("**Strengths**\n" + "\n".join(f"- {x}" for x in chosen.strengths))
            st.markdown("**Gaps**\n" + "\n".join(f"- {x}" for x in chosen.gaps))
            if chosen.red_flags:
                st.warning("**Red flags**\n" + "\n".join(f"- {x}" for x in chosen.red_flags))
            if chosen.bias_flagged:
                st.error(f"Bias guardrail: needs human review. {chosen.bias_note}")

        with st.expander("📎 Evidence behind each score"):
            st.dataframe(pd.DataFrame([s.model_dump() for s in chosen.scores]),
                         hide_index=True, width="stretch")
        with st.expander("🔒 What the agent actually saw (after the anonymizer hook)"):
            raw_col, anon_col = st.columns(2)
            raw_col.caption("Original resume (HR only)")
            raw_col.code(state.run.candidates[pick].raw_text, language=None, wrap_lines=True)
            anon_col.caption("Sent to the LLM")
            anon_col.code(anonymized_resume(state.run.candidates[pick])[0], language=None, wrap_lines=True)

# ---------------------------------------------------------------- tab 3: interview kits

with tab_kits:
    if not state.result or not state.result.get("kits"):
        st.info("Run a screening first. Kits are generated for shortlisted candidates.")
    else:
        kits = state.result["kits"]
        pick = st.selectbox("Shortlisted candidate", list(kits), format_func=label)
        kit, score = kits[pick], state.result["scores"][pick]
        st.markdown("**Focus areas:** " + " · ".join(kit.focus_areas))
        for i, q in enumerate(kit.questions, start=1):
            with st.expander(f"Q{i}. [{q.competency}] {q.question.replace('{{candidate}}', label(pick))}"):
                st.markdown(f"**Why ask:** {q.why_ask}")
                st.markdown(f"**A strong answer includes:** {q.good_answer}")
        st.download_button(
            "⬇️ Download interview kit (.docx)",
            data=interview_kit_docx(kit, score, label(pick), state.result["job_profile"].title),
            file_name=f"interview_kit_{pick}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            type="primary",
        )

# ---------------------------------------------------------------- tab 4: guardrail lab

with tab_lab:
    st.caption("Try the guardrails on your own text. Useful for showing them fire on demand.")
    pii_col, bias_col = st.columns(2)

    with pii_col:
        st.markdown("#### 🛡️ PII guardrail")
        pii_text = st.text_area(
            "LLM output to check",
            "Q4. You mentioned working under Rahul Verma on the fraud model. Walk me through your role.\n"
            "Portfolio: github.com/priya-sharma-ml, reach at +91 98765 43210.",
            height=140,
        )
        if st.button("Run PII guardrail"):
            state.lab_pii = check_pii(pii_text)
        if result := state.get("lab_pii"):
            st.markdown(f'Decision: <span class="tag {result.action}">{result.action}</span>',
                        unsafe_allow_html=True)
            st.code(result.cleaned, language=None, wrap_lines=True)
            if result.findings:
                st.dataframe(pd.DataFrame(result.findings), hide_index=True, width="stretch")

    with bias_col:
        st.markdown("#### 🛡️ Bias guardrail")
        bias_text = st.text_area(
            "Scoring reasoning to check",
            "Leadership (6/10): Strong platform leadership, but she graduated in 1995 and has three "
            "children, so may struggle with the pace of a fast-moving team.",
            height=140,
        )
        if st.button("Run bias guardrail"):
            with st.spinner(f"Asking {config.JUDGE_MODEL_ID}…"):
                state.lab_bias = check_bias(bias_text)
        if verdict := state.get("lab_bias"):
            action = "BLOCK" if verdict.biased else "PASS"
            st.markdown(f'Decision: <span class="tag {action}">{action}</span>', unsafe_allow_html=True)
            if verdict.attributes:
                st.write("Protected attributes: " + ", ".join(verdict.attributes))
            st.write(verdict.explanation)

# ---------------------------------------------------------------- tab 5: audit log

with tab_audit:
    if not state.run:
        st.info("Run a screening first.")
    else:
        st.dataframe(pd.DataFrame([e.model_dump() for e in state.run.audit.events]),
                     hide_index=True, width="stretch")
        dl1, dl2 = st.columns(2)
        if AUDIT_FILE.exists():
            dl1.download_button("⬇️ audit_log.jsonl", AUDIT_FILE.read_bytes(), "audit_log.jsonl")
        if SCORES_FILE.exists():
            dl2.download_button("⬇️ scores.csv", SCORES_FILE.read_bytes(), "scores.csv")
