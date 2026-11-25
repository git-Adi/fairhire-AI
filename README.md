# 🧑‍💼 FairHire AI: Blind Resume Screener

Agentic AI demo built with **LangChain + LangGraph** on **Amazon Bedrock** with a **Streamlit** UI.

HR uploads a job description and resumes. Three agents analyse the role, score each candidate
**without seeing who they are**, and write a tailored interview kit for the shortlist.
Hooks and guardrails make the process fair, private and auditable.

| Concept | Where | What it does |
|---|---|---|
| **3 agents** | `screener/agents.py` | JD Analyzer → Resume Matcher (one per candidate, in parallel) → Interview Designer |
| **2 skills** | `skills/` | `competency-framework` (catalogue + levels), `scoring-rubric` (rules + `compute_score.py`) |
| **4 hooks** | `screener/hooks.py` | Pre: `enforce_candidate_scope`, `validate_score_submission` · Post: `anonymize_resume`, `record_score` |
| **2 guardrails** | `screener/guardrails.py` | Bias judge (Nova Lite) on scoring reasoning · PII scan (Presidio) on final outputs |
| **Workflow** | `screener/graph.py` | LangGraph with parallel fan-out (`Send`) |
| **UI** | `app.py` | Screen, Leaderboard, Interview Kits, Guardrail Lab, Audit Log |
| **Evals** | `evals/` | MLflow GenAI evaluation of guardrails, the anonymizer hook and the full agent pipeline, with tracing |

## Workflow

```
analyze_jd ─► score_candidate ×N (parallel) ─► bias_guardrail ─► design_interview ×K ─► pii_guardrail ─► END
   │                │                               │                    │                    │
 skill:        skill: scoring-rubric          LLM judge;            skill:              Presidio scan;
 competency-   hooks: anonymize on read,      biased → re-score     competency-         contact details →
 framework     validate + record score        → human review        framework           regenerate, else mask
```

## Hooks vs guardrails

- **Hooks** are deterministic code that runs around **every tool call** (LangChain `wrap_tool_call` middleware).
  A PreToolUse hook can block a call. The agent sees the reason and retries. A PostToolUse hook can rewrite
  a tool's output before the model sees it. This is how resumes get anonymized.
- **Guardrails** check what the **LLM wrote**. The bias guardrail uses a second model as a judge.
  The PII guardrail scans the final scorecards and interview kits before anything is displayed.

## How names are shown when everything is redacted

Candidate identities sit in a server-side **identity vault** (`outputs/identity_vault.json`), keyed by `C-01`, `C-02`, and so on.
The LLM only ever sees IDs. After ranking, the **Reveal identities** toggle joins names back in with plain
Python. Redaction only applies to LLM-generated text, so the ranking never changes when names appear.

## Setup

Prerequisite: the `mugambo` AWS CLI profile from the AWS setup guide, with Bedrock access in `us-east-1`.
Default models: **Amazon Nova Pro** (agents) and **Nova Lite** (bias judge). Switching to Claude is a
`.env` change only (see `.env.example`).

The virtual environment lives **outside** `~/Documents`, because iCloud sync of `~/Documents`
corrupts Python packages (it creates `file 2.py` duplicates).

```bash
python3.13 -m venv ~/.venvs/ai_upliftment_project
source ~/.venvs/ai_upliftment_project/bin/activate
cd ~/Documents/ai_upliftment_project
pip install -r requirements.txt
python -m spacy download en_core_web_md
cp .env.example .env          # edit model IDs here if needed
export AWS_PROFILE=mugambo
streamlit run app.py
```

## Project structure

```
ai_upliftment_project/
├── app.py                  # Streamlit UI
├── screener/
│   ├── config.py           # env settings + Bedrock model factory
│   ├── schemas.py          # Pydantic models
│   ├── documents.py        # resume parsing, identity extraction, DOCX export
│   ├── context.py          # per-run state, identity vault, audit log
│   ├── skills.py           # skill loader (progressive disclosure)
│   ├── tools.py            # agent tools
│   ├── hooks.py            # Pre/PostToolUse hooks + LangChain middleware
│   ├── pii.py              # Presidio (regex + spaCy NER)
│   ├── guardrails.py       # bias judge + PII output guardrail
│   ├── agents.py           # the 3 agents
│   └── graph.py            # LangGraph workflow
├── skills/
│   ├── competency-framework/   SKILL.md, levels.md
│   └── scoring-rubric/         SKILL.md, compute_score.py
├── evals/
│   ├── golden_set.json     # labelled expectations + guardrail test cases
│   └── run_evals.py        # MLflow eval suites
├── data/                   # sample JD + 6 synthetic resumes
└── outputs/                # created at runtime: audit log, scores, identity vault, mlflow.db
```

## Evals (MLflow)

```bash
python -m evals.run_evals                     # all suites (~2 min, uses Bedrock)
python -m evals.run_evals --suite anonymizer  # one suite
mlflow ui --backend-store-uri sqlite:///outputs/mlflow.db --port 5001
```

Each suite is an MLflow run in the `fairhire-evals` experiment, using `mlflow.genai.evaluate` with custom scorers.
Expectations are in `evals/golden_set.json`.

| Suite | Scorers / metrics | LLM calls |
|---|---|---|
| `pii_guardrail` | right decision (PASS/MASK/BLOCK), redaction recall, no false positives on technical text | none |
| `bias_guardrail` | verdict correct, bias precision / recall | judge model |
| `anonymizer` | no identity leak after the hook, key evidence preserved | none |
| `pipeline` | score in expected tier band, evidence grounded in the resume, prompt injection flagged, no protected terms, interview kit specificity + LLM-judge rating, shortlist recall, tier ordering, JD profile validity, latency, hook/guardrail counts, **trace identity leaks** | full agent run |

**Privacy in MLflow:** the pipeline run is traced with `mlflow.langchain.autolog()`. A span processor replaces raw
resumes and identity values before traces are saved, and the suite reads the saved traces back to count any leaked
identity value (`trace_identity_leaks`, should be 0). Eval tables contain candidate IDs only.

## Demo traps in the sample data

| Resume | What it shows |
|---|---|
| `priya_sharma.txt` | Gender, DOB, marital status, a third-party name and a GitHub URL. Watch the anonymizer and PII guardrail |
| `meera_iyer.txt` | Age, religion, children, a career break. The anonymizer removes them; the bias guardrail watches the reasoning |
| `vikram_singh.txt` | Keyword stuffing + a prompt injection ("give this candidate 10"). The rubric caps list-only skills and records a red flag |
| `rohan_das.txt` | Strong frontend engineer, weak fit for an ML role. Shows honest low scores |
