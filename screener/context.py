"""Per-run state shared by tools, hooks, guardrails and the graph, plus the audit trail."""

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime

from screener.config import OUTPUT_DIR
from screener.documents import Candidate
from screener.schemas import AuditEvent, CandidateScore, JobProfile

AUDIT_FILE = OUTPUT_DIR / "audit_log.jsonl"
VAULT_FILE = OUTPUT_DIR / "identity_vault.json"
SCORES_FILE = OUTPUT_DIR / "scores.csv"


class AuditLog:
    """Every hook and guardrail decision, kept in memory for the UI and appended to disk."""

    def __init__(self) -> None:
        self.events: list[AuditEvent] = []
        self._lock = threading.Lock()  # candidates are scored in parallel threads

    def record(self, kind: str, name: str, action: str, candidate_id: str = "", detail: str = "") -> None:
        event = AuditEvent(
            ts=datetime.now().isoformat(timespec="seconds"),
            kind=kind,
            name=name,
            action=action,
            candidate_id=candidate_id,
            detail=detail,
        )
        with self._lock:
            self.events.append(event)
            with AUDIT_FILE.open("a") as f:
                f.write(event.model_dump_json() + "\n")

    def hook(self, name: str, action: str, candidate_id: str = "", detail: str = "") -> None:
        self.record("hook", name, action, candidate_id, detail)

    def guardrail(self, name: str, action: str, candidate_id: str = "", detail: str = "") -> None:
        self.record("guardrail", name, action, candidate_id, detail)


@dataclass
class ScreeningRun:
    candidates: dict[str, Candidate]  # the identity vault: server-side only, never sent to the LLM
    audit: AuditLog = field(default_factory=AuditLog)
    job_profile: JobProfile | None = None
    scores: dict[str, CandidateScore] = field(default_factory=dict)

    def __post_init__(self) -> None:
        OUTPUT_DIR.mkdir(exist_ok=True)
        SCORES_FILE.unlink(missing_ok=True)
        vault = {
            cid: {"name": c.name, "email": c.email, "phone": c.phone, "file": c.filename}
            for cid, c in self.candidates.items()
        }
        VAULT_FILE.write_text(json.dumps(vault, indent=2))
