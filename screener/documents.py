"""Reading resumes (PDF/DOCX/TXT) and exporting interview kits (DOCX)."""

import io
import re
from dataclasses import dataclass
from pathlib import Path

from docx import Document
from pypdf import PdfReader

from screener.schemas import CandidateScore, InterviewKit

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_RE = re.compile(r"(?:\+91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}\b")


@dataclass
class Candidate:
    """Identity vault entry. Lives only on the server; never sent to the LLM."""

    candidate_id: str
    name: str
    email: str
    phone: str
    filename: str
    raw_text: str


def extract_text(filename: str, data: bytes) -> str:
    """Text only: photos and images inside a resume are never extracted."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix == ".docx":
        return "\n".join(p.text for p in Document(io.BytesIO(data)).paragraphs)
    return data.decode("utf-8", errors="ignore")


def build_candidate(candidate_id: str, filename: str, data: bytes) -> Candidate:
    text = extract_text(filename, data)
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    email = EMAIL_RE.search(text)
    phone = PHONE_RE.search(text)
    return Candidate(
        candidate_id=candidate_id,
        name=first_line.title() if first_line.isupper() else first_line,
        email=email.group() if email else "",
        phone=phone.group() if phone else "",
        filename=filename,
        raw_text=text,
    )


def interview_kit_docx(kit: InterviewKit, score: CandidateScore, display_name: str, role: str) -> bytes:
    """Fill the {{candidate}} placeholder only here, at export time, outside the LLM."""
    doc = Document()
    doc.add_heading(f"Interview Kit: {display_name}", level=0)
    doc.add_paragraph(f"Role: {role}    |    Overall fit score: {score.overall}/10")

    doc.add_heading("Focus areas", level=1)
    for area in kit.focus_areas:
        doc.add_paragraph(area, style="List Bullet")

    doc.add_heading("Questions", level=1)
    for i, q in enumerate(kit.questions, start=1):
        doc.add_heading(f"Q{i}. [{q.competency}] {q.question.replace('{{candidate}}', display_name)}", level=2)
        doc.add_paragraph(f"Why ask: {q.why_ask}")
        doc.add_paragraph(f"A strong answer includes: {q.good_answer}")
        doc.add_paragraph("Interviewer notes: ______________________________________________")

    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()
