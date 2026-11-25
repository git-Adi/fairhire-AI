"""PII detection shared by the anonymizer hook and the output guardrail.

Two layers in one Presidio analyzer:
  1. Regex recognizers for fixed-format PII (Indian phones, profile URLs, personal details).
  2. spaCy named-entity recognition for free-text PII such as person names and locations.
"""

from functools import lru_cache

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from screener.config import SPACY_MODEL

# Tech words, degrees and competency names spaCy sometimes mistakes for names or places.
ALLOW_LIST = [
    "Python", "Java", "Go", "Spark", "PySpark", "Flask", "Django", "Kafka", "Jenkins", "Airflow",
    "Docker", "Kubernetes", "SageMaker", "Bedrock", "LangChain", "LangGraph", "PyTorch", "TensorFlow",
    "XGBoost", "LightGBM", "MLflow", "Evidently", "Snowflake", "OpenSearch", "FAISS", "LoRA", "BERT",
    "React", "Next.js", "Tailwind", "Hugging Face", "Great Expectations", "dbt", "EKS", "ECS", "EMR",
    "AI", "ML", "F1", "RAG", "LLM", "LLMs", "GenAI", "NLP", "NER", "C++", "SQL", "ETL", "API", "AWS",
    "CI", "CI/CD", "Tech Lead", "Career", "Hadoop", "Glue", "Lighthouse", "TypeScript", "JavaScript",
    "B.Sc.", "B.Tech", "B.E.", "M.Tech", "Cloud", "MLOps",
    "Programming", "ML & Modelling", "GenAI & LLMs", "Data Engineering", "Cloud & MLOps",
    "System Design", "Frontend Engineering", "Communication", "Leadership",
]

CONTACT_ENTITIES = ["EMAIL_ADDRESS", "PHONE_NUMBER", "IN_PHONE"]
OUTPUT_ENTITIES = ["PERSON", "PROFILE_URL", *CONTACT_ENTITIES]
RESUME_ENTITIES = [*OUTPUT_ENTITIES, "LOCATION", "PERSONAL_DETAIL"]

_RECOGNIZERS = [
    PatternRecognizer(
        supported_entity="IN_PHONE",
        patterns=[Pattern("in_phone", r"(?:\+91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}\b", 0.7)],
        context=["phone", "mobile", "contact", "call"],
    ),
    PatternRecognizer(
        supported_entity="PROFILE_URL",
        patterns=[Pattern(
            "profile_url",
            r"(?:https?://)?(?:www\.)?(?:linkedin\.com/in|github\.com|gitlab\.com)/[\w-]+",
            0.9,
        )],
    ),
    # Attributes that must never influence screening (protected characteristics).
    PatternRecognizer(
        supported_entity="PERSONAL_DETAIL",
        patterns=[
            Pattern(
                "labelled_field",
                r"(?i)\b(?:gender|sex|date of birth|dob|age|marital status|nationality|religion|caste)"
                r"\s*[:\-]\s*[^|\n,]+",
                0.9,
            ),
            Pattern(
                "family_status",
                r"(?i)\b(?:married|unmarried|single|divorced|widowed)\b(?:,?\s*\d+\s+(?:children|kids))?",
                0.6,
            ),
            Pattern("children", r"(?i)\b\d+\s+(?:children|kids)\b", 0.6),
        ],
    ),
]


@lru_cache
def _analyzer() -> AnalyzerEngine:
    nlp_engine = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": SPACY_MODEL}],
    }).create_engine()
    analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
    for recognizer in _RECOGNIZERS:
        analyzer.registry.add_recognizer(recognizer)
    return analyzer


@lru_cache
def _anonymizer() -> AnonymizerEngine:
    return AnonymizerEngine()


def find_pii(
    text: str,
    entities: list[str],
    known_names: list[str] | None = None,
    threshold: float = 0.6,
) -> list[RecognizerResult]:
    """Find PII spans. `known_names` are always flagged (e.g. the candidate's own name)."""
    ad_hoc = []
    if known_names:
        ad_hoc.append(PatternRecognizer(supported_entity="PERSON", deny_list=known_names))
    return _analyzer().analyze(
        text=text,
        language="en",
        entities=entities,
        score_threshold=threshold,
        allow_list=ALLOW_LIST,
        ad_hoc_recognizers=ad_hoc or None,
    )


def redact(text: str, findings: list[RecognizerResult], placeholder: str | None = None) -> str:
    """Replace each finding with `placeholder`, or with its entity type, e.g. <PERSON>."""
    if not findings:
        return text
    if placeholder:
        operators = {"DEFAULT": OperatorConfig("replace", {"new_value": placeholder})}
    else:
        operators = None  # Presidio default: <ENTITY_TYPE>
    return _anonymizer().anonymize(text=text, analyzer_results=findings, operators=operators).text


def describe(text: str, findings: list[RecognizerResult]) -> list[dict]:
    """Findings as plain dicts for logs and the UI."""
    return [
        {"type": f.entity_type, "text": text[f.start:f.end], "score": round(f.score, 2)}
        for f in sorted(findings, key=lambda f: f.start)
    ]
