"""
Shared utilities — IDENTICAL to the functions duplicated across the
Phase 1 / Phase 2 / Phase 5 / Phase 6 notebooks (clean_text, skill
normalization). Kept in one place here instead of copy-pasted, per the
notebooks' own "suggested future improvement" note.

Do NOT change SKILL_ALIASES without also re-running Phase 1 (job corpus)
with the same dictionary, or matching quality will silently degrade.
"""
import re
import unicodedata
import html

# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------
TAG_RE = re.compile(r"<[^>]{1,400}?>")
SCRIPT_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.S | re.I)
INLINE_WS_RE = re.compile(r"[ \t\x0b\f\r\u00a0\u2000-\u200a]+")
MANY_NL_RE = re.compile(r"\n{3,}")
ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\ufeff\u2060]")


def clean_text(value, keep_newlines: bool = True) -> str:
    """Normalize text while preserving useful structure. Identical to Phase 1/2."""
    if not isinstance(value, str) or not value.strip():
        return ""

    text = value
    for _ in range(3):
        new = html.unescape(text)
        if new == text:
            break
        text = new

    text = SCRIPT_RE.sub(" ", text)
    text = TAG_RE.sub("\n", text)
    text = ZERO_WIDTH_RE.sub("", text)
    text = unicodedata.normalize("NFKC", text)

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = INLINE_WS_RE.sub(" ", text)

    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln]
    text = "\n".join(lines)
    text = MANY_NL_RE.sub("\n\n", text)

    if not keep_newlines:
        text = text.replace("\n", " ")
        text = INLINE_WS_RE.sub(" ", text)

    return text.strip()


def smart_truncate(text: str, max_chars: int) -> str:
    """Head+tail truncation, identical to Phase 1/2."""
    if len(text) <= max_chars:
        return text
    head_budget = int(max_chars * 0.7)
    tail_budget = max_chars - head_budget - len("\n[... omitted ...]\n")
    return text[:head_budget] + "\n[... omitted ...]\n" + text[-tail_budget:]


# ---------------------------------------------------------------------------
# Skill normalization — identical SKILL_ALIASES dictionary to Phase 1
# ---------------------------------------------------------------------------
SKILL_ALIASES = {
    "powerbi": "power bi", "ms power bi": "power bi",
    "microsoft power bi": "power bi",
    "ms excel": "excel", "microsoft excel": "excel",
    "advanced excel": "excel", "ms office": "microsoft office",
    "office suite": "microsoft office", "msoffice": "microsoft office",
    "sql server": "microsoft sql server", "mssql": "microsoft sql server",
    "postgres": "postgresql", "ms sql": "microsoft sql server",
    "google data studio": "looker studio",
    "data visualisation": "data visualization",
    "data analytics": "data analysis",
    "statistical analysis": "statistics",
    "machine learning (ml)": "machine learning",
    "ml": "machine learning", "ai": "artificial intelligence",
    "nlp": "natural language processing",
    "etl pipelines": "etl", "etl processes": "etl",
    "js": "javascript", "reactjs": "react", "react.js": "react",
    "nodejs": "node.js", "node js": "node.js",
    "py": "python", "python3": "python",
    "c sharp": "c#", "golang": "go",
    "rest apis": "rest api", "restful api": "rest api",
    "ci cd": "ci/cd", "cicd": "ci/cd",
    "aws cloud": "aws", "amazon web services": "aws",
    "microsoft azure": "azure", "gcp": "google cloud",
    "k8s": "kubernetes",
    "ms project": "microsoft project",
    "erp systems": "erp", "sap erp": "sap",
    "kpi reporting": "kpi reporting",
    "stakeholder management": "stakeholder management",
    "project mgmt": "project management",
    "communication skills": "communication",
    "verbal communication": "communication",
    "written communication": "written communication",
    "team work": "teamwork", "team player": "teamwork",
    "problem-solving": "problem solving",
    "time-management": "time management",
    "attention to detail": "attention to detail",
    "interpersonal skills": "interpersonal skills",
    "leadership skills": "leadership",
    "analytical skills": "analytical thinking",
    "analytical thinking skills": "analytical thinking",
    "english language": "english", "arabic language": "arabic",
    "fluent english": "english", "native arabic": "arabic",
    "english (fluent)": "english",
}

ACRONYM_KEEP = {"sql", "aws", "gcp", "erp", "sap", "api", "etl", "bi", "qa",
                "hr", "ui", "ux", "iso", "css", "html", "php", "ios", "crm",
                "kpi", "cad", "plc", "hvac", "ccna", "pmp", "cpa", "acca"}

NOISE_RE = re.compile(
    r"^(ability to|able to|experience (in|with)|knowledge of|proficiency in|"
    r"strong |excellent |good |demonstrated |proven |solid |working )",
    re.I,
)


def normalize_skill(skill: str) -> str:
    if not isinstance(skill, str):
        return ""
    text = clean_text(skill, keep_newlines=False).lower()
    text = NOISE_RE.sub("", text)
    text = re.sub(r"[\(\)\[\]\.,;:]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" -\u2013\u2014/")

    if not text or len(text) < 2:
        return ""
    if len(text.split()) > 5:
        return ""

    text = SKILL_ALIASES.get(text, text)
    if text.endswith("s") and text[:-1] in SKILL_ALIASES.values():
        text = text[:-1]
    return SKILL_ALIASES.get(text, text)


def display_skill(canonical: str) -> str:
    if canonical in ACRONYM_KEEP:
        return canonical.upper()
    return " ".join(
        w.upper() if w in ACRONYM_KEEP else w.capitalize()
        for w in canonical.split()
    )


def normalize_skill_list(values) -> list:
    seen, out = set(), []
    for item in (values or []):
        canon = normalize_skill(item)
        if canon and canon not in seen:
            seen.add(canon)
            out.append(canon)
    return out
