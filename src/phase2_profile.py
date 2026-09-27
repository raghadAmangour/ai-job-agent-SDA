"""
Phase 2 — Candidate Profile.

Turns (resume text + user preferences) into a candidate_profile dict with
the exact same schema/normalization as the Phase 2 notebook's PATH B (live
candidate). Runs synchronously (single resume per request — no need for the
notebook's asyncio/semaphore machinery, which existed for batch processing).
"""
import hashlib
import json
import time
from datetime import datetime, timezone
from typing import List, Optional, Literal

import numpy as np
from pydantic import BaseModel, Field

from . import config
from .shared import clean_text, normalize_skill_list, display_skill

ExperienceLevel = Literal[
    "Internship", "Entry Level", "Mid Level", "Senior Level",
    "Executive", "Not Specified",
]
EducationLevel = Literal[
    "High School", "Diploma", "Bachelor's Degree",
    "Master's Degree", "PhD", "Not Specified",
]


class ResumeExtraction(BaseModel):
    full_name: Optional[str]
    email: Optional[str]
    phone: Optional[str]
    most_recent_title: Optional[str]

    hard_skills: List[str] = Field(description="Technical/domain skills, max 4 words each.")
    soft_skills: List[str] = Field(description="Behavioural/interpersonal skills, max 4 words each.")
    tools_technologies: List[str] = Field(description="Named tools, software, platforms, frameworks.")
    languages: List[str] = Field(description="Spoken/written languages, e.g. Arabic, English.")

    qualifications: List[str] = Field(description="Certifications, licences, degrees, memberships.")

    experience_level: ExperienceLevel
    experience_level_inferred: bool
    years_experience_total: Optional[float]
    past_titles: List[str]
    industries: List[str]

    education_level: List[EducationLevel]
    education_field: List[str]

    summary: Optional[str]
    extraction_confidence: float


SYSTEM_PROMPT = """
You are a resume information extraction system. You convert an unstructured
resume into a strict structured record.

GENERAL RULES

1. Never invent information. If it is not stated or clearly implied, use an
   empty list, null, or "Not Specified".
2. Extract from the resume content only. Ignore headers/footers/page numbers
   and decorative text.

SKILLS — THE MOST IMPORTANT PART

3. A skill is a short noun phrase of AT MOST 4 words, exactly like:
   GOOD: "SQL", "financial modeling", "patient care", "welding"
   BAD:  "Bachelor's degree in Computer Science"
   BAD:  "4 years of experience in data analysis"
4. Degrees, certifications, and licences are NOT skills — put them in
   `qualifications`.
5. Split skills into four buckets, each skill in exactly ONE bucket:
   - hard_skills: technical/domain capabilities
   - soft_skills: behavioural ("teamwork", "leadership")
   - tools_technologies: named products/platforms/standards ("Power BI", "SAP")
   - languages: spoken languages only ("Arabic", "English")
6. Use the canonical name of a tool: "Power BI" not "MS PowerBI".
7. If the resume has an explicit "Skills" section, extract every item from it.
   ALSO scan the experience bullet points for skills not listed there
   explicitly (e.g. "built dashboards in Tableau" implies "Tableau" and
   "dashboarding" even with no dedicated skills section).

EXPERIENCE

8. years_experience_total: if the work history has explicit date ranges
   (e.g. "Jan 2022 - Present", "2019-2021"), compute the total span across
   all roles in years, one decimal place. Overlapping roles should not be
   double-counted — use the union of the date ranges.
   If no dates are present, return null. Never guess from job titles alone.
9. experience_level: Internship/Entry Level (0-1y) / Mid Level (2-4y) /
   Senior Level (5-9y) / Executive (10+y or C-level/VP/Director titles).
   Base it on years_experience_total when available (set
   experience_level_inferred = true). If the resume states a level
   explicitly (e.g. "Senior" in the most recent title), that also counts
   as inferred = false only if the resume literally uses that word.
10. past_titles: list of previous job titles, most recent first, EXCLUDING
    the most recent/current one (which goes in most_recent_title).

EDUCATION

11. education_level uses ONLY the allowed enum values. List every degree
    held, not just the highest. education_field holds the subject of study
    for each entry, in the same order.

OTHER

12. full_name, email, phone: extract exactly as written. Null if absent.
13. summary: write 2-3 sentences in the third person describing the
    candidate's profile, based only on resume content (not invented).
14. industries: infer from employer descriptions or job context, not from
    company names alone unless the industry is obvious (e.g. "XYZ Bank").
15. extraction_confidence: 0.9+ for a detailed, well-structured resume,
    0.5 for a thin one, below 0.3 if the text looks corrupted or mostly
    unreadable (e.g. from a badly-parsed scanned PDF).
"""


def extract_resume_features(client, resume_text: str, usage_state: dict = None) -> dict:
    """Synchronous single-resume extraction with retry/backoff, mirroring
    Phase 2's extract_one() (minus the cache/semaphore, unneeded for one call)."""
    from .shared import smart_truncate

    text = smart_truncate(resume_text, config.MAX_RESUME_CHARS_SENT_TO_LLM)
    delay = 2.0
    last_exc = None

    for attempt in range(config.MAX_RETRIES):
        try:
            response = client.responses.parse(
                model=config.EXTRACTION_MODEL,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                text_format=ResumeExtraction,
            )
            from .usage_tracker import track_llm
            track_llm(usage_state, response, "phase2_extraction")
            return response.output_parsed.model_dump()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt == config.MAX_RETRIES - 1:
                raise RuntimeError(f"Resume extraction failed after retries: {exc}") from exc
            time.sleep(min(delay, 20))
            delay *= 2
    raise RuntimeError(f"Resume extraction failed: {last_exc}")


def embed_texts(client, texts: list, usage_state: dict = None) -> np.ndarray:
    """Batch embed via OpenAI embeddings API — same model as job_embeddings.npy."""
    response = client.embeddings.create(model=config.EMBEDDING_MODEL, input=texts)
    from .usage_tracker import track_embedding
    track_embedding(usage_state, response)
    vecs = np.array([d.embedding for d in response.data], dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def make_candidate_id(source: str) -> str:
    return hashlib.sha1(source.encode("utf-8")).hexdigest()[:16]


def build_embedding_text(row: dict) -> str:
    parts = [
        row.get("most_recent_title") or "",
        row.get("summary") or "",
        "Skills: " + ", ".join(display_skill(s) for s in row["all_skills_norm"]),
        "Industries: " + ", ".join(row.get("industries") or []),
    ]
    return clean_text(" . ".join(p for p in parts if p), keep_newlines=False)


def build_candidate_profile(client, resume_name: str, resume_raw_text: str,
                             preferences: dict, usage_state: dict = None) -> tuple:
    """Full Phase 2 PATH B pipeline for one resume. Returns
    (candidate_profile: dict, embedding: np.ndarray[1536])."""
    resume_clean = clean_text(resume_raw_text, keep_newlines=True)
    candidate_id = make_candidate_id(f"{resume_name}|{datetime.now(timezone.utc).isoformat()}")

    features = extract_resume_features(client, resume_clean, usage_state)

    normalized = {
        f"{field}_norm": normalize_skill_list(features.get(field, []))
        for field in ["hard_skills", "soft_skills", "tools_technologies", "languages"]
    }
    normalized["all_skills_norm"] = sorted(set(
        normalized["hard_skills_norm"]
        + normalized["tools_technologies_norm"]
        + normalized["soft_skills_norm"]
    ))

    candidate_profile = {
        "candidate_id": candidate_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_file": resume_name,
        **features,
        **normalized,
        "n_skills": len(normalized["all_skills_norm"]),
        **preferences,
    }

    embedding_text = build_embedding_text({
        "most_recent_title": candidate_profile.get("most_recent_title"),
        "summary": candidate_profile.get("summary"),
        "all_skills_norm": candidate_profile["all_skills_norm"],
        "industries": candidate_profile.get("industries") or [],
    })
    vector = embed_texts(client, [embedding_text], usage_state)[0]

    return candidate_profile, vector.astype(np.float32)


def profile_warnings(candidate_profile: dict) -> list:
    """Same consistency checks as Phase 2 §7.3."""
    warnings = []
    if not candidate_profile.get("desired_countries") and not candidate_profile.get("willing_to_relocate"):
        warnings.append(
            "desired_countries is empty and willing_to_relocate=False — location "
            "filtering in Phase 4 will have nothing to work with."
        )
    if candidate_profile.get("extraction_confidence", 1.0) < 0.4:
        warnings.append(
            f"Low extraction_confidence ({candidate_profile['extraction_confidence']}) — "
            "the resume text may have extracted poorly."
        )
    if not candidate_profile.get("all_skills_norm"):
        warnings.append("No skills were extracted — check the resume text quality.")
    return warnings
