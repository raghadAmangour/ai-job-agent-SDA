"""
Phase 6 — Cross-Encoder Re-ranking. Identical logic to the notebook, minus
the score-cache/jsonl persistence (not useful across independent Streamlit
sessions) and the adaptive-batch-size OOM handling (Streamlit Cloud has no
GPU, so there's no CUDA OOM path to adapt to — batch size is fixed and small).
"""
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import config
from .phase5_matching import cell_values

PII_FIELDS = config.PII_FIELDS


def text_section(label: str, value) -> str:
    values = cell_values(value)
    return f"{label}: {', '.join(values)}" if values else ""


def truncate_words(value, max_words: int) -> str:
    """Deterministically caps a long free-text field at max_words, appending
    '...' if cut. Used instead of letting the tokenizer silently truncate
    wherever it lands — this way only the tail of long prose fields (full
    job description, responsibilities) is ever dropped, never the short
    structured fields (skills, title, experience, education) built above.
    Handles both plain strings and list-like values (via cell_values) so a
    responsibilities field stored as a list doesn't get str()'d into
    bracket/quote noise."""
    text = ", ".join(cell_values(value))
    if not text:
        return ""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " ..."


def redact_pii(text: str, profile: dict) -> str:
    redacted = text
    for field in PII_FIELDS:
        value = profile.get(field)
        if value and len(str(value).strip()) >= 3:
            redacted = redacted.replace(str(value), "[REDACTED]")
    return redacted


def build_candidate_text(profile: dict) -> str:
    sections = [
        text_section("Current role", profile.get("most_recent_title")),
        text_section("Previous roles", profile.get("past_titles")),
        text_section("Hard skills", profile.get("hard_skills_norm")),
        text_section("Tools and technologies", profile.get("tools_technologies_norm")),
        text_section("Soft skills", profile.get("soft_skills_norm")),
        text_section("Languages", profile.get("languages_norm")),
        text_section("Experience level", profile.get("experience_level")),
        text_section("Total years of experience", profile.get("years_experience_total")),
        text_section("Industries", profile.get("industries")),
        text_section("Education level", profile.get("education_level")),
        text_section("Education fields", profile.get("education_field")),
        text_section("Professional qualifications", profile.get("qualifications")),
        text_section("Professional summary",
                     truncate_words(profile.get("summary"), config.CANDIDATE_SUMMARY_MAX_WORDS)),
    ]
    return redact_pii("\n".join(s for s in sections if s), profile)


def build_job_text(row: pd.Series) -> str:
    years = []
    if pd.notna(row.get("years_experience_min")):
        years.append(f"minimum {row.get('years_experience_min')}")
    if pd.notna(row.get("years_experience_max")):
        years.append(f"maximum {row.get('years_experience_max')}")
    sections = [
        text_section("Job title", row.get("title_clean")),
        text_section("Industry", row.get("industry")),
        text_section("Department", row.get("department")),
        text_section("Required skills", row.get("required_skills_norm")),
        text_section("Preferred skills", row.get("preferred_skills_norm")),
        text_section("Hard skills", row.get("hard_skills_norm")),
        text_section("Tools and technologies", row.get("tools_technologies_norm")),
        text_section("Soft skills", row.get("soft_skills_norm")),
        text_section("Languages", row.get("languages_norm")),
        text_section("Experience level", row.get("experience_level")),
        text_section("Experience years", years),
        text_section("Education level", row.get("education_level")),
        text_section("Education fields", row.get("education_field")),
        text_section("Qualifications", row.get("qualifications")),
        text_section("Responsibilities",
                     truncate_words(row.get("responsibilities"), config.RESPONSIBILITIES_MAX_WORDS)),
        text_section("Full job description",
                     truncate_words(row.get("jd_clean"), config.JOB_DESCRIPTION_MAX_WORDS)),
    ]
    return "\n".join(s for s in sections if s)


def load_reranker():
    """Loads the pinned multilingual cross-encoder. Call this once via
    st.cache_resource in app.py — NOT per request."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        config.RERANKER_MODEL, revision=config.RERANKER_MODEL_REVISION,
        trust_remote_code=config.RERANKER_TRUST_REMOTE_CODE,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        config.RERANKER_MODEL, revision=config.RERANKER_MODEL_REVISION,
        code_revision=config.RERANKER_CODE_REVISION,
        trust_remote_code=config.RERANKER_TRUST_REMOTE_CODE, torch_dtype=dtype,
    ).to(device)
    model.eval()
    return tokenizer, model, device


def score_pairs(tokenizer, model, device, candidate_text: str, job_texts: list,
                 batch_size: int = None) -> list:
    import torch
    batch_size = batch_size or config.INITIAL_BATCH_SIZE
    pairs = [[candidate_text, jt] for jt in job_texts]
    scores = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start:start + batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True,
                             max_length=config.MAX_PAIR_TOKENS, return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.inference_mode():
            logits = model(**encoded, return_dict=True).logits.view(-1).float().cpu().numpy()
        scores.extend(float(v) for v in logits)
    return scores


def rerank(matched_df: pd.DataFrame, candidate_profile: dict, tokenizer, model, device,
           pool_size: int = None, final_top_n: int = None) -> pd.DataFrame:
    """Full Phase 6 pipeline. Falls back to Phase 5 order on any model error."""
    pool_size = pool_size or config.PHASE6_POOL_SIZE if hasattr(config, "PHASE6_POOL_SIZE") else 50
    final_top_n = final_top_n or config.FINAL_TOP_N

    flag = matched_df.get("is_phase6_candidate")
    if flag is not None and flag.any():
        pool = matched_df.loc[flag.fillna(False)].copy()
    else:
        pool = matched_df.sort_values("phase5_rank").head(pool_size).copy()
    pool = pool.sort_values("phase5_rank").reset_index(drop=True)

    if len(pool) == 0:
        return pool

    candidate_text = build_candidate_text(candidate_profile)
    job_texts = [build_job_text(row) for _, row in pool.iterrows()]

    rerank_error = None
    raw_scores = [None] * len(pool)
    if tokenizer is not None and model is not None:
        try:
            raw_scores = score_pairs(tokenizer, model, device, candidate_text, job_texts)
        except Exception as exc:  # noqa: BLE001
            rerank_error = f"{type(exc).__name__}: {exc}"
    else:
        rerank_error = "reranker not loaded"

    pool["reranker_raw_score"] = pd.to_numeric(pd.Series(raw_scores), errors="coerce")

    score_available = pool["reranker_raw_score"].notna().all() if len(pool) else False
    if score_available:
        pool = pool.sort_values(
            ["reranker_raw_score", "objective_rank_score", "match_score",
             "scoring_confidence", "similarity", "phase5_rank"],
            ascending=[False, False, False, False, False, True], kind="mergesort",
        ).reset_index(drop=True)
        status = "no_comparison_pool" if len(pool) == 1 else "success"
    else:
        pool = pool.sort_values("phase5_rank", kind="mergesort").reset_index(drop=True)
        status = "fallback_phase5"

    pool.insert(0, "phase6_rank", range(1, len(pool) + 1))
    pool["rank_change"] = pool["phase5_rank"].astype(int) - pool["phase6_rank"]
    pool["rerank_status"] = status
    pool["rerank_error"] = rerank_error

    if score_available:
        pool["reranker_percentile"] = (pool["reranker_raw_score"].rank(method="average", pct=True) * 100.0).round(2)
    else:
        pool["reranker_percentile"] = np.nan

    return pool.head(final_top_n)


def to_phase7_payload(reranked_df: pd.DataFrame, candidate_id: str) -> dict:
    """Builds the privacy-safe final_top_jobs.json payload for Phase 7."""
    jobs = []
    for _, row in reranked_df.iterrows():
        jobs.append({
            "job_id": row.get("job_id"),
            "phase6_rank": int(row.get("phase6_rank")),
            "phase5_rank": int(row.get("phase5_rank")),
            "title": row.get("title_clean"),
            "company": row.get("company_clean"),
            "url": row.get("url_clean"),
            "location": {"city": row.get("city"), "country": row.get("country")},
            "job_description": row.get("jd_clean"),
            "reranker_raw_score": None if pd.isna(row.get("reranker_raw_score")) else float(row.get("reranker_raw_score")),
            "objective_rank_score": float(row.get("objective_rank_score")),
            "match_score": float(row.get("match_score")),
            "scoring_confidence": float(row.get("scoring_confidence")),
            "matched_core_skills": cell_values(row.get("matched_core_skills")),
            "missing_core_skills": cell_values(row.get("missing_core_skills")),
            "matched_qualifications": cell_values(row.get("matched_qualifications")),
            "missing_qualifications": cell_values(row.get("missing_qualifications")),
            "matched_education_fields": cell_values(row.get("matched_education_fields")),
            "missing_education_fields": cell_values(row.get("missing_education_fields")),
            "rerank_status": row.get("rerank_status"),
        })
    return {
        "candidate_id": candidate_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "reranker_model": config.RERANKER_MODEL,
        "jobs": jobs,
    }
