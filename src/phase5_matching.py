"""Phase 5 — Candidate Matching. Deterministic, explainable scoring, identical
to the notebook (equal-weight match_score + CRITIC-weighted objective_rank_score)."""
import json
import re
import unicodedata

import numpy as np
import pandas as pd

from .config import RANKING_CRITERIA, EDUCATION_HIERARCHY, EXPERIENCE_HIERARCHY, PHASE6_POOL_SIZE

_exp_order = {v.lower(): i for i, v in enumerate(EXPERIENCE_HIERARCHY)}
_edu_order = {v.lower(): i for i, v in enumerate(EDUCATION_HIERARCHY)}


def cell_values(value) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
        items = list(value)
    else:
        try:
            if pd.isna(value):
                return []
        except (TypeError, ValueError):
            pass
        items = [value]
    out = []
    for item in items:
        if item is None:
            continue
        try:
            if pd.isna(item):
                continue
        except (TypeError, ValueError):
            pass
        text = str(item).strip()
        if text and text.lower() not in {"nan", "none", "not specified"}:
            out.append(text)
    return out


def normalized_skill_set(value) -> set:
    return {str(v).strip().lower() for v in cell_values(value) if str(v).strip()}


def skill_weight(skill: str, idf_by_skill: dict) -> float:
    raw_idf = idf_by_skill.get(skill, 0.0)
    if not np.isfinite(raw_idf):
        return 0.0
    return max(float(raw_idf), 0.0)


def skill_coverage(job_skills, candidate_skills, idf_by_skill: dict) -> dict:
    required = normalized_skill_set(job_skills)
    candidate = normalized_skill_set(candidate_skills)
    if not required:
        return {"score": None, "matched": [], "missing": [], "required_count": 0}
    matched = sorted(required & candidate)
    missing = sorted(required - candidate)
    weights = {skill: skill_weight(skill, idf_by_skill) for skill in required}
    total_weight = sum(weights.values())
    if total_weight > 0:
        matched_weight = sum(weights[s] for s in matched)
        score = 100.0 * matched_weight / total_weight
    else:
        score = 100.0 * len(matched) / len(required)
    return {"score": round(score, 4), "matched": matched, "missing": missing, "required_count": len(required)}


def normalize_phrase(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value)).lower().strip()
    text = re.sub(r"[^\w+#.]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def phrase_similarity(left: str, right: str) -> float:
    a, b = normalize_phrase(left), normalize_phrase(right)
    if not a or not b:
        return 0.0
    a_tokens, b_tokens = set(a.split()), set(b.split())
    return 2.0 * len(a_tokens & b_tokens) / max(len(a_tokens) + len(b_tokens), 1)


def phrase_is_explicit_match(left: str, right: str) -> bool:
    a, b = normalize_phrase(left), normalize_phrase(right)
    if not a or not b:
        return False
    a_tokens, b_tokens = set(a.split()), set(b.split())
    return a_tokens == b_tokens or a_tokens.issubset(b_tokens) or b_tokens.issubset(a_tokens)


def is_education_credential(value: str) -> bool:
    tokens = set(normalize_phrase(value).split())
    degree_markers = {"degree", "bachelor", "bachelors", "master", "masters",
                      "phd", "doctorate", "diploma", "school"}
    return bool(tokens & degree_markers)


def phrase_list_coverage(job_values, candidate_values) -> dict:
    requirements = cell_values(job_values)
    candidate_items = cell_values(candidate_values)
    if not requirements:
        return {"score": None, "matched": [], "missing": []}
    best_pairs = []
    for requirement in requirements:
        if candidate_items:
            similarities = [(phrase_similarity(requirement, item), item) for item in candidate_items]
            best_similarity, best_item = max(similarities, key=lambda x: x[0])
        else:
            best_similarity, best_item = 0.0, None
        best_pairs.append((requirement, best_item, best_similarity))
    matched = [req for req, item, _ in best_pairs if item and phrase_is_explicit_match(req, item)]
    missing = [req for req, item, _ in best_pairs if not item or not phrase_is_explicit_match(req, item)]
    score = 100.0 * sum(sim for _, _, sim in best_pairs) / len(best_pairs)
    return {"score": round(score, 4), "matched": matched, "missing": missing}


def safe_float(value):
    values = cell_values(value)
    if not values:
        return None
    try:
        number = float(values[0])
        return number if np.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def ordinal_score(candidate_value, job_value, order: dict):
    candidate_items = [str(v).strip().lower() for v in cell_values(candidate_value)]
    job_items = [str(v).strip().lower() for v in cell_values(job_value)]
    candidate_ranks = [order[v] for v in candidate_items if v in order]
    job_ranks = [order[v] for v in job_items if v in order]
    if not candidate_ranks or not job_ranks:
        return None
    candidate_rank = max(candidate_ranks)
    required_rank = min(job_ranks)
    shortfall = max(required_rank - candidate_rank, 0)
    max_distance = max(len(order) - 1, 1)
    return round(100.0 * (1.0 - shortfall / max_distance), 4)


def years_experience_score(candidate_years, job_minimum):
    candidate_years = safe_float(candidate_years)
    job_minimum = safe_float(job_minimum)
    if candidate_years is None or job_minimum is None:
        return None
    if job_minimum <= 0 or candidate_years >= job_minimum:
        return 100.0
    return round(100.0 * max(candidate_years, 0.0) / job_minimum, 4)


def education_score(candidate: dict, job_row: pd.Series) -> dict:
    level_score = ordinal_score(candidate.get("education_level"), job_row.get("education_level"), _edu_order)
    field_result = phrase_list_coverage(job_row.get("education_field"), candidate.get("education_field"))
    return {"level_score": level_score, "field_score": field_result["score"],
            "matched_fields": field_result["matched"], "missing_fields": field_result["missing"]}


def bool_value(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() in {"true", "1", "yes"}


def experience_fit_score(candidate: dict, job_row: pd.Series) -> dict:
    years_score = years_experience_score(candidate.get("years_experience_total"), job_row.get("years_experience_min"))
    level_score = ordinal_score(candidate.get("experience_level"), job_row.get("experience_level"), _exp_order)
    score = years_score if years_score is not None else level_score
    return {"score": score, "years_score": years_score, "level_score": level_score}


def equal_weight_absolute_score(component_scores: dict) -> dict:
    active = {k: float(component_scores[k]) for k in RANKING_CRITERIA
              if component_scores.get(k) is not None and np.isfinite(float(component_scores[k]))}
    score = float(np.mean(list(active.values()))) if active else 0.0
    equal_weights = {k: (1.0 / len(active) if k in active else 0.0) for k in RANKING_CRITERIA}
    return {"match_score": round(score, 4),
            "scoring_coverage": round(len(active) / len(RANKING_CRITERIA), 4),
            "equal_weights": equal_weights}


def critic_objective_weights(score_frame: pd.DataFrame) -> dict:
    available = [c for c in RANKING_CRITERIA if c in score_frame and score_frame[c].notna().any()]
    if not available:
        return {c: 0.0 for c in RANKING_CRITERIA}
    matrix = score_frame[available].astype(float) / 100.0
    matrix = matrix.apply(lambda col: col.fillna(col.mean()), axis=0)
    std = matrix.std(ddof=0)
    informative = std[std > 0].index.tolist()
    if len(matrix) < 2 or not informative:
        neutral = 1.0 / len(available)
        return {c: (neutral if c in available else 0.0) for c in RANKING_CRITERIA}
    corr = matrix[informative].corr().fillna(0.0)
    information = std[informative] * (1.0 - corr).sum(axis=1)
    if information.sum() <= 0:
        neutral = 1.0 / len(available)
        return {c: (neutral if c in available else 0.0) for c in RANKING_CRITERIA}
    raw = (information / information.sum()).to_dict()
    return {c: float(raw.get(c, 0.0)) for c in RANKING_CRITERIA}


def critic_score(row: pd.Series, weights: dict) -> float:
    active = [(float(row[c]), weights[c]) for c in RANKING_CRITERIA
              if c in row and pd.notna(row[c]) and weights.get(c, 0.0) > 0]
    if not active:
        return float(row.get("match_score", 0.0))
    total_weight = sum(weight for _, weight in active)
    return round(sum(score * weight for score, weight in active) / total_weight, 4)


def match_job(job_row: pd.Series, candidate: dict, idf_by_skill: dict) -> dict:
    candidate_core_skills = sorted(set().union(
        normalized_skill_set(candidate.get("hard_skills_norm")),
        normalized_skill_set(candidate.get("soft_skills_norm")),
        normalized_skill_set(candidate.get("tools_technologies_norm")),
    ))
    candidate_all_skills = sorted(set(candidate_core_skills) | normalized_skill_set(candidate.get("languages_norm")))

    job_languages = normalized_skill_set(job_row.get("languages_norm"))
    stated_required = normalized_skill_set(job_row.get("required_skills_norm"))
    core_requirements = stated_required - job_languages
    if not core_requirements:
        core_requirements = set().union(
            normalized_skill_set(job_row.get("hard_skills_norm")),
            normalized_skill_set(job_row.get("soft_skills_norm")),
            normalized_skill_set(job_row.get("tools_technologies_norm")),
        ) - job_languages

    core = skill_coverage(core_requirements, candidate_core_skills, idf_by_skill)
    required = skill_coverage(job_row.get("required_skills_norm"), candidate_all_skills, idf_by_skill)
    preferred = skill_coverage(job_row.get("preferred_skills_norm"), candidate_all_skills, idf_by_skill)
    hard = skill_coverage(job_row.get("hard_skills_norm"), candidate.get("hard_skills_norm"), idf_by_skill)
    tools = skill_coverage(job_row.get("tools_technologies_norm"), candidate.get("tools_technologies_norm"), idf_by_skill)
    soft = skill_coverage(job_row.get("soft_skills_norm"), candidate.get("soft_skills_norm"), idf_by_skill)
    languages = skill_coverage(job_row.get("languages_norm"), candidate.get("languages_norm"), idf_by_skill)

    experience = experience_fit_score(candidate, job_row)
    education = education_score(candidate, job_row)
    job_prof_quals = [v for v in cell_values(job_row.get("qualifications")) if not is_education_credential(v)]
    cand_prof_quals = [v for v in cell_values(candidate.get("qualifications")) if not is_education_credential(v)]
    qualifications = phrase_list_coverage(job_prof_quals, cand_prof_quals)

    component_scores = {
        "core_skills_score": core["score"], "required_skills_score": required["score"],
        "preferred_skills_score": preferred["score"], "hard_skills_score": hard["score"],
        "tools_score": tools["score"], "soft_skills_score": soft["score"],
        "languages_score": languages["score"], "experience_fit_score": experience["score"],
        "experience_level_score": experience["level_score"], "years_experience_score": experience["years_score"],
        "education_level_score": education["level_score"], "education_field_score": education["field_score"],
        "education_score": education["field_score"], "qualifications_score": qualifications["score"],
    }

    absolute = equal_weight_absolute_score(component_scores)
    candidate_confidence = safe_float(candidate.get("extraction_confidence"))
    job_confidence = safe_float(job_row.get("extraction_confidence"))
    source_confidences = [v for v in (candidate_confidence, job_confidence) if v is not None]
    source_confidence = sum(source_confidences) / len(source_confidences) if source_confidences else 1.0
    experience_inferred = (bool_value(job_row.get("experience_level_inferred"))
                            or bool_value(candidate.get("experience_level_inferred")))
    active_reliabilities = []
    for criterion in RANKING_CRITERIA:
        if component_scores.get(criterion) is not None:
            active_reliabilities.append(
                source_confidence if criterion == "experience_fit_score" and experience_inferred else 1.0)
    evidence_reliability = float(np.mean(active_reliabilities)) if active_reliabilities else 0.0
    scoring_confidence = absolute["scoring_coverage"] * source_confidence * evidence_reliability

    evidence = {
        "matched_core_skills": core["matched"], "missing_core_skills": core["missing"],
        "matched_required_skills": required["matched"], "missing_required_skills": required["missing"],
        "matched_preferred_skills": preferred["matched"], "missing_preferred_skills": preferred["missing"],
        "matched_hard_skills": hard["matched"], "missing_hard_skills": hard["missing"],
        "matched_tools": tools["matched"], "missing_tools": tools["missing"],
        "matched_soft_skills": soft["matched"], "missing_soft_skills": soft["missing"],
        "matched_languages": languages["matched"], "missing_languages": languages["missing"],
        "matched_qualifications": qualifications["matched"], "missing_qualifications": qualifications["missing"],
        "matched_education_fields": education["matched_fields"], "missing_education_fields": education["missing_fields"],
    }

    return {
        **component_scores,
        "skill_match_score": core["score"] if core["score"] is not None else 0.0,
        "match_score": absolute["match_score"],
        "scoring_coverage": absolute["scoring_coverage"],
        "scoring_confidence": round(float(np.clip(scoring_confidence, 0.0, 1.0)), 4),
        **evidence,
        "match_evidence_json": json.dumps(evidence, ensure_ascii=False, sort_keys=True),
        "component_scores_json": json.dumps(component_scores, ensure_ascii=False, sort_keys=True),
        "effective_weights_json": json.dumps(absolute["equal_weights"], sort_keys=True),
    }


def score_and_rank(filtered_df: pd.DataFrame, candidate_profile: dict, idf_by_skill: dict) -> pd.DataFrame:
    """Full Phase 5 pipeline: score every survivor, CRITIC-weight, sort, rank."""
    if len(filtered_df) == 0:
        return filtered_df.copy()

    match_records = [match_job(row, candidate_profile, idf_by_skill) for _, row in filtered_df.iterrows()]
    score_df = pd.DataFrame(match_records, index=filtered_df.index)
    matched_df = pd.concat([filtered_df.copy(), score_df], axis=1)

    critic_weights = critic_objective_weights(score_df)
    matched_df["objective_rank_score"] = matched_df.apply(lambda row: critic_score(row, critic_weights), axis=1)
    matched_df["critic_weights_json"] = json.dumps(critic_weights, sort_keys=True)

    matched_df = matched_df.sort_values(
        ["objective_rank_score", "match_score", "preferred_skills_score",
         "scoring_confidence", "similarity", "post_filter_rank"],
        ascending=[False, False, False, False, False, True], kind="mergesort",
    ).reset_index(drop=True)

    matched_df.insert(0, "phase5_rank", range(1, len(matched_df) + 1))
    matched_df.insert(1, "candidate_id", candidate_profile["candidate_id"])
    matched_df["is_phase6_candidate"] = matched_df["phase5_rank"] <= PHASE6_POOL_SIZE
    return matched_df


def build_skill_vocabulary(jobs_df: pd.DataFrame) -> pd.DataFrame:
    """Rebuild Phase 1's skill_vocabulary.parquet (canonical skill -> IDF)
    directly from jobs_prepared.parquet, since it wasn't shipped separately."""
    from collections import Counter
    counter = Counter()
    for skills in jobs_df["all_skills_norm"]:
        counter.update(cell_values(skills))
    vocab_df = pd.DataFrame(counter.most_common(), columns=["skill_canonical", "job_count"])
    vocab_df["idf"] = np.log(len(jobs_df) / (1 + vocab_df["job_count"])).round(4)
    return vocab_df
