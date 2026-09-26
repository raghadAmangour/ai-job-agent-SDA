"""Phase 4 — Hard Filtering. Deterministic rule-based filters, identical to the notebook."""
import numpy as np
import pandas as pd

from .config import EDUCATION_HIERARCHY


def _is_unset(value) -> bool:
    return value in (None, "", [], "Not Specified")


def highest_education(levels: list) -> str:
    order = {lvl.lower(): i for i, lvl in enumerate(EDUCATION_HIERARCHY)}
    ranked = [lvl for lvl in (levels or []) if str(lvl).lower() in order]
    if not ranked:
        return None
    return max(ranked, key=lambda lvl: order[str(lvl).lower()])


def _cell_values(value) -> list:
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
    return [v for v in items if v is not None and not (isinstance(v, float) and pd.isna(v))]


def mask_country(df, allowed):
    if not allowed or "country" not in df.columns:
        return np.ones(len(df), dtype=bool)
    allowed_lower = {str(c).strip().lower() for c in allowed}

    def check(value):
        values = _cell_values(value)
        return bool(values) and any(str(v).strip().lower() in allowed_lower for v in values)
    return df["country"].apply(check).to_numpy()


def mask_city(df, allowed, enforce_for_remote=False):
    if not allowed or "city" not in df.columns:
        return np.ones(len(df), dtype=bool)
    allowed_lower = {str(c).strip().lower() for c in allowed}

    def check(value):
        values = _cell_values(value)
        return any(str(v).strip().lower() in allowed_lower for v in values)
    keep = df["city"].apply(check).to_numpy()

    if not enforce_for_remote and "work_arrangement" in df.columns:
        def is_remote(value):
            return any(str(v).strip().lower() == "remote" for v in _cell_values(value))
        keep = keep | df["work_arrangement"].apply(is_remote).to_numpy()
    return keep


def mask_work_arrangement(df, accepted):
    if not accepted or "work_arrangement" not in df.columns:
        return np.ones(len(df), dtype=bool)
    accepted_lower = {str(a).strip().lower() for a in accepted}

    def check(value):
        return any(str(v).strip().lower() in accepted_lower for v in _cell_values(value))
    matches = df["work_arrangement"].apply(check).to_numpy()

    if "work_arrangement_inferred" in df.columns:
        inferred = df["work_arrangement_inferred"].fillna(False).astype(bool).to_numpy()
        return matches | inferred
    return matches


def mask_employment_type(df, accepted):
    if not accepted or "employment_type" not in df.columns:
        return np.ones(len(df), dtype=bool)
    accepted_lower = {str(a).strip().lower() for a in accepted}

    def check(value):
        return any(str(v).strip().lower() in accepted_lower for v in _cell_values(value))
    return df["employment_type"].apply(check).to_numpy()


def mask_education(df, candidate_level, hierarchy=EDUCATION_HIERARCHY):
    if not candidate_level or "education_level" not in df.columns:
        return np.ones(len(df), dtype=bool)
    order = {level.strip().lower(): i for i, level in enumerate(hierarchy)}
    cand_idx = order.get(str(candidate_level).strip().lower())
    if cand_idx is None:
        return np.ones(len(df), dtype=bool)

    def check(value):
        values = _cell_values(value)
        if not values:
            return True
        recognized = [order.get(str(v).strip().lower()) for v in values]
        recognized = [i for i in recognized if i is not None]
        if not recognized:
            return True
        return any(idx <= cand_idx for idx in recognized)
    return df["education_level"].apply(check).to_numpy()


def mask_nationality(df, candidate_nationality, strict=False):
    if "nationality_requirement" not in df.columns:
        return np.ones(len(df), dtype=bool)
    open_values = {"", "none", "any", "open", "all", "no restriction", "nan", "not specified"}

    def check(value):
        values = _cell_values(value)
        normalized = [str(v).strip().lower() for v in values]
        if not normalized or any(v in open_values for v in normalized):
            return True
        if not candidate_nationality:
            return not strict
        cand = str(candidate_nationality).strip().lower()
        return any(cand in v for v in normalized)
    return df["nationality_requirement"].apply(check).to_numpy()


def mask_salary(df, cand_min, cand_max, cand_currency):
    needed_cols = {"salary_min", "salary_max", "salary_currency"}
    if (cand_min is None and cand_max is None) or not needed_cols.issubset(df.columns):
        return np.ones(len(df), dtype=bool)

    def scalar(value):
        values = _cell_values(value)
        return values[0] if values else None

    def check(row):
        job_min = scalar(row["salary_min"])
        job_max = scalar(row["salary_max"])
        job_currency = scalar(row["salary_currency"])
        if job_min is None and job_max is None:
            return True
        if cand_currency and job_currency is not None and str(job_currency).strip().lower() != str(cand_currency).strip().lower():
            return True
        lo = cand_min if cand_min is not None else float("-inf")
        hi = cand_max if cand_max is not None else float("inf")
        job_lo = job_min if job_min is not None else float("-inf")
        job_hi = job_max if job_max is not None else float("inf")
        return job_lo <= hi and job_hi >= lo
    return df.apply(check, axis=1).to_numpy()


def run_filter_funnel(retrieved_df: pd.DataFrame, candidate_profile: dict) -> tuple:
    """Runs the full Phase 4 funnel. Returns (filtered_df, funnel_records)."""
    candidate_highest_education = highest_education(candidate_profile.get("education_level"))
    willing_to_relocate = candidate_profile.get("willing_to_relocate")

    cfg = {
        "allowed_countries": None if willing_to_relocate else candidate_profile.get("desired_countries"),
        "allowed_cities": None if willing_to_relocate else candidate_profile.get("desired_cities"),
        "accepted_work_arrangements": (
            None if _is_unset(candidate_profile.get("work_arrangement_preference"))
            else [candidate_profile["work_arrangement_preference"]]
        ),
        "accepted_employment_types": (
            None if _is_unset(candidate_profile.get("employment_type_preference"))
            else [candidate_profile["employment_type_preference"]]
        ),
        "candidate_education_level": candidate_highest_education,
        "candidate_nationality": candidate_profile.get("nationality"),
        "salary_expectation_min": candidate_profile.get("salary_expectation_min"),
        "salary_expectation_max": candidate_profile.get("salary_expectation_max"),
        "salary_currency": candidate_profile.get("salary_currency"),
    }

    steps = [
        ("country", mask_country(retrieved_df, cfg["allowed_countries"])),
    ]
    current_df = retrieved_df.copy()
    funnel_records = []

    def apply_step(df, name, mask):
        before = len(df)
        survivors = df[mask]
        funnel_records.append({"step": name, "before": before, "after": len(survivors),
                                "dropped": before - len(survivors)})
        return survivors

    current_df = apply_step(current_df, "country", mask_country(current_df, cfg["allowed_countries"]))
    current_df = apply_step(current_df, "city", mask_city(current_df, cfg["allowed_cities"]))
    current_df = apply_step(current_df, "work_arrangement",
                             mask_work_arrangement(current_df, cfg["accepted_work_arrangements"]))
    current_df = apply_step(current_df, "employment_type",
                             mask_employment_type(current_df, cfg["accepted_employment_types"]))
    current_df = apply_step(current_df, "education",
                             mask_education(current_df, cfg["candidate_education_level"]))
    current_df = apply_step(current_df, "nationality",
                             mask_nationality(current_df, cfg["candidate_nationality"]))
    current_df = apply_step(current_df, "salary",
                             mask_salary(current_df, cfg["salary_expectation_min"],
                                         cfg["salary_expectation_max"], cfg["salary_currency"]))

    filtered_df = current_df.reset_index(drop=True)
    if len(filtered_df):
        filtered_df["post_filter_rank"] = range(1, len(filtered_df) + 1)
    return filtered_df, funnel_records
