"""Phase 8 — Agent. Pure-Python tools over already-computed Phase 5/6/7
outputs, wired to an OpenAI function-calling loop. No re-scoring, no
re-embedding — every fact the agent states must come from a tool call."""
import json
import re
from collections import Counter

import numpy as np

from . import config


def make_tools(jobs_by_id: dict, browse_pool):
    """Builds the tool functions + JSON-schemas bound to this candidate's
    already-computed results (closure over jobs_by_id / browse_pool)."""

    def _job_not_found(job_id):
        return {"error": "job_not_found", "job_id": job_id,
                "message": f"No job found with id {job_id} in the loaded results."}

    def list_top_jobs(limit: int = 10):
        jobs = list(jobs_by_id.values())
        limit = max(1, min(limit, len(jobs)))
        return [
            {
                "job_id": j["job_id"],
                "title": j["title"],
                "company": j["company"],
                "city": (j.get("location") or {}).get("city"),
                "match_score": j["match_score"],
                "phase6_rank": j.get("phase6_rank"),
            }
            for j in jobs[:limit]
        ]

    def get_job_explanation(job_id: str):
        return jobs_by_id.get(job_id) or _job_not_found(job_id)

    def compare_jobs(job_id_a: str, job_id_b: str):
        a, b = jobs_by_id.get(job_id_a), jobs_by_id.get(job_id_b)
        if a is None:
            return _job_not_found(job_id_a)
        if b is None:
            return _job_not_found(job_id_b)

        def brief(j):
            return {"job_id": j["job_id"], "title": j["title"], "company": j["company"],
                    "match_score": j["match_score"], "n_matched_skills": len(j.get("matched_skills", [])),
                    "n_missing_skills": len(j.get("missing_skills", [])), "missing_skills": j.get("missing_skills", [])}
        return {"job_a": brief(a), "job_b": brief(b)}

    def skill_gap_summary():
        counter = Counter()
        for job in jobs_by_id.values():
            counter.update(job.get("missing_skills", []))
        if not counter:
            return {"message": "No missing skills recorded — excellent match across all jobs.", "gaps": []}
        return {"gaps": [{"skill": s, "missing_in_n_jobs": c} for s, c in counter.most_common(10)],
                "n_jobs_analyzed": len(jobs_by_id)}

    def get_learning_plan(job_id: str):
        job = jobs_by_id.get(job_id)
        if job is None:
            return _job_not_found(job_id)
        return {"job_id": job_id, "title": job["title"], "learning_areas": job.get("learning_areas", [])}

    def browse_jobs(city: str = None, country: str = None, work_arrangement: str = None,
                     min_match_score: float = None, limit: int = 15):
        if browse_pool is None:
            return {"error": "no_browse_pool",
                    "message": "No wider job pool is loaded — use list_top_jobs instead."}
        df = browse_pool.copy()
        if city:
            df = df[df["city"].astype(str).str.contains(city, case=False, na=False)]
        if country:
            df = df[df["country"].astype(str).str.contains(country, case=False, na=False)]
        if work_arrangement and "work_arrangement" in df.columns:
            df = df[df["work_arrangement"].astype(str).str.lower() == work_arrangement.lower()]
        score_col = "match_score" if "match_score" in df.columns else "objective_rank_score"
        if min_match_score is not None:
            df = df[df[score_col] >= min_match_score]
        df = df.sort_values(score_col, ascending=False)
        cols = [c for c in ["job_id", "title_clean", "company_clean", "city", "country",
                             "work_arrangement", score_col] if c in df.columns]
        limit = max(1, min(limit, 50))
        results = df[cols].head(limit).to_dict(orient="records")
        return {"count": len(results), "total_matches": len(df), "results": results}

    def get_candidate_summary():
        if not jobs_by_id:
            return {"message": "No jobs loaded."}
        scores = [j["match_score"] for j in jobs_by_id.values()]
        best = max(jobs_by_id.values(), key=lambda j: j["match_score"])
        return {"n_jobs_evaluated": len(jobs_by_id), "avg_match_score": round(float(np.mean(scores)), 2),
                "best_match": {"job_id": best["job_id"], "title": best["title"],
                                "company": best["company"], "match_score": best["match_score"]}}

    tool_functions = {
        "list_top_jobs": list_top_jobs, "get_job_explanation": get_job_explanation,
        "compare_jobs": compare_jobs, "skill_gap_summary": skill_gap_summary,
        "get_learning_plan": get_learning_plan, "browse_jobs": browse_jobs,
        "get_candidate_summary": get_candidate_summary,
    }

    tools_schema = [
        {"type": "function", "name": "list_top_jobs",
         "description": "Top matching jobs for the candidate in the final ranking order. Use this for general questions like 'what are my top matching jobs?' or 'what's my best job?'.",
         "parameters": {"type": "object", "properties": {
             "limit": {"type": "integer", "description": "Number of jobs to return", "default": 10}}, "required": []}},
        {"type": "function", "name": "get_job_explanation",
         "description": "Full explanation for one job: why it matches, matched/missing skills, qualifications, education, and improvement suggestions.",
         "parameters": {"type": "object", "properties": {"job_id": {"type": "string"}}, "required": ["job_id"]}},
        {"type": "function", "name": "compare_jobs",
         "description": "Side-by-side comparison of two specific jobs by id.",
         "parameters": {"type": "object", "properties": {
             "job_id_a": {"type": "string"}, "job_id_b": {"type": "string"}}, "required": ["job_id_a", "job_id_b"]}},
        {"type": "function", "name": "skill_gap_summary",
         "description": "Most frequently missing skills across all matched jobs — answers 'what should I improve?'.",
         "parameters": {"type": "object", "properties": {}, "required": []}},
        {"type": "function", "name": "get_learning_plan",
         "description": "Learning/improvement suggestions tied to one specific job.",
         "parameters": {"type": "object", "properties": {"job_id": {"type": "string"}}, "required": ["job_id"]}},
        {"type": "function", "name": "browse_jobs",
         "description": "Filter/browse jobs by city, country, work arrangement, or minimum match score. Covers a wider pool than the top N.",
         "parameters": {"type": "object", "properties": {
             "city": {"type": "string"}, "country": {"type": "string"},
             "work_arrangement": {"type": "string", "enum": ["On-site", "Remote", "Hybrid", "Not Specified"]},
             "min_match_score": {"type": "number"}, "limit": {"type": "integer", "default": 15}}, "required": []}},
        {"type": "function", "name": "get_candidate_summary",
         "description": "Overall summary of the matching session: number of jobs, average score, best match.",
         "parameters": {"type": "object", "properties": {}, "required": []}},
    ]

    return tool_functions, tools_schema


SYSTEM_PROMPT_TEMPLATE = """
You are the conversational agent in an AI Job Matching system, talking
directly to the candidate (candidate_id: {candidate_id}).

You have access to tools that query already-computed matching results from
Phase 5 (scoring), Phase 6 (re-ranking), and Phase 7 (LLM explanations).
You do NOT compute or estimate match scores yourself — you only report what
the tools return.

STRICT GROUNDING RULES

1. Never state a match score, skill, qualification, or education fact about
   a specific job without first calling a tool to retrieve it.
2. Never invent a job_id. Only reference job_ids returned by a tool.
3. If a tool returns an error (e.g. job_not_found, no_browse_pool), explain
   the limitation to the user in plain language — do not pretend the data
   exists.
4. If the user asks something outside your tools' scope (e.g. "change my
   desired salary"), explain that this requires re-running earlier phases
   of the pipeline, not something you can do here.
5. Refer to jobs by title and company (and city if helpful). Do NOT show
   job_ids to the user unless they explicitly ask for them. Keep the
   job_ids in mind internally so you can call tools for follow-up
   questions about a specific job.
6. Keep answers concise and in the user's language (Arabic or English,
   mirror what they use).
7. You may call more than one tool in sequence if a question needs it.
8. Do not repeat the match score when explaining a job because it is already
   displayed separately in the user interface. Never describe a match score
   as a probability, chance, likelihood of getting the job, or likelihood
   of being hired.
9. Speak naturally to the user. Never use internal wording such as
   "evidence", "tool", "Phase 5", or "the matching evidence". Say things
   like "based on your resume and this job's requirements" instead.
10. Do not infer that the candidate is currently studying, currently enrolled,
    or pursuing a degree unless this is explicitly stated in the tool output.
    When a job requires current study or enrollment, describe it only as a
    job requirement and do not imply that the candidate is currently studying.
11. When explaining a specific job, use the job explanation returned by the
    tool as the source for matched and missing skills, qualifications, and
    education. Do not combine or reinterpret these categories.
12. Do not make your own overall judgment about which job is better, more
    suitable, or a closer match based on comparing tool results. Report the
    differences between jobs factually, such as matched skills, missing
    skills, and match scores when relevant to the user's question. Let the
    candidate make the final judgment.

Be warm and helpful, like a career advisor — but every factual claim must
trace back to a tool call.
"""


def run_tool_call(tool_functions: dict, name: str, arguments: dict) -> dict:
    fn = tool_functions.get(name)
    if fn is None:
        return {"error": "unknown_tool", "message": f"No such tool: {name}"}
    try:
        return fn(**arguments)
    except Exception as exc:  # noqa: BLE001
        return {"error": "tool_execution_failed", "message": repr(exc)[:300]}


def ask_agent(client, user_message: str, history: list, tool_functions: dict,
              tools_schema: list, system_prompt: str, usage_state: dict = None) -> tuple:
    """One conversational turn. Returns (assistant_text, updated_history, tool_calls_made)."""
    history = history + [{"role": "user", "content": user_message}]
    tool_calls_made = []
    from .usage_tracker import track_llm

    for _ in range(config.MAX_TOOL_ITERATIONS):
        response = client.responses.create(
            model=config.AGENT_MODEL, instructions=system_prompt, input=history, tools=tools_schema,
        )
        track_llm(usage_state, response, "phase8_chat")
        function_calls = [item for item in response.output if item.type == "function_call"]

        if not function_calls:
            assistant_text = response.output_text
            history = history + [{"role": "assistant", "content": assistant_text}]
            return assistant_text, history, tool_calls_made

        history = history + list(response.output)
        for call in function_calls:
            args = json.loads(call.arguments) if call.arguments else {}
            result = run_tool_call(tool_functions, call.name, args)
            tool_calls_made.append({"tool": call.name, "arguments": args})
            history = history + [{"type": "function_call_output", "call_id": call.call_id,
                                   "output": json.dumps(result, ensure_ascii=False)}]

    fallback = ("Sorry, I needed too many lookups to answer that and couldn't reach a final "
                "answer. Try rephrasing more simply.")
    history = history + [{"role": "assistant", "content": fallback}]
    return fallback, history, tool_calls_made


_JOB_ID_PATTERN = re.compile(r"\bjob_[a-zA-Z0-9_]{3,}\b")


def check_grounding(assistant_text: str, known_job_ids: set) -> list:
    mentioned = set(_JOB_ID_PATTERN.findall(assistant_text))
    return sorted(mentioned - known_job_ids)
