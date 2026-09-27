"""
Central configuration. Values mirror the CONFIG dicts scattered across the
original Phase 2/3/4/5/6/7/8 notebooks, collected here in one place.

IMPORTANT — model names:
The original notebooks call a model named "gpt-5.6-luna". Replace
EXTRACTION_MODEL / EXPLANATION_MODEL / AGENT_MODEL below with a model name
that actually exists on your OpenAI account if "gpt-5.6-luna" is not
available to you (e.g. "gpt-4o", "gpt-4o-mini", or whatever your account
has access to).
"""

# ---- Phase 2: candidate profile extraction ----
EXTRACTION_MODEL = "gpt-5.6-luna"
PROMPT_VERSION = "v2.0"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
MAX_RESUME_CHARS_SENT_TO_LLM = 8_000
MIN_RESUME_CHARS = 200
MAX_RETRIES = 5

# ---- Phase 3: retrieval ----
RETRIEVAL_TOP_K = 150
RETRIEVAL_MIN_SIMILARITY = None

# ---- Phase 5: matching ----
RANKING_CRITERIA = [
    "core_skills_score",
    "languages_score",
    "experience_fit_score",
    "education_field_score",
    "qualifications_score",
]
PHASE6_POOL_SIZE = 50

# ---- Phase 6: cross-encoder reranking ----
# Switched from Alibaba-NLP/gte-multilingual-reranker-base (~306M params,
# multilingual, requires trust_remote_code) to cross-encoder/ms-marco-MiniLM-L-6-v2
# (~22M params, English-only) since the app's resume/job text is English-only.
# Much lighter to download and run on Streamlit Cloud's free CPU tier.
# No trust_remote_code / custom code_revision needed for this model.
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_MODEL_REVISION = "d3c87ff86d02a312dc895fff50a815f5aaaee56b"
RERANKER_CODE_REVISION = None
RERANKER_TRUST_REMOTE_CODE = False
FINAL_TOP_N = 20
INITIAL_BATCH_SIZE = 8
# ms-marco-MiniLM-L-6-v2 has max_position_embeddings=512 (covers query+passage
# together). 2048 was fine for the previous larger model but overflows this
# one's position embeddings, so it's capped here to match.
MAX_PAIR_TOKENS = 512
# Manual, deterministic word caps applied to the long free-text fields before
# they reach the tokenizer, so truncation always drops the tail of a long
# job description / responsibilities / summary — never the short structured
# fields (skills, title, experience, education) built earlier in the text.
# ~130 words ≈ 512 tokens total once split across both sides of the pair
# plus the structured sections, with headroom to spare.
JOB_DESCRIPTION_MAX_WORDS = 150
RESPONSIBILITIES_MAX_WORDS = 60
CANDIDATE_SUMMARY_MAX_WORDS = 60
PII_FIELDS = ["full_name", "email", "phone", "source_file"]

# ---- Phase 7 / 8: LLM explanations + agent ----
EXPLANATION_MODEL = "gpt-5.6-luna"
AGENT_MODEL = "gpt-5.6-luna"
MAX_TOOL_ITERATIONS = 6

# ---- Data paths (bundled with the app, produced once by Phase 1) ----
DATA_DIR = "data"
JOBS_PARQUET = f"{DATA_DIR}/jobs_prepared.parquet"
JOB_EMBEDDINGS = f"{DATA_DIR}/job_embeddings.npy"
JOB_IDS = f"{DATA_DIR}/job_ids.npy"
SKILL_VOCAB_PARQUET = f"{DATA_DIR}/skill_vocabulary.parquet"

EDUCATION_HIERARCHY = ["High School", "Diploma", "Bachelor's Degree", "Master's Degree", "PhD"]
EXPERIENCE_HIERARCHY = ["Internship", "Entry Level", "Mid Level", "Senior Level", "Executive"]
