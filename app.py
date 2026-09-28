"""
AI Job Agent — Streamlit app.

Runs the live-candidate path of the job matching pipeline for one uploaded
resume, then offers a chat experience over the results.

The job corpus is prepared separately and loaded as static files from data/.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src import config
from src.resume_reader import read_resume, resume_quality_check
from src.phase2_profile import build_candidate_profile, profile_warnings
from src.phase3_retrieval import retrieve_jobs_for_candidate
from src.phase4_filtering import run_filter_funnel
from src.phase5_matching import score_and_rank, build_skill_vocabulary
from src.phase6_rerank import load_reranker, rerank, to_phase7_payload
from src.phase7_explain import explain_jobs
from src.phase8_agent import (
    make_tools,
    ask_agent,
    check_grounding,
    SYSTEM_PROMPT_TEMPLATE,
)
from src.usage_tracker import new_usage_state, estimate_cost_usd, total_tokens


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

LOGO_PATH = "assets/logo.png"

try:
    page_icon = Image.open(LOGO_PATH)
except Exception:
    page_icon = "🧭"

st.set_page_config(
    page_title="Masar — AI Job Agent",
    page_icon=page_icon,
    layout="wide",
)


MAX_CHAT_TURNS_PER_SESSION = 12


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading job data...")
def load_job_data():
    jobs_df = pd.read_parquet(config.JOBS_PARQUET)

    job_vectors = np.load(config.JOB_EMBEDDINGS)

    job_ids = np.load(
        config.JOB_IDS,
        allow_pickle=True
    )

    if Path(config.SKILL_VOCAB_PARQUET).exists():
        vocab_df = pd.read_parquet(
            config.SKILL_VOCAB_PARQUET
        )
    else:
        vocab_df = build_skill_vocabulary(jobs_df)

    idf_by_skill = dict(
        zip(
            vocab_df["skill_canonical"].astype(str),
            vocab_df["idf"].astype(float),
        )
    )

    return (
        jobs_df,
        job_vectors,
        job_ids,
        idf_by_skill,
    )


@st.cache_resource(
    show_spinner=(
        "Loading the semantic matching model "
        "(first run only, ~1-2 min)..."
    )
)
def get_reranker():
    try:
        return load_reranker()
    except Exception:
        return None, None, None


# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

def get_openai_client():
    """
    Load the OpenAI API key from Streamlit Secrets or environment variables.

    The API key is intentionally not exposed in the user interface.
    """

    api_key = (
        st.secrets.get("OPENAI_API_KEY", "")
        or os.environ.get("OPENAI_API_KEY", "")
    )

    if not api_key:
        return None

    from openai import OpenAI

    return OpenAI(api_key=api_key)


# ---------------------------------------------------------------------------
# Sidebar — user preferences
# ---------------------------------------------------------------------------

st.sidebar.header("⚙️ Settings")

st.sidebar.subheader("Your Preferences")

desired_countries = st.sidebar.text_input(
    "Desired countries (comma-separated)",
    "Saudi Arabia",
)

desired_cities = st.sidebar.text_input(
    "Desired cities (optional)",
    "",
)

willing_to_relocate = st.sidebar.checkbox(
    "Willing to relocate anywhere",
    value=False,
)

work_arrangement_preference = st.sidebar.selectbox(
    "Work arrangement",
    [
        "Not Specified",
        "Remote",
        "Hybrid",
        "On-site",
    ],
)

employment_type_preference = st.sidebar.selectbox(
    "Employment type",
    [
        "Not Specified",
        "Full-time",
        "Part-time",
        "Contract",
        "Internship",
        "Temporary",

    ],
)

nationality = st.sidebar.text_input(
    "Nationality",
    "Saudi",
)

salary_min = st.sidebar.number_input(
    "Minimum expected salary (optional)",
    min_value=0,
    value=0,
    step=500,
)

salary_max = st.sidebar.number_input(
    "Maximum expected salary (optional)",
    min_value=0,
    value=0,
    step=500,
)

salary_currency = st.sidebar.text_input(
    "Salary currency",
    "SAR",
)

st.sidebar.markdown("---")

# NOTE: defaulted to False for demo stability — Phase 6 (semantic reranking)
# loads a torch/transformers cross-encoder model, which is heavy for
# Streamlit Community Cloud's free CPU tier and can trigger the platform's
# CPU throttle / makes cold starts much slower after the app sleeps. The
# phase is still fully implemented (src/phase6_rerank.py) and documented in
# the write-up — it's just off by default here so the live demo stays fast
# and reliable. Check this box to demonstrate it live if resources allow.
use_reranker = st.sidebar.checkbox(
    "Use semantic reranking",
    value=False,
    help=(
        "Re-ranks the top matches with an AI cross-encoder model for finer "
        "context understanding. Off by default in this demo to keep the "
        "app fast and avoid Streamlit Cloud's free-tier CPU limits — the "
        "underlying matching (Phase 5) is unaffected either way."
    ),
)

final_top_n = st.sidebar.slider(
    "Number of final jobs to display",
    5,
    20,
    10,
)

st.sidebar.markdown("---")

with st.sidebar.expander("📊 About the Job Dataset"):

    st.markdown(
        """
        Masar is powered by a database of over **9,600 real job postings**.

        Each job has already been analyzed by AI to understand its
        requirements in detail, matched against a library of more than
        **35,000 standardized skills**.

        This preparation is done in advance (offline), so the app stays
        fast when you use it — it only needs to focus on analyzing
        your resume.
        """
    )

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    <div style="text-align: center; font-size: 0.8rem; opacity: 0.8;">
        © 2026 Masar Assistant<br><br>
        Developed by<br>
        Raghad Almangour<br>
        Rawan Alotaibi<br>
        Atheer Alzaedi<br>
        Gori Alwabel<br><br>
        Powered by AI Technology
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Main — resume upload + analysis
# ---------------------------------------------------------------------------

col_logo, col_title = st.columns([1, 6])

with col_logo:
    st.image(LOGO_PATH, width=80)

with col_title:
    st.title("Masar")
    st.caption("AI Job Agent")

st.caption(
    "Upload your resume and get personalized job matches "
    "based on your skills, experience, and preferences."
)


tab_match, tab_chat = st.tabs(
    [
        "🔍 Job Matching",
        "💬 Chat with the Agent",
    ]
)


# ---------------------------------------------------------------------------
# Job Matching
# ---------------------------------------------------------------------------

with tab_match:

    uploaded_resume = st.file_uploader(
        "Upload your resume",
        type=[
            "pdf",
            "docx",
            "txt",
            "md",
        ],
        help="Supported formats: PDF, DOCX, TXT, and Markdown.",
    )

    run_clicked = st.button(
        "🚀 Run Analysis",
        type="primary",
        use_container_width=True,
    )


    # -----------------------------------------------------------------------
    # Run analysis
    # -----------------------------------------------------------------------

    if run_clicked:

        client = get_openai_client()

        if client is None:
            st.error(
                "The application is not properly configured. "
                "Please contact the administrator."
            )
            st.stop()


        if uploaded_resume is None:
            st.error(
                "Please upload your resume first."
            )
            st.stop()


        # -------------------------------------------------------------------
        # User preferences
        # -------------------------------------------------------------------

        preferences = {
            "desired_countries": [
                c.strip()
                for c in desired_countries.split(",")
                if c.strip()
            ],

            "desired_cities": [
                c.strip()
                for c in desired_cities.split(",")
                if c.strip()
            ],

            "work_arrangement_preference":
                work_arrangement_preference,

            "employment_type_preference":
                employment_type_preference,

            "willing_to_relocate":
                willing_to_relocate,

            "nationality":
                nationality,

            "salary_expectation_min":
                salary_min or None,

            "salary_expectation_max":
                salary_max or None,

            "salary_currency":
                salary_currency,

            "salary_period":
                "month",

            "availability":
                "Immediate",

            "notes":
                "",
        }


        # -------------------------------------------------------------------
        # Progress
        # -------------------------------------------------------------------

        progress = st.progress(
            0,
            text="Reading your resume...",
        )

        # Token/cost tracker for THIS run only (per-session, not shared
        # between users — see src/usage_tracker.py)
        usage_state = new_usage_state()


        # -------------------------------------------------------------------
        # Save uploaded resume temporarily
        # -------------------------------------------------------------------

        suffix = Path(
            uploaded_resume.name
        ).suffix

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as tmp:

            tmp.write(
                uploaded_resume.getvalue()
            )

            tmp_path = tmp.name


        try:
            raw_text = read_resume(
                tmp_path
            )
        finally:
            os.unlink(
                tmp_path
            )


        # -------------------------------------------------------------------
        # Resume quality check
        # -------------------------------------------------------------------

        quality = resume_quality_check(
            raw_text
        )

        if quality["likely_scanned_or_empty"]:

            st.warning(
                "The file appears to be scanned or nearly empty. "
                "Results may be inaccurate."
            )


        # -------------------------------------------------------------------
        # Candidate profile
        # -------------------------------------------------------------------

        progress.progress(
            10,
            text="Analyzing your profile...",
        )

        try:

            candidate_profile, candidate_vector = (
                build_candidate_profile(
                    client,
                    uploaded_resume.name,
                    raw_text,
                    preferences,
                    usage_state,
                )
            )

        except Exception as exc:

            st.error(
                f"Failed to analyze your resume: {exc}"
            )

            st.stop()


        for warning in profile_warnings(
            candidate_profile
        ):
            st.warning(
                warning
            )


        # -------------------------------------------------------------------
        # Job retrieval
        # -------------------------------------------------------------------

        progress.progress(
            30,
            text="Finding relevant jobs...",
        )

        (
            jobs_df,
            job_vectors,
            job_ids,
            idf_by_skill,
        ) = load_job_data()


        retrieved_df = retrieve_jobs_for_candidate(
            jobs_df,
            job_vectors,
            candidate_vector,
            config.RETRIEVAL_TOP_K,
        )


        # -------------------------------------------------------------------
        # Hard filtering
        # -------------------------------------------------------------------

        progress.progress(
            45,
            text="Applying your preferences...",
        )

        filtered_df, funnel = run_filter_funnel(
            retrieved_df,
            candidate_profile,
        )


        if len(filtered_df) == 0:

            progress.empty()

            st.error(
                "No jobs matched your current preferences. "
                "Try broadening your country or city preferences, "
                "or enable \"Willing to relocate anywhere\"."
            )

            with st.expander(
                "Filtering Details"
            ):

                st.dataframe(
                    pd.DataFrame(funnel)
                )

            st.stop()


        # -------------------------------------------------------------------
        # Match scoring
        # -------------------------------------------------------------------

        progress.progress(
            60,
            text="Calculating job match scores...",
        )

        matched_df = score_and_rank(
            filtered_df,
            candidate_profile,
            idf_by_skill,
        )


        # -------------------------------------------------------------------
        # Semantic reranking
        # -------------------------------------------------------------------

        if use_reranker:

            progress.progress(
                75,
                text=(
                    "Refining job matches "
                    "(this may take a minute the first time)..."
                ),
            )

            tokenizer, model, device = get_reranker()


            if model is None:

                st.warning(
                    "Semantic reranking could not be loaded. "
                    "Showing results using the standard matching score."
                )

                reranked_df = (
                    matched_df
                    .head(final_top_n)
                    .copy()
                )

                reranked_df.insert(
                    0,
                    "phase6_rank",
                    range(
                        1,
                        len(reranked_df) + 1,
                    ),
                )

                reranked_df["rerank_status"] = (
                    "unavailable"
                )

            else:

                reranked_df = rerank(
                    matched_df,
                    candidate_profile,
                    tokenizer,
                    model,
                    device,
                    final_top_n=final_top_n,
                )

        else:

            reranked_df = (
                matched_df
                .head(final_top_n)
                .copy()
            )

            reranked_df.insert(
                0,
                "phase6_rank",
                range(
                    1,
                    len(reranked_df) + 1,
                ),
            )

            reranked_df["rerank_status"] = (
                "skipped_by_user"
            )


        phase6_payload = to_phase7_payload(
            reranked_df,
            candidate_profile["candidate_id"],
        )


        # -------------------------------------------------------------------
        # Generate explanations
        # -------------------------------------------------------------------

        progress.progress(
            90,
            text="Generating job insights...",
        )


        try:

            explained_jobs = explain_jobs(
                client,
                phase6_payload["jobs"],
                usage_state=usage_state,
            )

        except Exception as exc:

            st.warning(
                f"Some explanations could not be generated ({exc}). "
                "Scores will still be displayed."
            )


            explained_jobs = [

                {
                    "job_id":
                        j["job_id"],

                    "title":
                        j["title"],

                    "company":
                        j["company"],

                    "url":
                        j.get("url"),

                    "location":
                        j["location"],

                    "match_score":
                        j["match_score"],

                    "scoring_confidence":
                        j["scoring_confidence"],

                    "why_match":
                        "(Explanation unavailable)",

                    "matched_skills":
                        j["matched_core_skills"],

                    "missing_skills":
                        j["missing_core_skills"],

                    "matched_qualifications":
                        j["matched_qualifications"],

                    "missing_qualifications":
                        j["missing_qualifications"],

                    "matched_education":
                        j["matched_education_fields"],

                    "missing_education":
                        j["missing_education_fields"],

                    "learning_areas":
                        [],

                }

                for j in phase6_payload["jobs"]
            ]


        progress.progress(
            100,
            text="Complete!",
        )

        progress.empty()


        # -------------------------------------------------------------------
        # Persist results for display + chat
        # -------------------------------------------------------------------

        st.session_state["candidate_profile"] = (
            candidate_profile
        )

        st.session_state["explained_jobs"] = (
            explained_jobs
        )

        st.session_state["browse_pool"] = (
            matched_df
        )

        st.session_state["funnel"] = (
            funnel
        )

        st.session_state["chat_history"] = []

        st.session_state["chat_display"] = []

        st.session_state["chat_turns"] = 0

        st.session_state["usage_state"] = usage_state


    # -----------------------------------------------------------------------
    # Display results
    # -----------------------------------------------------------------------

    if "explained_jobs" in st.session_state:

        jobs = st.session_state[
            "explained_jobs"
        ]

        profile = st.session_state[
            "candidate_profile"
        ]


        st.success(
            f"Found {len(jobs)} suitable jobs "
            f"for {profile.get('most_recent_title', 'your profile')}."
        )

        # ---------------------------------------------------------------
        # Token usage / cost — this is where the output lives.
        # Click the expander below to open it.
        # ---------------------------------------------------------------
        run_usage = st.session_state.get("usage_state")

        if run_usage:

            run_cost = estimate_cost_usd(
                run_usage,
                config.EMBEDDING_MODEL,
                config.EXTRACTION_MODEL,
            )

            with st.expander("📊 Token Usage", expanded=False):

                u1, u2, u3, u4 = st.columns(4)

                u1.metric(
                    "LLM Input Tokens",
                    f"{run_usage['llm_input_tokens']:,}",
                )

                u2.metric(
                    "LLM Output Tokens",
                    f"{run_usage['llm_output_tokens']:,}",
                )

                u3.metric(
                    "Embedding Tokens",
                    f"{run_usage['embedding_tokens']:,}",
                )

                u4.metric(
                    "Total Tokens",
                    f"{total_tokens(run_usage):,}",
                )

                st.caption(
                    f"Calls per phase: {run_usage['calls']} — "
                    f"Estimated cost: ${run_cost:.4f} "
                    f"(official {config.EXTRACTION_MODEL} short-context "
                    f"pricing: $0.20 / $1.20 per 1M input/output tokens, "
                    f"{config.EMBEDDING_MODEL}: $0.02 per 1M tokens)."
                )


        with st.expander(
            "📋 Filtering Summary"
        ):

            st.dataframe(
                pd.DataFrame(
                    st.session_state["funnel"]
                ),
                use_container_width=True,
            )


        for job in jobs:

            with st.container(
                border=True
            ):

                col1, col2 = st.columns(
                    [4, 1]
                )


                with col1:

                    st.subheader(
                        f"{job['title']} — {job['company']}"
                    )

                    loc = (
                        job.get("location")
                        or {}
                    )

                    st.caption(
                        f"{loc.get('city', '')}, "
                        f"{loc.get('country', '')}"
                    )


                with col2:

                    st.metric(
                        "Match Score",
                        f"{job['match_score']:.0f}/100",
                    )


                st.write(
                    job["why_match"]
                )

                if job.get("low_score_note"):

                    st.caption(
                        f"ℹ️ {job['low_score_note']}"
                    )


                c1, c2 = st.columns(
                    2
                )


                with c1:

                    st.markdown(
                        "**✅ Matching Skills**"
                    )

                    st.write(
                        ", ".join(
                            job["matched_skills"]
                        )
                        or "—"
                    )


                with c2:

                    st.markdown(
                        "**❌ Missing Skills**"
                    )

                    st.write(
                        ", ".join(
                            job["missing_skills"]
                        )
                        or "—"
                    )


                if job.get(
                    "learning_areas"
                ):

                    with st.expander(
                        "💡 Development Suggestions"
                    ):

                        for la in job[
                            "learning_areas"
                        ]:

                            st.write(
                                f"- "
                                f"{la['missing_skill']}"
                                f" → "
                                f"{la['learning_area']}"
                            )


                if job.get("url"):

                    st.link_button(
                        "View Original Job Posting",
                        job["url"],
                    )


# ---------------------------------------------------------------------------
# Chat with the Agent
# ---------------------------------------------------------------------------

with tab_chat:

    if "explained_jobs" not in st.session_state:

        st.info(
            'Run the analysis first from the '
            '"Job Matching" tab before starting the chat.'
        )

    else:

        jobs_by_id = {
            j["job_id"]: j
            for j in st.session_state[
                "explained_jobs"
            ]
        }


        browse_pool = st.session_state.get(
            "browse_pool"
        )


        tool_functions, tools_schema = make_tools(
            jobs_by_id,
            browse_pool,
        )


        system_prompt = (
            SYSTEM_PROMPT_TEMPLATE.format(
                candidate_id=st.session_state[
                    "candidate_profile"
                ]["candidate_id"]
            )
        )


        known_job_ids = set(
            jobs_by_id.keys()
        )


        # -------------------------------------------------------------------
        # Display previous messages
        # -------------------------------------------------------------------

        for turn in st.session_state.get(
            "chat_display",
            [],
        ):

            with st.chat_message(
                turn["role"]
            ):

                st.write(
                    turn["content"]
                )


        remaining = (
            MAX_CHAT_TURNS_PER_SESSION
            - st.session_state.get(
                "chat_turns",
                0,
            )
        )


        col_remaining, col_clear = st.columns(
            [4, 1]
        )

        with col_remaining:

            st.caption(
                f"Messages remaining in this session: {remaining}"
            )

        with col_clear:

            if st.button(
                "🗑️ Clear Chat History",
                use_container_width=True,
            ):

                st.session_state["chat_history"] = []

                st.session_state["chat_display"] = []

                st.session_state["chat_turns"] = 0

                st.rerun()


        user_msg = st.chat_input(
            "Ask me about your results, e.g., Which job matches me best?"
        )


        if user_msg:

            if remaining <= 0:

                st.warning(
                    "You have reached the maximum number "
                    "of messages for this session. "
                    "Run the analysis again to start a new session."
                )


            else:

                client = get_openai_client()


                if client is None:

                    st.error(
                        "The application is not properly configured. "
                        "Please contact the administrator."
                    )


                else:

                    st.session_state.setdefault(
                        "chat_display",
                        [],
                    ).append(
                        {
                            "role":
                                "user",

                            "content":
                                user_msg,
                        }
                    )


                    with st.chat_message(
                        "user"
                    ):

                        st.write(
                            user_msg
                        )


                    with st.chat_message(
                        "assistant"
                    ):

                        with st.spinner(
                            "Thinking..."
                        ):

                            (
                                reply,
                                history,
                                calls,
                            ) = ask_agent(
                                client,
                                user_msg,
                                st.session_state.get(
                                    "chat_history",
                                    [],
                                ),
                                tool_functions,
                                tools_schema,
                                system_prompt,
                                usage_state=st.session_state.get(
                                    "usage_state"
                                ),
                            )


                        st.write(
                            reply
                        )


                        unknown_ids = (
                            check_grounding(
                                reply,
                                known_job_ids,
                            )
                        )


                        if unknown_ids:

                            st.caption(
                                "⚠️ Verification warning: "
                                f"Unknown job IDs mentioned: "
                                f"{unknown_ids}"
                            )


                    st.session_state[
                        "chat_history"
                    ] = history


                    st.session_state.setdefault(
                        "chat_display",
                        [],
                    ).append(
                        {
                            "role":
                                "assistant",

                            "content":
                                reply,
                        }
                    )


                    st.session_state[
                        "chat_turns"
                    ] = (
                        st.session_state.get(
                            "chat_turns",
                            0,
                        ) + 1
                    )
