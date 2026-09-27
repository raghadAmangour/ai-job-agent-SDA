"""
Token usage tracking. Pass a `usage_state` dict (stored in st.session_state,
NOT a module-level global — a global would leak between different users'
sessions on a shared Streamlit Cloud process) through the pipeline calls,
and this module accumulates real token counts from each OpenAI response.
"""

# Reference prices in USD per 1,000,000 tokens. Verify against your account
# at https://platform.openai.com/docs/pricing before trusting these for a
# real cost report — prices change and custom/preview models may differ.
PRICING_PER_MILLION = {
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "text-embedding-3-large": {"input": 0.13, "output": 0.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    # Official OpenAI Pricing docs — Standard tier, SHORT context
    # (this app's prompts are all short: a resume, one job's evidence, or
    # one chat turn — never near the long-context threshold).
    "gpt-5.6-luna": {"input": 0.20, "cached_input": 0.02, "cache_writes": 0.25, "output": 1.20},
}


def new_usage_state() -> dict:
    return {"embedding_tokens": 0, "llm_input_tokens": 0, "llm_output_tokens": 0, "calls": {}}


def track_llm(usage_state: dict, response, phase: str) -> None:
    """Call right after any client.responses.create()/.parse() call."""
    if usage_state is None:
        return
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    usage_state["llm_input_tokens"] += getattr(usage, "input_tokens", 0) or 0
    usage_state["llm_output_tokens"] += getattr(usage, "output_tokens", 0) or 0
    usage_state["calls"][phase] = usage_state["calls"].get(phase, 0) + 1


def track_embedding(usage_state: dict, response, phase: str = "phase2_embedding") -> None:
    """Call right after client.embeddings.create()."""
    if usage_state is None:
        return
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    tokens = getattr(usage, "total_tokens", None) or getattr(usage, "prompt_tokens", 0) or 0
    usage_state["embedding_tokens"] += tokens
    usage_state["calls"][phase] = usage_state["calls"].get(phase, 0) + 1


def estimate_cost_usd(usage_state: dict, embedding_model: str, llm_model: str) -> float:
    if usage_state is None:
        return 0.0
    emb_price = PRICING_PER_MILLION.get(embedding_model, {"input": 0.02})["input"]
    llm_price = PRICING_PER_MILLION.get(llm_model, {"input": 0.0, "output": 0.0})
    embedding_cost = (usage_state["embedding_tokens"] / 1_000_000) * emb_price
    llm_cost = (
        usage_state["llm_input_tokens"] / 1_000_000 * llm_price.get("input", 0.0)
        + usage_state["llm_output_tokens"] / 1_000_000 * llm_price.get("output", 0.0)
    )
    return embedding_cost + llm_cost


def total_tokens(usage_state: dict) -> int:
    if usage_state is None:
        return 0
    return usage_state["embedding_tokens"] + usage_state["llm_input_tokens"] + usage_state["llm_output_tokens"]
