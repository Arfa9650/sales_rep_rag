"""
Tool registry and implementations for the scratch agent.
Tools are chosen by the model in the Decide step; the loop runs registry[tool_id].fn(**tool_input).
No mock data: extract_insights uses the LLM over the provided profile text.
search_web uses DuckDuckGo (free, no API key) to fetch external knowledge.
"""

from typing import Any, Callable, Dict, Optional

# Default number of search results to return (keeps context size manageable)
DEFAULT_SEARCH_MAX_RESULTS = 5

# Tool shape: id, description, parameters (schema), fn
ToolSpec = Dict[str, Any]

EXTRACT_INSIGHTS_PROMPT = (
    "From this company profile, extract key facts, pain points, opportunities, and differentiators. "
    "Return a concise bullet list. Do not invent information; only summarize what is in the profile.\n\n"
    "Profile:\n{profile_text}"
)


def search_web(query: str, max_results: Optional[int] = None) -> str:
    """
    Search the web via DuckDuckGo (free, no API key). Returns title, URL, and snippet for each result.
    Use this to look up company info, industry trends, or supporting evidence not in the profile.
    """
    max_results = max_results if max_results is not None else DEFAULT_SEARCH_MAX_RESULTS
    query = (query or "").strip()
    if not query:
        return "No search query provided. Pass a non-empty 'query' string."
    try:
        from ddgs import DDGS
    except ImportError:
        return "Search unavailable: install with 'pip install ddgs'."
    try:
        ddgs = DDGS()
        results = list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        return f"Search failed: {e}"
    if not results:
        return "No results found for that query."
    parts = []
    for i, r in enumerate(results, 1):
        title = r.get("title") or ""
        href = r.get("href") or ""
        body = r.get("body") or ""
        parts.append(f"[{i}] {title}\nURL: {href}\n{body}")
    return "\n\n".join(parts)


def _make_extract_insights_fn(profile_text_in_scope: str) -> Callable[..., str]:
    """Build extract_insights that closes over profile_text (used when profile is in scope for this run)."""

    def _extract_insights(profile_text: Optional[str] = None) -> str:
        from local_llm import complete
        text = (profile_text or "").strip() or profile_text_in_scope
        if not text:
            return "No profile text provided to extract insights from."
        prompt = EXTRACT_INSIGHTS_PROMPT.format(profile_text=text[:8000])
        return complete(prompt)

    return _extract_insights


def _extract_insights_no_profile(profile_text: Optional[str] = None) -> str:
    """Fallback when no profile in scope; profile should be in task context. Model can pass a snippet via tool_input."""
    from local_llm import complete
    if not profile_text or not str(profile_text).strip():
        return "No profile text in this call. The company profile is in the task context above; use it to reason and then stop with your answer."
    prompt = EXTRACT_INSIGHTS_PROMPT.format(profile_text=str(profile_text)[:8000])
    return complete(prompt)


def _safe_call(fn: Callable[..., Any], **kwargs: Any) -> Any:
    """Run tool fn with kwargs; let caller handle exceptions for visibility."""
    return fn(**kwargs)


def get_tool_registry(profile_text: Optional[str] = None) -> Dict[str, ToolSpec]:
    """
    Registry of tools the model can choose from.
    When profile_text is provided (sales-rep flow), extract_insights uses it. Otherwise the tool
    expects profile to be in the task context or passed via tool_input.
    """
    if profile_text is not None:
        fn = _make_extract_insights_fn(profile_text)
        # Model can call with empty args; we use closed-over profile
        params = "optional profile_text; the profile is already in scope for this run"
    else:
        fn = _extract_insights_no_profile
        params = "optional profile_text; if not provided, profile is in the task context"
    registry = {
        "extract_insights": {
            "id": "extract_insights",
            "description": "Extract key facts, pain points, and opportunities from the company profile text. Use this to structure the profile before proposing value hypothesis and messaging.",
            "parameters": {"profile_text": params},
            "fn": fn,
        },
        "search_web": {
            "id": "search_web",
            "description": "Search the web (DuckDuckGo) for current information. Use when you need company info, industry trends, or supporting evidence not in the profile. Pass 'query' (search string) and optionally 'max_results' (default 5).",
            "parameters": {"query": "search query string", "max_results": "optional, number of results (default 5)"},
            "fn": search_web,
        },
    }
    return registry


def run_tool(registry: Dict[str, ToolSpec], tool_id: str, tool_input: Dict[str, Any]) -> Any:
    """
    Execute the tool selected by the model. Called from the agent loop after Decide.
    Raises on unknown tool_id; caller catches and passes error into Observe/memory.
    """
    if tool_id not in registry:
        raise ValueError(f"Unknown tool: {tool_id}. Available: {list(registry.keys())}")
    spec = registry[tool_id]
    fn = spec["fn"]
    # Pass through tool_input; extract_insights may have profile_text or empty dict
    return _safe_call(fn, **tool_input)
