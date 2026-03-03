"""
Tool registry and implementations for the sales-rep agent.
Tools (sales-rep flow):
  search_web                — DuckDuckGo (no API key)
  extract_insights          — LLM over the prospect profile text
  consult_oracle            — GPT-4o-mini for synthesis / when stuck

Tools (prospect finder flow):
  search_linkedin_companies — DuckDuckGo site:linkedin.com/company query
  search_web_general        — plain DuckDuckGo fallback when LinkedIn is blocked
  fetch_company_info        — enrich a discovered company name with website/facts
"""

import logging
from typing import Any, Callable, Dict, Optional

DEFAULT_SEARCH_MAX_RESULTS = 5
ToolSpec = Dict[str, Any]
logger = logging.getLogger("sales_rep.agent")

EXTRACT_INSIGHTS_PROMPT = (
    "From this company profile, extract key facts, pain points, opportunities, and differentiators. "
    "Return a concise bullet list. Do not invent information; only summarise what is in the profile.\n\n"
    "Profile:\n{profile_text}"
)


# ---------------------------------------------------------------------------
# search_web
# ---------------------------------------------------------------------------

def search_web(query: str, max_results: Optional[int] = None) -> str:
    """
    Search the web via DuckDuckGo (free, no API key).
    Returns title, URL, and snippet for each result.
    """
    max_results = max_results if max_results is not None else DEFAULT_SEARCH_MAX_RESULTS
    query = (query or "").strip()
    if not query:
        return "No search query provided."
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
        return "No results found."
    parts = []
    for i, r in enumerate(results, 1):
        title = r.get("title") or ""
        href = r.get("href") or ""
        body = r.get("body") or ""
        parts.append(f"[{i}] {title}\nURL: {href}\n{body}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# extract_insights
# ---------------------------------------------------------------------------

def _make_extract_insights_fn(profile_text_in_scope: str) -> Callable[..., str]:
    def _extract_insights(profile_text: Optional[str] = None) -> str:
        from local_llm import complete
        text = (profile_text or "").strip() or profile_text_in_scope
        if not text:
            return "No profile text provided."
        prompt = EXTRACT_INSIGHTS_PROMPT.format(profile_text=text[:8_000])
        return complete(prompt)
    return _extract_insights


def _extract_insights_no_profile(profile_text: Optional[str] = None) -> str:
    from local_llm import complete
    if not profile_text or not str(profile_text).strip():
        return "No profile text provided. The profile is in the task context; reason from it and stop."
    prompt = EXTRACT_INSIGHTS_PROMPT.format(profile_text=str(profile_text)[:8_000])
    return complete(prompt)


# ---------------------------------------------------------------------------
# consult_oracle  (GPT-4o-mini)
# ---------------------------------------------------------------------------

def consult_oracle(question: str, context: str) -> str:
    """
    Call GPT-4o-mini for synthesis when the local model has gathered enough facts
    but needs higher-quality reasoning to produce a sharp VALUE HYPOTHESIS or MESSAGING ANGLE.

    Args:
        question: the specific synthesis question, e.g.
                  "What is the strongest value hypothesis for selling K2X to Zones Pakistan?"
        context:  all gathered facts from memory + search results (trimmed to ~4000 chars)
    """
    from config import OPENAI_API_KEY, OPENAI_MODEL
    api_key = OPENAI_API_KEY
    if not api_key:
        return (
            "Oracle unavailable: OPENAI_API_KEY is not set. "
            "Add it to your .env file or environment and try again."
        )
    question = (question or "").strip()
    context = (context or "").strip()[:6_000]
    if not question:
        return "Oracle call skipped: no question provided."

    logger.info("=== ORACLE CALL === model=%s  question=%s", OPENAI_MODEL, question[:200])
    try:
        import openai
    except ImportError:
        return "Oracle unavailable: install with 'pip install openai'."
    try:
        client = openai.OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise B2B sales intelligence analyst. "
                        "Given research notes about a prospect, produce sharp, specific, "
                        "evidence-grounded output. Be concise and direct. "
                        "Label anything not directly evidenced as [ASSUMPTION]."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Research notes:\n{context}\n\n"
                        f"Question: {question}"
                    ),
                },
            ],
            max_tokens=700,
            temperature=0.3,
        )
        result = resp.choices[0].message.content or "(Oracle returned empty response.)"
        logger.info("=== ORACLE RESPONSE ===\n%s", result[:800])
        return result
    except Exception as exc:
        logger.warning("Oracle call failed: %s", exc)
        return f"Oracle call failed: {exc}"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def _safe_call(fn: Callable[..., Any], **kwargs: Any) -> Any:
    return fn(**kwargs)


def get_tool_registry(profile_text: Optional[str] = None) -> Dict[str, ToolSpec]:
    """
    Return the full tool registry.
    When profile_text is provided, extract_insights closes over it so the model
    can call it with empty args.
    """
    if profile_text is not None:
        extract_fn = _make_extract_insights_fn(profile_text)
        extract_params: Any = "optional profile_text; the profile is already in scope for this run"
    else:
        extract_fn = _extract_insights_no_profile
        extract_params = "profile_text (string) — pass the prospect profile text to extract"

    return {
        "search_web": {
            "id": "search_web",
            "description": (
                "Search the web (DuckDuckGo) for current information about the prospect or industry. "
                "Use when you need company details, industry trends, or supporting evidence. "
                "Pass 'query' (a specific search string) and optionally 'max_results' (default 5)."
            ),
            "parameters": {
                "query": "specific search string, e.g. 'Zones Pakistan IT managed services clients'",
                "max_results": "optional int, default 5",
            },
            "fn": search_web,
        },
        "extract_insights": {
            "id": "extract_insights",
            "description": (
                "Extract key facts, pain points, and opportunities from the prospect profile text. "
                "Use ONCE to structure the raw profile before further research. "
                "Do not call this if you have already extracted insights."
            ),
            "parameters": {"profile_text": extract_params},
            "fn": extract_fn,
        },
        "consult_oracle": {
            "id": "consult_oracle",
            "description": (
                "Call a higher-intelligence AI model (GPT-4o-mini) for synthesis and analysis. "
                "OPTIONAL — Two use cases: "
                "(1) After gathering facts: use when you need help formulating a sharp VALUE HYPOTHESIS "
                "or MESSAGING ANGLE; pass 'question' (what to synthesise) and 'context' (gathered facts, ~2000-4000 chars). "
                "If you are already confident, you may skip the oracle and stop. "
                "(2) When stuck: If you don't know what to search for or how to find answers, you MAY call "
                "consult_oracle with question e.g. 'What search queries would help find information about [prospect/industry]?' "
                "and context=what you know so far. Use at most once when stuck; then use the suggested queries with search_web."
            ),
            "parameters": {
                "question": "the specific synthesis question for the oracle",
                "context": "all gathered facts from memory and search results (~2000-4000 chars)",
            },
            "fn": consult_oracle,
        },
    }


def run_tool(registry: Dict[str, ToolSpec], tool_id: str, tool_input: Dict[str, Any]) -> Any:
    """Execute the selected tool. Raises ValueError on unknown tool_id."""
    if tool_id not in registry:
        raise ValueError(f"Unknown tool: '{tool_id}'. Available: {list(registry.keys())}")
    return _safe_call(registry[tool_id]["fn"], **tool_input)


# ---------------------------------------------------------------------------
# Prospect finder tools
# ---------------------------------------------------------------------------

def search_linkedin_companies(
    location: str,
    industry_hint: str = "",
    seller_description: str = "",
    max_results: int = 8,
) -> str:
    """
    Search DuckDuckGo for LinkedIn company pages matching a location + optional industry hint.
    Constructs a query like: site:linkedin.com/company "location" "industry_hint"
    Returns title, LinkedIn URL, and snippet for each result.
    Falls back to plain web search if LinkedIn results are blocked or empty.
    """
    location = (location or "").strip()
    industry_hint = (industry_hint or "").strip()
    if not location:
        return "No location provided for LinkedIn company search."

    # Build a targeted LinkedIn query
    parts = [f'site:linkedin.com/company "{location}"']
    if industry_hint:
        parts.append(f'"{industry_hint}"')
    linkedin_query = " ".join(parts)

    logger.info("=== LINKEDIN SEARCH === query=%s", linkedin_query)
    result = search_web(linkedin_query, max_results=max_results)

    # If LinkedIn is blocked or returns nothing useful, fall back to plain search
    linkedin_blocked = (
        "No results found" in result
        or "Search failed" in result
        or result.strip() == ""
        or "linkedin.com" not in result.lower()
    )
    if linkedin_blocked:
        fallback_parts = [f'companies "{location}"']
        if industry_hint:
            fallback_parts.append(f'"{industry_hint}"')
        fallback_parts.append("company list")
        fallback_query = " ".join(fallback_parts)
        logger.info("=== LINKEDIN FALLBACK (plain web) === query=%s", fallback_query)
        result = search_web(fallback_query, max_results=max_results)
        return f"[LinkedIn blocked/empty — using web fallback]\n\n{result}"

    return result


def search_web_general(query: str, max_results: Optional[int] = None) -> str:
    """
    General-purpose DuckDuckGo web search for the prospect finder.
    Use when LinkedIn is inaccessible or when you need broader company information.
    Same as search_web but with a distinct tool_id so the agent can track usage.
    """
    return search_web(query, max_results=max_results)


def fetch_company_info(company_name: str, location: str = "") -> str:
    """
    Enrich a discovered company with website, industry, size, and key facts.
    Runs a DuckDuckGo search: "{company_name}" {location} company website industry size employees
    Returns web snippets useful for building the prospect card.
    """
    company_name = (company_name or "").strip()
    if not company_name:
        return "No company name provided."
    loc_str = f" {location.strip()}" if location else ""
    query = f'"{company_name}"{loc_str} company website industry size employees'
    logger.info("=== FETCH COMPANY INFO === company=%s  location=%s", company_name, location)
    return search_web(query, max_results=5)


def get_prospect_finder_registry(my_company_description: str = "") -> Dict[str, ToolSpec]:
    """
    Tool registry for the Prospect Finder agent.
    Includes LinkedIn search, general web fallback, company enrichment, and oracle.
    """
    return {
        "search_linkedin_companies": {
            "id": "search_linkedin_companies",
            "description": (
                "Search DuckDuckGo for LinkedIn company pages in a specific location and industry. "
                "This is the PRIMARY tool for discovering new prospects. "
                "Pass 'location' (required, e.g. 'Islamabad Pakistan'), "
                "'industry_hint' (optional, e.g. 'IT services'), "
                "'seller_description' (optional, brief seller summary), "
                "'max_results' (optional, default 8). "
                "Automatically falls back to general web search if LinkedIn is blocked."
            ),
            "parameters": {
                "location": "city, region, or country to search in (required)",
                "industry_hint": "optional industry or sector, e.g. 'manufacturing', 'fintech'",
                "seller_description": "optional brief seller summary to guide relevance",
                "max_results": "optional int, default 8",
            },
            "fn": search_linkedin_companies,
        },
        "search_web_general": {
            "id": "search_web_general",
            "description": (
                "General-purpose DuckDuckGo web search for prospect discovery or research. "
                "Use as a fallback when LinkedIn search fails, or to search for industry lists, "
                "directories, or company news. Pass 'query' (specific search string)."
            ),
            "parameters": {
                "query": "specific search string, e.g. 'top IT companies in Karachi Pakistan'",
                "max_results": "optional int, default 5",
            },
            "fn": search_web_general,
        },
        "fetch_company_info": {
            "id": "fetch_company_info",
            "description": (
                "Enrich a discovered company with website, industry, size, and key facts. "
                "Use AFTER you have a company name from search_linkedin_companies. "
                "Pass 'company_name' and optionally 'location' for more precise results."
            ),
            "parameters": {
                "company_name": "company name to enrich, e.g. 'Zones IT Solutions'",
                "location": "optional location for disambiguation, e.g. 'Pakistan'",
            },
            "fn": fetch_company_info,
        },
        "consult_oracle": {
            "id": "consult_oracle",
            "description": (
                "Call GPT-4o-mini for guidance. OPTIONAL — two use cases: "
                "(1) When stuck or confused about what to search or how to find prospects: "
                "ask 'What search queries would help find [industry] companies in [location]?' "
                "with context = current task summary. Use at most once when stuck. "
                "(2) When you have all companies found and want to synthesise the final list: "
                "ask to format and validate the list. "
                "Pass 'question' and 'context' (~2000-4000 chars of gathered facts)."
            ),
            "parameters": {
                "question": "specific question for the oracle",
                "context": "gathered facts, current findings, or task description (~2000-4000 chars)",
            },
            "fn": consult_oracle,
        },
    }
