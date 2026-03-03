"""
LangGraph workflow for the Prospect Finder agent.

Graph topology:
    START → reason → decide → act → observe → update → reflect
                ↑                                          |
                └──────────── (loop) ───────────────── route
                                                          |
                                               compile_prospects → END

Goal: Given a seller description + location + target count, find and enrich N
      real prospect companies and return them as a structured list.

Prompt design notes (small local models):
  - reason_node  : gap analysis only; never write the final list
  - decide_node  : adaptive rules based on tools_used; oracle optional when stuck
  - reflect_node : fill-in-the-blank format (same as sales-rep graph)
  - compile_prospects_node : single clean LLM call; JSON output with fallback
"""

import json
import logging
import re
from typing import Any, Dict, List, Literal, Optional, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from config import (
    DECIDE_PARSE_RETRIES,
    MEMORY_RECENT_K,
    MIN_CONFIDENCE_TO_STOP,
    MODEL_ERROR_RETRIES,
)
from decisions import Decision, parse_decision, parse_reflection
from local_llm import complete, complete_structured
from tools import run_tool

logger = logging.getLogger("sales_rep.agent")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class ProspectFinderState(TypedDict):
    # Immutable inputs (set once at start)
    task: str                       # full task description (seller + goal + location + count)
    target_count: int               # how many prospects to find
    location: str                   # target location string
    my_company_summary: str         # one-liner: who the seller is
    industry_focus: str             # optional industry/sector hint
    max_steps: int
    tool_descriptions: str
    # Accumulated across steps
    step: int
    tools_used: List[str]
    memory_items: List[str]
    turn_history: List[Dict[str, Any]]
    found_companies: List[str]      # raw company names/LinkedIn URLs discovered so far
    # Per-step outputs (overwritten each step)
    reason_text: str
    decision: Dict[str, Any]
    tool_result: str
    tool_error: str
    observation: str
    reflection: Dict[str, Any]
    # Final output (set by compile_prospects_node)
    final_prospects: List[Dict[str, Any]]


def _cfg(config: RunnableConfig, key: str, default: Any = None) -> Any:
    return (config or {}).get("configurable", {}).get(key, default)


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _log_prompt(tag: str, prompt: str) -> None:
    logger.info("[%s] PROMPT  len=%d", tag, len(prompt))
    if len(prompt) <= 10_000:
        logger.info("[%s] PROMPT:\n%s", tag, prompt)
    else:
        logger.info("[%s] PROMPT (first 5000):\n%s", tag, prompt[:5_000])
        logger.info("[%s] PROMPT (last 5000):\n%s", tag, prompt[-5_000:])


def _log_response(tag: str, response: str) -> None:
    logger.info("[%s] RESPONSE  len=%d:\n%s", tag, len(response or ""), response or "")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _call_model(prompt: str, step_name: str) -> str:
    last_err: Optional[Exception] = None
    for attempt in range(MODEL_ERROR_RETRIES + 1):
        try:
            return complete(prompt) or ""
        except Exception as exc:
            last_err = exc
            logger.warning("Model error (%s) attempt %d: %s", step_name, attempt + 1, exc)
            if attempt == MODEL_ERROR_RETRIES:
                raise
    raise last_err or RuntimeError(f"{step_name} failed after retries")


def _decision_from_dict(d: Dict[str, Any]) -> Decision:
    return Decision(
        next_action=d.get("next_action", ""),
        tool_id=d.get("tool_id") or None,
        tool_input=d.get("tool_input") or {},
        confidence=float(d.get("confidence", 0)),
        should_stop=bool(d.get("should_stop", False)),
        should_revise=bool(d.get("should_revise", False)),
        reasoning=d.get("reasoning", ""),
    )


def _decision_to_dict(dec: Decision) -> Dict[str, Any]:
    return {
        "next_action": dec.next_action,
        "tool_id": dec.tool_id,
        "tool_input": dec.tool_input,
        "confidence": dec.confidence,
        "should_stop": dec.should_stop,
        "should_revise": dec.should_revise,
        "reasoning": dec.reasoning,
    }


def _extract_company_names_from_text(text: str) -> List[str]:
    """
    Heuristically pull company names from search result text.
    Looks for LinkedIn URL patterns and bold-ish title lines.
    """
    names: List[str] = []
    # LinkedIn URLs like linkedin.com/company/company-name
    for m in re.finditer(r"linkedin\.com/company/([\w\-]+)", text, re.IGNORECASE):
        slug = m.group(1).replace("-", " ").title()
        if slug and slug not in names:
            names.append(slug)
    # Result titles like "[1] Company Name\nURL:"
    for m in re.finditer(r"\[\d+\]\s+(.+?)\n", text):
        candidate = m.group(1).strip()
        if candidate and len(candidate) < 80 and candidate not in names:
            names.append(candidate)
    return names[:20]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def reason_node(state: ProspectFinderState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Step 1 – Reason: gap analysis.
    How many companies found vs target? What searches still need doing?
    """
    new_step = state["step"] + 1
    logger.info("========== PROSPECT STEP %d / %d ==========", new_step, state["max_steps"])
    logger.info("=== REASON NODE (prospect finder) ===")

    found = state.get("found_companies") or []
    target = state["target_count"]
    memory_str = "\n".join(state.get("memory_items") or []) or "(no prior findings yet)"
    history_str = (
        "\n".join(
            f"Turn {i + 1}: {t.get('action', '')} -> {t.get('observation', '')[:200]}"
            for i, t in enumerate((state.get("turn_history") or [])[-MEMORY_RECENT_K:])
        )
        or "(none)"
    )
    tools_used_str = ", ".join(state.get("tools_used") or []) or "none yet"

    prompt = (
        f"Seller: {state.get('my_company_summary', 'N/A')}\n"
        f"Target: Find {target} prospect companies in {state.get('location', 'N/A')}"
        + (f" | Industry focus: {state['industry_focus']}" if state.get("industry_focus") else "")
        + f"\n\nCompanies found so far ({len(found)}/{target}): {', '.join(found[:20]) or 'none'}\n\n"
        f"Prior search findings:\n{memory_str}\n\n"
        f"Recent turns:\n{history_str}\n\n"
        f"Tools used so far: [{tools_used_str}]\n\n"
        "Your task: Analyse the situation in ONE short paragraph.\n"
        "1. How many unique companies have been found vs the target?\n"
        "2. What is still missing or needs enriching?\n"
        "3. What is the single best next step (LinkedIn search, general web search, enrich a company, oracle, or compile)?\n\n"
        "If you are unsure what to search, you may optionally call consult_oracle for search query suggestions.\n"
        "IMPORTANT: Do NOT write the final prospect list yet."
    )
    _log_prompt("Reason", prompt)
    try:
        out = _call_model(prompt, "reason")
        _log_response("Reason", out)
        logger.info("Reasoning (summary): %s", (out or "")[:400])
    except Exception as exc:
        logger.error("Reason node failed: %s", exc)
        out = f"(Reasoning failed: {exc})"

    return {"step": new_step, "reason_text": out}


def decide_node(state: ProspectFinderState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Step 2 – Decide: choose next tool or stop.
    Adaptive rules based on tools_used and found_companies count.
    """
    logger.info("=== DECIDE NODE (prospect finder) ===")

    found = state.get("found_companies") or []
    target = state["target_count"]
    tools_used: List[str] = state.get("tools_used") or []
    tools_used_str = ", ".join(tools_used) or "none yet"
    linkedin_count = sum(1 for t in tools_used if t == "search_linkedin_companies")
    general_count = sum(1 for t in tools_used if t == "search_web_general")
    fetch_count = sum(1 for t in tools_used if t == "fetch_company_info")
    oracle_used = "consult_oracle" in tools_used
    memory_str = (
        "\n".join((state.get("memory_items") or [])[-MEMORY_RECENT_K:])
        or "(no prior findings)"
    )

    rules: List[str] = [
        f"Tools you have already called this run: [{tools_used_str}]",
        f"Companies discovered so far ({len(found)}/{target}): {', '.join(found[:15]) or 'none'}",
        "RULES:",
    ]

    if oracle_used and len(found) >= target:
        rules.append(
            "- You have used the oracle AND found enough companies. Set should_stop=true now."
        )
    elif len(found) >= target and fetch_count >= max(1, len(found) // 2):
        rules.append(
            f"- You have found {len(found)}/{target} companies and done enrichment. "
            "Set should_stop=true to compile the final list."
        )
    elif len(found) >= target:
        rules.append(
            f"- You have found {len(found)}/{target} companies. "
            "Call fetch_company_info on companies that still need enrichment, "
            "then set should_stop=true."
        )
    elif len(found) > 0:
        rules.append(
            f"- You have {len(found)}/{target} companies. "
            "Do more searches to find the remaining {target - len(found)}."
        )
        if fetch_count < len(found):
            rules.append(
                "- Enrich companies you have found with fetch_company_info."
            )
    elif oracle_used:
        rules.append(
            "- You used the oracle for search ideas. Now use those suggestions with search_linkedin_companies "
            "or search_web_general."
        )
    elif linkedin_count == 0 and general_count == 0:
        rules.append(
            "- You have not searched yet. Call search_linkedin_companies first. "
            "If unsure what to search, you MAY optionally call consult_oracle for query ideas."
        )
    elif linkedin_count >= 1 and len(found) == 0:
        rules.append(
            "- LinkedIn search returned no useful companies. "
            "Try search_web_general with a different query (e.g. 'top companies [location] [industry]'), "
            "or try consult_oracle for alternative search approaches (optional)."
        )
    else:
        rules.append(
            f"- You have done {linkedin_count} LinkedIn search(es) and {general_count} general search(es). "
            "Continue searching to reach the target, or enrich found companies with fetch_company_info."
        )

    if oracle_used:
        rules.append("- Do NOT call consult_oracle again; it was already used.")

    rules_str = "\n".join(rules)

    base_prompt = (
        f"Seller: {state.get('my_company_summary', 'N/A')}\n"
        f"Target: Find {target} companies in {state.get('location', 'N/A')}"
        + (f" | Industry: {state['industry_focus']}" if state.get("industry_focus") else "")
        + f"\n\nYour analysis:\n{(state.get('reason_text') or '')[:700]}\n\n"
        f"Prior findings:\n{memory_str}\n\n"
        f"{rules_str}\n\n"
        f"Available tools:\n{state['tool_descriptions']}\n\n"
        "Respond with a JSON object — keys MUST be exactly:\n"
        "  next_action (string), tool_id (string or null), tool_input (object),\n"
        "  confidence (0.0-1.0), should_stop (bool), should_revise (bool), reasoning (string)\n\n"
        "JSON only — no other text."
    )

    prompt = base_prompt
    last_error: Optional[Exception] = None

    for attempt in range(DECIDE_PARSE_RETRIES + 1):
        _log_prompt("Decide", prompt)
        try:
            raw = _call_model(prompt, "decide")
            _log_response("Decide", raw)
            structured = complete_structured(prompt)
            decision = parse_decision(raw, structured_fallback=structured)
            logger.info(
                "Decision: next_action=%s  tool_id=%s  should_stop=%s  confidence=%.2f  reasoning=%s",
                decision.next_action,
                decision.tool_id or "(none)",
                decision.should_stop,
                decision.confidence,
                (decision.reasoning or "")[:300],
            )
            return {"decision": _decision_to_dict(decision)}
        except Exception as exc:
            last_error = exc
            logger.warning("parse_decision attempt %d failed: %s", attempt + 1, exc)
            if attempt < DECIDE_PARSE_RETRIES:
                prompt = (
                    "Your previous response was invalid. Respond ONLY with a valid JSON object "
                    "containing exactly: next_action, tool_id, tool_input (object), confidence (0-1), "
                    "should_stop (bool), should_revise (bool), reasoning (string)."
                )

    logger.error("Decide node exhausted retries: %s", last_error)
    return {
        "decision": _decision_to_dict(
            Decision(next_action="continue", reasoning=f"Decide failed: {last_error}")
        )
    }


def act_node(state: ProspectFinderState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Step 3 – Act: run the chosen tool.
    After search tools, extract company names from results and append to found_companies.
    """
    tool_registry = _cfg(config, "tool_registry") or {}
    decision = _decision_from_dict(state.get("decision") or {})
    logger.info("=== ACT NODE (prospect finder) ===")

    if not decision.tool_id:
        logger.info("  No tool selected; skipping Act.")
        return {"tool_result": "", "tool_error": ""}

    logger.info("=== TOOL CALL === tool_id=%s  tool_input=%s", decision.tool_id, decision.tool_input)

    try:
        result = run_tool(tool_registry, decision.tool_id, decision.tool_input)
        result_str = str(result)
        logger.info("=== TOOL RESULT (%s) === (first 1500 chars)\n%s", decision.tool_id, result_str[:1_500])
    except Exception as exc:
        logger.warning("Tool error (%s): %s", decision.tool_id, exc)
        return {"tool_result": "", "tool_error": str(exc)}

    # Extract newly discovered company names from search results
    new_companies: List[str] = []
    if decision.tool_id in ("search_linkedin_companies", "search_web_general"):
        new_companies = _extract_company_names_from_text(result_str)
        existing = state.get("found_companies") or []
        merged = list(existing)
        for c in new_companies:
            if c not in merged:
                merged.append(c)
        logger.info("  Companies extracted: %s", new_companies[:10])
        logger.info("  Total found companies: %d", len(merged))
        return {"tool_result": result_str, "tool_error": "", "found_companies": merged}

    return {"tool_result": result_str, "tool_error": ""}


def observe_node(state: ProspectFinderState, config: RunnableConfig) -> Dict[str, Any]:
    """Step 4 – Observe: convert tool result/error into a readable observation."""
    logger.info("=== OBSERVE NODE (prospect finder) ===")
    decision = _decision_from_dict(state.get("decision") or {})
    tool_id = decision.tool_id or ""
    tool_result = state.get("tool_result") or ""
    tool_error = state.get("tool_error") or ""

    if tool_error:
        obs = f"Tool {tool_id} failed: {tool_error}"
    elif tool_id:
        obs = f"Tool {tool_id} result: {tool_result[:2_500]}"
    else:
        obs = "No tool was used this step."

    logger.info("  Observation: %s", obs[:400])
    return {"observation": obs}


def update_node(state: ProspectFinderState, config: RunnableConfig) -> Dict[str, Any]:
    """Step 5 – Update: append step to memory/history and record tool used."""
    logger.info("=== UPDATE NODE (prospect finder) ===")
    decision = _decision_from_dict(state.get("decision") or {})
    step = state["step"]
    observation = state.get("observation") or ""
    tool_id = decision.tool_id or "none"
    found = state.get("found_companies") or []

    new_memory_item = (
        f"Step {step}: tool={tool_id}  found={len(found)} companies  "
        f"observation: {observation[:300]}"
    )
    new_turn = {
        "action": f"tool={tool_id}" if tool_id != "none" else decision.next_action,
        "observation": observation,
    }
    logger.info("  Memory item: %s", new_memory_item[:200])

    new_tools_used = list(state.get("tools_used") or [])
    if tool_id != "none":
        new_tools_used.append(tool_id)
        logger.info("  Tools used (updated): %s", new_tools_used)

    return {
        "memory_items": list(state.get("memory_items") or []) + [new_memory_item],
        "turn_history": list(state.get("turn_history") or []) + [new_turn],
        "tools_used": new_tools_used,
    }


def reflect_node(state: ProspectFinderState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Step 6 – Reflect: structured self-critique.
    Fill-in-the-blank format so small models can follow it reliably.
    """
    logger.info("=== REFLECT NODE (prospect finder) ===")
    observation = state.get("observation") or ""
    memory_str = (
        "\n".join((state.get("memory_items") or [])[-MEMORY_RECENT_K:])
        or "(none)"
    )
    decision = _decision_from_dict(state.get("decision") or {})
    found = state.get("found_companies") or []
    target = state["target_count"]
    tools_used_str = ", ".join(state.get("tools_used") or []) or "none"

    prompt = (
        f"Goal: Find {target} prospects in {state.get('location', 'N/A')}\n"
        f"Companies found so far ({len(found)}/{target}): {', '.join(found[:15]) or 'none'}\n"
        f"Tools used: {tools_used_str}\n\n"
        f"Latest finding:\n{observation[:1_000]}\n\n"
        f"All findings:\n{memory_str}\n\n"
        "Complete each line (be brief):\n"
        "Companies confirmed found: [comma-separated list or 'none']\n"
        "What is still missing: [one sentence, or write 'nothing — target reached']\n"
        f"Confidence I can stop and compile the final list (0.0-1.0): [number only]\n"
        "Should I keep searching (yes/no): [yes or no]\n"
        "Reason: [one sentence]"
    )
    _log_prompt("Reflect", prompt)
    try:
        raw = _call_model(prompt, "reflect")
        _log_response("Reflect", raw)
        parsed = parse_reflection(raw)
    except Exception as exc:
        logger.warning("Reflect node failed: %s", exc)
        parsed = {"confidence": decision.confidence, "should_revise": False, "critique": str(exc)}

    logger.info(
        "  Reflection: confidence=%.2f  should_revise=%s  critique=%s",
        parsed.get("confidence", 0),
        parsed.get("should_revise"),
        (parsed.get("critique") or "")[:300],
    )
    return {"reflection": parsed}


def compile_prospects_node(state: ProspectFinderState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Final node: one dedicated LLM call to format all gathered information
    into a clean JSON list of enriched prospect objects.
    Falls back to simple name extraction if JSON parsing fails.
    """
    logger.info("=== COMPILE PROSPECTS NODE ===")

    memory_str = "\n".join(state.get("memory_items") or []) or "(no prior findings)"
    found = state.get("found_companies") or []
    target = state["target_count"]

    prompt = (
        "You are compiling a final list of prospect companies.\n\n"
        f"Seller: {state.get('my_company_summary', 'N/A')}\n"
        f"Target: {target} companies in {state.get('location', 'N/A')}"
        + (f" | Industry: {state['industry_focus']}" if state.get("industry_focus") else "")
        + f"\n\nRaw companies discovered: {', '.join(found) or 'none'}\n\n"
        f"Research gathered:\n{memory_str}\n\n"
        "Instructions:\n"
        "- From the research above, compile the best matching companies.\n"
        f"- Return UP TO {target} companies.\n"
        "- For each company, fill in as many fields as the research supports.\n"
        "- Mark unknown fields as null.\n"
        "- Output ONLY a valid JSON array, nothing else:\n\n"
        '[\n'
        '  {\n'
        '    "company_name": "...",\n'
        '    "linkedin_url": "https://linkedin.com/company/... or null",\n'
        '    "website": "https://... or null",\n'
        '    "industry": "...",\n'
        '    "size_estimate": "e.g. 50-200 employees or null",\n'
        '    "location": "...",\n'
        '    "key_facts": ["fact 1", "fact 2"]\n'
        '  }\n'
        ']'
    )
    _log_prompt("CompileProspects", prompt)
    final_prospects: List[Dict[str, Any]] = []
    try:
        raw = _call_model(prompt, "compile_prospects")
        _log_response("CompileProspects", raw)

        # Try to extract JSON array
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(0))
                if isinstance(parsed, list):
                    final_prospects = parsed
                    logger.info("=== PROSPECTS COMPILED === count=%d", len(final_prospects))
            except json.JSONDecodeError as exc:
                logger.warning("JSON parse failed: %s", exc)

        # Fallback: build minimal records from found_companies list
        if not final_prospects and found:
            logger.info("Falling back to minimal prospect records from found_companies list")
            final_prospects = [
                {
                    "company_name": name,
                    "linkedin_url": None,
                    "website": None,
                    "industry": state.get("industry_focus") or None,
                    "size_estimate": None,
                    "location": state.get("location"),
                    "key_facts": [],
                }
                for name in found[:target]
            ]

    except Exception as exc:
        logger.error("compile_prospects_node failed: %s", exc)
        final_prospects = [
            {
                "company_name": name,
                "linkedin_url": None,
                "website": None,
                "industry": None,
                "size_estimate": None,
                "location": state.get("location"),
                "key_facts": [],
            }
            for name in found[:target]
        ]

    logger.info("=== FINAL PROSPECTS ===")
    for i, p in enumerate(final_prospects, 1):
        logger.info("  [%d] %s | %s | %s", i, p.get("company_name"), p.get("industry"), p.get("website"))

    return {"final_prospects": final_prospects}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_after_reflect(state: ProspectFinderState) -> Literal["reason", "compile_prospects"]:
    """
    After Reflect, decide whether to keep searching or compile the final list.
    """
    decision = _decision_from_dict(state.get("decision") or {})
    reflection = state.get("reflection") or {}
    step = state["step"]
    max_steps = state["max_steps"]
    found = state.get("found_companies") or []
    target = state["target_count"]

    should_revise = decision.should_revise or reflection.get("should_revise", False)
    confidence_ok = decision.confidence >= MIN_CONFIDENCE_TO_STOP

    if step >= max_steps:
        logger.info("Route → compile_prospects  (max_steps=%d reached)", max_steps)
        return "compile_prospects"

    if len(found) >= target and not should_revise:
        logger.info("Route → compile_prospects  (found %d >= target %d)", len(found), target)
        return "compile_prospects"

    if decision.should_stop and confidence_ok:
        logger.info("Route → compile_prospects  (should_stop + confidence=%.2f)", decision.confidence)
        return "compile_prospects"

    if decision.should_stop and not confidence_ok:
        logger.info(
            "Route → reason  (rejecting stop: confidence %.2f < %.2f)",
            decision.confidence,
            MIN_CONFIDENCE_TO_STOP,
        )
        return "reason"

    logger.info("Route → reason  (default: keep searching, found %d/%d)", len(found), target)
    return "reason"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(checkpointing: bool = False):
    builder = StateGraph(ProspectFinderState)

    builder.add_node("reason", reason_node)
    builder.add_node("decide", decide_node)
    builder.add_node("act", act_node)
    builder.add_node("observe", observe_node)
    builder.add_node("update", update_node)
    builder.add_node("reflect", reflect_node)
    builder.add_node("compile_prospects", compile_prospects_node)

    builder.add_edge(START, "reason")
    builder.add_edge("reason", "decide")
    builder.add_edge("decide", "act")
    builder.add_edge("act", "observe")
    builder.add_edge("observe", "update")
    builder.add_edge("update", "reflect")
    builder.add_conditional_edges(
        "reflect",
        route_after_reflect,
        {"reason": "reason", "compile_prospects": "compile_prospects"},
    )
    builder.add_edge("compile_prospects", END)

    if checkpointing:
        try:
            from langgraph.checkpoint.memory import MemorySaver
            logger.info("ProspectFinder graph: compiling with MemorySaver")
            return builder.compile(checkpointer=MemorySaver())
        except ImportError:
            logger.warning("MemorySaver not available; compiling without checkpointing")

    return builder.compile(debug=False)


def make_initial_state(
    task: str,
    target_count: int,
    location: str,
    my_company_summary: str,
    industry_focus: str,
    max_steps: int,
    tool_descriptions: str,
) -> ProspectFinderState:
    return ProspectFinderState(
        task=task,
        target_count=target_count,
        location=location,
        my_company_summary=my_company_summary,
        industry_focus=industry_focus,
        max_steps=max_steps,
        tool_descriptions=tool_descriptions,
        step=0,
        tools_used=[],
        memory_items=[],
        turn_history=[],
        found_companies=[],
        reason_text="",
        decision={},
        tool_result="",
        tool_error="",
        observation="",
        reflection={},
        final_prospects=[],
    )
