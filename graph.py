"""
LangGraph workflow for the RAG sales-rep agent.

Graph topology:
    START → reason → decide → act → observe → update → reflect
                ↑                                          |
                └──────────── (loop) ───────────────── route
                                                          |
                                                   generate_answer → END

Prompt design principles for small local models (qwen/deepseek 8b):
  - reason_node  : gap-analysis only; "Do NOT write a VALUE HYPOTHESIS yet"
  - decide_node  : no RAG re-retrieval; tracks tools_used; consult_oracle trigger
  - reflect_node : structured fill-in-the-blank (no free-form JSON expected)
  - generate_answer : clean, tool-instruction-free prompt; [ASSUMPTION] labelling
  RAG retrieval only in reason_node and generate_answer_node (saves context budget).
"""

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
import rag

logger = logging.getLogger("sales_rep.agent")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    # Immutable inputs (set once at start)
    task: str               # full context document (seller + prospect + profile)
    retrieval_query: str
    max_steps: int
    tool_descriptions: str
    # Short run-level metadata used by node prompts
    my_company_summary: str   # one-liner: who the seller is
    company_name: str         # prospect company
    industry: str             # prospect industry
    # Accumulated across steps
    step: int
    tools_used: List[str]               # tool_ids called so far (for dedup rules)
    memory_items: List[str]
    turn_history: List[Dict[str, Any]]
    # Per-step outputs (overwritten each step)
    reason_text: str
    decision: Dict[str, Any]
    tool_result: str
    tool_error: str
    observation: str
    reflection: Dict[str, Any]
    # Final answer (set by generate_answer node)
    final_response: str


def _cfg(config: RunnableConfig, key: str, default: Any = None) -> Any:
    return (config or {}).get("configurable", {}).get(key, default)


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _log_prompt(tag: str, prompt: str) -> None:
    logger.info("[%s] PROMPT  len=%d", tag, len(prompt))
    if len(prompt) <= 12_000:
        logger.info("[%s] PROMPT:\n%s", tag, prompt)
    else:
        logger.info("[%s] PROMPT (first 6000):\n%s", tag, prompt[:6_000])
        logger.info("[%s] PROMPT (last 6000):\n%s", tag, prompt[-6_000:])


def _log_response(tag: str, response: str) -> None:
    logger.info("[%s] RESPONSE  len=%d:\n%s", tag, len(response or ""), response or "")


def _log_retrieved(query: str, chunks: List[Any]) -> None:
    logger.info("=== VECTOR DB FETCH === chunks=%d", len(chunks))
    logger.info("  query: %s", query[:400])
    for i, c in enumerate(chunks):
        content = getattr(c, "page_content", str(c)) or ""
        meta = getattr(c, "metadata", {}) or {}
        logger.info("  [chunk %d] source=%-15s  len=%d", i + 1, meta.get("source", "?"), len(content))
        logger.info("  [chunk %d] content:\n%s", i + 1, content[:2_000] if len(content) > 2_000 else content)
    if not chunks:
        logger.info("  (no chunks returned)")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_INSUFFICIENT_PHRASES = ("insufficient", "need more", "need additional", "lack ", "not enough")


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


def _format_retrieved(chunks: List[Any]) -> str:
    if not chunks:
        return "(No chunks retrieved.)"
    return "\n\n---\n\n".join(getattr(c, "page_content", str(c)) for c in chunks)


def _retrieve(state: AgentState, retriever: Any, tag: str) -> str:
    """Run retriever and log results. Returns formatted string."""
    if not retriever:
        return ""
    query = (
        state.get("retrieval_query")
        or f"{state.get('company_name', '')} {state.get('industry', '')}".strip()
        or state["task"][:300]
    )
    try:
        chunks = retriever.invoke(query)
        _log_retrieved(query, chunks)
        return _format_retrieved(chunks) if chunks else ""
    except Exception as exc:
        logger.warning("Retrieval failed [%s]: %s", tag, exc)
        return ""


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


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def reason_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Step 1 – Reason: gap analysis only.
    RAG retrieval happens HERE (and in generate_answer). Not in decide/reflect.
    """
    retriever = _cfg(config, "retriever")
    new_step = state["step"] + 1
    logger.info("========== STEP %d / %d ==========", new_step, state["max_steps"])
    logger.info("=== REASON NODE ===")

    retrieved_str = _retrieve(state, retriever, "reason")

    memory_str = "\n".join(state.get("memory_items") or []) or "(no prior findings yet)"
    history_str = (
        "\n".join(
            f"Turn {i + 1}: {t.get('action', '')} -> {t.get('observation', '')[:250]}"
            for i, t in enumerate((state.get("turn_history") or [])[-MEMORY_RECENT_K:])
        )
        or "(none)"
    )
    tools_used_str = ", ".join(state.get("tools_used") or []) or "none yet"

    prompt = (
        f"Seller: {state.get('my_company_summary', 'N/A')}\n"
        f"Prospect: {state.get('company_name', 'N/A')} | {state.get('industry', 'N/A')}\n\n"
        f"Prior findings:\n{memory_str}\n\n"
        f"Recent turns:\n{history_str}\n\n"
        + (f"Knowledge base context:\n{retrieved_str}\n\n" if retrieved_str else "")
        + f"Tools used so far this run: [{tools_used_str}]\n\n"
        "Your task: Analyse the situation in ONE short paragraph.\n"
        "1. What specific facts do you have about the prospect?\n"
        "2. What is still unknown or unclear?\n"
        "3. What is the single best next step (search, extract, oracle synthesis, or stop)?\n\n"
        "If you are unsure what to search for or how to proceed, you may (optionally) call "
        "consult_oracle to get suggested search queries or next steps.\n\n"
        "IMPORTANT: Do NOT write a VALUE HYPOTHESIS, MESSAGING ANGLE, or SUPPORTING EVIDENCE yet."
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


def decide_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Step 2 – Decide: choose the next action.
    No RAG re-retrieval here (saves token budget). Tracks tools_used for dedup rules.
    consult_oracle is optional (at end for synthesis, or when stuck for search ideas).
    """
    logger.info("=== DECIDE NODE ===")

    memory_str = (
        "\n".join((state.get("memory_items") or [])[-MEMORY_RECENT_K:])
        or "(no prior findings)"
    )
    tools_used: List[str] = state.get("tools_used") or []
    tools_used_str = ", ".join(tools_used) or "none yet"
    search_count = sum(1 for t in tools_used if t == "search_web")
    extract_count = sum(1 for t in tools_used if t == "extract_insights")
    oracle_used = "consult_oracle" in tools_used

    # Build adaptive rules block
    rules: List[str] = [
        f"Tools you have already called this run: [{tools_used_str}]",
        "RULES:",
    ]
    if oracle_used:
        rules.append(
            "- consult_oracle was already called. You MUST set should_stop=true now. "
            "The oracle synthesis is in your prior findings above — use it to generate the final answer."
        )
    elif search_count >= 1 and extract_count >= 1:
        rules.append(
            "- You have searched AND extracted insights. "
            "You MAY call consult_oracle to get a sharper synthesis, OR if you are already confident "
            "(e.g. confidence >= 0.7) you MAY set should_stop=true and skip the oracle."
        )
    elif search_count >= 2:
        rules.append(
            f"- You have done {search_count} web searches. "
            "You MAY call consult_oracle to synthesise, OR set should_stop=true if you are "
            "confident enough (optional oracle)."
        )
    elif search_count == 1:
        rules.append(
            "- You have done 1 web search. "
            "You may call consult_oracle to synthesise, or do one more targeted search if a key "
            "fact is still missing, or set should_stop=true if confident (optional oracle)."
        )
    else:
        rules.append(
            "- You have not searched yet. Call search_web with a specific query, "
            "or extract_insights on the profile. "
            "If you don't know what to search for: you MAY call consult_oracle with "
            "question='What should I search for to learn about [prospect/industry]?' and "
            "context=current task summary (optional). Do NOT stop yet."
        )

    if search_count > 0:
        rules.append("- Do NOT repeat a search_web query you have already used.")
    if extract_count > 0:
        rules.append("- Do NOT call extract_insights again; you already have that result.")

    rules_str = "\n".join(rules)

    base_prompt = (
        f"Seller: {state.get('my_company_summary', 'N/A')}\n"
        f"Prospect: {state.get('company_name', 'N/A')} | {state.get('industry', 'N/A')}\n\n"
        f"Your analysis this step:\n{(state.get('reason_text') or '')[:700]}\n\n"
        f"Prior findings (memory):\n{memory_str}\n\n"
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
                "Decision: next_action=%s  tool_id=%s  should_stop=%s  should_revise=%s  "
                "confidence=%.2f  reasoning=%s",
                decision.next_action,
                decision.tool_id or "(none)",
                decision.should_stop,
                decision.should_revise,
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


def act_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Step 3 – Act: run the chosen tool (if any).
    search_web results are also embedded into the vector store for future retrieval.
    """
    tool_registry = _cfg(config, "tool_registry") or {}
    vector_store = _cfg(config, "vector_store")
    decision = _decision_from_dict(state.get("decision") or {})
    logger.info("=== ACT NODE ===")

    if not decision.tool_id:
        logger.info("  No tool selected; skipping Act.")
        return {"tool_result": "", "tool_error": ""}

    logger.info("=== TOOL CALL === tool_id=%s  tool_input=%s", decision.tool_id, decision.tool_input)
    if decision.tool_id == "search_web":
        logger.info("=== SEARCH RUNNING === query=%s", decision.tool_input.get("query", ""))
    elif decision.tool_id == "consult_oracle":
        logger.info("=== ORACLE CONSULTING === question=%s", decision.tool_input.get("question", "")[:200])

    try:
        result = run_tool(tool_registry, decision.tool_id, decision.tool_input)
        result_str = str(result)
        logger.info(
            "=== TOOL RESULT (%s) === (first 1500 chars)\n%s",
            decision.tool_id,
            result_str[:1_500],
        )
    except Exception as exc:
        logger.warning("Tool error (%s): %s", decision.tool_id, exc)
        return {"tool_result": "", "tool_error": str(exc)}

    # Embed search results into vector store so future reason/generate nodes can retrieve them
    if decision.tool_id == "search_web" and decision.tool_input.get("query") and vector_store is not None:
        q = decision.tool_input["query"]
        max_r = decision.tool_input.get("max_results", 5)
        logger.info("=== VECTOR STORE (save) === query=%s", q)
        try:
            extra_docs = rag.search_results_as_docs(q, max_results=max_r)
            logger.info("  Saving %d search docs to vector store", len(extra_docs))
            for j, d in enumerate(extra_docs):
                content = getattr(d, "page_content", "") or ""
                meta = getattr(d, "metadata", {}) or {}
                logger.info(
                    "  [doc %d] source=%s  url=%s  len=%d",
                    j + 1, meta.get("source"), meta.get("url", ""), len(content),
                )
                logger.info(
                    "  [doc %d] content:\n%s",
                    j + 1,
                    content[:1_200] if len(content) > 1_200 else content,
                )
            if extra_docs and hasattr(vector_store, "add_documents"):
                vector_store.add_documents(extra_docs)
                logger.info("  Vector store: added %d docs", len(extra_docs))
            else:
                logger.info("  Vector store: add_documents unavailable or no docs to add")
        except Exception as exc:
            logger.warning("  Failed to add search docs to vector store: %s", exc)

    return {"tool_result": result_str, "tool_error": ""}


def observe_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Step 4 – Observe: convert tool result/error into a readable observation."""
    logger.info("=== OBSERVE NODE ===")
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


def update_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Step 5 – Update: append step to memory/history and record tool used."""
    logger.info("=== UPDATE NODE ===")
    decision = _decision_from_dict(state.get("decision") or {})
    step = state["step"]
    observation = state.get("observation") or ""
    tool_id = decision.tool_id or "none"

    new_memory_item = (
        f"Step {step}: action={decision.next_action}  tool={tool_id}  "
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


def reflect_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Step 6 – Reflect: structured self-critique.
    Uses fill-in-the-blank format (no free-form JSON) so small models can follow it reliably.
    No RAG re-retrieval here.
    """
    logger.info("=== REFLECT NODE ===")
    observation = state.get("observation") or ""
    memory_str = (
        "\n".join((state.get("memory_items") or [])[-MEMORY_RECENT_K:])
        or "(none)"
    )
    decision = _decision_from_dict(state.get("decision") or {})
    tools_used_str = ", ".join(state.get("tools_used") or []) or "none"
    oracle_used = "consult_oracle" in (state.get("tools_used") or [])

    prompt = (
        f"Prospect: {state.get('company_name', 'N/A')} | {state.get('industry', 'N/A')}\n"
        f"Tools used this run: {tools_used_str}\n\n"
        f"Latest finding:\n{observation[:1_000]}\n\n"
        f"All findings so far:\n{memory_str}\n\n"
        + (
            "NOTE: consult_oracle has already been called. Do NOT suggest more searching.\n\n"
            if oracle_used
            else ""
        )
        + "Complete each line (be brief and specific):\n"
        "Assumptions I am making: [one sentence]\n"
        "Key facts confirmed about the prospect: [one sentence]\n"
        "What is still missing: [one sentence, or write 'nothing critical']\n"
        "Confidence I can write all three sections right now (0.0-1.0): [number only]\n"
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


def generate_answer_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Final node: dedicated LLM call for the structured three-section answer.
    Prompt is clean (no tool instructions). [ASSUMPTION] labels unconfirmed claims.
    """
    retriever = _cfg(config, "retriever")
    logger.info("=== GENERATE ANSWER NODE ===")

    retrieved_str = _retrieve(state, retriever, "generate_answer")
    memory_str = "\n".join(state.get("memory_items") or []) or "(no prior findings)"

    # Check if oracle synthesis is in memory — surface it prominently
    oracle_synthesis = ""
    for item in reversed(state.get("memory_items") or []):
        if "tool=consult_oracle" in item:
            oracle_synthesis = item
            break

    oracle_section = (
        f"\nOracle synthesis (use this as primary input):\n{oracle_synthesis}\n"
        if oracle_synthesis
        else ""
    )

    prompt = (
        "You are writing a concise sales pitch brief.\n\n"
        f"Seller: {state.get('my_company_summary', 'N/A')}\n"
        f"Prospect: {state.get('company_name', 'N/A')} ({state.get('industry', 'N/A')})\n\n"
        f"Research gathered:\n{memory_str}\n"
        + oracle_section
        + (f"\nSupporting context from knowledge base:\n{retrieved_str}\n" if retrieved_str else "")
        + "\nInstructions:\n"
        "- Use ONLY confirmed facts from the research above.\n"
        "- Label any claim that is not directly evidenced as [ASSUMPTION].\n"
        "- Be specific: name the prospect, name the seller's capability, say WHY it matters to them.\n"
        "- Output ONLY the three sections below, nothing else.\n\n"
        "VALUE HYPOTHESIS: [one or two sentences — specific value the seller delivers to THIS prospect]\n\n"
        "MESSAGING ANGLE: [one sentence — the opening line of the pitch, tailored to the prospect]\n\n"
        "SUPPORTING EVIDENCE:\n"
        "- [fact or [ASSUMPTION]]\n"
        "- [fact or [ASSUMPTION]]\n"
        "- [fact or [ASSUMPTION]]"
    )
    _log_prompt("GenerateAnswer", prompt)
    try:
        final = _call_model(prompt, "generate_answer")
        _log_response("GenerateAnswer", final)
        logger.info("=== FINAL ANSWER GENERATED ===\n%s", final[:800])
    except Exception as exc:
        logger.error("generate_answer_node failed: %s", exc)
        final = state.get("observation") or "Unable to generate answer."

    return {"final_response": final}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_after_reflect(state: AgentState) -> Literal["reason", "generate_answer"]:
    """
    After Reflect, decide whether to loop back to Reason or produce the final answer.
    Oracle-used state always routes to generate_answer (oracle is the last tool).
    """
    decision = _decision_from_dict(state.get("decision") or {})
    reflection = state.get("reflection") or {}
    step = state["step"]
    max_steps = state["max_steps"]
    tools_used: List[str] = state.get("tools_used") or []
    oracle_used = "consult_oracle" in tools_used

    should_revise = decision.should_revise or reflection.get("should_revise", False)
    confidence_ok = decision.confidence >= MIN_CONFIDENCE_TO_STOP
    insufficient_but_no_tool = (
        decision.should_stop
        and not decision.tool_id
        and any(p in (decision.reasoning or "").lower() for p in _INSUFFICIENT_PHRASES)
    )

    if step >= max_steps:
        logger.info("Route → generate_answer  (max_steps=%d reached)", max_steps)
        return "generate_answer"

    # Oracle was called → stop iterating regardless of other flags
    if oracle_used and decision.should_stop:
        logger.info("Route → generate_answer  (oracle used + should_stop)")
        return "generate_answer"
    if oracle_used and not should_revise:
        logger.info("Route → generate_answer  (oracle used, not revising)")
        return "generate_answer"

    if should_revise and step < max_steps:
        logger.info("Route → reason  (should_revise=True, step=%d)", step)
        return "reason"

    if decision.should_stop and insufficient_but_no_tool:
        logger.info("Route → reason  (rejecting stop: insufficient info, no tool used)")
        return "reason"

    if decision.should_stop and not confidence_ok:
        logger.info(
            "Route → reason  (rejecting stop: confidence %.2f < %.2f)",
            decision.confidence,
            MIN_CONFIDENCE_TO_STOP,
        )
        return "reason"

    if decision.should_stop and confidence_ok:
        logger.info("Route → generate_answer  (should_stop=True, confidence=%.2f)", decision.confidence)
        return "generate_answer"

    logger.info("Route → reason  (default: not ready to stop)")
    return "reason"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(checkpointing: bool = False):
    builder = StateGraph(AgentState)

    builder.add_node("reason", reason_node)
    builder.add_node("decide", decide_node)
    builder.add_node("act", act_node)
    builder.add_node("observe", observe_node)
    builder.add_node("update", update_node)
    builder.add_node("reflect", reflect_node)
    builder.add_node("generate_answer", generate_answer_node)

    builder.add_edge(START, "reason")
    builder.add_edge("reason", "decide")
    builder.add_edge("decide", "act")
    builder.add_edge("act", "observe")
    builder.add_edge("observe", "update")
    builder.add_edge("update", "reflect")
    builder.add_conditional_edges(
        "reflect",
        route_after_reflect,
        {"reason": "reason", "generate_answer": "generate_answer"},
    )
    builder.add_edge("generate_answer", END)

    if checkpointing:
        try:
            from langgraph.checkpoint.memory import MemorySaver
            logger.info("Graph: compiling with MemorySaver checkpointing")
            return builder.compile(checkpointer=MemorySaver())
        except ImportError:
            logger.warning("MemorySaver not available; compiling without checkpointing")

    return builder.compile(debug=False)


def make_initial_state(
    task: str,
    max_steps: int,
    tool_descriptions: str,
    retrieval_query: str = "",
    my_company_summary: str = "",
    company_name: str = "",
    industry: str = "",
) -> AgentState:
    return AgentState(
        task=task,
        retrieval_query=retrieval_query,
        max_steps=max_steps,
        tool_descriptions=tool_descriptions,
        my_company_summary=my_company_summary,
        company_name=company_name,
        industry=industry,
        step=0,
        tools_used=[],
        memory_items=[],
        turn_history=[],
        reason_text="",
        decision={},
        tool_result="",
        tool_error="",
        observation="",
        reflection={},
        final_response="",
    )
