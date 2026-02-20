"""
LangGraph workflow for the RAG sales-rep agent.

Graph topology:
    START → reason → decide → act → observe → update → reflect
                ↑                                          |
                └──────────── (loop) ───────────────── route
                                                          |
                                                   generate_answer → END

Each node receives AgentState; non-serialisable runtime objects (retriever,
vector_store, tool_registry) are injected via config["configurable"].
"""

import logging
import re
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from config import (
    DECIDE_PARSE_RETRIES,
    ENABLE_CHECKPOINTING,
    GRAPH_RECURSION_LIMIT,
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
    task: str
    retrieval_query: str
    max_steps: int
    tool_descriptions: str
    # Accumulated across steps
    step: int
    memory_items: List[str]
    turn_history: List[Dict[str, Any]]
    # Per-step outputs (overwritten each step)
    reason_text: str
    decision: Dict[str, Any]    # serialised Decision fields
    tool_result: str
    tool_error: str
    observation: str
    reflection: Dict[str, Any]  # {confidence, should_revise, critique}
    # Final answer (set by generate_answer node)
    final_response: str


def _cfg(config: RunnableConfig, key: str, default: Any = None) -> Any:
    """Safely pull a value from config["configurable"]."""
    return (config or {}).get("configurable", {}).get(key, default)


# ---------------------------------------------------------------------------
# Logging helpers (all logging for the graph lives here)
# ---------------------------------------------------------------------------

def _log_prompt(tag: str, prompt: str) -> None:
    logger.info("[%s] prompt  len=%d", tag, len(prompt))
    if len(prompt) <= 12_000:
        logger.info("[%s] PROMPT:\n%s", tag, prompt)
    else:
        logger.info("[%s] PROMPT (first 6000):\n%s", tag, prompt[:6_000])
        logger.info("[%s] PROMPT (last 6000):\n%s", tag, prompt[-6_000:])


def _log_response(tag: str, response: str) -> None:
    logger.info("[%s] RESPONSE len=%d:\n%s", tag, len(response or ""), response or "")


def _log_retrieved(query: str, chunks: List[Any]) -> None:
    logger.info("=== VECTOR DB FETCH === chunks=%d", len(chunks))
    logger.info("  query: %s", query[:400])
    for i, c in enumerate(chunks):
        content = getattr(c, "page_content", str(c)) or ""
        meta = getattr(c, "metadata", {}) or {}
        logger.info("  [chunk %d] source=%-15s len=%d", i + 1, meta.get("source", "?"), len(content))
        logger.info("  [chunk %d] content:\n%s", i + 1, content[:2_000] if len(content) > 2_000 else content)
    if not chunks:
        logger.info("  (no chunks returned)")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_INSUFFICIENT_PHRASES = ("insufficient", "need more", "need additional", "lack ", "not enough")


def _call_model(prompt: str, step_name: str) -> str:
    """Call Ollama with retries."""
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


def _build_context(state: AgentState, retriever: Optional[Any] = None) -> str:
    """Assemble context string: task + memory + recent turns + optional RAG chunks."""
    history_str = "\n".join(
        f"Turn {i + 1}: {t.get('action', '')} -> {t.get('observation', '')[:300]}"
        for i, t in enumerate((state.get("turn_history") or [])[-MEMORY_RECENT_K:])
    ) or "(no turns yet)"

    memory_str = "\n".join(state.get("memory_items") or []) or "(no memory yet)"

    context = (
        f"Task:\n{state['task']}\n\n"
        f"Memory (prior findings):\n{memory_str}\n\n"
        f"Recent turns:\n{history_str}"
    )

    if retriever:
        query = state.get("retrieval_query") or state["task"][:500]
        try:
            chunks = retriever.invoke(query)
            _log_retrieved(query, chunks)
            if chunks:
                context += f"\n\nRelevant context retrieved from knowledge base:\n{_format_retrieved(chunks)}"
        except Exception as exc:
            logger.warning("Retrieval failed: %s", exc)

    return context


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
    Step 1 – Reason: what do we know, what is still missing?
    Increments the step counter and injects fresh RAG chunks into context.
    """
    retriever = _cfg(config, "retriever")
    new_step = state["step"] + 1
    logger.info("========== STEP %d / %d ==========", new_step, state["max_steps"])
    logger.info("=== REASON NODE ===")

    context = _build_context(state, retriever=retriever)
    prompt = (
        f"{context}\n\n"
        "Briefly state: what do you know so far and what is still missing to write a "
        "VALUE HYPOTHESIS, MESSAGING ANGLE, and SUPPORTING EVIDENCE? One short paragraph."
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
    Step 2 – Decide: what action to take next (tool call or stop).
    Retries up to DECIDE_PARSE_RETRIES times if the JSON cannot be parsed.
    """
    retriever = _cfg(config, "retriever")
    logger.info("=== DECIDE NODE ===")
    context = _build_context(state, retriever=retriever)

    base_prompt = (
        f"{context}\n\n"
        f"Your reasoning so far: {(state.get('reason_text') or '')[:600]}\n\n"
        f"Available tools:\n{state['tool_descriptions']}\n\n"
        "Decide: What should I do next?\n"
        "- If you lack company/industry details: set should_stop=false and call a tool.\n"
        "  Use search_web with a concrete query, or extract_insights on the profile.\n"
        "- Only set should_stop=true when you have enough for VALUE HYPOTHESIS, MESSAGING ANGLE, "
        "and SUPPORTING EVIDENCE.\n"
        "- Do NOT stop with 'insufficient information' — search first, then synthesise.\n\n"
        "Respond with a JSON object with exactly these keys:\n"
        "  next_action (string), tool_id (string or null), tool_input (object, e.g. {\"query\": \"...\"}),\n"
        "  confidence (0.0-1.0), should_stop (bool), should_revise (bool), reasoning (string)"
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
                    "Your previous response was invalid. Respond ONLY with a JSON object "
                    "containing: next_action, tool_id, tool_input (object), confidence (0-1), "
                    "should_stop (bool), should_revise (bool), reasoning (string)."
                )

    logger.error("Decide node exhausted retries: %s", last_error)
    # Return a safe fallback decision so the graph can continue
    return {
        "decision": _decision_to_dict(Decision(
            next_action="continue",
            reasoning=f"Decide failed: {last_error}",
        ))
    }


def act_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Step 3 – Act: run the chosen tool (if any).
    When search_web is used, search results are also added to the vector store.
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

    try:
        result = run_tool(tool_registry, decision.tool_id, decision.tool_input)
        result_str = str(result)
        logger.info("=== TOOL RESULT (%s) === (first 1500 chars)\n%s", decision.tool_id, result_str[:1_500])
    except Exception as exc:
        logger.warning("Tool error (%s): %s", decision.tool_id, exc)
        return {"tool_result": "", "tool_error": str(exc)}

    # After search_web: add results to vector store for future retrieval
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
                logger.info("  [doc %d] content:\n%s", j + 1, content[:1_200] if len(content) > 1_200 else content)
            if extra_docs and hasattr(vector_store, "add_documents"):
                vector_store.add_documents(extra_docs)
                logger.info("  Vector store: added %d docs", len(extra_docs))
            else:
                logger.info("  Vector store: add_documents unavailable or no docs to add")
        except Exception as exc:
            logger.warning("  Failed to add search docs: %s", exc)

    return {"tool_result": result_str, "tool_error": ""}


def observe_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Step 4 – Observe: convert tool result/error into a human-readable observation."""
    logger.info("=== OBSERVE NODE ===")
    decision = _decision_from_dict(state.get("decision") or {})
    tool_id = decision.tool_id or ""
    tool_result = state.get("tool_result") or ""
    tool_error = state.get("tool_error") or ""

    if tool_error:
        obs = f"Tool {tool_id} failed: {tool_error}"
    elif tool_id:
        obs = f"Tool {tool_id} result: {tool_result[:2_000]}"
    else:
        obs = "No tool was used this step."

    logger.info("  Observation: %s", obs[:400])
    return {"observation": obs}


def update_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Step 5 – Update: append this step to memory and turn history."""
    logger.info("=== UPDATE NODE ===")
    decision = _decision_from_dict(state.get("decision") or {})
    step = state["step"]
    observation = state.get("observation") or ""
    tool_id = decision.tool_id or "none"

    new_memory_item = (
        f"Step {step}: action={decision.next_action}  tool={tool_id}  "
        f"observation: {observation[:250]}"
    )
    new_turn = {
        "action": f"tool={tool_id}" if tool_id != "none" else decision.next_action,
        "observation": observation,
    }
    logger.info("  Memory item: %s", new_memory_item[:200])

    return {
        "memory_items": list(state.get("memory_items") or []) + [new_memory_item],
        "turn_history": list(state.get("turn_history") or []) + [new_turn],
    }


def reflect_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Step 6 – Reflect: self-critique; is the information sufficient to stop?"""
    retriever = _cfg(config, "retriever")
    logger.info("=== REFLECT NODE ===")
    context = _build_context(state, retriever=retriever)
    observation = state.get("observation") or ""
    decision = _decision_from_dict(state.get("decision") or {})

    prompt = (
        f"{context}\n\n"
        f"Last observation: {observation}\n\n"
        "Evaluate the current state:\n"
        "- What assumptions are you making?\n"
        "- Is your information sufficient to write VALUE HYPOTHESIS, MESSAGING ANGLE, "
        "and SUPPORTING EVIDENCE right now?\n"
        "- Confidence score (0.0-1.0)?\n"
        "- If not sufficient, recommend continuing with a tool (e.g. search_web).\n"
        "- should_revise=true means keep looping; should_revise=false means ready to answer."
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
    Final node: one dedicated LLM call that produces the three-section structured answer
    grounded in all accumulated memory + one last retrieval pass.
    """
    retriever = _cfg(config, "retriever")
    logger.info("=== GENERATE ANSWER NODE ===")

    # Last retrieval pass for fresh grounding
    retrieved_str = ""
    if retriever:
        query = state.get("retrieval_query") or state["task"][:500]
        try:
            chunks = retriever.invoke(query)
            _log_retrieved(query, chunks)
            if chunks:
                retrieved_str = _format_retrieved(chunks)
        except Exception as exc:
            logger.warning("Final retrieval failed: %s", exc)

    memory_str = "\n".join(state.get("memory_items") or []) or "(no prior findings)"

    prompt = (
        f"Task:\n{state['task']}\n\n"
        f"Prior findings (memory):\n{memory_str}\n\n"
        + (f"Retrieved context (knowledge base):\n{retrieved_str}\n\n" if retrieved_str else "")
        + "Using ONLY the information above, write your final answer with EXACTLY these headers "
        "(no additional text before or after):\n"
        "VALUE HYPOTHESIS: <one or two sentences>\n"
        "MESSAGING ANGLE: <one or two sentences>\n"
        "SUPPORTING EVIDENCE: <bullets or short paragraph>"
    )
    _log_prompt("GenerateAnswer", prompt)
    try:
        final = _call_model(prompt, "generate_answer")
        _log_response("GenerateAnswer", final)
        logger.info("=== FINAL ANSWER GENERATED ===\n%s", final[:600])
    except Exception as exc:
        logger.error("generate_answer_node failed: %s", exc)
        # Fall back to last observation
        final = state.get("observation") or "Unable to generate answer."

    return {"final_response": final}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_after_reflect(
    state: AgentState,
) -> Literal["reason", "generate_answer"]:
    """
    After Reflect, decide whether to loop back to Reason or produce the final answer.
    Four cases:
      1. should_revise → reason
      2. max_steps reached → generate_answer
      3. should_stop + confidence OK + not "insufficient without tool" → generate_answer
      4. everything else → reason (continue iterating)
    """
    decision = _decision_from_dict(state.get("decision") or {})
    reflection = state.get("reflection") or {}
    step = state["step"]
    max_steps = state["max_steps"]

    should_revise = decision.should_revise or reflection.get("should_revise", False)
    confidence_ok = decision.confidence >= MIN_CONFIDENCE_TO_STOP
    insufficient_but_no_tool = (
        decision.should_stop
        and not decision.tool_id
        and any(p in (decision.reasoning or "").lower() for p in _INSUFFICIENT_PHRASES)
    )

    if should_revise and step < max_steps:
        logger.info("Route → reason  (should_revise=True, step=%d)", step)
        return "reason"

    if step >= max_steps:
        logger.info("Route → generate_answer  (max_steps=%d reached)", max_steps)
        return "generate_answer"

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
        logger.info(
            "Route → generate_answer  (should_stop=True, confidence=%.2f)",
            decision.confidence,
        )
        return "generate_answer"

    # Default: keep iterating
    logger.info("Route → reason  (default: not ready to stop)")
    return "reason"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(checkpointing: bool = False):
    """
    Build and compile the LangGraph StateGraph.

    Args:
        checkpointing: if True, attach a MemorySaver for in-process state persistence.

    Returns:
        A compiled LangGraph app ready for .invoke() / .stream().
    """
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
) -> AgentState:
    """Return a clean initial AgentState."""
    return AgentState(
        task=task,
        retrieval_query=retrieval_query,
        max_steps=max_steps,
        tool_descriptions=tool_descriptions,
        step=0,
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
