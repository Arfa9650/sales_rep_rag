"""
Sales-rep agent: RAG setup, logging scaffold, and entry point.
The agent loop itself is managed by LangGraph (see graph.py).
"""

import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from config import ENABLE_CHECKPOINTING, GRAPH_RECURSION_LIMIT, MAX_STEPS
from graph import build_graph, make_initial_state
from tools import get_tool_registry
import rag

logger = logging.getLogger("sales_rep.agent")

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_run_log(log_dir: str) -> logging.FileHandler:
    """
    Add a FileHandler for the run.  Suppresses noisy third-party loggers so
    logs/run_*.txt only contains agent thinking, retrieval, and search events.
    """
    os.makedirs(log_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"run_{stamp}.txt")
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    for name in ("sales_rep.agent", "sales_rep.rag"):
        logging.getLogger(name).setLevel(logging.DEBUG)
    for name in ("primp", "reqwest", "hyper_util", "h2", "rustls",
                 "httpcore", "hpack", "httpx", "ddgs", "cookie_store"):
        logging.getLogger(name).setLevel(logging.WARNING)
    return handler


# ---------------------------------------------------------------------------
# Output parser
# ---------------------------------------------------------------------------

def _parse_sales_rep_output(final_response: str) -> Dict[str, str]:
    """
    Parse final response into value_hypothesis, messaging_angle, supporting_evidence.
    Primary: header regex.  Fallback: one reformatting LLM call.  Last resort: raw text.
    """
    out: Dict[str, str] = {
        "value_hypothesis": "",
        "messaging_angle": "",
        "supporting_evidence": "",
    }
    text = (final_response or "").strip()
    patterns = [
        (r"VALUE\s*HYPOTHESIS\s*[:\-]\s*(.+?)(?=MESSAGING|SUPPORTING|$)", "value_hypothesis"),
        (r"MESSAGING\s*(?:ANGLE)?\s*[:\-]\s*(.+?)(?=VALUE|SUPPORTING|$)", "messaging_angle"),
        (r"SUPPORTING\s*EVIDENCE(?:\s*OR\s*ASSUMPTIONS)?\s*[:\-]\s*(.+?)(?=VALUE|MESSAGING|$)", "supporting_evidence"),
    ]
    for pattern, key in patterns:
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if m:
            out[key] = m.group(1).strip()
    if any(out.values()):
        return out
    # Fallback: ask the model to reformat once
    try:
        from local_llm import complete
        prompt = (
            "Convert this sales-rep response into exactly three short sections. "
            "Output ONLY the following format:\n"
            "VALUE HYPOTHESIS: <one or two sentences>\n"
            "MESSAGING ANGLE: <one or two sentences>\n"
            "SUPPORTING EVIDENCE: <bullets or short paragraph>\n\n"
            f"Response to convert:\n{text[:4_000]}"
        )
        structured = complete(prompt) or ""
        for pattern, key in patterns:
            m = re.search(pattern, structured.strip(), re.DOTALL | re.IGNORECASE)
            if m:
                out[key] = m.group(1).strip()
        if any(out.values()):
            return out
    except Exception:
        pass
    out["value_hypothesis"] = text[:1_500] or "(No value hypothesis extracted)"
    if not out["messaging_angle"]:
        out["messaging_angle"] = "(No messaging angle extracted)"
    if not out["supporting_evidence"]:
        out["supporting_evidence"] = "(No supporting evidence extracted)"
    return out


# ---------------------------------------------------------------------------
# Task template
# ---------------------------------------------------------------------------

SALES_REP_TASK_TEMPLATE = """You are a sales rep assistant.

Who you represent (your company / who you are selling for):
{my_company_description}

Prospect (the company you are exploring or contacting):
Company: {company_name}
Industry: {industry}
Public website or profile text:
{profile_text}

Your goal: Propose (1) a value hypothesis, (2) a suggested messaging angle, (3) supporting evidence or assumptions.
You MUST use tools when information is thin: use search_web to look up the company and industry, and extract_insights on the profile.
Do NOT stop with "insufficient information" — iterate: search, then synthesise.
Only stop when you have enough to write concrete VALUE HYPOTHESIS, MESSAGING ANGLE, and SUPPORTING EVIDENCE.

When you stop, your final answer MUST use these exact headers:
VALUE HYPOTHESIS: <your value hypothesis>
MESSAGING ANGLE: <your suggested messaging angle>
SUPPORTING EVIDENCE: <supporting evidence or assumptions>"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_sales_rep_flow(
    my_company_description: str,
    prospect_company_name: str,
    prospect_industry: str,
    prospect_profile_text: str,
    max_steps: Optional[int] = None,
    run_initial_search: bool = True,
) -> Dict[str, str]:
    """
    Full pipeline:
      1. Build RAG index (my_company + prospect + optional web search).
      2. Compile LangGraph and invoke with streaming.
      3. Parse the structured three-section answer.
      4. Write run log to logs/run_YYYYMMDD_HHMMSS.txt.
    """
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    file_handler = _setup_run_log(log_dir)
    try:
        max_steps = max_steps if max_steps is not None else MAX_STEPS
        logger.info(
            "=== RUN START === prospect=%s  industry=%s  max_steps=%d",
            prospect_company_name, prospect_industry, max_steps,
        )

        # ---- 1. Build RAG index ------------------------------------------------
        search_docs: List[Any] = []
        if run_initial_search and (prospect_company_name or prospect_industry):
            q = f"{prospect_company_name or ''} {prospect_industry or ''}".strip()
            if q:
                logger.info("=== INITIAL SEARCH === query=%s", q)
                search_docs = rag.search_results_as_docs(q, max_results=5)
                logger.info("Initial search: %d docs added to RAG index", len(search_docs))

        documents = rag.build_documents(
            my_company_description,
            prospect_company_name,
            prospect_industry,
            prospect_profile_text,
            search_docs=search_docs or None,
        )
        logger.info("=== RAG INDEX === source docs=%d", len(documents))
        chunked = rag.chunk_documents(documents)
        logger.info("After chunking: %d chunks stored in vector DB", len(chunked))
        vector_store = rag.create_vector_store(chunked)
        retriever = rag.get_retriever(vector_store)

        # ---- 2. Build task and tool registry -----------------------------------
        task = SALES_REP_TASK_TEMPLATE.format(
            my_company_description=(my_company_description or "").strip(),
            company_name=prospect_company_name or "N/A",
            industry=prospect_industry or "N/A",
            profile_text=(prospect_profile_text or "").strip(),
        )
        registry = get_tool_registry(profile_text=(prospect_profile_text or "").strip())
        tool_descriptions = "\n".join(
            f"- {tid}: {spec.get('description', '')} (params: {spec.get('parameters', {})})"
            for tid, spec in registry.items()
        )
        retrieval_query = (
            f"Value hypothesis messaging angle evidence: "
            f"Company: {prospect_company_name or 'N/A'}, Industry: {prospect_industry or 'N/A'}."
        )

        # ---- 3. Build and run LangGraph ----------------------------------------
        app = build_graph(checkpointing=ENABLE_CHECKPOINTING)
        initial_state = make_initial_state(
            task=task,
            max_steps=max_steps,
            tool_descriptions=tool_descriptions,
            retrieval_query=retrieval_query,
        )
        run_config: Dict[str, Any] = {
            "configurable": {
                "retriever": retriever,
                "vector_store": vector_store,
                "tool_registry": registry,
            },
            "recursion_limit": GRAPH_RECURSION_LIMIT,
        }
        if ENABLE_CHECKPOINTING:
            run_config["configurable"]["thread_id"] = (
                f"{prospect_company_name or 'run'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

        final_state: Dict[str, Any] = {}
        logger.info("=== GRAPH STREAM START ===")
        for step_event in app.stream(initial_state, config=run_config, stream_mode="updates"):
            for node_name, updates in step_event.items():
                updated_keys = [k for k, v in (updates or {}).items() if v not in (None, "", [], {})]
                logger.info("  [graph] node=%s  updated_keys=%s", node_name, updated_keys)
            final_state.update(step_event.get(list(step_event)[-1], {}))

        # Retrieve full final state via invoke if stream didn't capture it
        if not final_state.get("final_response"):
            final_state = app.invoke(initial_state, config=run_config)

        final_response = (final_state or {}).get("final_response") or ""
        logger.info("=== GRAPH COMPLETE === final_response len=%d", len(final_response))

        # ---- 4. Parse structured output ----------------------------------------
        parsed = _parse_sales_rep_output(final_response)
        logger.info(
            "=== RUN COMPLETE ===\n"
            "VALUE HYPOTHESIS: %s\n"
            "MESSAGING ANGLE: %s\n"
            "SUPPORTING EVIDENCE: %s",
            parsed.get("value_hypothesis", "")[:300],
            parsed.get("messaging_angle", "")[:300],
            parsed.get("supporting_evidence", "")[:300],
        )
        return parsed

    finally:
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    my_company = (
        "K2X Technologies: We provide AI-driven software solutions for industrial companies "
        "to improve operational efficiency and reduce downtime."
    )
    result = run_sales_rep_flow(
        my_company_description=my_company,
        prospect_company_name="Antonx",
        prospect_industry="Software Solutions",
        prospect_profile_text="antonx.com",
    )
    print("VALUE HYPOTHESIS:", result["value_hypothesis"])
    print("MESSAGING ANGLE:", result["messaging_angle"])
    print("SUPPORTING EVIDENCE:", result["supporting_evidence"])
