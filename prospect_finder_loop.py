"""
Prospect Finder entry point.
Mirrors agent_loop.py but runs the ProspectFinder LangGraph and returns
a list of enriched prospect company dicts instead of a sales pitch.
"""

import logging
import os
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from config import ENABLE_CHECKPOINTING, GRAPH_RECURSION_LIMIT, MAX_STEPS
from prospect_finder_graph import build_graph, make_initial_state
from tools import get_prospect_finder_registry

logger = logging.getLogger("sales_rep.agent")

# ---------------------------------------------------------------------------
# Constants (reused from agent_loop for consistency)
# ---------------------------------------------------------------------------

_LOG_MESSAGE_MAX = 500
_PREVIEW_MAX = 400

PROSPECT_FINDER_TASK_TEMPLATE = """Seller (who you represent):
{my_company_description}

Goal: Find {num_prospects} real companies in {location} that would be valuable prospects
for the seller described above.
{industry_focus_line}
Use search_linkedin_companies to discover companies, then fetch_company_info to enrich each one.
For each found company, record: name, LinkedIn URL, website, industry, estimated size, key facts."""


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_prospect_log(log_dir: str) -> logging.FileHandler:
    os.makedirs(log_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"prospect_run_{stamp}.txt")
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


def _sanitise_prospect_updates(updates: Dict[str, Any]) -> Dict[str, Any]:
    """Produce a small dict safe for SSE."""
    if not updates:
        return {}
    out: Dict[str, Any] = {}
    for k, v in updates.items():
        if v is None or v == "" or v == [] or v == {}:
            continue
        if k == "reason_text" and isinstance(v, str):
            out[k] = v[:_LOG_MESSAGE_MAX] + ("..." if len(v) > _LOG_MESSAGE_MAX else "")
        elif k == "observation" and isinstance(v, str):
            out[k] = v[:_PREVIEW_MAX] + ("..." if len(v) > _PREVIEW_MAX else "")
        elif k == "decision" and isinstance(v, dict):
            out[k] = {
                "tool_id": v.get("tool_id"),
                "should_stop": v.get("should_stop"),
                "confidence": v.get("confidence"),
            }
        elif k in ("step", "found_companies"):
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_prospect_finder_flow(
    my_company_description: str,
    location: str,
    num_prospects: int,
    industry_focus: str = "",
    max_steps: Optional[int] = None,
    event_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> List[Dict[str, Any]]:
    """
    Find num_prospects real companies in location that match the seller's profile.
    Returns a list of enriched prospect dicts:
      company_name, linkedin_url, website, industry, size_estimate, location, key_facts.

    If event_sink is provided, emits SSE-style events:
      start, log, step, tool, done.
    """
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    file_handler = _setup_prospect_log(log_dir)
    event_handler: Optional[logging.Handler] = None

    if event_sink:
        def _emit_log(record: logging.LogRecord) -> None:
            msg = record.getMessage()
            if len(msg) > _LOG_MESSAGE_MAX:
                msg = msg[:_LOG_MESSAGE_MAX] + "..."
            try:
                event_sink({"type": "log", "level": record.levelname, "message": msg, "logger": record.name})
            except Exception:
                pass

        class EventSinkHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                try:
                    _emit_log(record)
                except Exception:
                    self.handleError(record)

        event_handler = EventSinkHandler()
        event_handler.setLevel(logging.DEBUG)
        logging.getLogger("sales_rep.agent").addHandler(event_handler)
        logging.getLogger("sales_rep.rag").addHandler(event_handler)

    try:
        max_steps = max_steps if max_steps is not None else MAX_STEPS
        logger.info(
            "=== PROSPECT FINDER START === location=%s  count=%d  industry=%s  max_steps=%d",
            location, num_prospects, industry_focus or "any", max_steps,
        )

        registry = get_prospect_finder_registry(my_company_description=my_company_description)
        tool_descriptions = "\n".join(
            f"- {tid}: {spec.get('description', '')} (params: {spec.get('parameters', {})})"
            for tid, spec in registry.items()
        )

        industry_focus_line = (
            f"Industry focus: {industry_focus}\n" if industry_focus else ""
        )
        task = PROSPECT_FINDER_TASK_TEMPLATE.format(
            my_company_description=(my_company_description or "").strip(),
            num_prospects=num_prospects,
            location=location or "N/A",
            industry_focus_line=industry_focus_line,
        )

        my_company_summary = (my_company_description or "").strip()[:200]

        if event_sink:
            event_sink({
                "type": "start",
                "location": location or "N/A",
                "num_prospects": num_prospects,
                "industry_focus": industry_focus or "",
                "company_summary": my_company_summary[:100],
            })

        app = build_graph(checkpointing=ENABLE_CHECKPOINTING)
        initial_state = make_initial_state(
            task=task,
            target_count=num_prospects,
            location=location or "N/A",
            my_company_summary=my_company_summary,
            industry_focus=industry_focus or "",
            max_steps=max_steps,
            tool_descriptions=tool_descriptions,
        )
        run_config: Dict[str, Any] = {
            "configurable": {"tool_registry": registry},
            "recursion_limit": GRAPH_RECURSION_LIMIT,
        }
        if ENABLE_CHECKPOINTING:
            run_config["configurable"]["thread_id"] = (
                f"prospect_{location}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

        final_state: Dict[str, Any] = {}
        logger.info("=== PROSPECT GRAPH STREAM START ===")
        for step_event in app.stream(initial_state, config=run_config, stream_mode="updates"):
            for node_name, updates in step_event.items():
                updated_keys = [k for k, v in (updates or {}).items() if v not in (None, "", [], {})]
                logger.info("  [graph] node=%s  updated_keys=%s", node_name, updated_keys)
                if event_sink:
                    step_num = (updates or {}).get("step") or final_state.get("step") or 0
                    event_sink({
                        "type": "step",
                        "node": node_name,
                        "step_number": step_num,
                        "updates": _sanitise_prospect_updates(updates or {}),
                    })
                    if node_name == "observe":
                        dec = final_state.get("decision") or {}
                        tool_id = dec.get("tool_id")
                        if tool_id:
                            obs = (updates or {}).get("observation") or ""
                            prev = obs[:_PREVIEW_MAX] + ("..." if len(obs) > _PREVIEW_MAX else "")
                            event_sink({
                                "type": "tool",
                                "tool_id": tool_id,
                                "input": dec.get("tool_input") or {},
                                "result_preview": prev,
                            })
            final_state.update(step_event.get(list(step_event)[-1], {}))

        if not final_state.get("final_prospects"):
            final_state = app.invoke(initial_state, config=run_config)

        prospects = final_state.get("final_prospects") or []
        logger.info("=== PROSPECT FINDER COMPLETE === prospects=%d", len(prospects))
        for i, p in enumerate(prospects, 1):
            logger.info("  [%d] %s | %s | %s", i, p.get("company_name"), p.get("industry"), p.get("website"))

        if event_sink:
            event_sink({
                "type": "done",
                "result": {"prospects": prospects},
                "tools_used": final_state.get("tools_used"),
                "steps": final_state.get("step"),
            })
        return prospects

    except Exception as e:
        logger.error("Prospect finder failed: %s", e)
        if event_sink:
            try:
                event_sink({"type": "done", "error": str(e), "result": None})
            except Exception:
                pass
        raise

    finally:
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()
        if event_handler:
            logging.getLogger("sales_rep.agent").removeHandler(event_handler)
            logging.getLogger("sales_rep.rag").removeHandler(event_handler)


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    my_company = (
        "K2X Technologies: We provide AI-driven software solutions for industrial companies "
        "to improve operational efficiency and reduce downtime."
    )
    results = run_prospect_finder_flow(
        my_company_description=my_company,
        location="Islamabad, Pakistan",
        num_prospects=5,
        industry_focus="IT services",
    )
    for i, p in enumerate(results, 1):
        print(f"\n[{i}] {p.get('company_name')}")
        print(f"    LinkedIn : {p.get('linkedin_url')}")
        print(f"    Website  : {p.get('website')}")
        print(f"    Industry : {p.get('industry')}")
        print(f"    Size     : {p.get('size_estimate')}")
        print(f"    Facts    : {p.get('key_facts')}")
