"""
FastAPI backend for the sales-rep RAG agent.
POST /api/run streams SSE events (start, log, step, tool, done) for the React webapp.
"""

import json
import queue
import threading
from typing import Any, Dict, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

import agent_loop
import prospect_finder_loop

app = FastAPI(title="Sales Rep RAG Agent API", version="0.1.0")

# CORS: allow React app; configurable via API_CORS_ORIGINS (comma-separated)
_origins = ["http://localhost:3000", "http://127.0.0.1:3000"]
try:
    import config as _config
    if getattr(_config, "API_CORS_ORIGINS", ""):
        _origins = [x.strip() for x in str(_config.API_CORS_ORIGINS).split(",") if x.strip()]
except Exception:
    pass

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _sse_event(data: Dict[str, Any]) -> str:
    return f"data: {json.dumps(data)}\n\n"


@app.get("/health")
@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/run")
def run_agent(body: dict) -> StreamingResponse:
    """
    Run the sales-rep flow. Request body (JSON):
      my_company_description, prospect_company_name, prospect_industry, prospect_profile_text,
      optional max_steps, optional run_initial_search (default true).

    Response: Server-Sent Events stream until done.
    Event types: start, log, step, tool, done.
    """
    event_queue: queue.Queue = queue.Queue()

    def event_sink(ev: Dict[str, Any]) -> None:
        try:
            event_queue.put(ev)
        except Exception:
            pass

    def run_in_thread() -> None:
        try:
            agent_loop.run_sales_rep_flow(
                my_company_description=body.get("my_company_description") or "",
                prospect_company_name=body.get("prospect_company_name") or "",
                prospect_industry=body.get("prospect_industry") or "",
                prospect_profile_text=body.get("prospect_profile_text") or "",
                max_steps=body.get("max_steps"),
                run_initial_search=body.get("run_initial_search", True),
                event_sink=event_sink,
            )
        except Exception as e:
            event_sink({"type": "done", "error": str(e), "result": None})

    thread = threading.Thread(target=run_in_thread)
    thread.start()

    def generate() -> Any:
        while True:
            try:
                ev = event_queue.get(timeout=60)
            except queue.Empty:
                continue
            yield _sse_event(ev)
            if ev.get("type") == "done":
                break

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/find-prospects")
def find_prospects(body: dict) -> StreamingResponse:
    """
    Find prospect companies. Request body (JSON):
      my_company_description (string)
      location               (string, e.g. "Islamabad, Pakistan")
      num_prospects          (int, how many to find)
      industry_focus         (string, optional)
      max_steps              (int, optional)

    Response: Server-Sent Events stream until done.
    Event types: start, log, step, tool, done.
    done.result = { "prospects": [ { company_name, linkedin_url, website,
                                     industry, size_estimate, location, key_facts }, ... ] }
    """
    event_queue: queue.Queue = queue.Queue()

    def event_sink(ev: Dict[str, Any]) -> None:
        try:
            event_queue.put(ev)
        except Exception:
            pass

    def run_in_thread() -> None:
        try:
            prospect_finder_loop.run_prospect_finder_flow(
                my_company_description=body.get("my_company_description") or "",
                location=body.get("location") or "",
                num_prospects=int(body.get("num_prospects") or 5),
                industry_focus=body.get("industry_focus") or "",
                max_steps=body.get("max_steps"),
                event_sink=event_sink,
            )
        except Exception as e:
            event_sink({"type": "done", "error": str(e), "result": None})

    thread = threading.Thread(target=run_in_thread)
    thread.start()

    def generate() -> Any:
        while True:
            try:
                ev = event_queue.get(timeout=60)
            except queue.Empty:
                continue
            yield _sse_event(ev)
            if ev.get("type") == "done":
                break

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
