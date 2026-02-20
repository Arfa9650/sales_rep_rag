"""
Decision types and parsing for the agent loop.
parse_decision: extract Decision from raw LLM text (JSON embedded in freeform output).
parse_reflection: extract confidence / should_revise from reflection text.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Decision:
    next_action: str = ""
    tool_id: Optional[str] = None
    tool_input: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    should_stop: bool = False
    should_revise: bool = False
    reasoning: str = ""


def _extract_json_obj(text: str) -> Optional[Dict]:
    """Try to extract the first valid JSON object from arbitrary text."""
    if not text:
        return None
    # Strip markdown code fences first
    cleaned = re.sub(r"```(?:json)?\s*", "", text.strip())
    cleaned = re.sub(r"```", "", cleaned).strip()
    # Try the full cleaned text
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # Find the outermost { ... } block
    start = cleaned.find("{")
    if start == -1:
        return None
    depth, end = 0, -1
    for i, ch in enumerate(cleaned[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end == -1:
        return None
    try:
        obj = json.loads(cleaned[start : end + 1])
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return None


def _coerce_tool_input(raw: Any, tool_id: Optional[str]) -> Dict[str, Any]:
    """Ensure tool_input is always a dict."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        s = raw.strip()
        try:
            parsed = json.loads(s)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        # Plain string → guess the right key
        if tool_id == "search_web":
            return {"query": s}
        return {"profile_text": s}
    return {}


def parse_decision(raw: str, structured_fallback: Optional[Dict] = None) -> Decision:
    """
    Parse LLM output into a Decision.
    Tries JSON extraction from raw first, then structured_fallback dict.
    """
    obj = _extract_json_obj(raw or "")
    if obj is None and isinstance(structured_fallback, dict):
        obj = structured_fallback
    if obj is None:
        obj = {}

    try:
        confidence = float(obj.get("confidence", 0))
    except (TypeError, ValueError):
        confidence = 0.0

    tool_id_raw = obj.get("tool_id") or None
    if isinstance(tool_id_raw, str):
        tool_id_raw = tool_id_raw.strip() or None

    return Decision(
        next_action=str(obj.get("next_action") or ""),
        tool_id=tool_id_raw,
        tool_input=_coerce_tool_input(obj.get("tool_input"), tool_id_raw),
        confidence=confidence,
        should_stop=bool(obj.get("should_stop", False)),
        should_revise=bool(obj.get("should_revise", False)),
        reasoning=str(obj.get("reasoning") or ""),
    )


def parse_reflection(raw: str) -> Dict[str, Any]:
    """
    Parse reflection output: returns dict with confidence, should_revise, critique.
    Falls back to heuristic scanning when no JSON is found.
    """
    result: Dict[str, Any] = {
        "confidence": 0.5,
        "should_revise": False,
        "critique": (raw or "")[:500],
    }
    obj = _extract_json_obj(raw or "")
    if obj:
        try:
            result["confidence"] = float(obj["confidence"])
        except (KeyError, TypeError, ValueError):
            pass
        if "should_revise" in obj:
            result["should_revise"] = bool(obj["should_revise"])
        if "critique" in obj:
            result["critique"] = str(obj["critique"])
        return result

    # Heuristic: scan for confidence value
    m = re.search(r"confidence[:\s]+([01](?:\.[0-9]+)?|0\.[0-9]+)", raw or "", re.IGNORECASE)
    if m:
        try:
            result["confidence"] = float(m.group(1))
        except ValueError:
            pass

    # Heuristic: should_revise
    revise_signals = ("revise", "continue", "need more", "need additional", "insufficient", "not enough", "more info")
    stop_signals = ("sufficient", "ready to answer", "high confidence", "should stop", "can stop")
    lower = (raw or "").lower()
    if any(s in lower for s in revise_signals):
        result["should_revise"] = True
    if any(s in lower for s in stop_signals):
        result["should_revise"] = False

    return result
