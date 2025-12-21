"""LangGraph-powered schema reasoning with iterative validation."""

import json
import re
import textwrap
from typing import Any, Dict, List, TypedDict

from agentic.ollama_client import run_ollama

try:  # pragma: no cover - import guard for optional dependency clarity
    from langgraph.graph import StateGraph, END
except ImportError as exc:  # pragma: no cover - handled at import time
    raise ImportError(
        "LangGraph is required for agentic schema reasoning. Install it via 'pip install langgraph langchain-core'."
    ) from exc


PROMPT_HEADER = textwrap.dedent(
    """
    You are an expert assistant that maps natural-language healthcare search questions to structured schema filters.
    Respond ONLY with valid JSON following the provided schema. Each filter must use a field defined in the schema snippet.
    Do not hallucinate new field names. If you are unsure, iterate until you find a valid field.
    """
)

INSTRUCTIONS = textwrap.dedent(
    """
    Output format:
    {
      "filters": [
        {"field_path": "Type.field", "operator": "=", "value": "VALUE"}
      ]
    }
    Allowed operators: =, >, <, >=, <=, contains.
    NEVER use brackets, array indices, or prefixes like [Provider] or Facilities[*]. Use dotted Type.field paths only.
    Reasoning loop:
      1. Draft candidate filters using the schema reference below.
      2. Re-check every field_path against the schema. If any path is invalid, iterate again until all are valid.
      3. Return ONLY the final JSON (no Markdown fences, no comments).
    """
)

FEW_SHOTS = textwrap.dedent(
    """
    Examples (imitate this style, no Markdown fences):
    Question: Find active cardiology providers in Los Angeles hospitals that accept Blue Shield PPO
    Response:
    {"filters": [
      {"field_path": "Affiliation.status", "operator": "=", "value": "ACTIVE"},
      {"field_path": "Specialty.name", "operator": "=", "value": "Cardiology"},
      {"field_path": "Facility.location.city", "operator": "=", "value": "Los Angeles"},
      {"field_path": "Facility.type", "operator": "=", "value": "HOSPITAL"},
      {"field_path": "Facility.plansAccepted.name", "operator": "=", "value": "Blue Shield PPO"}
    ]}

    Question: Show inactive oncology clinics in Seattle
    Response:
    {"filters": [
      {"field_path": "Affiliation.status", "operator": "=", "value": "INACTIVE"},
      {"field_path": "Specialty.name", "operator": "=", "value": "Oncology"},
      {"field_path": "Facility.location.city", "operator": "=", "value": "Seattle"},
      {"field_path": "Facility.type", "operator": "=", "value": "CLINIC"}
    ]}

    Question: List providers accepting Cigna Choice plan in Texas
    Response:
    {"filters": [
      {"field_path": "Facility.location.state", "operator": "=", "value": "TX"},
      {"field_path": "Facility.plansAccepted.name", "operator": "=", "value": "Cigna Choice"}
    ]}
    """
)

MAX_ATTEMPTS = 3


class AgentState(TypedDict, total=False):
    question: str
    schema: Dict[str, Any]
    mentions: List[Dict[str, Any]]
    filters: List[Dict[str, Any]]
    invalid_paths: List[str]
    feedback: str
    attempts: int
    raw_response: str
    force_retry: bool


_GRAPH = None


def _format_schema_fields(schema: Dict[str, Any]) -> str:
    types = schema.get("types", {})
    lines = ["Schema field reference (auto-generated):"]
    for type_name in sorted(types):
        field_names = sorted(types[type_name].get("fields", {}).keys())
        if not field_names:
            continue
        dotted = ", ".join(f"{type_name}.{fname}" for fname in field_names)
        lines.append(f"- {dotted}")
    lines.append("Use only these exact Type.field combinations or others explicitly shown in the schema JSON payload.")
    return "\n".join(lines)


def _strip_markdown_fences(raw: str) -> str:
    text = raw.strip()
    if not text.startswith("```"):
        return text

    newline_idx = text.find("\n")
    if newline_idx == -1:
        body = text.lstrip("`")
    else:
        body = text[newline_idx + 1 :]

    fence_idx = body.rfind("```")
    if fence_idx != -1:
        body = body[:fence_idx]

    return body.strip()


def _strip_inline_comments(raw: str) -> str:
    return re.sub(r"//.*?$", "", raw, flags=re.MULTILINE)


def _extract_json_block(raw: str) -> str | None:
    start = raw.find("{")
    if start == -1:
        return None
    depth = 0
    for idx in range(start, len(raw)):
        ch = raw[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return raw[start : idx + 1]
    return None


def _build_prompt(state: AgentState) -> str:
    schema_fields = _format_schema_fields(state["schema"])
    payload = {
        "question": state["question"],
        "schema": state["schema"],
        "mentions": state["mentions"],
        "invalid_paths": state.get("invalid_paths", []),
        "feedback": state.get("feedback", ""),
    }
    return (
        f"{PROMPT_HEADER}\n{INSTRUCTIONS}\n{schema_fields}\n{FEW_SHOTS}\n###PAYLOAD###\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )


def _parse_response(raw: str) -> Dict[str, Any]:
    candidates = [raw]
    fenced = _strip_markdown_fences(raw)
    if fenced not in candidates:
        candidates.append(fenced)
    uncommented = _strip_inline_comments(fenced)
    if uncommented not in candidates:
        candidates.append(uncommented)
    json_block = _extract_json_block(uncommented)
    if json_block and json_block not in candidates:
        candidates.append(json_block)

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:  # pragma: no cover - agent formatting edge cases
            last_error = exc

    raise ValueError(f"Agent response was not JSON: {raw}") from last_error


def _field_path_exists(field_path: str, schema: Dict[str, Any]) -> bool:
    parts = field_path.split(".")
    if len(parts) < 2:
        return False
    types = schema.get("types", {})
    current_type = parts[0]
    if current_type not in types:
        return False

    fields = types[current_type].get("fields", {})
    for name in parts[1:]:
        if name not in fields:
            return False
        field_type = fields[name]
        base_type = field_type[:-2] if field_type.endswith("[]") else field_type
        if base_type in types:
            fields = types[base_type].get("fields", {})
        else:
            fields = {}
    return True


def _generate_filters(state: AgentState) -> AgentState:
    prompt = _build_prompt(state)
    raw = run_ollama(prompt)
    attempts = state.get("attempts", 0) + 1

    try:
        parsed = _parse_response(raw)
        return {
            **state,
            "filters": parsed.get("filters", []),
            "raw_response": raw,
            "attempts": attempts,
            "feedback": "",
            "force_retry": False,
        }
    except ValueError:
        # Force another iteration with explicit feedback so the model removes chatter.
        return {
            **state,
            "filters": [],
            "raw_response": raw,
            "attempts": attempts,
            "invalid_paths": ["<non-json-response>"] if not state.get("force_retry") else state.get("invalid_paths", []),
            "feedback": "Your previous response was not valid JSON. Respond with ONLY JSON matching the schema instructions.",
            "force_retry": True,
        }


def _validate_filters(state: AgentState) -> AgentState:
    if state.get("force_retry"):
        return state

    schema = state["schema"]
    filters = state.get("filters", [])
    invalid = []
    cleaned: List[Dict[str, Any]] = []

    for f in filters:
        fp = f.get("field_path")
        val = f.get("value")
        if not fp or val is None or not _field_path_exists(fp, schema):
            invalid.append(fp or "<missing field_path>")
            continue
        cleaned.append({
            "field_path": fp,
            "operator": f.get("operator", "="),
            "value": val,
        })

    new_state: AgentState = {
        **state,
        "filters": cleaned,
        "invalid_paths": invalid,
        "feedback": "" if not invalid else (
            "Invalid field_path values: "
            + ", ".join(sorted(set(invalid)))
            + ". Use only Type.field paths listed in the schema reference."
        ),
    }
    return new_state


def _validation_router(state: AgentState) -> str:
    if state.get("force_retry") and state.get("attempts", 0) < MAX_ATTEMPTS:
        return "retry"
    if state.get("invalid_paths") and state.get("attempts", 0) < MAX_ATTEMPTS:
        return "retry"
    return "complete"


def _build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("generate", _generate_filters)
    builder.add_node("validate", _validate_filters)
    builder.set_entry_point("generate")
    builder.add_edge("generate", "validate")
    builder.add_conditional_edges(
        "validate",
        _validation_router,
        {"retry": "generate", "complete": END},
    )
    return builder.compile()


def _get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = _build_graph()
    return _GRAPH


def infer_filters(text: str, schema: Dict[str, Any], mentions: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    graph = _get_graph()
    initial_state: AgentState = {
        "question": text,
        "schema": schema,
        "mentions": mentions,
        "filters": [],
        "invalid_paths": [],
        "feedback": "",
        "attempts": 0,
        "force_retry": False,
    }

    result: AgentState = graph.invoke(initial_state)

    if result.get("invalid_paths"):
        invalid = ", ".join(sorted(set(result["invalid_paths"])))
        raise ValueError(
            f"Agent could not generate schema-valid filters after {MAX_ATTEMPTS} attempts: {invalid}"
        )

    return result.get("filters", [])
