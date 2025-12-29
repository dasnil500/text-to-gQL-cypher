import json
import os
import shutil
import subprocess
from typing import Any, Dict

_DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "gpt-oss:20b")
_MOCK_RESPONSES = {
    "active cardiology": [
        {"field_path": "Affiliation.status", "operator": "=", "value": "ACTIVE"},
        {"field_path": "Specialty.name", "operator": "=", "value": "Cardiology"},
        {"field_path": "Facility.location.city", "operator": "=", "value": "Los Angeles"},
        {"field_path": "Facility.type", "operator": "=", "value": "HOSPITAL"},
        {"field_path": "Facility.plansAccepted.name", "operator": "=", "value": "Blue Shield PPO"},
    ],
    "inactive oncology": [
        {"field_path": "Affiliation.status", "operator": "=", "value": "INACTIVE"},
        {"field_path": "Specialty.name", "operator": "=", "value": "Oncology"},
        {"field_path": "Facility.location.city", "operator": "=", "value": "Seattle"},
        {"field_path": "Facility.type", "operator": "=", "value": "CLINIC"},
    ],
    "cigna choice": [
        {"field_path": "Facility.location.state", "operator": "=", "value": "TX"},
        {"field_path": "Facility.plansAccepted.name", "operator": "=", "value": "Cigna Choice"},
    ],
    "open appointments": [
        {"field_path": "Facility.location.city", "operator": "=", "value": "Austin"},
        {"field_path": "Facility.type", "operator": "=", "value": "URGENT_CARE"},
        {"field_path": "Appointment.availabilityStatus", "operator": "=", "value": "OPEN"},
    ],
}


class OllamaUnavailableError(RuntimeError):
    pass


def _extract_question(prompt: str) -> str:
    marker = "###PAYLOAD###"
    if marker not in prompt:
        return ""
    payload_str = prompt.split(marker, 1)[-1].strip()
    try:
        data = json.loads(payload_str)
        return data.get("question", "")
    except json.JSONDecodeError:
        return ""


def _mock_completion(prompt: str) -> str:
    question = _extract_question(prompt).lower()
    payload = None
    for key, filters in _MOCK_RESPONSES.items():
        if key in question:
            payload = filters
            break
    if payload is None:
        payload = _MOCK_RESPONSES.get("active cardiology")
    return json.dumps({"filters": payload})


def run_ollama(prompt: str, model: str = _DEFAULT_MODEL, timeout: int = None) -> str:
    if os.environ.get("MOCK_OLLAMA") == "1":
        return _mock_completion(prompt)

    executable = shutil.which("ollama")
    if executable is None:
        raise OllamaUnavailableError("Ollama CLI not found on PATH. Install Ollama or set MOCK_OLLAMA=1 for dry runs.")

    proc = subprocess.run(
        [executable, "run", model],
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=True,
    )
    output = proc.stdout.decode("utf-8").strip()
    return output
