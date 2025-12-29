"""Run the 30-query evaluation with spaCy NER disabled."""
import os
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    os.environ.setdefault("OLLAMA_MODEL", "gemma3:1b")
    os.environ.pop("MOCK_OLLAMA", None)
    os.environ["ABLATION_DISABLE_NER"] = "1"
    os.environ.pop("ABLATION_DISABLE_VALIDATION", None)
    os.environ.pop("ABLATION_DISABLE_GRAPHQL", None)
    runpy.run_path("tests/run_eval_30.py", run_name="__main__")


if __name__ == "__main__":
    main()
