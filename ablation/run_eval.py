"""Run the 30-query evaluation with the default (no ablation) stack."""
import os
import runpy
import importlib
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_SPACY_MODEL = "en_core_web_sm"
_LOCAL_WHEEL = ROOT / "en_core_web_sm-3.7.1-py3-none-any.whl"


def _ensure_spacy_model() -> None:
    """Install the bundled spaCy model wheel if the model is missing."""
    try:
        import spacy  # type: ignore

        spacy.load(_SPACY_MODEL)
        return
    except Exception:
        pass

    if not _LOCAL_WHEEL.exists():
        raise RuntimeError(
            f"Missing spaCy model '{_SPACY_MODEL}' and bundled wheel {_LOCAL_WHEEL.name} not found."
        )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--disable-pip-version-check",
            str(_LOCAL_WHEEL),
        ],
        check=True,
    )

    importlib.invalidate_caches()
    import spacy  # type: ignore  # re-import after installation

    try:
        spacy.load(_SPACY_MODEL)
    except OSError as exc:
        raise RuntimeError(
            f"Installed {_LOCAL_WHEEL.name} but spaCy still cannot load '{_SPACY_MODEL}'."
        ) from exc


def main() -> None:
    _ensure_spacy_model()
    os.environ.setdefault("OLLAMA_MODEL", "gemma3:1b")
    os.environ.pop("MOCK_OLLAMA", None)
    os.environ.pop("ABLATION_DISABLE_NER", None)
    os.environ.pop("ABLATION_DISABLE_VALIDATION", None)
    os.environ.pop("ABLATION_DISABLE_GRAPHQL", None)
    runpy.run_path("tests/run_eval_30.py", run_name="__main__")


if __name__ == "__main__":
    main()
