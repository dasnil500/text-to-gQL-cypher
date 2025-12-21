from typing import List

import spacy
from spacy.language import Language


_SPACY_MODEL: Language | None = None


def _get_spacy_model() -> Language:
    global _SPACY_MODEL
    if _SPACY_MODEL is None:
        try:
            _SPACY_MODEL = spacy.load("en_core_web_trf")
        except OSError:
            _SPACY_MODEL = spacy.load("en_core_web_sm")
    return _SPACY_MODEL


def extract_mentions(text: str) -> List[dict]:
    """Return spaCy entity spans for downstream agentic reasoning."""
    model = _get_spacy_model()
    doc = model(text)
    mentions: List[dict] = []

    for ent in doc.ents:
        mentions.append(
            {
                'text': ent.text,
                'label': ent.label_,
                'span': (ent.start_char, ent.end_char),
            }
        )

    return mentions
