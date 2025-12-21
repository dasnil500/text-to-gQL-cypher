STATE_NAME_MAP = {
    'california': 'CA',
    'new york': 'NY'
}

STATE_CODE_MAP = {
    'ca': 'CA',
    'ny': 'NY'
}

KNOWN_STATE_CODES = set(STATE_NAME_MAP.values()) | set(STATE_CODE_MAP.values())


def normalize_state(text):
    t = text.strip().lower()
    if not t:
        return ''

    if t in STATE_NAME_MAP:
        return STATE_NAME_MAP[t]
    if t in STATE_CODE_MAP:
        return STATE_CODE_MAP[t]

    # check if any multi-word state name appears inside the phrase (e.g., "Los Angeles California")
    for name, code in STATE_NAME_MAP.items():
        if name in t:
            return code

    # otherwise, scan tokens for two-letter codes
    for token in t.replace(',', ' ').split():
        if token in STATE_CODE_MAP:
            return STATE_CODE_MAP[token]

    return text.upper()


def normalize_status(text):
    t = text.strip().lower()
    if t in ('active', 'in network', 'participating'):
        return 'ACTIVE'
    return text.upper()
