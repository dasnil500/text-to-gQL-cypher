import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MOCK_OLLAMA", "1")

from pipeline.run import process


SCENARIOS = [
    {
        'text': 'Find active cardiology providers in Los Angeles hospitals that accept Blue Shield',
        'expect': ['"ACTIVE"', '"Cardiology"', '"Los Angeles"', '"Blue Shield PPO"'],
        'expect_cypher': [
            '[:PROVIDER_AFFILIATIONS]',
            "n4.city = 'Los Angeles'",
            "n5.name = 'Blue Shield PPO'"
        ]
    },
    {
        'text': 'Show inactive oncology clinics in Seattle',
        'expect': ['"INACTIVE"', '"Oncology"', '"Seattle"', 'primaryFacility: { type'],
        'expect_cypher': [
            "n1.status = 'INACTIVE'",
            "n2.name = 'Oncology'",
            "n4.city = 'Seattle'"
        ]
    },
    {
        'text': 'List providers accepting Cigna Choice plan in Texas',
        'expect': ['"Cigna Choice"', 'state: { eq: "TX"'],
        'expect_cypher': [
            '[:FACILITY_PLANS_ACCEPTED]',
            "n3.name = 'Cigna Choice'",
            "n2.state = 'TX'"
        ]
    },
    {
        'text': 'Find providers with open appointments in Austin urgent care centers',
        'expect': ['appointments', '"OPEN"', '"Austin"', '"URGENT_CARE"'],
        'expect_cypher': [
            '[:PROVIDER_APPOINTMENTS]',
            "n3.availabilityStatus = 'OPEN'",
            "n2.city = 'Austin'"
        ]
    }
]


def main():
    ok = True
    unique_queries = set()

    for scenario in SCENARIOS:
        res = process(
            scenario['text'],
            schema_path='schema/schema.json'
        )
        query = res['query']
        cypher = res.get('cypher', '')
        unique_queries.add(query.strip())
        for snippet in scenario['expect']:
            if snippet not in query:
                print(f"FAIL: missing snippet '{snippet}' for query '{scenario['text']}'")
                ok = False
        for snippet in scenario.get('expect_cypher', []):
            if snippet not in cypher:
                print(f"FAIL: missing cypher snippet '{snippet}' for query '{scenario['text']}'")
                ok = False

    if len(unique_queries) < len(SCENARIOS):
        print('FAIL: multiple natural language queries collapsed into identical GraphQL output')
        ok = False

    if ok:
        print('OK: scenario checks passed')
        return 0

    return 2


if __name__ == '__main__':
    sys.exit(main())
