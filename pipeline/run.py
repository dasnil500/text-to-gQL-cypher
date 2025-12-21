import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nlp.ner import extract_mentions
from ir.logical_plan import LogicalPlan, Join, Filter
from ir.validator import validate_plan
from compiler.graphql_compiler import compile_graphql
from compiler.cypher_compiler import graphql_to_cypher
from agentic.schema_reasoner import infer_filters


def load_schema(path='schema/schema.json'):
    with open(path, 'r') as f:
        return json.load(f)


def select_defaults_for_root(root):
    if root == 'Provider':
        return ['providerId', 'name']
    return ['id']


def field_path_exists(field_path, schema):
    parts = field_path.split('.')
    types = schema.get('types', {})
    if not parts:
        return False
    base = parts[0]
    if base not in types:
        return False
    fields = types[base]['fields']
    for p in parts[1:]:
        if p not in fields:
            return False
        ftype = fields[p]
        if ftype.endswith('[]'):
            ftype = ftype[:-2]
        if ftype in types:
            fields = types[ftype]['fields']
        else:
            fields = {}
    return True


def build_plan_from_text(text, schema):
    mentions = extract_mentions(text)
    agent_filters = infer_filters(text, schema, mentions)

    # choose a root: prefer Provider if in schema
    root = 'Provider' if 'Provider' in schema.get('types', {}) else list(schema.get('types', {}).keys())[0]

    # build joins: build simple path from root to any linked types
    joins = []
    # naive: use schema relations directly to include typical joins
    relations = schema.get('relations', [])
    for a, b in relations:
        if a == root or any(j.from_type == a for j in joins):
            # pick via name as the lowercase of b
            joins.append(Join(a, b, b.lower()))

    # build filters from mentions using intent mapping
    filters = []
    rejected_filters = []
    filter_signatures = set()
    for f in agent_filters:
        fp = f.get('field_path')
        op = f.get('operator', '=')
        val = f.get('value')
        if not fp or val is None:
            continue
        if field_path_exists(fp, schema):
            signature = (fp, op, json.dumps(val, sort_keys=True))
            if signature in filter_signatures:
                continue
            filters.append(Filter(fp, op, val))
            filter_signatures.add(signature)
        else:
            rejected_filters.append(fp)

    select = select_defaults_for_root(root)

    return LogicalPlan(root=root, joins=joins, filters=filters, select=select), rejected_filters


def process(text, schema_path='schema/schema.json'):
    schema = load_schema(schema_path)
    plan, rejected_filters = build_plan_from_text(text, schema)
    errors = validate_plan(plan, schema)
    if rejected_filters:
        raise ValueError(f"Rejected attributes not in schema: {sorted(set(rejected_filters))}")
    if errors:
        raise ValueError(f"Logical plan validation failed: {errors}")
    query = compile_graphql(plan, schema)
    cypher = graphql_to_cypher(query, schema)
    return {
        'text': text,
        'plan': plan,
        'query': query,
        'cypher': cypher,
        'rejected_filters': rejected_filters
    }


if __name__ == '__main__':
    import sys
    text = ' '.join(sys.argv[1:]) or 'Find active providers in California hospitals'
    out = process(text)
    print(out['query'])
    print('\n--- Cypher ---')
    print(out['cypher'])
