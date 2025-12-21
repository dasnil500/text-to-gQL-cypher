from collections import deque


def _render_value(v):
    # strings quoted
    if isinstance(v, str):
        return f'"{v}"'
    return str(v)


def _nest_field_parts(parts, value):
    # parts: ['affiliations','facility','location','state'] -> render nested GraphQL input
    if not parts:
        return ''
    head = parts[0]
    if len(parts) == 1:
        return f"{head}: {{ eq: { _render_value(value) } }}"
    return f"{head}: {{ { _nest_field_parts(parts[1:], value) } }}"


def _build_type_graph(schema):
    graph = {}
    types = schema.get('types', {})
    for t, info in types.items():
        edges = []
        for fname, ftype in info.get('fields', {}).items():
            base = ftype[:-2] if ftype.endswith('[]') else ftype
            if base in types:
                edges.append((fname, base))
        graph[t] = edges
    return graph


def _find_field_chain(root_type, target_type, graph):
    if root_type == target_type:
        return []
    q = deque([(root_type, [])])
    visited = set([root_type])
    while q:
        node, chain = q.popleft()
        if node == target_type:
            return chain
        for field_name, nxt in graph.get(node, []):
            if nxt in visited:
                continue
            visited.add(nxt)
            q.append((nxt, chain + [field_name]))
    return None


def _resolve_attribute_path(type_name, attr_parts, schema):
    if not attr_parts:
        return []
    types = schema.get('types', {})
    current_type = type_name
    resolved = []
    for idx, attr in enumerate(attr_parts):
        type_info = types.get(current_type)
        if not type_info:
            return None
        fields = type_info.get('fields', {})
        field_name = None
        for candidate in fields:
            if candidate.lower() == attr.lower():
                field_name = candidate
                break
        if not field_name:
            return None
        resolved.append(field_name)
        ftype = fields[field_name]
        base = ftype[:-2] if ftype.endswith('[]') else ftype
        if base in types:
            current_type = base
        else:
            if idx < len(attr_parts) - 1:
                return None
    return resolved


def compile_graphql(plan, schema):
    """Compile LogicalPlan into a GraphQL query string using the schema's root queries.
    This is deterministic and schema-driven but schema-agnostic in code.
    """
    root_type = plan.root
    # map type->root query name: try to find plural in root_queries that matches lowercased root
    root_query = None
    for q in schema.get('root_queries', []):
        if q.lower().startswith(root_type.lower()):
            root_query = q
            break
    if not root_query:
        # fallback to first root_query
        root_query = schema.get('root_queries', [])[0]

    graph = _build_type_graph(schema)

    # group filters by top-level type after root
    where_parts = []
    for f in plan.filters:
        # e.g., f.field_path = 'Affiliation.status' or 'Facility.location.state'
        parts = f.field_path.split('.')
        top = parts[0]
        rest = parts[1:]

        chain = _find_field_chain(root_type, top, graph)
        attr_path = _resolve_attribute_path(top, rest, schema)
        if chain is None or attr_path is None:
            continue

        nested_parts = chain + attr_path
        where_parts.append(_nest_field_parts(nested_parts, f.value))

    where_block = '\n    '.join(where_parts)
    select_block = '\n    '.join(plan.select)

    query = f"""
query {{
  {root_query}(where: {{
    {where_block}
  }}) {{
    {select_block}
  }}
}}
""".strip()

    return query
