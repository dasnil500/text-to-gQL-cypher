import json
import re
from typing import Dict, List, Sequence, Tuple

_OPERATOR_MAP = {
    'eq': '=',
    'ne': '<>',
    'neq': '<>',
    'gt': '>',
    'lt': '<',
    'gte': '>=',
    'lte': '<=',
    'in': 'IN'
}


class CypherBuilder:
    """Build Cypher MATCH/WHERE clauses deterministically."""

    def __init__(self, root_type: str, schema: Dict):
        self.schema = schema
        self.types = schema.get('types', {})
        if root_type not in self.types:
            raise ValueError(f"Unknown root type '{root_type}' in schema")
        self.alias_map: Dict[Tuple[str, ...], Tuple[str, str]] = {(): ('root', root_type)}
        self.alias_counter = 1
        self.match_clauses: List[str] = []
        self.conditions: List[str] = []

    def add_condition(self, path_fields: Sequence[str], operator: str, value):
        if not path_fields:
            return
        relation_fields = tuple(path_fields[:-1])
        property_field = path_fields[-1]
        alias, node_type = self._ensure_path(relation_fields)
        actual_field, field_type = _resolve_field(node_type, property_field, self.types)
        if field_type in self.types:
            raise ValueError(f"Field '{actual_field}' on '{node_type}' resolves to object; expected scalar")
        cypher_op = _OPERATOR_MAP.get(operator.lower(), '=')
        formatted_value = _format_value(value)
        condition = self._render_condition(alias, actual_field, cypher_op, formatted_value)
        self.conditions.append(condition)

    def build(self, select_fields: Sequence[str]) -> str:
        root_alias, root_type = self.alias_map[()]
        clauses = [f"MATCH ({root_alias}:{root_type})"]
        clauses.extend(self.match_clauses)
        if self.conditions:
            clauses.append("WHERE " + " AND ".join(self.conditions))
        clauses.append(_render_return_clause(select_fields, root_alias, root_type, self.types))
        return "\n".join(clauses)

    def _render_condition(self, alias: str, field: str, operator: str, value_literal: str) -> str:
        if operator == 'IN' and not value_literal.startswith('['):
            # ensure list literal for IN operations
            value_literal = f"[{value_literal}]"
        return f"{alias}.{field} {operator} {value_literal}"

    def _ensure_path(self, relation_fields: Tuple[str, ...]) -> Tuple[str, str]:
        if relation_fields in self.alias_map:
            return self.alias_map[relation_fields]
        if not relation_fields:
            return self.alias_map[()]
        parent_tuple = relation_fields[:-1]
        parent_alias, parent_type = self._ensure_path(parent_tuple)
        field_name = relation_fields[-1]
        actual_field, field_type = _resolve_field(parent_type, field_name, self.types)
        base_type = _base_type(field_type)
        if base_type not in self.types:
            raise ValueError(f"Field '{actual_field}' on '{parent_type}' does not reference another type")
        alias = f"n{self.alias_counter}"
        self.alias_counter += 1
        rel_label = _relationship_label(parent_type, actual_field)
        self.match_clauses.append(
            f"MATCH ({parent_alias}:{parent_type})-[:{rel_label}]->({alias}:{base_type})"
        )
        self.alias_map[relation_fields] = (alias, base_type)
        return alias, base_type


def graphql_to_cypher(graphql_query: str, schema: Dict) -> str:
    """Convert deterministic GraphQL query (from compile_graphql) into Cypher."""
    query = graphql_query.strip()
    if not query:
        raise ValueError('GraphQL query is empty')
    root_query = _extract_root_query(query)
    root_type = _infer_root_type(root_query, schema)

    where_block, where_end = _extract_block(query, 'where')
    selection_block, _ = _extract_selection_block(query, where_end)

    filters = _parse_filters(where_block)
    select_fields = _parse_selection_fields(selection_block)

    builder = CypherBuilder(root_type, schema)
    for path_fields, operator, value in filters:
        builder.add_condition(path_fields, operator, value)
    return builder.build(select_fields)


def _parse_filters(where_block: str) -> List[Tuple[List[str], str, object]]:
    filters: List[Tuple[List[str], str, object]] = []
    for raw_line in where_block.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parsed = _parse_filter_line(line)
        if parsed:
            filters.append(parsed)
    return filters


def _parse_filter_line(line: str) -> Tuple[List[str], str, object]:
    # capture nested field names ending with operator token, e.g., affiliations, facility, eq
    tokens = re.findall(r'([A-Za-z0-9_]+)\s*:', line)
    if len(tokens) < 2:
        raise ValueError(f"Unable to parse filter line: {line}")
    operator = tokens[-1]
    path_fields = tokens[:-1]
    value_segment = line.split(f"{operator}:", 1)[1].strip()
    value_raw = value_segment.split('}', 1)[0].strip().rstrip(',')
    value = _parse_value(value_raw)
    return path_fields, operator, value


def _parse_value(value_raw: str):
    try:
        return json.loads(value_raw)
    except json.JSONDecodeError:
        lowered = value_raw.lower()
        if lowered in {'true', 'false'}:
            return lowered == 'true'
        return value_raw.strip('"')


def _parse_selection_fields(selection_block: str) -> List[str]:
    fields = []
    for raw_line in selection_block.splitlines():
        field = raw_line.strip()
        if not field:
            continue
        fields.append(field)
    return fields


def _extract_root_query(query: str) -> str:
    match = re.search(r'query\s*\{\s*(\w+)\s*\(', query)
    if not match:
        raise ValueError('Unable to locate root query in GraphQL input')
    return match.group(1)


def _infer_root_type(root_query: str, schema: Dict) -> str:
    types = schema.get('types', {})
    for type_name in types:
        if root_query.lower().startswith(type_name.lower()):
            return type_name
    if not types:
        raise ValueError('Schema contains no types')
    return next(iter(types))


def _extract_block(query: str, keyword: str) -> Tuple[str, int]:
    idx = query.find(keyword)
    if idx == -1:
        raise ValueError(f"Unable to find '{keyword}' block in GraphQL query")
    brace_idx = query.find('{', idx)
    if brace_idx == -1:
        raise ValueError(f"Missing opening '{{' after '{keyword}'")
    return _extract_braced_block(query, brace_idx)


def _extract_selection_block(query: str, start_idx: int) -> Tuple[str, int]:
    brace_idx = query.find('{', start_idx)
    if brace_idx == -1:
        raise ValueError('Missing selection block in GraphQL query')
    return _extract_braced_block(query, brace_idx)


def _extract_braced_block(text: str, open_idx: int) -> Tuple[str, int]:
    if text[open_idx] != '{':
        raise ValueError('Expected brace at provided index')
    depth = 1
    i = open_idx + 1
    start = i
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
        i += 1
    if depth != 0:
        raise ValueError('Unbalanced braces in GraphQL block')
    return text[start:i - 1], i


def _resolve_field(type_name: str, field_name: str, types: Dict) -> Tuple[str, str]:
    type_info = types.get(type_name, {})
    fields = type_info.get('fields', {})
    for actual, ftype in fields.items():
        if actual.lower() == field_name.lower():
            return actual, ftype
    raise ValueError(f"Field '{field_name}' not found on type '{type_name}'")


def _base_type(field_type: str) -> str:
    if field_type.endswith('[]'):
        return field_type[:-2]
    return field_type


def _relationship_label(parent_type: str, field_name: str) -> str:
    return f"{_normalize_label(parent_type)}_{_normalize_label(field_name)}"


def _normalize_label(name: str) -> str:
    snake = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name)
    return snake.upper()


def _render_return_clause(select_fields: Sequence[str], root_alias: str, root_type: str, types: Dict) -> str:
    if not select_fields:
        return f"RETURN DISTINCT {root_alias}"
    resolved_fields = []
    for field in select_fields:
        actual_field, _ = _resolve_field(root_type, field, types)
        resolved_fields.append(f"{root_alias}.{actual_field} AS {actual_field}")
    return "RETURN DISTINCT " + ", ".join(resolved_fields)


def _format_value(value) -> str:
    if isinstance(value, str):
        escaped = value.replace("'", "\\'")
        return f"'{escaped}'"
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if value is None:
        return 'null'
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        inner = ", ".join(_format_value(v) for v in value)
        return f"[{inner}]"
    return f"'{str(value)}'"
