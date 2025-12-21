def validate_plan(plan, schema):
    """Validate that types/fields in the LogicalPlan exist in the schema."""
    errors = []

    types = schema.get("types", {})

    # validate root exists
    if plan.root not in types:
        errors.append(f"Unknown root type: {plan.root}")

    # validate joins
    for j in plan.joins:
        if j.from_type not in types:
            errors.append(f"Unknown join from type: {j.from_type}")
        if j.to_type not in types:
            errors.append(f"Unknown join to type: {j.to_type}")

    # validate filters
    for f in plan.filters:
        parts = f.field_path.split('.')
        t = parts[0]
        if t not in types:
            errors.append(f"Filter references unknown type: {t}")
        else:
            # walk fields
            cur = types[t]['fields']
            for p in parts[1:]:
                # arrays may be represented as Name[]
                if p not in cur:
                    errors.append(f"Unknown field '{p}' on type {t}")
                    break
                # get next type if complex
                ftype = cur[p]
                if ftype.endswith('[]'):
                    ftype = ftype[:-2]
                if ftype in types:
                    cur = types[ftype]['fields']
                else:
                    # primitive, stop walking
                    break

    return errors
