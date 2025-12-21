from dataclasses import dataclass
from typing import List

@dataclass
class Filter:
    field_path: str
    operator: str
    value: str

@dataclass
class Join:
    from_type: str
    to_type: str
    via: str

@dataclass
class LogicalPlan:
    root: str
    joins: List[Join]
    filters: List[Filter]
    select: List[str]
