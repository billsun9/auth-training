\
from dataclasses import dataclass, asdict
from typing import Dict, Any

@dataclass
class Example:
    id: str
    split: str
    regime: str
    source: str
    authorized: bool
    style: str
    mechanism: str
    domain: str
    candidate_action: str
    prompt: str
    target: Dict[str, Any]
    metadata: Dict[str, Any]

    def to_dict(self):
        return asdict(self)
