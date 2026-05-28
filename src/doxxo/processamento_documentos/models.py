from typing import Dict, Any
from dataclasses import dataclass
from uuid import uuid4

@dataclass
class Fragmento:
    page_content: str
    metadata: Dict[str, Any]
    id: str = None

    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid4())

        self.metadata['id'] = self.id