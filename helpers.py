import json
from typing import (List, Dict, Any)


def read_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        data = json.load(f)
    return data