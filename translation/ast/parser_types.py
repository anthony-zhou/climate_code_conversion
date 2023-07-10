from typing import TypedDict

class Dependency(TypedDict):
    source: str
    calls: list[str]
