class Dependency(dict):
    source: str
    calls: list[str]
