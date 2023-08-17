import re
import lsp


def get_module_names_from_use_statements(uri: str):
    """
    There must be a way to write this more readably...
    """

    use_pattern = re.compile(
        r"\buse\s+([a-zA-Z_]\w*)(?:\s*,\s*only\s*:\s*(\w+(?:\s*,\s*\w+)*))?"
    )
    result = []

    with open(uri) as f:
        lines = f.readlines()
        concatenated_line = ""
        start_line_number = None

        for i, line in enumerate(lines):
            # Handle line continuation
            if line.strip().endswith("&"):
                if not concatenated_line:
                    start_line_number = i
                concatenated_line += line.rstrip("& \n")
                continue
            else:
                concatenated_line += line

            # Ignore lines that start with a comment
            if not concatenated_line.strip().startswith("!"):
                match = use_pattern.search(concatenated_line)
                if match:
                    module_name = match.group(1)
                    char_number = match.start(1)
                    only_names = match.group(2)
                    only_list = (
                        [name.strip() for name in only_names.split(",")]
                        if only_names
                        else []
                    )
                    result.append(
                        {
                            "name": module_name,
                            "range": {
                                "start": {
                                    "line": start_line_number
                                    if start_line_number is not None
                                    else i,
                                    "character": char_number,
                                },
                                "end": {
                                    "line": start_line_number
                                    if start_line_number is not None
                                    else i,
                                    "character": char_number + len(module_name),
                                },
                            },
                            "only": only_list,
                        }
                    )

            # Reset concatenated line and line numbers
            concatenated_line = ""
            start_line_number = None

    return result


def find_module(root_path: str, uri: str, module: dict):
    """
    Use the fortls API to find the definition for a module.
    """
    init_request = lsp.initialize_request(root_path)
    def_request = lsp.make_request(
        "textDocument/definition",
        {
            "textDocument": {"uri": uri},
            "position": {
                "line": module["range"]["start"]["line"],
                "character": module["range"]["start"]["character"] + 1,
            },
        },
    )

    response = lsp.submit_request(init_request + def_request)[1]

    return response["result"]


def get_module_sources(root_path: str, uri: str):
    modules = get_module_names_from_use_statements(uri)
    modules = [
        {**module, "definition": find_module(root_path, uri, module)}
        for module in modules
    ]
    return modules


if __name__ == "__main__":
    root_path = (
        "/Users/anthony/Documents/climate_code_conversion/dependency_graphs/source"
    )
    uri = "/Users/anthony/Documents/climate_code_conversion/dependency_graphs/source/client.f90"

    modules = get_module_sources(root_path, uri)

    import pprint

    pprint.pprint(modules)
