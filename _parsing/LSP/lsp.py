import os
import json
import subprocess
import io


def make_request(method: str, params: object):
    obj = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    string = json.dumps(obj)
    return f"Content-Length: {len(string)}\r\nContent-Type: application/vscode-jsonrpc; charset=utf8\r\n\r\n{string}"


def symbol_request(uri: str):
    return make_request(
        "textDocument/documentSymbol",
        {"textDocument": {"uri": uri}},
    )


def initialize_request(root_path: str):
    return make_request("initialize", {"rootPath": root_path})


def submit_request(formatted_req: str):
    """
    Submit the formatted request to fortls. Returns an array of responses, since formatted_req may contain multiple requests.
    """
    command = [
        "fortls",
        "--incl_suffixes",
        ".f90",
        "--enable_code_actions",
        "--incremental_sync",
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = process.communicate(input=formatted_req)

    if process.returncode == 0:
        buf = io.StringIO(stdout)
        n = len(buf.getvalue())

        arr = []

        while buf.tell() < n:
            content_length = int(buf.readline().split(":")[1])
            _content_type = buf.readline()
            _empty_line = buf.readline()
            j = buf.read(content_length)
            obj = json.loads(j)
            arr.append(obj)

        return arr
    else:
        raise ChildProcessError(stderr)


def get_document_symbols(root_path: str, uri: str):
    initialize = initialize_request(root_path)
    symbols = symbol_request(uri)
    req = initialize + symbols

    res = submit_request(req)
    symbol_response = res[1]

    return symbol_response['result']


if __name__ == "__main__":
    symbols = get_document_symbols(
        root_path="/Users/anthony/Documents/climate_code_conversion/dependency_graphs/source",
        uri="/Users/anthony/Documents/climate_code_conversion/dependency_graphs/source/client.f90",
    )

    import pprint

    pprint.pprint(symbols)
