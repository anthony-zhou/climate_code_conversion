from io import TextIOWrapper
from . import parser


class DAG:
    def __init__(self, filename: str):
        self.filename = filename
        self.public_functions, self.dag = self._parse(filename)

    def _parse(self, filename: str):
        # parse the file, get back public functions and definitions
        public_functions = parser.get_public_functions(filename)
        dag = parser.get_dag(filename)

        return public_functions, dag

    def classify_dependencies(self, function_name: str):
        """
        Given a function, find its dependencies and return which ones are internal and external.
        """
        dependencies = parser.filter_for_dependencies(self.dag, function_name)

        external = [dep for dep in dependencies if dep[1]["source"] == "not_found"]

        internal = [dep for dep in dependencies if dep[1]["source"] != "not_found"]

        return external, internal
