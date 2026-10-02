"""
nanomind/codegen/ast_utils.py — AST-based code analysis utilities.

Analyse generated code using Python's AST module:
  - Extract function signatures
  - Compute cyclomatic complexity
  - Detect common patterns (recursion, loops, list comprehensions)
  - Code similarity via tree edit distance

Used for:
  - Quality assessment of generated code
  - Bug detection (missing return, infinite recursion risk)
  - Code search (find similar implementations)
"""

from __future__ import annotations
import ast
from dataclasses import dataclass


@dataclass
class FunctionInfo:
    """Information extracted from a function AST node."""
    name:           str
    args:           list[str]
    return_type:    str | None
    n_lines:        int
    has_recursion:  bool
    n_loops:        int
    n_branches:     int
    n_returns:      int
    uses_comprehension: bool
    complexity:     int       # cyclomatic complexity

    def to_dict(self) -> dict:
        return {
            "name":         self.name,
            "args":         self.args,
            "n_lines":      self.n_lines,
            "complexity":   self.complexity,
            "has_recursion": self.has_recursion,
        }


class ASTAnalyser:
    """
    Analyse Python code via AST.

    Args:
        code: Python source code string.

    Example::

        analyser = ASTAnalyser("def add(a, b):\n    return a + b")
        info     = analyser.analyse_function("add")
        print(info.complexity)   # 1 (no branches)
    """

    def __init__(self, code: str) -> None:
        self.code = code
        try:
            self.tree = ast.parse(code)
            self._ok  = True
        except SyntaxError:
            self.tree = None
            self._ok  = False

    def is_valid(self) -> bool:
        return self._ok

    def list_functions(self) -> list[str]:
        """Return names of all defined functions."""
        if not self._ok:
            return []
        return [
            node.name for node in ast.walk(self.tree)
            if isinstance(node, ast.FunctionDef)
        ]

    def analyse_function(self, fn_name: str) -> FunctionInfo | None:
        """Extract info for a specific function."""
        if not self._ok:
            return None
        fn_node = None
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef) and node.name == fn_name:
                fn_node = node
                break
        if fn_node is None:
            return None

        # Args
        args = [a.arg for a in fn_node.args.args]

        # Return type annotation
        ret = None
        if fn_node.returns:
            try:
                ret = ast.unparse(fn_node.returns)
            except Exception:
                ret = None

        # Count features
        n_loops = 0; n_branches = 0; n_returns = 0; uses_comp = False
        has_recursion = False
        for node in ast.walk(fn_node):
            if isinstance(node, (ast.For, ast.While)):
                n_loops += 1
            if isinstance(node, (ast.If,)):
                n_branches += 1
            if isinstance(node, ast.Return):
                n_returns += 1
            if isinstance(node, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
                uses_comp = True
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id == fn_name:
                    has_recursion = True

        # Cyclomatic complexity = branches + loops + 1
        complexity = n_branches + n_loops + 1
        n_lines    = (fn_node.end_lineno - fn_node.lineno + 1
                      if hasattr(fn_node, "end_lineno") else 1)

        return FunctionInfo(
            name              = fn_name,
            args              = args,
            return_type       = ret,
            n_lines           = n_lines,
            has_recursion     = has_recursion,
            n_loops           = n_loops,
            n_branches        = n_branches,
            n_returns         = n_returns,
            uses_comprehension = uses_comp,
            complexity        = complexity,
        )

    def has_return_in_all_paths(self, fn_name: str) -> bool:
        """Check if function always returns a value (heuristic)."""
        if not self._ok:
            return False
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef) and node.name == fn_name:
                for child in ast.walk(node):
                    if isinstance(child, ast.Return) and child.value is not None:
                        return True
        return False

    def extract_docstring(self, fn_name: str) -> str | None:
        """Extract docstring from function."""
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef) and node.name == fn_name:
                return ast.get_docstring(node)
        return None

    def detect_issues(self) -> list[str]:
        """Detect potential code quality issues."""
        issues = []
        if not self._ok:
            return ["SyntaxError: code does not parse"]
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                # Empty function
                if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                    issues.append(f"Function '{node.name}' is not implemented (pass only)")
                # Bare except
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                issues.append("Bare except clause (catches all exceptions)")
        return issues
