"""
nanomind/gnn/ast_parser.py — AST → CodeGraph converter.

Uses Python's built-in ``ast`` module to parse Python source code
into an Abstract Syntax Tree and converts it to a CodeGraph.

AST Node Types (sampled):
  Module, FunctionDef, AsyncFunctionDef, ClassDef
  Return, Delete, Assign, AugAssign, AnnAssign
  For, While, If, With, Try, ExceptHandler
  Import, ImportFrom, Global, Nonlocal, Expr, Pass, Break, Continue
  BoolOp, BinOp, UnaryOp, Lambda, IfExp, Dict, Set, ListComp
  Call, Attribute, Subscript, Name, Constant, List, Tuple

Each AST node becomes a graph node.
Edges:
  EDGE_AST_CHILD:   parent → child
  EDGE_AST_NEXT:    sibling → sibling (sequential order)
"""

from __future__ import annotations
import ast
import torch
from nanomind.gnn.graph import CodeGraph, EDGE_AST_CHILD, EDGE_AST_NEXT

# Node type vocabulary (62 types)
_AST_TYPES = [
    "Module", "FunctionDef", "AsyncFunctionDef", "ClassDef",
    "Return", "Delete", "Assign", "AugAssign", "AnnAssign",
    "For", "AsyncFor", "While", "If", "With", "AsyncWith",
    "Raise", "Try", "ExceptHandler", "Assert", "Import", "ImportFrom",
    "Global", "Nonlocal", "Expr", "Pass", "Break", "Continue",
    "BoolOp", "BinOp", "UnaryOp", "Lambda", "IfExp",
    "Dict", "Set", "ListComp", "SetComp", "DictComp", "GeneratorExp",
    "Await", "Yield", "YieldFrom", "Compare", "Call", "FormattedValue",
    "JoinedStr", "Constant", "Attribute", "Subscript", "Starred",
    "Name", "List", "Tuple", "Slice", "Load", "Store", "Del",
    "Add", "Sub", "Mult", "MatMult", "Div", "Mod", "Pow",
    "Unknown",
]
TYPE2ID = {t: i for i, t in enumerate(_AST_TYPES)}
N_NODE_TYPES = len(_AST_TYPES)


def _type_id(node) -> int:
    return TYPE2ID.get(type(node).__name__, TYPE2ID["Unknown"])


def _one_hot(idx: int, size: int) -> torch.Tensor:
    v = torch.zeros(size)
    v[idx] = 1.0
    return v


class ASTParser:
    """
    Parse Python source code into a :class:`CodeGraph`.

    Each AST node becomes a graph node with a one-hot feature vector
    encoding its node type. Edges connect parent→child and sibling→sibling.

    Args:
        add_sibling_edges: Add EDGE_AST_NEXT edges between siblings.

    Example::

        parser = ASTParser()
        graph  = parser.parse("def foo(x): return x + 1")
        print(graph.n_nodes, graph.n_edges)
    """

    def __init__(self, add_sibling_edges: bool = True) -> None:
        self.add_sibling_edges = add_sibling_edges

    def parse(self, source: str) -> CodeGraph:
        """
        Parse Python source into a CodeGraph.

        Args:
            source: Python source code string.

        Returns:
            :class:`CodeGraph`.
        """
        try:
            tree = ast.parse(source)
        except SyntaxError:
            # Return empty graph on parse failure
            return CodeGraph(
                node_features = torch.zeros(1, N_NODE_TYPES),
                edge_index    = torch.zeros(2, 0, dtype=torch.long),
                node_labels   = ["<error>"],
                source        = source,
            )

        nodes:       list[tuple] = []   # (node_id, type_id, label)
        edges_src:   list[int]   = []
        edges_dst:   list[int]   = []
        edge_types:  list[int]   = []
        node_map:    dict        = {}   # id(ast_node) → node_id

        def visit(node, parent_id: int | None = None):
            nid   = len(nodes)
            label = type(node).__name__
            nodes.append((nid, _type_id(node), label))
            node_map[id(node)] = nid

            if parent_id is not None:
                edges_src.append(parent_id)
                edges_dst.append(nid)
                edge_types.append(EDGE_AST_CHILD)

            children = list(ast.iter_child_nodes(node))
            prev_id  = None
            for child in children:
                visit(child, parent_id=nid)
                child_id = node_map[id(child)]
                if self.add_sibling_edges and prev_id is not None:
                    edges_src.append(prev_id)
                    edges_dst.append(child_id)
                    edge_types.append(EDGE_AST_NEXT)
                prev_id = child_id

        visit(tree)

        features = torch.stack([_one_hot(n[1], N_NODE_TYPES) for n in nodes])
        labels   = [n[2] for n in nodes]
        if edges_src:
            ei = torch.tensor([edges_src, edges_dst], dtype=torch.long)
            et = torch.tensor(edge_types, dtype=torch.long)
        else:
            ei = torch.zeros(2, 0, dtype=torch.long)
            et = torch.zeros(0, dtype=torch.long)

        return CodeGraph(
            node_features = features,
            edge_index    = ei,
            edge_types    = et,
            node_labels   = labels,
            source        = source,
        )

    def parse_function(self, func_source: str) -> CodeGraph:
        """Parse a single function definition."""
        return self.parse(func_source)
