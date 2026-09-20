"""
nanomind/gnn/dataflow.py — Data flow analysis for code graphs.

Data flow analysis tracks how variable values propagate through code.
Used for:
  - Variable misuse detection (VarMisuse task, Allamanis et al., 2018)
  - Null pointer dereference detection
  - Type inference

Reaching Definitions:
  A definition d reaches point p if there exists a path from d to p
  in the CFG along which d is not redefined.

This module adds DATA_FLOW edges to AST graphs based on simple
variable def-use analysis within Python code.

Reference:
  Allamanis et al. (2018) "Learning to Represent Programs with Graphs"
  https://arxiv.org/abs/1711.00740
"""

from __future__ import annotations
import ast
import torch
from nanomind.gnn.graph import CodeGraph, EDGE_DATA_FLOW


class DataFlowAnalyser:
    """
    Analyse data flow in Python code and add def-use edges.

    Finds all variable definitions and uses in a function.
    Adds EDGE_DATA_FLOW edges from definition site to use sites.

    Args:
        parser: :class:`ASTParser` to get base graph.

    Example::

        analyser = DataFlowAnalyser()
        graph    = analyser.analyse("def f(x):\n  y = x + 1\n  return y")
        # graph now has DATA_FLOW edges: x_def→x_use, y_def→y_use
    """

    def __init__(self) -> None:
        self._defs: dict[str, list[int]] = {}   # var_name → [node_ids]
        self._uses: dict[str, list[int]] = {}

    def _collect_defs_uses(self, tree, node_map: dict) -> None:
        """Collect definition and use sites for each variable."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        nid = node_map.get(id(target))
                        if nid is not None:
                            self._defs.setdefault(target.id, []).append(nid)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                nid = node_map.get(id(node))
                if nid is not None:
                    self._uses.setdefault(node.id, []).append(nid)

    def analyse(self, source: str) -> CodeGraph:
        """
        Parse source and add data flow edges.

        Returns:
            :class:`CodeGraph` with DATA_FLOW edges added.
        """
        from nanomind.gnn.ast_parser import ASTParser
        parser   = ASTParser()
        graph    = parser.parse(source)

        # Re-parse to get node map
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return graph

        self._defs.clear()
        self._uses.clear()

        # Build a node_map by traversal order
        nodes = list(ast.walk(tree))
        node_map = {id(n): i for i, n in enumerate(nodes)}
        N_ast = len(nodes)

        self._collect_defs_uses(tree, node_map)

        # Build data flow edges
        extra_src, extra_dst, extra_types = [], [], []
        for var, def_ids in self._defs.items():
            for def_nid in def_ids:
                for use_nid in self._uses.get(var, []):
                    if def_nid < graph.n_nodes and use_nid < graph.n_nodes:
                        extra_src.append(def_nid)
                        extra_dst.append(use_nid)
                        extra_types.append(EDGE_DATA_FLOW)

        if not extra_src:
            return graph

        ei  = graph.edge_index
        et  = graph.edge_types
        new_ei = torch.tensor([extra_src, extra_dst], dtype=torch.long)
        new_et = torch.tensor(extra_types, dtype=torch.long)

        combined_ei = torch.cat([ei, new_ei], dim=1) if ei.shape[1] > 0 else new_ei
        combined_et = torch.cat([et, new_et]) if et is not None else new_et

        return CodeGraph(
            node_features = graph.node_features,
            edge_index    = combined_ei,
            edge_types    = combined_et,
            node_labels   = graph.node_labels,
            source        = source,
        )

    def def_use_pairs(self) -> dict[str, dict]:
        """Return def-use summary per variable."""
        return {
            var: {"n_defs": len(self._defs.get(var, [])),
                  "n_uses": len(self._uses.get(var, []))}
            for var in set(list(self._defs) + list(self._uses))
        }
