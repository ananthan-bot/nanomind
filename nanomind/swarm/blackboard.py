"""
nanomind/swarm/blackboard.py — Shared Blackboard architecture for collaborative multi-agent problem solving.
"""
from typing import Dict, Any, List, Optional, Callable
import time


class Blackboard:
    """
    Shared working memory blackboard accessible to all agents in the swarm.
    Maintains structured problem state, solution candidates, artifacts, and change history.
    """

    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._history: List[Dict[str, Any]] = []
        self._watchers: Dict[str, List[Callable[[str, Any, Any], None]]] = {}

    def write(self, key: str, value: Any, author_agent_id: str):
        """Write or update entry on blackboard and notify watchers."""
        old_value = self._data.get(key)
        self._data[key] = value

        event = {
            "timestamp": time.time(),
            "author": author_agent_id,
            "key": key,
            "old_value": old_value,
            "new_value": value,
        }
        self._history.append(event)

        if key in self._watchers:
            for watcher in self._watchers[key]:
                watcher(key, old_value, value)

    def read(self, key: str, default: Any = None) -> Any:
        """Read value from blackboard."""
        return self._data.get(key, default)

    def watch(self, key: str, callback: Callable[[str, Any, Any], None]):
        """Subscribe to modifications of a specific blackboard key."""
        if key not in self._watchers:
            self._watchers[key] = []
        self._watchers[key].append(callback)

    def snapshot(self) -> Dict[str, Any]:
        """Get copy of entire current blackboard state."""
        return dict(self._data)

    def get_history(self) -> List[Dict[str, Any]]:
        return list(self._history)
