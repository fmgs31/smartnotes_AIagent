"""
Tool-result refiners: transparent one-shot follow-up calls that repair a tool
result before the planner sees it.

Server quirks that would otherwise force the planner into an extra planning
step (or a retry loop) are handled here, keeping the workaround logic out of
the planning loop and out of the prompts. The agent runs a single refinement
pass after each tool call: the first refiner that applies gets one follow-up
call to the same tool, and that follow-up's result replaces the original.
"""
import json
import logging
import re
from abc import ABC, abstractmethod

log = logging.getLogger("agent")


class ToolResultRefiner(ABC):
    """
    Inspects a finished tool call and, when it applies, supplies replacement
    arguments for ONE follow-up call to the same tool.

    Contract: refined arguments must not re-trigger `applies` — the agent does
    a single refinement pass, never a loop.
    """

    @abstractmethod
    def applies(self, tool_name: str, arguments: dict, result: str) -> bool:
        """True when this refiner should repair the given call's result."""

    @abstractmethod
    def refined_arguments(self, arguments: dict, result: str) -> dict | None:
        """Arguments for the follow-up call, or None to keep the original result."""


class TabAwareInspectRefiner(ToolResultRefiner):
    """
    google_workspace_mcp's inspect_doc_structure fetches the document with
    includeTabsContent=True, which leaves the legacy top-level body empty — so
    without a tab_id every doc (all docs are tab-based since Oct 2024) reports
    total_length=0 alongside a tabs list. A planner acting on that 0 loops on
    invalid index-0 writes.

    Repair: re-run the inspection against the first tab taken from the result
    itself, so the planner receives the real total_length in the same
    planning step. Never assumes a fixed tab id like 't.0'.
    """

    _TOOL = "inspect_doc_structure"

    def applies(self, tool_name: str, arguments: dict, result: str) -> bool:
        if tool_name != self._TOOL or arguments.get("tab_id"):
            return False
        payload = self._payload(result)
        if not payload:
            return False
        return payload.get("total_length") == 0 and bool(payload.get("tabs"))

    def refined_arguments(self, arguments: dict, result: str) -> dict | None:
        payload = self._payload(result) or {}
        tab_id = self._first_tab_id(payload.get("tabs", []))
        if not tab_id:
            return None
        return {**arguments, "tab_id": tab_id}

    @classmethod
    def _first_tab_id(cls, tabs: list) -> str | None:
        for tab in tabs:
            if tab.get("tab_id"):
                return tab["tab_id"]
            child = cls._first_tab_id(tab.get("child_tabs", []))
            if child:
                return child
        return None

    @staticmethod
    def _payload(result: str) -> dict | None:
        """Extract the JSON block embedded in the tool's text result."""
        start, end = result.find("{"), result.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            return json.loads(result[start:end + 1])
        except ValueError:
            return None

class InsertIndexBoundsRefiner(ToolResultRefiner):
    """
    inspect_doc_structure's total_length is the segment END index, but the
    Docs API requires an insertion index strictly BELOW it — so a
    modify_doc_text insertion at start_index=total_length always fails with
    "Index X must be less than the end index of the referenced segment, Y".

    Repair: retry once at Y-1 (taken from the API's own error, never computed
    from local assumptions) — the last valid insertion index, just before the
    document's final newline, i.e. the visual end of the doc.
    """

    _TOOL = "modify_doc_text"
    _BOUND_ERROR = re.compile(
        r"must be less than the end index of the referenced segment,\s*(\d+)"
    )

    def applies(self, tool_name: str, arguments: dict, result: str) -> bool:
        if tool_name != self._TOOL:
            return False
        if arguments.get("end_index") is not None:  # insertion only, never replacement
            return False
        match = self._BOUND_ERROR.search(result)
        if not match:
            return False
        last_valid = int(match.group(1)) - 1
        return last_valid >= 1 and arguments.get("start_index") != last_valid

    def refined_arguments(self, arguments: dict, result: str) -> dict | None:
        match = self._BOUND_ERROR.search(result)
        if not match:
            return None
        return {**arguments, "start_index": int(match.group(1)) - 1}


# Tuple becuase Defaults are evaluated once and shared across every instance. Mutable default (list)
# is the classic Python foot-gun: any code doing agent._tool_refiners.append(...) would
# silently mutate the default for ALL agents. A tuple is immutable, so the shared-default is safe.
# Also communicates intent: refiner set is fixed at construction, order matters (first applicable wins).
DEFAULT_REFINERS: tuple[ToolResultRefiner, ...] = (
    TabAwareInspectRefiner(),
    InsertIndexBoundsRefiner(),
)
