from enum import StrEnum

from pydantic import BaseModel, Field, field_validator
from typing import Any, Dict


class EventType(StrEnum):
    STEP_PLANNED = "step_planned"
    RESULT = "result"
    ERROR = "error"
    TRANSCRIPT = "transcript"


class InboundRequest(BaseModel):
    text: str = Field(..., min_length=1)
    mcp_token: str = Field(..., min_length=1)
    user_guidelines: str | None = None
    planner_model: str | None = None


class NextStepPlan(BaseModel):
    done: bool
    last_step_check: str = ""
    next_tool: str = ""
    arguments: Dict[str, Any] = Field(default_factory=dict)
    explanation: str | None = None
    full_task_summary: str = ""
    items: list["DriveItem"] = Field(default_factory=list)

    @field_validator("arguments", mode="before")
    @classmethod
    def _coerce_empty_string(cls, v: Any) -> Any:
        if v == "" or v is None:
            return {}
        return v

    @field_validator("last_step_check", "next_tool", "full_task_summary", mode="before")
    @classmethod
    def _coerce_none_to_empty_string(cls, v: Any) -> Any:
        return "" if v is None else v


class ExecutedToolStep(BaseModel):
    step: int
    server_id: str
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    mcp_result: Any = None


class DriveItem(BaseModel):
    name: str
    id: str
    type: str
    location: str = "/"
    access_freq: int = 0


class ProcedureSummary(BaseModel):
    text: str
    items: list[DriveItem] = Field(default_factory=list)


class ErrorEvent(BaseModel):
    type: str = EventType.ERROR
    detail: str
