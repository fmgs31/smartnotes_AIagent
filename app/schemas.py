from pydantic import BaseModel, Field
from typing import Any, Dict


class InboundRequest(BaseModel):
    text: str = Field(..., min_length=1)
    mcp_token: str = Field(..., min_length=1)
    user_extra_info: str | None = None
    planner_model: str | None = None


class NextStepPlan(BaseModel):
    done: bool
    tool_name: str = ""
    arguments: Dict[str, Any] = Field(default_factory=dict)
    explanation: str | None = None


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


class ProcedureSummary(BaseModel):
    text: str
    items: list[DriveItem] = Field(default_factory=list)


class OutboundResponse(BaseModel):
    summary: str = ""
    items: list[DriveItem] = Field(default_factory=list)
