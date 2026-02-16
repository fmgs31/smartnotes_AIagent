from pydantic import BaseModel, Field
from typing import Any, Dict


class InboundRequest(BaseModel):
    text: str = Field(..., min_length=1)
    mcp_token: str = Field(..., min_length=1)


class ToolCallPlan(BaseModel):
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class NextStepPlan(BaseModel):
    done: bool
    tool_name: str = ""
    arguments: Dict[str, Any] = Field(default_factory=dict)
    explanation: str | None = None


class ExecutedToolStep(BaseModel):
    step: int
    server_id: str
    tool_call: ToolCallPlan
    mcp_result: Any


class OutboundResponse(BaseModel):
    server_id: str | None = None
    tool_call: ToolCallPlan | None = None
    mcp_result: Any = None
    steps: list[ExecutedToolStep] = Field(default_factory=list)
