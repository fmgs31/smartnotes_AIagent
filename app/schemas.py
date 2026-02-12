from pydantic import BaseModel, Field
from typing import Any, Dict, Optional


class InboundRequest(BaseModel):
    text: str = Field(..., min_length=1)
    mcp_token: str = Field(..., min_length=1)
    server_id: Optional[str] = None
    allowed_tools: Optional[list[str]] = None


class ToolCallPlan(BaseModel):
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class OutboundResponse(BaseModel):
    server_id: str
    tool_call: ToolCallPlan
    mcp_result: Any
