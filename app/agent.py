import os
import logging
import httpx

from collections.abc import AsyncIterable
from typing import Any, Literal
from pydantic import BaseModel
from agents.mcp import MCPServerStreamableHttp
from agents import (
    Agent as OpenAIAgent,
    Runner,
    ItemHelpers,
    HostedMCPTool,
)
from agents.tool import MCPToolApprovalFunctionResult
from agents.model_settings import ModelSettings
from agents.tracing import set_tracing_disabled


logger = logging.getLogger(__name__)


def on_approval_request(tool_call):
    logger.info("waiting for approval: %s", tool_call)
    return MCPToolApprovalFunctionResult(approved=True, reason="auto-approved")


class Agent:
    """The default agent impl."""

    def __init__(self, instruction, model, tools=[]):
        # Determine model based on model_source
        set_tracing_disabled(True)
        mcp_tools = []
        for tool in tools:
            logger.info("Adding tool: %s", tool)
            mcp_tool = HostedMCPTool(
                tool_config={
                    "type": "mcp",
                    "server_label": tool["id"],
                    "server_url": tool["url"],
                    "require_approval": "never",
                },
                # on_approval_request=on_approval_request,
            )
            mcp_tools.append(mcp_tool)
        if model == "gemini":
            model = "litellm/gemini/gemini-2.5-flash"
        else:
            model = "gpt-5-mini"
        self.agent = OpenAIAgent(
            name="default",
            instructions=instruction,
            model=model,
            tools=mcp_tools,
        )

    async def stream(self, inputs) -> AsyncIterable[dict[str, Any]]:
        try:
            stream = Runner.run_streamed(self.agent, inputs, max_turns=100)
            # Stream events as they come in
            async for event in stream.stream_events():
                if event.type == "raw_response_event":
                    continue
                if event.type == "agent_updated_stream_event":
                    continue
                if event.type == "run_item_stream_event":
                    logger.info("Run item event: %s", event)
                    # Handle different types of run items
                    if event.name == "tool_called":
                        # Tool call event
                        # Extract tool call information
                        tool_name = event.item.raw_item.name
                        tool_call_id = event.item.raw_item.id
                        arguments = event.item.raw_item.arguments
                        yield {
                            "task_complete": False,
                            "content": f"""
                            <tool>
                            <func>{tool_name}</func>
                            <id>{tool_call_id}</id>
                            <args>{arguments}</args>
                            </tool>
                            """,
                        }
                    elif event.name == "tool_output":
                        # Tool output event
                        tool_call_id = event.item.raw_item["call_id"]
                        tool_output = event.item.raw_item["output"]
                        yield {
                            "task_complete": False,
                            "content": f"""
                            <tool>
                            <id>{tool_call_id}</id>
                            <result>{tool_output}</result>
                            </tool>
                            """,
                        }
                    elif event.name == "message_output_created":
                        yield {
                            "task_complete": False,
                            "content": f"{ItemHelpers.text_message_output(event.item)}",
                        }
            # Parse the final response to determine completion status and format
            yield {
                "task_complete": True,
                "content": "",
            }
        except Exception as e:
            logger.error("Error in stream method: %s", e)
            yield {
                "task_complete": False,
                "content": f"An error occurred: {str(e)}",
            }
