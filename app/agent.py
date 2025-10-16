import os
import logging
import httpx

from contextlib import AsyncExitStack
from collections.abc import AsyncIterable
from typing import Any, Literal
from pydantic import BaseModel
from agents.mcp import MCPServerStreamableHttp
from agents import (
    Agent as OpenAIAgent,
    Runner,
    Model,
    OpenAIResponsesModel,
    ModelProvider,
    RunConfig,
    ItemHelpers,
    HostedMCPTool,
    set_default_openai_client,
)
from agents.tool import MCPToolApprovalFunctionResult
from agents.model_settings import ModelSettings
from agents.tracing import set_tracing_disabled
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


def on_approval_request(tool_call):
    logger.info("waiting for approval: %s", tool_call)
    return MCPToolApprovalFunctionResult(approved=True, reason="auto-approved")


key = os.getenv("AMBIENT_API_KEY", "")
ambient_client = AsyncOpenAI(base_url="https://api.ambient.xyz", api_key=key)


class AmbientModelProvider(ModelProvider):
    def get_model(self, model_name: str | None) -> Model:
        return OpenAIResponsesModel(model=model_name, openai_client=ambient_client)


class Agent:
    """The default agent impl."""

    def __init__(self, instruction, model, tools=[]):
        # Determine model based on model_source
        set_tracing_disabled(True)
        # mcp_tools = []
        # for tool in tools:
        #     logger.info("Adding tool: %s", tool)
        #     mcp_tool = HostedMCPTool(
        #         tool_config={
        #             "type": "mcp",
        #             "server_label": tool["id"],
        #             "server_url": tool["url"],
        #             "require_approval": "never",
        #         },
        #     )
        #     mcp_tools.append(mcp_tool)
        self.instructions = instruction
        self.model = model
        self.tools = tools

    async def init(self) -> (OpenAIAgent, RunConfig):
        mcp_list = []
        for tool in self.tools:
            server = MCPServerStreamableHttp(
                name=tool["name"],
                cache_tools_list=True,
                client_session_timeout_seconds=30,
                params={
                    "url": tool["url"],
                },
            )
            try:
                await server.connect()
                mcp_list.append(server)
            except Exception as e:
                logger.error("Failed to connect to MCP server %s: %s", tool["url"], e)
                await server.cleanup()
        if self.model == "openai":
            return (
                OpenAIAgent(
                    name="default",
                    instructions=self.instructions,
                    mcp_servers=mcp_list,
                    # tools=mcp_tools,
                    model="gpt-5-mini",
                ),
                RunConfig(),
            )
        if self.model == "ambient":
            return (
                OpenAIAgent(
                    name="default",
                    instructions=self.instructions,
                    mcp_servers=mcp_list,
                    # tools=mcp_tools,
                ),
                RunConfig(model_provier=AmbientModelProvider()),
            )
        return (
            OpenAIAgent(
                name="default",
                instructions=self.instructions,
                mcp_servers=mcp_list,
                # tools=mcp_tools,
                model="litellm/gemini/gemini-2.5-flash",
            ),
            RunConfig(),
        )

    async def stream(self, inputs) -> AsyncIterable[dict[str, Any]]:
        try:
            agent, run_config = await self.init()
            stream = Runner.run_streamed(
                agent,
                inputs,
                max_turns=100,
                run_config=run_config,
            )
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
                        tool_call_id = event.item.raw_item.call_id
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
