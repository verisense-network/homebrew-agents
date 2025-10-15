#!/usr/bin/env python3
import httpx
import os
import asyncpg
import logging

from a2a.server.request_handlers import DefaultRequestHandler
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
from a2a.server.tasks import (
    BasePushNotificationSender,
    InMemoryPushNotificationConfigStore,
    InMemoryTaskStore,
)
from a2a.server.request_handlers.jsonrpc_handler import JSONRPCHandler
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.agent import Agent
from app.agent_executor import MultiAgentExecutor

logger = logging.getLogger(__name__)


class AgentFactory:
    db_url: str

    def __init__(self, db_url: str):
        self.db_url = db_url

    async def _fetch_agent_config(self, agent_id: str) -> Optional[Dict[str, Any]]:
        try:
            conn = await asyncpg.connect(self.db_url)
            try:
                row = await conn.fetchrow(
                    """
                    SELECT id, name, description, model, prompt, mcps, created_at, updated_at
                    FROM homebrew_agents
                    WHERE id = $1::uuid
                    """,
                    agent_id,
                )
                if row:
                    return {
                        "id": str(row["id"]),
                        "name": row["name"],
                        "description": row["description"],
                        "model": row["model"],
                        "prompt": row["prompt"],
                        "mcps": list(row["mcps"]) if row["mcps"] is not None else [],
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"],
                    }
            finally:
                await conn.close()
        except Exception as e:
            logger.warning("DB fetch failed for agent_id=%s: %s", agent_id, e)
            return None

    def _build_agent_card(self, cfg: Dict[str, Any]) -> AgentCard:
        agent_id = cfg["id"]
        """Build Agent Card"""
        return AgentCard(
            name=cfg["name"],
            description=cfg.get("description") or "",
            version="1.0.0",
            # TODO
            url=os.environ.get("APP_URL", "http://34.71.62.169:8080") + "/" + agent_id,
            defaultInputModes=["text", "text/plain"],
            defaultOutputModes=["text", "text/plain"],
            capabilities=AgentCapabilities(streaming=True),
            skills=[
                AgentSkill(
                    id="main",
                    name="main",
                    description=cfg.get("description", "Main capability"),
                    tags=[],
                    examples=[],
                ),
            ],
        )

    async def _fetch_mcp_configs(self, mcp_ids: List[str]) -> List[Dict[str, Any]]:
        """Fetch MCP configurations from database"""
        if not mcp_ids:
            return []
        try:
            conn = await asyncpg.connect(self.db_url)
            try:
                rows = await conn.fetch(
                    """
                    SELECT id, name, description, url, provider
                    FROM mcp_servers
                    WHERE id = ANY($1::text[])
                    """,
                    mcp_ids,
                )
                return [
                    {
                        "id": row["id"],
                        "name": row["name"],
                        "description": row["description"],
                        "url": row["url"],
                        "provider": row["provider"],
                    }
                    for row in rows
                ]
            finally:
                await conn.close()
        except Exception as e:
            logger.warning(f"Failed to fetch MCP configs: {e}")
            return []

    async def build(self, agent_id: str) -> JSONRPCHandler:
        """Build LLM Agent"""
        agent_config = self._fetch_agent_config(agent_id)
        # TODO assert not NONE
        agent_card = self._build_agent_card(agent_config)
        tools = await self._fetch_mcp_configs(agent_config["mcps"])
        model_name = agent_config.get("model", "gemini")
        if model_name == "openai":
            model = "gpt-5-mini"
        else:
            model = "litellm/gemini/gemini-2.5-flash"
        agent = Agent(
            instruction=agent_config.get("prompt", ""),
            model=model,
            tools=tools,
        )
        httpx_client = httpx.AsyncClient()
        push_config_store = InMemoryPushNotificationConfigStore()
        push_sender = BasePushNotificationSender(
            httpx_client=httpx_client, config_store=push_config_store
        )
        http_handler = DefaultRequestHandler(
            agent_executor=MultiAgentExecutor(agent),
            task_store=InMemoryTaskStore(),
            push_config_store=push_config_store,
            push_sender=push_sender,
        )
        return JSONRPCHandler(
            agent_card=agent_card,
            request_handler=http_handler,
            extended_agent_card=None,
            extended_card_modifier=None,
        )
