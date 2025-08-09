from google.adk.agents import Agent
from homebrew_agent.agent_executor import ADKAgentExecutor
from homebrew_agent.logging_config import setup_logging
from typing import Any, Dict, Optional, Tuple
import asyncpg
import json
import os

from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
from google.adk.agents import LlmAgent
from pydantic import BaseModel, ConfigDict
from starlette.applications import Starlette
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.apps import A2AStarletteApplication
from a2a.server.tasks import InMemoryTaskStore

logger = setup_logging(__name__)


class AgentInfo(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    agent: Agent
    card: AgentCard
    app: Starlette


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
                    SELECT name, description, model, prompt, mcps
                    FROM agents
                    WHERE agent_id = $1
                    """,
                    agent_id,
                )
                if row:
                    return {
                        "name": row["name"],
                        "description": row["description"],
                        "model": row["model"],
                        "prompt": row["prompt"],
                        "mcps": list(row["mcps"]) if row["mcps"] is not None else [],
                    }
            finally:
                await conn.close()
        except Exception as e:
            logger.warning("DB fetch failed for agent_id=%s: %s", agent_id, e)

        # Mock data
        MOCKS = {
            "auditor": {
                "name": "Llm_Auditor",
                "description": "Audits and refines answers with web cross-checking.",
                "model": "gemini-2.0-flash",
                "prompt": "You are a rigorous LLM answer auditor. Critically evaluate and improve answers.",
                "mcps": [],
            },
            "writer": {
                "name": "Pro_Writer",
                "description": "Writes and edits copy in a clean, factual tone.",
                "model": "gemini-2.0-flash",
                "prompt": "You are a professional technical writer. Write clear and accurate content.",
                "mcps": [],
            },
        }
        return MOCKS.get(agent_id)

    def _build_agent_card(self, cfg: Dict[str, Any]) -> AgentCard:
        """Build Agent Card"""
        return AgentCard(
            name=cfg["name"],
            description=cfg.get("description") or "",
            version="1.0.0",
            url=os.environ.get("APP_URL", "http://localhost:8080"),
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

    def _build_agent(self, cfg: Dict[str, Any]) -> LlmAgent:
        """Build LLM Agent"""
        logger.info(f"Building agent with config: {json.dumps(cfg, indent=2)}")

        model = cfg.get("model", "gemini-2.0-flash")
        name = cfg.get("name", "default_agent")
        description = cfg.get("description", "A helpful AI assistant")
        instruction = cfg.get("prompt", "You are a helpful assistant.")

        try:
            agent = LlmAgent(
                model=model,
                name=name,
                description=description,
                instruction=instruction,
            )
            logger.info(f"Successfully created agent: {agent.name}")
            return agent
        except Exception as e:
            logger.error(f"Failed to create agent: {e}")
            raise

    def _build_a2a_substarlette(
        self, agent_card: AgentCard, llm_agent: LlmAgent, task_store: InMemoryTaskStore
    ) -> Starlette:
        """Manually build A2A compatible Starlette application"""
        logger.info(f"Building A2A subapp manually for agent: {llm_agent.name}")

        request_handler = DefaultRequestHandler(
            agent_executor=ADKAgentExecutor(
                agent=llm_agent,
            ),
            task_store=task_store,
        )

        a2a_app = A2AStarletteApplication(
            agent_card=agent_card, http_handler=request_handler
        )

        routes = a2a_app.routes()
        logger.info(f"Routes: {routes}")
        app = Starlette(
            routes=routes,
            middleware=[],
        )
        return app

    async def build_agent_starlette_by_id(self, agent_id: str) -> Optional[AgentInfo]:
        cfg = await self._fetch_agent_config(agent_id)
        if not cfg:
            return None

        agent = self._build_agent(cfg)
        card = self._build_agent_card(cfg)
        app = self._build_a2a_substarlette(card, agent, InMemoryTaskStore())

        return AgentInfo(agent=agent, card=card, app=app)
