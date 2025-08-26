import asyncio
import functools
import logging
import os
import re
import json
from typing import Any, Dict, Optional, AsyncIterator

import asyncpg
import click
import uvicorn

# A2A related imports
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

# Google ADK imports
from google.adk.agents import LlmAgent
from google.adk.a2a.utils.agent_to_a2a import to_a2a

from dotenv import load_dotenv
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse, JSONResponse, StreamingResponse
from starlette.routing import Route

from homebrew_agent.agent_executor import ADKAgentExecutor
from homebrew_agent.manager import AgentManager
from homebrew_agent.logging_config import setup_logging
from sensespace_did import verify_token

load_dotenv()

# Setup rich logging
logger = setup_logging(__name__)


def make_sync(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))

    return wrapper


AGENT_PATH_RE = re.compile(r"^/([^/]+)(/.*)?$")

agent_manager = AgentManager(os.environ.get("DB_URL"))
BEARER_RE = re.compile(r"^Bearer\s+(.+)$", re.IGNORECASE)


def _extract_bearer_from_scope(scope) -> Optional[str]:
    for name, value in scope.get("headers", []):
        if name.lower() == b"authorization":
            try:
                m = BEARER_RE.match(value.decode("latin-1"))
                if m:
                    return m.group(1).strip()
            except Exception:
                return None
    return None


class AgentDispatchMiddleware:
    """Dynamic dispatch middleware"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        token = _extract_bearer_from_scope(scope)
        if token:
            logger.info(f"Got Bearer token (len={len(token)}).")
            try:
                payload = await verify_token(token)
                if payload and payload.success:
                    logger.info(f"Verified token: {payload}")
                else:
                    logger.error(f"Failed to verify token: {payload}")
                    return await self._json(send, 401, {"error": "Unauthorized"})
            except Exception as e:
                logger.error(f"Failed to verify token: {e}")
                return await self._json(send, 401, {"error": "Unauthorized"})
        else:
            logger.info("No Bearer token found.")
            return await self._json(send, 401, {"error": "Unauthorized"})

        path = scope.get("path", "")

        if path == "/healthz":

            async def _send_ok():
                await send(
                    {"type": "http.response.start", "status": 200, "headers": []}
                )
                await send({"type": "http.response.body", "body": b"ok"})

            return await _send_ok()

        m = AGENT_PATH_RE.match(path)
        if not m:
            return await self._not_found(send)

        agent_id = m.group(1)
        rest = m.group(2) or "/"

        logger.info(f"Routing to agent '{agent_id}', path: {rest}")
        agent_info = await agent_manager.get_or_create_agent_by_id(agent_id, token)
        if agent_info is None:
            return await self._json(
                send, 404, {"error": f"Agent '{agent_id}' not found"}
            )
        sub_app = agent_info.app

        # Modify scope
        new_scope = dict(scope)
        new_scope["path"] = rest
        new_scope["raw_path"] = rest.encode("utf-8")

        # Forward to sub-application
        try:
            return await sub_app(new_scope, receive, send)
        except Exception as e:
            logger.error(f"Error in sub-app: {e}", exc_info=True)
            return await self._json(send, 500, {"error": str(e)})

    async def _not_found(self, send):
        await send({"type": "http.response.start", "status": 404, "headers": []})
        await send({"type": "http.response.body", "body": b"Not Found"})

    async def _json(self, send, status: int, data: Dict[str, Any]):
        body = json.dumps(data).encode("utf-8")
        headers = [(b"content-type", b"application/json")]
        await send(
            {"type": "http.response.start", "status": status, "headers": headers}
        )
        await send({"type": "http.response.body", "body": body})


@click.command()
@click.option("--host", default="0.0.0.0")
@click.option("--port", default=8080)
@click.option(
    "--debug", is_flag=True, help="Enable debug mode (overrides LOG_LEVEL env)"
)
@make_sync
async def main(host, port, debug):
    if debug:
        # Debug flag overrides environment LOG_LEVEL
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled via CLI flag")

    base = Starlette()
    app = AgentDispatchMiddleware(base)

    # Use LOG_LEVEL from env unless debug flag is set
    uvicorn_log_level = (
        "debug" if debug else os.environ.get("LOG_LEVEL", "info").lower()
    )

    config = uvicorn.Config(
        app, host=host, port=port, log_level=uvicorn_log_level, reload=debug
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    main()
