import asyncio
import functools
import logging
import os
import re
from typing import Any, Dict, Optional

import asyncpg
import click
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from google.adk.agents import LlmAgent

# 如果你的工程需要自定义执行器，解开并按需接入：
# from llm_auditor.agent_executor import ADKAgentExecutor

from dotenv import load_dotenv
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse, JSONResponse

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def make_sync(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))

    return wrapper


# -----------------------
# 数据访问（DB 优先，失败则 mock）
# -----------------------
async def fetch_agent_config(
    agent_id: str, pool: Optional[asyncpg.Pool]
) -> Optional[Dict[str, Any]]:
    if pool is not None:
        try:
            row = await pool.fetchrow(
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
        except Exception as e:
            logger.warning("DB fetch failed for agent_id=%s: %s", agent_id, e)

    # --- mock ---
    MOCKS = {
        "auditor": {
            "name": "Llm_Auditor",
            "description": "Audits and refines answers with web cross-checking.",
            "model": "openai-4o",
            "prompt": "You are a rigorous LLM answer auditor.",
            "mcps": ["kGfo3DfYduMVGRkf6kbmAKTYpfqQXjKN1nFvgHpfHqghu5rGn"],
        },
        "writer": {
            "name": "Pro_Writer",
            "description": "Writes and edits copy in a clean, factual tone.",
            "model": "openai-4o",
            "prompt": "You are a precise technical writer.",
            "mcps": [],
        },
    }
    return MOCKS.get(agent_id)


def build_agent(cfg: Dict[str, Any]) -> LlmAgent:
    logger.info(f"Building agent: {cfg}")
    return LlmAgent(
        name=cfg["name"],
        description=cfg.get("description") or "",
        model=cfg.get("model"),
        system_prompt=cfg.get("prompt"),
        # 若需把 mcps 接成工具，可在这里或 handler 里注入
    )


def build_agent_card(cfg: Dict[str, Any]) -> AgentCard:
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
                id="critic",
                name="critic",
                description="Critical analysis.",
                tags=[],
                examples=[],
            ),
            AgentSkill(
                id="reviser",
                name="reviser",
                description="Editorial revision.",
                tags=[],
                examples=[],
            ),
        ],
    )


def build_a2a_subapp(
    agent_card: AgentCard, llm_agent: LlmAgent, task_store: InMemoryTaskStore
) -> Starlette:
    # 如果需要自定义执行器，替换下面的 DefaultRequestHandler
    handler = DefaultRequestHandler(
        # agent_executor=ADKAgentExecutor(agent=llm_agent),
        agent=llm_agent,  # 若你的版本支持直接传 agent
        task_store=task_store,
    )
    a2a_app = A2AStarletteApplication(agent_card=agent_card, http_handler=handler)
    return Starlette(routes=a2a_app.routes(), middleware=[])


class AppState:
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
        self.task_store = InMemoryTaskStore()
        self.cache: Dict[str, Starlette] = {}  # agent_id -> sub Starlette app

    async def ensure_pool(self):
        if self.pool is None:
            db_url = os.environ.get("DATABASE_URL")
            if db_url:
                self.pool = await asyncpg.create_pool(
                    dsn=db_url, min_size=1, max_size=5
                )

    async def get_sub_app(self, agent_id: str) -> Optional[Starlette]:
        if agent_id in self.cache:
            return self.cache[agent_id]
        cfg = await fetch_agent_config(agent_id, self.pool)
        if not cfg:
            return None
        agent = build_agent(cfg)
        card = build_agent_card(cfg)
        sub = build_a2a_subapp(card, agent, self.task_store)
        self.cache[agent_id] = sub
        return sub


app_state = AppState()

AGENT_PATH_RE = re.compile(r"^/([^/]+)(/.*)?$")  # /{agent_id}[/rest]


class AgentDispatchMiddleware:
    """
    动态分发到 /{agent_id} 子应用，并**流式转发**下游 ASGI 响应（SSE/分块）。
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            # 非 HTTP（如 WebSocket）直接交给下游
            return await self.app(scope, receive, send)

        path = scope.get("path", "")
        if path == "/healthz":
            # 简单健康检查（非流式）
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

        sub_app = await app_state.get_sub_app(agent_id)
        if sub_app is None:
            return await self._json(send, 404, {"error": "Agent not found"})

        # 修改 scope：把前缀去掉，让子应用看到原始 A2A 路由
        new_scope = dict(scope)
        new_scope["path"] = rest
        orig_root = scope.get("root_path", "")
        new_scope["root_path"] = f"{orig_root}/{agent_id}"

        # 直接把客户端的 send 传给子应用，**零拷贝/零缓冲**转发流式响应
        return await sub_app(new_scope, receive, send)

    async def _not_found(self, send):
        await send({"type": "http.response.start", "status": 404, "headers": []})
        await send({"type": "http.response.body", "body": b"Not Found"})

    async def _json(self, send, status: int, data: Dict[str, Any]):
        import json as _json

        body = _json.dumps(data).encode("utf-8")
        headers = [(b"content-type", b"application/json")]
        await send(
            {"type": "http.response.start", "status": status, "headers": headers}
        )
        await send({"type": "http.response.body", "body": body})


@click.command()
@click.option("--host", default="0.0.0.0")
@click.option("--port", default=8080)
@make_sync
async def main(host, port):
    await app_state.ensure_pool()

    # 基础应用仅提供 healthz（非流式路径），其余由中间件接管
    base = Starlette()

    # 包一层中间件，做动态分发 + 流式透传
    app = AgentDispatchMiddleware(base)

    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    main()
