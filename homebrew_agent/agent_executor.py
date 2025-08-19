import time
import asyncio
import concurrent.futures
import logging
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    Part,
    TaskState,
    TextPart,
)
from a2a.utils import new_agent_text_message, new_task
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.agents import Agent
from google.genai import types
from pydantic import BaseModel
from a2a.types import AgentCard
from starlette.applications import Starlette

logger = logging.getLogger(__name__)


class ADKAgentExecutor(AgentExecutor):

    def __init__(
        self,
        agent: Agent,
        status_message="Processing request...",
        artifact_name="response",
    ):
        """Initialize a generic ADK agent executor.

        Args:
            agent: The ADK agent instance
            status_message: Message to display while processing
            artifact_name: Name for the response artifact
        """
        self.agent = agent
        self.status_message = status_message
        self.artifact_name = artifact_name
        self.runner = Runner(
            app_name=agent.name,
            agent=agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Cancel the execution of a specific task."""
        raise NotImplementedError(
            "Cancellation is not implemented for ADKAgentExecutor."
        )

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        if not context.message:
            raise Exception("Message should be present in request context")
        query = context.get_user_input()
        task = context.current_task or new_task(context.message)
        await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        if context.call_context:
            user_id = context.call_context.user.user_name
        else:
            user_id = "a2a_user"

        logger.info(f"Starting agent execution for user: {user_id}, task: {task.id}")
        session = None
        try:
            # Update status with custom message
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(self.status_message, task.context_id, task.id),
            )
            # Process with ADK agent
            session = await self.runner.session_service.create_session(
                app_name=self.agent.name,
                user_id=user_id,
                state={},
                session_id=task.context_id,
            )
            content = types.Content(
                role="user", parts=[types.Part.from_text(text=query)]
            )

            logger.info(f"Running agent with query: {query[:100]}...")

            content = types.Content(
                role="user", parts=[types.Part.from_text(text=query)]
            )

            events = []
            try:
                # 如果 ADK 还有 run_stream / astream，也可用，思路相同
                async for ev in self.runner.run_async(
                    user_id=user_id,
                    session_id=session.id,
                    new_message=content,
                    # 某些版本支持：raise_on_error=True
                ):
                    # 把“错误事件”当作异常立即抛
                    if getattr(ev, "type", None) in {"error", "on_error"}:
                        raise RuntimeError(f"Runner error event: {ev}")
                    events.append(ev)
            except Exception:
                logger.exception("runner.run_async failed")
                raise

            if not events:
                raise RuntimeError("Runner produced no events (likely upstream crash).")

            logger.info("Agent execution completed, processing %d events", len(events))

            # Process events asynchronously
            for event in events:
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, "text") and part.text:
                            await updater.update_status(
                                TaskState.working,
                                new_agent_text_message(
                                    part.text, task.context_id, task.id
                                ),
                            )
                        elif hasattr(part, "function_call"):
                            # Log or handle function calls if needed
                            logger.debug(
                                f"Function call detected: {part.function_call}"
                            )

            await updater.complete()
            logger.info(f"Agent execution completed successfully for task: {task.id}")
            await asyncio.sleep(1)  # Use asyncio.sleep instead of time.sleep

        except Exception as e:
            logger.error(f"Error during agent execution: {e}", exc_info=True)

            # Check if this is an MCP-related error
            error_msg = str(e)
            if (
                "mcp" in error_msg.lower()
                or "cancelled" in error_msg.lower()
                or "wouldblock" in error_msg.lower()
            ):
                error_msg = "Agent execution failed due to tool connection issues. Please try again or contact support if the problem persists."
                logger.warning(
                    "MCP-related error detected, providing user-friendly error message"
                )

            await updater.update_status(
                TaskState.failed,
                new_agent_text_message(error_msg, task.context_id, task.id),
                final=True,
            )
        finally:
            # Ensure proper cleanup
            if session:
                try:
                    # Clean up session if needed
                    pass
                except Exception as cleanup_error:
                    logger.error(f"Error during cleanup: {cleanup_error}")
