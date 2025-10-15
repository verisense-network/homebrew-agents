import logging
import uuid

from typing import Any
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    InternalError,
    InvalidParamsError,
    Part,
    Role,
    Task,
    TaskState,
    TaskStatus,
    Message,
    TextPart,
    DataPart,
    FilePart,
    UnsupportedOperationError,
)
from a2a.utils import new_agent_text_message
from a2a.utils.errors import ServerError

from app.agent import Agent


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiAgentExecutor(AgentExecutor):
    """MultiAgentExecutor handles requests."""

    def __init__(self, agent: Agent):
        self.agent = agent

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        error = self._validate_request(context)
        if error:
            raise ServerError(error=InvalidParamsError())

        inputs = parse_input(context.message)
        logger.info("Parsed inputs: %s", inputs)
        task = context.current_task
        if not task:
            task = create_task(context.message)  # type: ignore
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        try:
            async for item in self.agent.stream(inputs):
                logger.info("Streaming item: %s", item)
                if item["task_complete"]:
                    await updater.complete()
                    break
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(
                        item["content"],
                        task.context_id,
                        task.id,
                    ),
                )
        except Exception as e:
            logger.error("An error occurred while streaming the response: %s", e)
            raise ServerError(error=InternalError()) from e

    def _validate_request(self, context: RequestContext) -> bool:
        return False

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise ServerError(error=UnsupportedOperationError())


def create_task(request: Message) -> Task:
    if not request.role:
        raise TypeError("Message role cannot be None")
    if not request.parts:
        raise ValueError("Message parts cannot be empty")
    return Task(
        status=TaskStatus(state=TaskState.submitted),
        id=request.task_id or str(uuid.uuid4()),
        context_id=request.context_id or str(uuid.uuid4()),
        history=[],
    )


def parse_input(request: Message) -> list[dict[str, Any]]:
    if not request.parts:
        raise ValueError("Message parts cannot be empty")
    inputs = []
    for part in request.parts:
        if part.root.metadata is not None and not part.root.metadata["from_user"]:
            role = "assistant"
        else:
            role = "user"
        if isinstance(part.root, TextPart):
            inputs.append({"role": role, "content": part.root.text})
        elif isinstance(part.root, DataPart):
            inputs.append({"role": role, "content": part.root.data})
        elif isinstance(part.root, FilePart):
            inputs.append({"role": role, "content": part.root.file})
    return inputs
