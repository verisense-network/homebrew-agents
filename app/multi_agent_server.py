#!/usr/bin/env python3

import contextlib
import json
import logging
import traceback
import hashlib
import sensespace_did

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any
from starlette.routing import Route
from pydantic import ValidationError

from a2a.auth.user import UnauthenticatedUser
from a2a.auth.user import User as A2AUser
from a2a.server.context import ServerCallContext
from a2a.server.request_handlers.jsonrpc_handler import JSONRPCHandler
from a2a.types import (
    A2AError,
    A2ARequest,
    CancelTaskRequest,
    DeleteTaskPushNotificationConfigRequest,
    GetAuthenticatedExtendedCardRequest,
    GetTaskPushNotificationConfigRequest,
    GetTaskRequest,
    InternalError,
    InvalidParamsError,
    InvalidRequestError,
    JSONParseError,
    JSONRPCError,
    JSONRPCErrorResponse,
    JSONRPCRequest,
    JSONRPCResponse,
    ListTaskPushNotificationConfigRequest,
    MethodNotFoundError,
    SendMessageRequest,
    SendStreamingMessageRequest,
    SendStreamingMessageResponse,
    SetTaskPushNotificationConfigRequest,
    TaskResubscriptionRequest,
    UnsupportedOperationError,
)
from a2a.utils.constants import (
    AGENT_CARD_WELL_KNOWN_PATH,
    DEFAULT_RPC_URL,
    EXTENDED_AGENT_CARD_PATH,
    PREV_AGENT_CARD_WELL_KNOWN_PATH,
)

# HTTP extension header constant
HTTP_EXTENSION_HEADER = 'X-A2A-Extensions'
from a2a.utils.errors import MethodNotImplementedError
from app.agent_handler import AgentFactory


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from fastapi import FastAPI
    from sse_starlette.sse import EventSourceResponse
    from starlette.applications import Starlette
    from starlette.authentication import BaseUser
    from starlette.exceptions import HTTPException
    from starlette.requests import Request
    from starlette.responses import JSONResponse, Response
    from starlette.status import HTTP_413_REQUEST_ENTITY_TOO_LARGE

    _package_starlette_installed = True
else:
    FastAPI = Any
    try:
        from sse_starlette.sse import EventSourceResponse
        from starlette.applications import Starlette
        from starlette.authentication import BaseUser
        from starlette.exceptions import HTTPException
        from starlette.requests import Request
        from starlette.responses import JSONResponse, Response
        from starlette.status import HTTP_413_REQUEST_ENTITY_TOO_LARGE

        _package_starlette_installed = True
    except ImportError:
        _package_starlette_installed = False
        # Provide placeholder types for runtime type hinting when dependencies are not installed.
        # These will not be used if the code path that needs them is guarded by _http_server_installed.
        EventSourceResponse = Any
        Starlette = Any
        BaseUser = Any
        HTTPException = Any
        Request = Any
        JSONResponse = Any
        Response = Any
        HTTP_413_REQUEST_ENTITY_TOO_LARGE = Any

MAX_CONTENT_LENGTH = 1_000_000


class StarletteUserProxy(A2AUser):
    """Adapts the Starlette User class to the A2A user representation."""

    def __init__(self, claims: dict):
        self._user = claims['sub']

    @property
    def is_authenticated(self) -> bool:
        """Returns whether the current user is authenticated."""
        return True

    @property
    def user_name(self) -> str:
        """Returns the user name of the current user."""
        return self._user


class CallContextBuilder(ABC):
    """A class for building ServerCallContexts using the Starlette Request."""

    @abstractmethod
    def build(self, request: Request) -> ServerCallContext:
        """Builds a ServerCallContext from a Starlette Request."""


class DefaultCallContextBuilder(CallContextBuilder):
    """A default implementation of CallContextBuilder."""

    async def build(self, request: Request) -> ServerCallContext:
        """Builds a ServerCallContext from a Starlette Request.

        Args:
            request: The incoming Starlette Request object.

        Returns:
            A ServerCallContext instance populated with user and state
            information from the request.
        """
        state = {}
        state['headers'] = dict(request.headers)
        user: A2AUser = UnauthenticatedUser()
        with contextlib.suppress(Exception):
            token = request.headers['authorization']
            logger.info(f"Authorization token: {token}")
            result = await sensespace_did.verify_token(token.removeprefix('Bearer '))
            if result.success:
                user = StarletteUserProxy(result.claims)
            else:
                user = UnauthenticatedUser()
        return ServerCallContext(
            user=user,
            state=state,
        )


class MultiAgentApplication:
    """Class for A2A JSONRPC applications.

    Handles incoming JSON-RPC requests, routes them to the appropriate
    handler methods, and manages response generation including Server-Sent Events
    (SSE).
    """

    # Method-to-model mapping for centralized routing
    A2ARequestModel = (
        SendMessageRequest
        | SendStreamingMessageRequest
        | GetTaskRequest
        | CancelTaskRequest
        | SetTaskPushNotificationConfigRequest
        | GetTaskPushNotificationConfigRequest
        | ListTaskPushNotificationConfigRequest
        | DeleteTaskPushNotificationConfigRequest
        | TaskResubscriptionRequest
        | GetAuthenticatedExtendedCardRequest
    )

    METHOD_TO_MODEL: dict[str, type[A2ARequestModel]] = {
        model.model_fields['method'].default: model
        for model in A2ARequestModel.__args__
    }

    def __init__(  # noqa: PLR0913
        self,
        agents: AgentFactory,
    ) -> None:
        """Initializes the JSONRPCApplication."""
        if not _package_starlette_installed:
            raise ImportError(
                'Packages `starlette` and `sse-starlette` are required to use the'
                ' `JSONRPCApplication`. They can be added as a part of `a2a-sdk`'
                ' optional dependencies, `a2a-sdk[http-server]`.'
            )
        self.agents = agents
        self._context_builder = DefaultCallContextBuilder()

    def _generate_error_response(
        self, request_id: str | int | None, error: JSONRPCError | A2AError
    ) -> JSONResponse:
        """Creates a Starlette JSONResponse for a JSON-RPC error.

        Logs the error based on its type.

        Args:
            request_id: The ID of the request that caused the error.
            error: The `JSONRPCError` or `A2AError` object.

        Returns:
            A `JSONResponse` object formatted as a JSON-RPC error response.
        """
        error_resp = JSONRPCErrorResponse(
            id=request_id,
            error=error if isinstance(error, JSONRPCError) else error.root,
        )

        log_level = (
            logging.ERROR
            if not isinstance(error, A2AError)
            or isinstance(error.root, InternalError)
            else logging.WARNING
        )
        logger.log(
            log_level,
            "Request Error (ID: %s): Code=%s, Message='%s'%s",
            request_id,
            error_resp.error.code,
            error_resp.error.message,
            ', Data=' + str(error_resp.error.data)
            if error_resp.error.data
            else '',
        )
        return JSONResponse(
            error_resp.model_dump(mode='json', exclude_none=True),
            status_code=200,
        )

    async def _handle_requests(self, request: Request) -> Response:  # noqa: PLR0911
        """Handles incoming POST requests to the main A2A endpoint.

        Parses the request body as JSON, validates it against A2A request types,
        dispatches it to the appropriate handler method, and returns the response.
        Handles JSON parsing errors, validation errors, and other exceptions,
        returning appropriate JSON-RPC error responses.

        Args:
            request: The incoming Starlette Request object.

        Returns:
            A Starlette Response object (JSONResponse or EventSourceResponse).

        Raises:
            (Implicitly handled): Various exceptions are caught and converted
            into JSON-RPC error responses by this method.
        """
        request_id = None
        body = None

        try:
            agent_id = request.path_params['agent_id']
            # TODO check db
            # if not agent_id or agent_id not in self.handlers:
            #     return self._generate_error_response(
            #         request_id,
            #         A2AError(
            #             root=InvalidRequestError(message='Agent not found')
            #         ),
            #     )

            body = await request.json()
            if isinstance(body, dict):
                request_id = body.get('id')
                # Ensure request_id is valid for JSON-RPC response (str/int/None only)
                if request_id is not None and not isinstance(
                    request_id, str | int
                ):
                    request_id = None
            # Treat very large payloads as invalid request (-32600) before routing
            with contextlib.suppress(Exception):
                content_length = int(request.headers.get('content-length', '0'))
                if content_length and content_length > MAX_CONTENT_LENGTH:
                    return self._generate_error_response(
                        request_id,
                        A2AError(
                            root=InvalidRequestError(
                                message='Payload too large'
                            )
                        ),
                    )
            logger.debug('Request body: %s', body)
            # 1) Validate base JSON-RPC structure only (-32600 on failure)
            try:
                base_request = JSONRPCRequest.model_validate(body)
            except ValidationError as e:
                logger.exception('Failed to validate base JSON-RPC request')
                return self._generate_error_response(
                    request_id,
                    A2AError(
                        root=InvalidRequestError(data=json.loads(e.json()))
                    ),
                )

            # 2) Route by method name; unknown -> -32601, known -> validate params (-32602 on failure)
            method = base_request.method

            model_class = self.METHOD_TO_MODEL.get(method)
            if not model_class:
                return self._generate_error_response(
                    request_id, A2AError(root=MethodNotFoundError())
                )
            try:
                specific_request = model_class.model_validate(body)
            except ValidationError as e:
                logger.exception('Failed to validate base JSON-RPC request')
                return self._generate_error_response(
                    request_id,
                    A2AError(
                        root=InvalidParamsError(data=json.loads(e.json()))
                    ),
                )

            # 3) Build call context and wrap the request for downstream handling
            call_context = await self._context_builder.build(request)
            if not call_context.user.is_authenticated:
                return self._generate_error_response(
                    None, A2AError(root=InvalidRequestError(message='Authentication required'))
                )

            request_id = specific_request.id
            a2a_request = A2ARequest(root=specific_request)
            request_obj = a2a_request.root
            handler = self.agents.build(agent_id)
            if isinstance(
                request_obj,
                TaskResubscriptionRequest | SendStreamingMessageRequest,
            ):
                return await self._process_streaming_request(
                    handler, request_id, a2a_request, call_context
                )

            return await self._process_non_streaming_request(
                handler, request_id, a2a_request, call_context
            )
        except MethodNotImplementedError:
            traceback.print_exc()
            return self._generate_error_response(
                request_id, A2AError(root=UnsupportedOperationError())
            )
        except json.decoder.JSONDecodeError as e:
            traceback.print_exc()
            return self._generate_error_response(
                None, A2AError(root=JSONParseError(message=str(e)))
            )
        except HTTPException as e:
            if e.status_code == HTTP_413_REQUEST_ENTITY_TOO_LARGE:
                return self._generate_error_response(
                    request_id,
                    A2AError(
                        root=InvalidRequestError(message='Payload too large')
                    ),
                )
            raise e
        except Exception as e:
            logger.exception('Unhandled exception')
            return self._generate_error_response(
                request_id, A2AError(root=InternalError(message=str(e)))
            )

    async def _process_streaming_request(
        self,
        handler: JSONRPCHandler,
        request_id: str | int | None,
        a2a_request: A2ARequest,
        context: ServerCallContext,
    ) -> Response:
        """Processes streaming requests (message/stream or tasks/resubscribe).

        Args:
            request_id: The ID of the request.
            a2a_request: The validated A2ARequest object.
            context: The ServerCallContext for the request.

        Returns:
            An `EventSourceResponse` object to stream results to the client.
        """
        request_obj = a2a_request.root
        handler_result: Any = None
        if isinstance(
            request_obj,
            SendStreamingMessageRequest,
        ):
            handler_result = handler.on_message_send_stream(
                request_obj, context
            )
        elif isinstance(request_obj, TaskResubscriptionRequest):
            handler_result = handler.on_resubscribe_to_task(
                request_obj, context
            )

        return self._create_response(context, handler_result)

    async def _process_non_streaming_request(
        self,
        handler: JSONRPCHandler,
        request_id: str | int | None,
        a2a_request: A2ARequest,
        context: ServerCallContext,
    ) -> Response:
        """Processes non-streaming requests (message/send, tasks/get, tasks/cancel, tasks/pushNotificationConfig/*).

        Args:
            request_id: The ID of the request.
            a2a_request: The validated A2ARequest object.
            context: The ServerCallContext for the request.

        Returns:
            A `JSONResponse` object containing the result or error.
        """
        request_obj = a2a_request.root
        handler_result: Any = None
        match request_obj:
            case SendMessageRequest():
                handler_result = await handler.on_message_send(
                    request_obj, context
                )
            case CancelTaskRequest():
                handler_result = await handler.on_cancel_task(
                    request_obj, context
                )
            case GetTaskRequest():
                handler_result = await handler.on_get_task(
                    request_obj, context
                )
            case SetTaskPushNotificationConfigRequest():
                handler_result = (
                    await handler.set_push_notification_config(
                        request_obj,
                        context,
                    )
                )
            case GetTaskPushNotificationConfigRequest():
                handler_result = (
                    await handler.get_push_notification_config(
                        request_obj,
                        context,
                    )
                )
            case ListTaskPushNotificationConfigRequest():
                handler_result = (
                    await handler.list_push_notification_config(
                        request_obj,
                        context,
                    )
                )
            case DeleteTaskPushNotificationConfigRequest():
                handler_result = (
                    await handler.delete_push_notification_config(
                        request_obj,
                        context,
                    )
                )
            case GetAuthenticatedExtendedCardRequest():
                handler_result = (
                    await handler.get_authenticated_extended_card(
                        request_obj,
                        context,
                    )
                )
            case _:
                logger.error(
                    'Unhandled validated request type: %s', type(request_obj)
                )
                error = UnsupportedOperationError(
                    message=f'Request type {type(request_obj).__name__} is unknown.'
                )
                handler_result = JSONRPCErrorResponse(
                    id=request_id, error=error
                )

        return self._create_response(context, handler_result)

    def _create_response(
        self,
        context: ServerCallContext,
        handler_result: (
            AsyncGenerator[SendStreamingMessageResponse]
            | JSONRPCErrorResponse
            | JSONRPCResponse
        ),
    ) -> Response:
        """Creates a Starlette Response based on the result from the request handler.

        Handles:
        - AsyncGenerator for Server-Sent Events (SSE).
        - JSONRPCErrorResponse for explicit errors returned by handlers.
        - Pydantic RootModels (like GetTaskResponse) containing success or error
        payloads.

        Args:
            context: The ServerCallContext provided to the request handler.
            handler_result: The result from a request handler method. Can be an
                async generator for streaming or a Pydantic model for non-streaming.

        Returns:
            A Starlette JSONResponse or EventSourceResponse.
        """
        headers = {}
        if exts := context.activated_extensions:
            headers[HTTP_EXTENSION_HEADER] = ', '.join(sorted(exts))
        if isinstance(handler_result, AsyncGenerator):
            # Result is a stream of SendStreamingMessageResponse objects
            async def event_generator(
                stream: AsyncGenerator[SendStreamingMessageResponse],
            ) -> AsyncGenerator[dict[str, str]]:
                async for item in stream:
                    yield {'data': item.root.model_dump_json(exclude_none=True)}

            return EventSourceResponse(
                event_generator(handler_result), headers=headers
            )
        if isinstance(handler_result, JSONRPCErrorResponse):
            return JSONResponse(
                handler_result.model_dump(
                    mode='json',
                    exclude_none=True,
                ),
                headers=headers,
            )

        return JSONResponse(
            handler_result.root.model_dump(mode='json', exclude_none=True),
            headers=headers,
        )

    async def _handle_get_agent_card(self, request: Request) -> JSONResponse:
        """Handles GET requests for the agent card endpoint.

        Args:
            request: The incoming Starlette Request object.

        Returns:
            A JSONResponse containing the agent card data.
        """
        agent_id = request.path_params.get('agent_id')
        if not agent_id or agent_id not in self.agent_cards:
            return self._generate_error_response(
                None,
                A2AError(
                    root=InvalidRequestError(message='Agent not found')
                ),
            )
        return JSONResponse(
            self.agent_cards[agent_id].model_dump(
                exclude_none=True,
                by_alias=True,
            )
        )

    def routes(
        self,
    ) -> list[Route]:
        """Returns the Starlette Routes for handling A2A requests.

        Args:
            agent_card_url: The URL path for the agent card endpoint.
            rpc_url: The URL path for the A2A JSON-RPC endpoint (POST requests).
            extended_agent_card_url: The URL for the authenticated extended agent card endpoint.

        Returns:
            A list of Starlette Route objects.
        """
        routes = [
            Route(
                "/{agent_id}",
                self._handle_requests,
                methods=["POST"],
                name="a2a_handler",
            ),
            Route(
                "/{agent_id}/.well-known/agent.json",
                self._handle_get_agent_card,
                methods=["GET"],
                name="agent_card_deprecated",
            ),
            Route(
                "/{agent_id}/.well-known/agent-card.json",
                self._handle_get_agent_card,
                methods=["GET"],
                name="agent_card",
            ),
        ]
        return routes

    def build(
        self,
        **kwargs: Any,
    ) -> Starlette:
        """Builds and returns the Starlette application instance.

        Args:
            **kwargs: Additional keyword arguments to pass to the Starlette constructor.

        Returns:
            A configured Starlette application instance.
        """
        app = Starlette(**kwargs)
        app.routes.extend(self.routes())
        return app
