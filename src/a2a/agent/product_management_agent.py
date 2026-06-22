import logging
import os
from collections.abc import AsyncIterable
from typing import Annotated, Any, Literal

from agent_framework import Agent, AgentSession, ChatContext, tool
from agent_framework.openai import OpenAIChatClient, OpenAIChatOptions
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)
load_dotenv()


# region Chat Service Configuration


def get_chat_client() -> OpenAIChatClient:
    """Return Azure OpenAI chat client using the v1 API with managed identity."""
    endpoint = os.getenv("gpt_endpoint")
    deployment_name = os.getenv("gpt_deployment")

    if not endpoint:
        raise ValueError("gpt_endpoint is required")
    if not deployment_name:
        raise ValueError("gpt_deployment is required")

    return OpenAIChatClient(
        model=deployment_name,
        base_url=f"{endpoint.rstrip('/')}/openai/v1/",
        credential=DefaultAzureCredential(),
    )


# endregion


# region Get Products


@tool(
    name="get_products",
    description="Retrieves a set of products based on a natural language user query.",
)
def get_products(
    question: Annotated[
        str,
        "Natural language query to retrieve products, e.g. 'What kinds of paint rollers do you have in stock?'",
    ],
) -> list[dict[str, Any]] | str:
    try:
        return [
            {
                "id": "1",
                "name": "Eco-Friendly Paint Roller",
                "type": "Paint Roller",
                "description": "A high-quality, eco-friendly paint roller for smooth finishes.",
                "punchLine": "Roll with the best, paint with the rest!",
                "price": 15.99,
            },
            {
                "id": "2",
                "name": "Premium Paint Brush Set",
                "type": "Paint Brush",
                "description": "A set of premium paint brushes for detailed work and fine finishes.",
                "punchLine": "Brush up your skills with our premium set!",
                "price": 25.49,
            },
            {
                "id": "3",
                "name": "All-Purpose Paint Tray",
                "type": "Paint Tray",
                "description": "A durable paint tray suitable for all types of rollers and brushes.",
                "punchLine": "Tray it, paint it, love it!",
                "price": 9.99,
            },
        ]
    except Exception as exc:
        return f"Product recommendation failed: {exc!s}"


# endregion


# region Response Format


class ResponseFormat(BaseModel):
    """Response format model to direct how the model should respond."""

    status: Literal["input_required", "completed", "error"] = "input_required"
    message: str


# endregion


# region Agent Framework Agent


class AgentFrameworkProductManagementAgent:
    """Wraps Microsoft Agent Framework agents for Zava product management tasks."""

    agent: Agent
    session: AgentSession | None = None
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def __init__(self):
        chat_service = get_chat_client()

        marketing_agent = Agent(
            client=chat_service,
            name="MarketingAgent",
            instructions=(
                "You specialize in planning and recommending marketing strategies for products. "
                "This includes identifying target audiences, making product descriptions better, "
                "and suggesting promotional tactics. Your goal is to help businesses effectively "
                "market their products and reach their desired customers."
            ),
        )

        ranker_agent = Agent(
            client=chat_service,
            name="RankerAgent",
            instructions=(
                "You specialize in ranking and recommending products based on various criteria. "
                "This includes analyzing product features, customer reviews, and market trends "
                "to provide tailored suggestions. Your goal is to help customers find the best "
                "products for their needs."
            ),
        )

        product_agent = Agent(
            client=chat_service,
            name="ProductAgent",
            instructions=(
                "You specialize in handling product-related requests from customers and employees. "
                "This includes providing product lists, prices, and descriptions exactly as they "
                "exist in the product catalog. You MUST use the get_products tool to answer all "
                "product-related questions. You MUST NEVER answer product questions from your own "
                "knowledge. Do not make up product information. Use only product information from "
                "the get_products tool."
            ),
            tools=get_products,
        )

        self.agent = Agent(
            client=chat_service,
            name="ProductManagerAgent",
            instructions=(
                "Your role is to carefully analyze the user's request and respond as best as you can. "
                "Your primary goal is precise and efficient delegation so customers and employees "
                "receive accurate and specialized assistance promptly. Whenever a query is related "
                "to retrieving product information, you MUST delegate the task to the ProductAgent. "
                "Use the MarketingAgent for marketing-related queries and the RankerAgent for product "
                "ranking and recommendation tasks. You may use these agents together to provide "
                "comprehensive responses.\n\n"
                "IMPORTANT: You must ALWAYS respond with a valid JSON object in the following format:\n"
                '{"status": "<status>", "message": "<your response>"}\n\n'
                "Where status is one of: input_required, completed, or error. Use completed when "
                "the task is finished, input_required when you need more information, and error when "
                "something went wrong. Never respond with plain text."
            ),
            tools=[product_agent.as_tool(), marketing_agent.as_tool(), ranker_agent.as_tool()],
        )

    async def invoke(self, user_input: str, session_id: str) -> dict[str, Any]:
        await self._ensure_session_exists(session_id)
        response = await self.agent.run(
            messages=user_input,
            session=self.session,
            options=OpenAIChatOptions(response_format=ResponseFormat),
        )
        return self._get_agent_response(response.text)

    async def stream(self, user_input: str, session_id: str) -> AsyncIterable[dict[str, Any]]:
        await self._ensure_session_exists(session_id)
        chunks: list[str] = []

        async for chunk in self.agent.run_stream(messages=user_input, session=self.session):
            if chunk.text:
                chunks.append(str(chunk.text))

        if chunks:
            yield self._get_agent_response("".join(chunks))

    def _get_agent_response(self, message: ChatContext | str) -> dict[str, Any]:
        text = str(message)
        default_response = {
            "is_task_complete": True,
            "require_user_input": False,
            "content": text,
        }

        try:
            structured_response = ResponseFormat.model_validate_json(text)
        except ValidationError:
            logger.info("Message did not come in JSON format.")
            return default_response
        except Exception:
            logger.error("An unexpected error occurred while processing the message.")
            return {
                "is_task_complete": False,
                "require_user_input": True,
                "content": "We are unable to process your request at the moment. Please try again.",
            }

        response_map = {
            "input_required": {"is_task_complete": False, "require_user_input": True},
            "error": {"is_task_complete": False, "require_user_input": True},
            "completed": {"is_task_complete": True, "require_user_input": False},
        }
        response = response_map.get(structured_response.status)
        if response:
            return {**response, "content": structured_response.message}
        return default_response

    async def _ensure_session_exists(self, session_id: str) -> None:
        if self.session is None or self.session.service_session_id != session_id:
            self.session = self.agent.create_session(session_id=session_id)


# endregion
