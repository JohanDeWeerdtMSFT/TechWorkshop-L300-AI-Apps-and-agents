import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    PromptAgentDefinition,
    PromptAgentDefinitionTextOptions,
    TextResponseFormatJsonSchema,
)
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

from app.agents.agent_initializer import initialize_agent
from app.agents.tool_definitions import get_tools_for_agent_oneshot
from services.handoff_service import IntentClassification


PROMPTS = ROOT / "prompts"

AGENTS = [
    {
        "agent_type": "customer_loyalty",
        "name": "customer-loyalty",
        "description": "Zava Customer Loyalty Agent",
        "prompt": "CustomerLoyaltyAgentPrompt.txt",
    },
    {
        "agent_type": "inventory_agent",
        "name": "inventory-agent",
        "description": "Zava Inventory Agent",
        "prompt": "InventoryAgentPrompt.txt",
    },
    {
        "agent_type": "interior_designer",
        "name": "interior-designer",
        "description": "Zava Interior Design Agent",
        "prompt": "InteriorDesignAgentPrompt.txt",
    },
    {
        "agent_type": "cora",
        "name": "cora",
        "description": "Cora - Zava Shopping Assistant",
        "prompt": "ShopperAgentPrompt.txt",
    },
    {
        "agent_type": "cart_manager",
        "name": "cart-manager",
        "description": "Zava Cart Manager Agent",
        "prompt": "CartManagerPrompt.txt",
    },
]


def read_prompt(filename: str) -> str:
    return (PROMPTS / filename).read_text(encoding="utf-8")


def deploy_handoff_agent(project_client: AIProjectClient, model: str) -> None:
    with project_client:
        agent = project_client.agents.create_version(
            agent_name="handoff-service",
            description="Zava Handoff Service Agent",
            definition=PromptAgentDefinition(
                model=model,
                text=PromptAgentDefinitionTextOptions(
                    format=TextResponseFormatJsonSchema(
                        name="IntentClassification",
                        schema=IntentClassification.model_json_schema(),
                    )
                ),
                instructions=read_prompt("HandoffAgentPrompt.txt"),
            ),
        )
        print(f"Created handoff-service agent, ID: {agent.id}")


async def main() -> None:
    load_dotenv()

    project_endpoint = os.environ["FOUNDRY_ENDPOINT"]
    model = os.environ["gpt_deployment"]
    project_client = AIProjectClient(
        endpoint=project_endpoint,
        credential=DefaultAzureCredential(),
    )

    for config in AGENTS:
        tools = await get_tools_for_agent_oneshot(config["agent_type"])
        initialize_agent(
            project_client=project_client,
            model=model,
            name=config["name"],
            description=config["description"],
            instructions=read_prompt(config["prompt"]),
            tools=tools,
        )

    deploy_handoff_agent(project_client, model)


if __name__ == "__main__":
    asyncio.run(main())
