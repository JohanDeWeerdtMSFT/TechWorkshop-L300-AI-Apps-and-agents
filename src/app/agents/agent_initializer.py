from typing import List, Any
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import PromptAgentDefinition
from dotenv import load_dotenv

load_dotenv()

def initialize_agent(project_client : AIProjectClient, model : str, name : str, description : str, instructions : str, tools : List[Any]):
    agent = project_client.agents.create_version(
        agent_name=name,
        description=description,
        definition=PromptAgentDefinition(
            model=model,
            instructions=instructions,
            tools=tools
        )
    )
    print(f"Created {name} agent, ID: {agent.id}")
