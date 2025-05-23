import asyncio
import yaml
from mcp_servers.multiMCP import MultiMCP
from agent.agent_loop2 import AgentLoop
from agent.agentSession import Step
from pprint import pprint
from agent.exceptions import HumanInterventionError
from agent.utils import show_input_dialog

BANNER = """
──────────────────────────────────────────────────────
🔸  Agentic Query Assistant  🔸
Type your question and press Enter.
Type 'exit' or 'quit' to leave.
──────────────────────────────────────────────────────
"""


async def interactive() -> None:
    print(BANNER)
    print("Loading MCP Servers...")
    with open("config/mcp_server_config.yaml", "r") as f:
        profile = yaml.safe_load(f)
        mcp_servers_list = profile.get("mcp_servers", [])
        configs = list(mcp_servers_list)

    # Initialize MCP + Dispatcher
    multi_mcp = MultiMCP(server_configs=configs)
    await multi_mcp.initialize()
    loop = AgentLoop(
        perception_prompt_path="prompts/perception_prompt.txt",
        decision_prompt_path="prompts/decision_prompt.txt",
        multi_mcp=multi_mcp,
        strategy="exploratory"
    )
    while True:

        query = input("🟢  You: ").strip()
        if query.lower() in {"exit", "quit"}:
            print("👋  Goodbye!")
            break


        response = await loop.run(query)
        # response = await loop.run("What is 4 + 4?")
        # pprint(f"🔵  Agent: {response.state['final_answer']}\n {response.state['reasoning_note']}\n")
        print(f"🔵 Agent: {response.state['solution_summary']}\n")

        follow = input("\n\nContinue? (press Enter) or type 'exit': ").strip()
        if follow.lower() in {"exit", "quit"}:
            print("👋  Goodbye!")
            break

async def handle_human_input(agent_loop: AgentLoop, input_text: str):
    """Handle human input from UI"""
    agent_loop.set_human_input(input_text)


async def on_tool_failure(agent_loop: AgentLoop, step: Step, tool_name: str, tool_args: dict, error: str):
    """Called when a tool fails and human intervention is needed"""
    if agent_loop.human_intervention["enabled"]:
        try:
            # Show input dialog to user with detailed information
            input_text = await show_input_dialog(
                f"Tool Execution Failed - Human Intervention Required\n"
                f"Step {step.index}: {step.description}\n"
                f"Tool: {tool_name}\n"
                f"Arguments: {tool_args}\n"
                f"Error: {error}\n"
                f"Attempt: {step.attempts + 1} of {agent_loop.max_lifelines}\n"
                f"Lifelines remaining: {agent_loop.max_lifelines - step.attempts}\n"
                f"Please provide the expected output:"
            )
            await handle_human_input(agent_loop, input_text)
        except HumanInterventionError as he:
            print(f"\n❌ {str(he)}")
            # The error will be handled by the agent loop
            raise

if __name__ == "__main__":
    asyncio.run(interactive())
