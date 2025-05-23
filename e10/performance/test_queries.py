# performance/test_queries.py

import os
import csv
import time
import logging
import asyncio
import yaml
from pathlib import Path
from typing import List, Dict

# Add the root directory to Python path
import sys
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from agent.agent_loop2 import AgentLoop
from mcp_servers.multiMCP import MultiMCP
from memory.session_log import extract_session_state

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
NUM_QUERIES = 2  # Number of queries to process
SLEEP_TIME = 5    # Sleep time between queries in seconds
INPUT_FILE = Path(__file__).parent / "test_queries_input.csv"
OUTPUT_FILE = Path(__file__).parent / "test_queries_output.csv"

class QueryTester:
    def __init__(self):
        """Initialize the query tester with agent and MCP"""
        # Load server configs from yaml
        with open(root_dir / "config" / "mcp_server_config.yaml", "r") as f:
            server_configs = yaml.safe_load(f)
            
        # Initialize MCP + Dispatcher
        self.multi_mcp = MultiMCP(server_configs=server_configs)
        
        # Initialize agent loop
        self.agent_loop = AgentLoop(
            perception_prompt_path="prompts/perception_prompt.txt",
            decision_prompt_path="prompts/decision_prompt.txt",
            multi_mcp=self.multi_mcp,
            strategy="exploratory"
        )
        
    async def execute_query(self, query: str, tools: str, complexity: str) -> Dict:
        """
        Execute a single query through the agent and capture the results.
        
        Args:
            query: The query to execute
            tools: Tools needed for the query
            complexity: Complexity level of the query
            
        Returns:
            Dict containing query results including plan, output, and tool usage
        """
        try:
            # Run the query through the agent
            session = await self.agent_loop.run(query)
            
            # Extract session state using the new function
            session_state = extract_session_state(session)

            logger.info(f"Session: {session}")
            logger.info(f"Session State: {session_state}")
            
            # Format the final plan - handle both list and string cases
            final_plan = session_state["final_plan"]
            if isinstance(final_plan, list):
                final_plan = "\n".join(final_plan)
            
            # Get the final answer, falling back to solution_summary if needed
            final_answer = session_state["final_answer"]
            
            # Log the tool usage for analysis
            tool_usage = session_state["tool_usage"]
            if tool_usage:
                logger.info(f"Tool usage for query '{query}':")
                for tool in tool_usage:
                    logger.info(f"- {tool['tool_name']}: {tool['status']} (Step {tool['step_index']})")
            
            logger.info(f"Final Plan: {final_plan}")
            logger.info(f"Final Answer: {final_answer}")    
            logger.info(f"Tool Usage: {tool_usage}")

            return {
                "query": query,
                "tools": tools,
                "complexity": complexity,
                "final_plan": final_plan,
                "output": final_answer,
                "tool_usage": tool_usage  # Include tool usage in the result
            }
            
        except Exception as e:
            logger.error(f"Error executing query '{query}': {str(e)}")
            return {
                "query": query,
                "tools": tools,
                "complexity": complexity,
                "final_plan": f"Error: {str(e)}",
                "output": "Failed to execute query",
                "tool_usage": []  # Empty tool usage for failed queries
            }

async def main():
    """Main function to process queries one at a time"""
    try:
        # Initialize query tester
        tester = QueryTester()
        
        # Create output file with headers (overwriting any existing file)
        with open(OUTPUT_FILE, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['Query', 'Tools Needed', 'Complexity', 'Final Plan', 'Output'])
            writer.writeheader()
        
        # Process queries one at a time
        with open(INPUT_FILE, 'r') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= NUM_QUERIES:
                    break
                    
                query = row['Query']
                tools = row['Tools Needed']
                complexity = row['Complexity']
                
                logger.info(f"Processing query {i+1}/{NUM_QUERIES}: {query}")
                
                # Execute query
                result = await tester.execute_query(query, tools, complexity)
                
                # Write result immediately
                with open(OUTPUT_FILE, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=['Query', 'Tools Needed', 'Complexity', 'Final Plan', 'Output'])
                    writer.writerow({
                        'Query': result['query'],
                        'Tools Needed': result['tools'],
                        'Complexity': result['complexity'],
                        'Final Plan': result['final_plan'],
                        'Output': result['output']
                    })
                
                # Sleep between queries
                if i < NUM_QUERIES - 1:
                    logger.info(f"Sleeping for {SLEEP_TIME} seconds...")
                    await asyncio.sleep(SLEEP_TIME)
        
        logger.info(f"All queries processed. Results written to {OUTPUT_FILE}")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())