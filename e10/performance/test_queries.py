# performance/test_queries.py

import os
import csv
import time
import logging
import asyncio
from pathlib import Path
from typing import List, Dict

# Add the root directory to Python path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from agent.agent_loop2 import AgentLoop2
from agent.model_manager import ModelManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
NUM_QUERIES = 10  # Number of queries to process
SLEEP_TIME = 5    # Sleep time between queries in seconds
INPUT_FILE = "test_queries_input.csv"
OUTPUT_FILE = "test_queries_output.csv"

class QueryTester:
    def __init__(self):
        """Initialize the query tester with agent and model manager"""
        self.model_manager = ModelManager()
        self.agent_loop = AgentLoop2(self.model_manager)
        
    async def execute_query(self, query: str) -> Dict:
        """Execute a single query using the agent"""
        try:
            # Run the query through the agent
            session = await self.agent_loop.run(query)
            
            # Extract the final plan and output
            final_plan = session.current_plan if session.current_plan else "No plan generated"
            output = session.perception_result.get("solution_summary", "No output generated")
            
            return {
                "query": query,
                "final_plan": final_plan,
                "output": output
            }
            
        except Exception as e:
            logger.error(f"Error executing query '{query}': {str(e)}")
            return {
                "query": query,
                "final_plan": f"Error: {str(e)}",
                "output": "Failed to execute query"
            }

async def main():
    """Main function to process queries"""
    try:
        # Initialize query tester
        tester = QueryTester()
        
        # Read queries from input file
        queries = []
        with open(INPUT_FILE, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                queries.append(row['Query'])
        
        # Process specified number of queries
        queries = queries[:NUM_QUERIES]
        logger.info(f"Processing {len(queries)} queries...")
        
        # Process each query
        results = []
        for i, query in enumerate(queries, 1):
            logger.info(f"Processing query {i}/{len(queries)}: {query}")
            
            # Execute query
            result = await tester.execute_query(query)
            results.append(result)
            
            # Sleep between queries
            if i < len(queries):
                logger.info(f"Sleeping for {SLEEP_TIME} seconds...")
                await asyncio.sleep(SLEEP_TIME)
        
        # Write results to output file
        with open(OUTPUT_FILE, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['Query', 'Final Plan', 'Output'])
            writer.writeheader()
            for result in results:
                writer.writerow({
                    'Query': result['query'],
                    'Final Plan': result['final_plan'],
                    'Output': result['output']
                })
        
        logger.info(f"Results written to {OUTPUT_FILE}")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())