import os
import json
import yaml
import csv
import logging
from pathlib import Path
from typing import List, Dict
from google import genai
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

NUM_QUERIES = 100  # Configurable number of queries to generate

class QueryGenerator:
    def __init__(self):
        """Initialize the query generator with configuration"""
        self.root = Path(__file__).parent.parent
        self.config_path = self.root / "config" / "models.json"
        self.profile_path = self.root / "config" / "profiles.yaml"
        
        # Load configuration
        self.config = json.loads(self.config_path.read_text())
        self.profile = yaml.safe_load(self.profile_path.read_text())
        
        # Initialize Gemini client
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        self.client = genai.Client(api_key=api_key)
        
        # Get model info
        self.text_model_key = self.profile["llm"]["text_generation"]
        self.model_info = self.config["models"][self.text_model_key]
        
        # Load prompt template
        self.prompt_path = self.root / "performance" / "query_generator_prompt.txt"
        self.prompt_template = self.prompt_path.read_text()
        
        # Query distribution
        self.complexity_distribution = {
            "very_low": 0.3,  # 30% very low complexity
            "low": 0.3,       # 30% low complexity
            "medium": 0.25,   # 25% medium complexity
            "high": 0.15      # 15% high complexity
        }
        
        # Number of queries to generate
        self.num_queries = NUM_QUERIES  # Use the config value
        
    def _generate_with_llm(self, prompt: str) -> str:
        """Generate text using the LLM"""
        try:
            response = self.client.models.generate_content(
                model=self.model_info["model"],
                contents=prompt
            )
            
            # Safely extract response text
            try:
                return response.text.strip()
            except AttributeError:
                try:
                    return response.candidates[0].content.parts[0].text.strip()
                except Exception:
                    return str(response)
                    
        except Exception as e:
            logger.error(f"Error generating with LLM: {str(e)}")
            raise
    
    def _parse_llm_response(self, response: str) -> List[Dict]:
        """Parse the LLM response into a list of query dictionaries"""
        queries = []
        try:
            # Split response into lines and process each line
            lines = response.strip().split('\n')
            for line in lines:
                if not line.strip():
                    continue
                    
                # Parse CSV format
                parts = [part.strip() for part in line.split(',')]
                if len(parts) >= 3:
                    # Convert complexity to lowercase to match our distribution keys
                    complexity = parts[2].lower()
                    # Validate complexity level
                    if complexity not in self.complexity_distribution:
                        logger.warning(f"Invalid complexity level '{complexity}', skipping query")
                        continue
                        
                    query = {
                        "query": parts[0],
                        "tools": [tool.strip() for tool in parts[1].split('|')],
                        "complexity": complexity  # Store in lowercase
                    }
                    queries.append(query)
                    
            if not queries:
                raise ValueError("No valid queries were parsed from the LLM response")
            
            return queries
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {str(e)}")
            raise
    
    def generate_queries(self) -> List[Dict]:
        """Generate queries using the LLM"""
        # Prepare the prompt with the number of queries
        prompt = self.prompt_template.replace("10", str(self.num_queries))
        
        # Generate queries using LLM
        logger.info("Generating queries with LLM...")
        response = self._generate_with_llm(prompt)
        
        # Parse the response
        queries = self._parse_llm_response(response)
        
        # Validate the distribution
        complexity_counts = {level: 0 for level in self.complexity_distribution}
        for query in queries:
            complexity_counts[query["complexity"]] += 1
            
        # Log the distribution
        logger.info("Query complexity distribution:")
        for level, count in complexity_counts.items():
            percentage = (count / self.num_queries) * 100
            logger.info(f"{level}: {count} queries ({percentage:.1f}%)")
            
        return queries
    
    def write_queries_to_csv(self, queries: List[Dict], output_file: str):
        """Write the generated queries to a CSV file"""
        try:
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Query', 'Tools Needed', 'Complexity'])
                
                for query in queries:
                    writer.writerow([
                        query['query'],
                        '|'.join(query['tools']),
                        query['complexity']
                    ])
                    
            logger.info(f"Queries written to {output_file}")
            
        except Exception as e:
            logger.error(f"Error writing queries to CSV: {str(e)}")
            raise

def main():
    """Main function to generate and write queries"""
    try:
        # Initialize query generator
        generator = QueryGenerator()
        
        # Generate queries
        queries = generator.generate_queries()
        
        # Write queries to CSV
        output_file = Path(__file__).parent / "test_queries_input.csv"
        generator.write_queries_to_csv(queries, output_file)
        
        logger.info("Query generation completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()