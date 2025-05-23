# agent/tool_simulation.py

import random
from dataclasses import dataclass
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class FailureType:
    type: str
    message: str
    probability: float

class ToolSimulation:
    def __init__(self, config: dict):
        self.enabled = config.get("enabled", False)
        self.failure_rate = config.get("failure_rate", 0)  # Default to 0% failure rate
        self.failure_types = [
            FailureType(**ft) for ft in config.get("failure_types", [])
        ]
        
    def should_fail(self) -> bool:
        """Determine if the current tool execution should fail based on failure rate"""
        if not self.enabled:
            return False
        return random.random() * 100 < self.failure_rate
        
    def get_failure(self) -> FailureType:
        """Get a random failure type based on probabilities"""
        if not self.failure_types:
            return FailureType(
                type="generic",
                message="Simulated tool failure",
                probability=1.0
            )
            
        # Normalize probabilities
        total_prob = sum(ft.probability for ft in self.failure_types)
        if total_prob == 0:
            return self.failure_types[0]
            
        # Select failure type based on probability
        r = random.random() * total_prob
        cum_prob = 0
        for ft in self.failure_types:
            cum_prob += ft.probability
            if r <= cum_prob:
                return ft
        return self.failure_types[-1]  # Fallback to last failure type

    def simulate_tool_execution(self, tool_name: str, tool_args: dict) -> dict:
        """Simulate a tool execution with probability-based failure when enabled"""
        if not self.enabled:
            return {"status": "success", "result": "Simulation disabled"}
            
        # Check if this execution should fail based on failure rate
        if self.should_fail():
            failure = self.get_failure()
            logger.info(f"🔧 Simulating {failure.type} failure for tool {tool_name}")
            raise Exception(f"Simulated {failure.type} error: {failure.message}")
            
        return {"status": "success", "result": "Simulation succeeded"}