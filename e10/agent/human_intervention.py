# agent/human_intervention.py

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any

@dataclass
class HumanIntervention:
    """Data class to track human interventions in the agent workflow"""
    timestamp: datetime
    step_index: int
    tool_name: str
    tool_arguments: Dict[str, Any]
    error_message: str
    human_input: str
    attempt_number: int
    lifelines_remaining: int
    was_successful: bool = False
    next_step_decision: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "step_index": self.step_index,
            "tool_name": self.tool_name,
            "tool_arguments": self.tool_arguments,
            "error_message": self.error_message,
            "human_input": self.human_input,
            "attempt_number": self.attempt_number,
            "lifelines_remaining": self.lifelines_remaining,
            "was_successful": self.was_successful,
            "next_step_decision": self.next_step_decision
        }
    
# agent/exceptions.py

class HumanInterventionError(Exception):
    """Custom exception for human intervention related errors.
    
    This exception is raised when:
    1. Human intervention is disabled but attempted
    2. Timeout occurs while waiting for human input
    3. Human input is cancelled
    4. Any other error occurs during human intervention process
    """
    def __init__(self, message: str, error_type: str = "general"):
        self.message = message
        self.error_type = error_type
        super().__init__(self.message)

    def __str__(self):
        return f"HumanInterventionError ({self.error_type}): {self.message}"