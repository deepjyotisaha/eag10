# agent/human_intervention.py

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any
from config.log_config import setup_logging
from agent.utils import show_input_dialog
from agent.exceptions import HumanInterventionError

logger = setup_logging(__name__)

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

class HumanInterventionHandler:
    """Handler class for managing human interventions in the agent workflow"""
    
    def __init__(self, max_lifelines: int = 3):
        self.max_lifelines = max_lifelines

    async def get_human_input(self, step, tool_name: str, tool_args: dict, error: str) -> HumanIntervention:
        """Get input from human when tool fails, planning input needed, or clarification required"""
        # Calculate remaining lifelines
        remaining_lifelines = self.max_lifelines - step.attempts

        # Determine the type of intervention needed
        intervention_type = "tool_error"
        if tool_name == "planning_input_request":
            intervention_type = "planning"
        elif tool_name == "clarification_request":
            intervention_type = "clarification"

        # Build the appropriate message based on intervention type
        if intervention_type == "tool_error":
            prompt = (
                f"\n🤖 Tool Execution Failed - Human Intervention Required\n"
                f"Step {step.index}: {step.description}\n"
                f"Tool: {tool_name}\n"
                f"Arguments: {tool_args}\n"
                f"Error: {error}\n"
                f"Attempt: {step.attempts + 1} of {self.max_lifelines}\n"
                f"Lifelines remaining: {remaining_lifelines}\n"
                f"Please provide the expected output for the tool execution:"
            )
        elif intervention_type == "planning":
            prompt = (
                f"\n🤖 Planning Input Required - Human Intervention\n"
                f"Step {step.index}: {step.description}\n"
                f"Current Plan: {tool_args.get('message', 'No plan details available')}\n"
                f"Error: {error}\n"
                f"Attempt: {step.attempts + 1} of {self.max_lifelines}\n"
                f"Lifelines remaining: {remaining_lifelines}\n"
                f"Please provide guidance for replanning this step:"
            )
        else:  # clarification
            prompt = (
                f"\n🤖 Clarification Required - Human Intervention\n"
                f"Step {step.index}: {step.description}\n"
                f"Context: {tool_args.get('message', 'No context available')}\n"
                f"Attempt: {step.attempts + 1} of {self.max_lifelines}\n"
                f"Lifelines remaining: {remaining_lifelines}\n"
                f"Please provide clarification for this step:"
            )

        logger.info(prompt)
        
        try:
            # Use the show_input_dialog function from main.py
            human_input = await show_input_dialog(prompt)
            
            # Create and return the intervention record
            return HumanIntervention(
                timestamp=datetime.now(),
                step_index=step.index,
                tool_name=tool_name,
                tool_arguments=tool_args,
                error_message=error,
                human_input=human_input,
                attempt_number=step.attempts + 1,
                lifelines_remaining=remaining_lifelines,
                was_successful=True
            )
            
        except Exception as e:
            raise HumanInterventionError(f"Error getting human input: {str(e)}", "general")