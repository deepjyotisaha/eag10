import uuid
import json
import datetime
import yaml
from perception.perception import Perception
from decision.decision import Decision
from action.executor import run_user_code
from agent.agentSession import AgentSession, PerceptionSnapshot, Step, ToolCode
from memory.session_log import live_update_session
from memory.memory_search import MemorySearch
from mcp_servers.multiMCP import MultiMCP
from config.log_config import setup_logging

logger = setup_logging(__name__)

GLOBAL_PREVIOUS_FAILURE_STEPS = 3

class AgentLoop:
    def __init__(self, perception_prompt_path: str, decision_prompt_path: str, multi_mcp: MultiMCP, strategy: str = "exploratory"):
        self.perception = Perception(perception_prompt_path)
        self.decision = Decision(decision_prompt_path, multi_mcp)
        self.multi_mcp = multi_mcp
        self.strategy = strategy
        
        # Load configuration from profiles.yaml
        with open("config/profiles.yaml", "r") as f:
            config = yaml.safe_load(f)
            self.max_steps = config["strategy"]["max_steps"]
            self.max_lifelines = config["strategy"]["max_lifelines_per_step"]
        
        self.total_steps_executed = 0
        self.step_retries = {}  # Track retries per step

    async def run(self, query: str):
        session = AgentSession(
            session_id=str(uuid.uuid4()), 
            original_query=query
        )
        session_memory = []
        self.log_session_start(session, query)
        self.total_steps_executed = 0  # Reset step counter for new session
        self.step_retries = {}  # Reset retries

        memory_results = self.search_memory(query)
        perception_result = self.run_perception(query, memory_results, memory_results)
        logger.info("\n📋 [Perception Result]: \n%s", json.dumps(perception_result, indent=2, ensure_ascii=False))
        session.add_perception(PerceptionSnapshot(**perception_result))

        if perception_result.get("original_goal_achieved"):
            self.handle_perception_completion(session, perception_result)
            return session

        decision_output = self.make_initial_decision(query, perception_result)
        logger.info("\n📝 [Decision Output]: \n%s", json.dumps(decision_output, indent=2, ensure_ascii=False))
        step = session.add_plan_version(decision_output["plan_text"], [self.create_step(decision_output)])
        live_update_session(session)
        logger.info(f"\n📝 [Decision Plan Text: V{len(session.plan_versions)}]:")
        for line in session.plan_versions[-1]["plan_text"]:
            logger.info(f"  {line}")

        #First implementation the max steps and max retries
        # If a plan fails, 
        while step:
            # Check max steps before executing
            logger.info(f"\n⚙️ [Out of {self.max_steps} steps, trying to execute step number {self.total_steps_executed + 1}]")
            if self.total_steps_executed >= self.max_steps:
                logger.info(f"\n⚠️ Maximum steps ({self.max_steps}) reached. Stopping execution.")
                conclusion_step = self.create_conclusion_step(
                    "Maximum steps reached",
                    f"Execution stopped after {self.total_steps_executed} steps due to step limit."
                )
                session.add_plan_version(
                    [f"Step {self.total_steps_executed + 1}: Maximum steps reached"],
                    [conclusion_step]
                )
                session.state.update({
                    "original_goal_achieved": False,
                    "final_answer": f"Execution stopped after {self.total_steps_executed} steps. Please try a more specific query or break down your request into smaller parts.",
                    "confidence": 0.8,
                    "reasoning_note": "Step limit reached",
                    "solution_summary": f"Maximum steps ({self.max_steps}) reached. Please refine your query."
                })
                live_update_session(session)
                break

            step_result = await self.execute_step(step, session, session_memory)
            if step_result is None:
                logger.info("\n❌ No steps.")
                break

            # Track step execution
            self.total_steps_executed += 1

            logger.info("\n⚙️ [Evaluating Step Summary]: \n%s", json.dumps(step_result.to_dict(), indent=2, ensure_ascii=False))
            step = self.evaluate_step(step_result, session, query)

        return session

    def log_session_start(self, session, query):
        logger.info("\n=== LIVE AGENT SESSION TRACE ===")
        logger.info(f"Session ID: {session.session_id}")
        logger.info(f"Query: {query}")

    def search_memory(self, query):
        logger.info("Searching Recent Conversation History")
        searcher = MemorySearch()
        results = searcher.search_memory(query)
        if not results:
            logger.info("❌ No matching memory entries found.\n")
        else:
            logger.info("\n🎯 Top Matches:\n")
            for i, res in enumerate(results, 1):
                logger.info(f"[{i}] File: {res['file']}\nQuery: {res['query']}\nResult Requirement: {res['result_requirement']}\nSummary: {res['solution_summary']}\n")
        return results

    def run_perception(self, query, memory_results, session_memory=None, snapshot_type="user_query", current_plan=None):
        combined_memory = (memory_results or []) + (session_memory or [])
        perception_input = self.perception.build_perception_input(
            raw_input=query, 
            memory=combined_memory, 
            current_plan=current_plan, 
            snapshot_type=snapshot_type
        )
        logger.info("\n📋 [Perception Input]: \n%s", json.dumps(perception_input, indent=2, ensure_ascii=False))
        perception_result = self.perception.run(perception_input)
        #logger.info("\n📋 [Perception Result]: \n%s", json.dumps(perception_result, indent=2, ensure_ascii=False))
        return perception_result

    def handle_perception_completion(self, session, perception_result):
        logger.info("\n✅ Perception fully answered the query.")
        session.state.update({
            "original_goal_achieved": True,
            "final_answer": perception_result.get("solution_summary", "Answer ready."),
            "confidence": perception_result.get("confidence", 0.95),
            "reasoning_note": perception_result.get("reasoning", "Handled by perception."),
            "solution_summary": perception_result.get("solution_summary", "Answer ready.")
        })
        live_update_session(session)

    def make_initial_decision(self, query, perception_result):
        decision_input = {
            "plan_mode": "initial",
            "planning_strategy": self.strategy,
            "original_query": query,
            "perception": perception_result
        }
        decision_output = self.decision.run(decision_input)
        return decision_output

    def create_step(self, decision_output):
        return Step(
            index=decision_output["step_index"],
            description=decision_output["description"],
            type=decision_output["type"],
            code=ToolCode(tool_name="raw_code_block", tool_arguments={"code": decision_output["code"]}) if decision_output["type"] == "CODE" else None,
            conclusion=decision_output.get("conclusion"),
            attempts=0,  # Initialize attempts counter
            was_replanned=False,  # Initialize replan flag
            parent_index=None  # Initialize parent index
        )

    async def execute_step(self, step, session, session_memory):
        logger.info(f"\n⚙️ [Step {step.index}] {step.description}")

        if step.type == "CODE":
            logger.info("%s\n⚙️  [EXECUTING CODE]\n%s", "-" * 50, step.code.tool_arguments["code"])
            executor_response = await run_user_code(step.code.tool_arguments["code"], self.multi_mcp)
            step.execution_result = executor_response
            #import pdb; pdb.set_trace()
            ## Human in loop for tool execution needs to go here
            step.status = "completed"

            logger.info("\n⚙️ [Executor Response]: \n%s", json.dumps(executor_response, indent=2, ensure_ascii=False))

            logger.info("\n⚙️ [Decoding executor response via perception]")

            perception_result = self.run_perception(
                query=executor_response.get('result', 'Tool Failed'),
                memory_results=session_memory,
                current_plan=session.plan_versions[-1]["plan_text"],
                snapshot_type="step_result"
            )

            logger.info("\n📋 [Post-Execution Perception Result]: \n%s", json.dumps(perception_result, indent=2, ensure_ascii=False))
            step.perception = PerceptionSnapshot(**perception_result)

            if not step.perception or not step.perception.local_goal_achieved:
                failure_memory = {
                    "query": step.description,
                    "result_requirement": "Tool failed",
                    "solution_summary": str(step.execution_result)[:300]
                }
                session_memory.append(failure_memory)

                if len(session_memory) > GLOBAL_PREVIOUS_FAILURE_STEPS:
                    session_memory.pop(0)

            live_update_session(session)
            logger.info("\n🔁 [Post-Execution Step Summary]: \n%s", json.dumps(step.to_dict(), indent=2, ensure_ascii=False))
            return step

        elif step.type == "CONCLUDE":
            logger.info(f"\n💡 Conclusion: {step.conclusion}")
            step.execution_result = step.conclusion
            step.status = "completed"

            perception_result = self.run_perception(
                query=step.conclusion,
                memory_results=session_memory,
                current_plan=session.plan_versions[-1]["plan_text"],
                snapshot_type="step_result"
            )
            logger.info("\n📋 [Post-Conclusion Perception Result]: \n%s", json.dumps(perception_result, indent=2, ensure_ascii=False))
            step.perception = PerceptionSnapshot(**perception_result)
            session.mark_complete(step.perception, final_answer=step.conclusion)
            live_update_session(session)
            return None

        elif step.type == "NOP":
            logger.info(f"\n❓ Clarification needed: {step.description}")
            step.status = "clarification_needed"
            live_update_session(session)
            return None

    def evaluate_step(self, step, session, query):
        if step.perception.original_goal_achieved:
            logger.info("\n✅ Goal achieved.")
            session.mark_complete(step.perception)
            live_update_session(session)
            return None
        elif step.perception.local_goal_achieved:
            logger.info("\n✅ Local Goal achieved, planning next step.")
            return self.get_next_step(session, query, step)
        else:
            logger.info("\n🔁 Step unhelpful. Replanning.")
            
            # Check if we've exceeded max retries for this step
            if step.attempts >= self.max_lifelines:
                logger.info(f"\n⚠️ Maximum retries ({self.max_lifelines}) reached for step {step.index}.")
                conclusion_step = self.create_conclusion_step(
                    "Maximum retries reached",
                    f"Step {step.index} failed after {step.attempts} attempts. Please try a different approach."
                )
                session.add_plan_version(
                    [f"Step {step.index}: Maximum retries reached"],
                    [conclusion_step]
                )
                session.state.update({
                    "original_goal_achieved": False,
                    "final_answer": f"Step {step.index} failed after {step.attempts} attempts. Please try a different approach.",
                    "confidence": 0.8,
                    "reasoning_note": "Step retry limit reached",
                    "solution_summary": f"Maximum retries ({self.max_lifelines}) reached for step {step.index}. Please try a different approach."
                })
                live_update_session(session)
                return None

            # Increment attempts counter
            step.attempts += 1
            logger.info(f"\n🔄 Attempt {step.attempts} of {self.max_lifelines} for step {step.index}")

            # Replan the step
            decision_output = self.decision.run({
                "plan_mode": "mid_session",
                "planning_strategy": self.strategy,
                "original_query": query,
                "current_plan_version": len(session.plan_versions),
                "current_plan": session.plan_versions[-1]["plan_text"],
                "completed_steps": [s.to_dict() for s in session.plan_versions[-1]["steps"] if s.status == "completed"],
                "current_step": step.to_dict()
            })

            logger.info("\n📝 [Post-Replanning Decision Output]: \n%s", json.dumps(decision_output, indent=2, ensure_ascii=False))

            # Create new step with incremented attempts
            new_step = self.create_step(decision_output)
            new_step.attempts = step.attempts  # Carry over the attempts count
            new_step.was_replanned = True     # Mark as replanned
            new_step.parent_index = step.index # Track original step

            step = session.add_plan_version(decision_output["plan_text"], [new_step])

            logger.info(f"\n📝 [Decision Plan Text: V{len(session.plan_versions)}]:")
            for line in session.plan_versions[-1]["plan_text"]:
                logger.info(f"  {line}")

            return step

    def get_next_step(self, session, query, step):
        next_index = step.index + 1
        total_steps = len(session.plan_versions[-1]["plan_text"])
        logger.info(f"\n🔄 [Next Step Index: {next_index}]")
        logger.info(f"🔄 [Total Steps: {total_steps}]")
        if next_index < total_steps:
            decision_output = self.decision.run({
                "plan_mode": "mid_session",
                "planning_strategy": self.strategy,
                "original_query": query,
                "current_plan_version": len(session.plan_versions),
                "current_plan": session.plan_versions[-1]["plan_text"],
                "completed_steps": [s.to_dict() for s in session.plan_versions[-1]["steps"] if s.status == "completed"],
                "current_step": step.to_dict()
            })

            logger.info("\n📝 [Post-Next-Step-Planning Decision Output]: \n%s", json.dumps(decision_output, indent=2, ensure_ascii=False))

            step = session.add_plan_version(decision_output["plan_text"], [self.create_step(decision_output)])

            logger.info(f"\n📝 [Decision Plan Text: V{len(session.plan_versions)}]:")
            for line in session.plan_versions[-1]["plan_text"]:
                logger.info(f"  {line}")

            return step

        else:
            logger.info("\n✅ No more steps.")
            return None

    def create_conclusion_step(self, title: str, message: str) -> Step:
        """Create a conclusion step when max steps is reached."""
        return Step(
            index=self.total_steps_executed + 1,
            description=title,
            type="CONCLUDE",
            conclusion=message,
            status="completed"
        )