import json
from pathlib import Path
from datetime import datetime
from config.log_config import setup_logging

logger = setup_logging(__name__)

def get_store_path(session_id: str, base_dir: str = "memory/session_logs") -> Path:
    """
    Construct the full path to the session file based on current date and session ID.
    Format: memory/session_logs/YYYY/MM/DD/<session_id>.json
    """
    now = datetime.now()
    day_dir = Path(base_dir) / str(now.year) / f"{now.month:02d}" / f"{now.day:02d}"
    day_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{session_id}.json"
    return day_dir / filename


def simplify_session_id(session_id: str) -> str:
    """
    Return the simplified (short) version of the session ID for display/logging.
    """
    return session_id.split("-")[0]


def append_session_to_store(session_obj, base_dir: str = "memory/session_logs") -> None:
    """
    Save the session object as a standalone file. If a file already exists and is corrupt,
    it will be overwritten with fresh data.
    """
    def datetime_handler(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    session_data = session_obj.to_json()
    session_data["_session_id_short"] = simplify_session_id(session_data["session_id"])

    store_path = get_store_path(session_data["session_id"], base_dir)

    if store_path.exists():
        try:
            with open(store_path, "r", encoding="utf-8") as f:
                existing = f.read().strip()
                if existing:
                    json.loads(existing)  # verify valid JSON
        except json.JSONDecodeError:
            print(f"⚠️ Warning: Corrupt JSON detected in {store_path}. Overwriting.")

    with open(store_path, "w", encoding="utf-8") as f:
        json.dump(session_data, f, indent=2, default=datetime_handler)

    print(f"✅ Session stored: {store_path}")


def live_update_session(session_obj, base_dir: str = "memory/session_logs") -> None:
    """
    Update (or overwrite) the session file with latest data.
    In per-file format, this is identical to append.
    """
    try:
        append_session_to_store(session_obj, base_dir)
        print("📝 Session live-updated.")
    except Exception as e:
        print(f"❌ Failed to update session: {e}")

# memory/session_log.py

from typing import Dict, List, Optional, Tuple

def extract_session_state(session) -> Dict:
    """
    Extract key details from a session object by reading the session log file.
    
    Args:
        session: The session object from AgentSession
        
    Returns:
        Dict containing:
        - final_plan: List of plan steps
        - final_steps: List of executed steps
        - final_answer: The final answer from the session
        - tool_usage: List of tools used with their status
    """
    try:
        # Get session ID from the session object
        session_id = session.session_id
        
        # Get the session log file path using existing get_store_path
        session_file = get_store_path(session_id)
        
        # Read the session file
        if not session_file.exists():
            raise FileNotFoundError(f"Session log file not found: {session_file}")
            
        with open(session_file, 'r') as f:
            session_data = json.load(f)
            
        # Get the state snapshot which contains the final state
        state_snapshot = session_data.get("state_snapshot", {})
        if not state_snapshot:
            return {
                "final_plan": [],
                "final_steps": [],
                "final_answer": "",
                "tool_usage": [],
                "session_id": session_id,
                "original_query": session_data.get("original_query", ""),
                "timestamp": datetime.now().isoformat()
            }
            
        # Extract data from state snapshot
        final_plan = state_snapshot.get("final_plan", [])
        final_steps = state_snapshot.get("final_steps", [])
        final_answer = state_snapshot.get("final_answer", "")
        
        # Extract tool usage from steps
        tool_usage = []
        for step in final_steps:
            if step.get("type") == "CODE":
                code = step.get("code", {})
                tool_name = code.get("tool_name")
                if tool_name:
                    execution_result = step.get("execution_result", {})
                    status = execution_result.get("status", "unknown")
                    tool_usage.append({
                        "tool_name": tool_name,
                        "status": status,
                        "step_index": step.get("index"),
                        "description": step.get("description"),
                        "result": execution_result.get("result", ""),
                        "error": execution_result.get("error", None),
                        "execution_time": execution_result.get("execution_time", ""),
                        "total_time": execution_result.get("total_time", "")
                    })
        
        return {
            "final_plan": final_plan,
            "final_steps": final_steps,
            "final_answer": final_answer,
            "tool_usage": tool_usage,
            "session_id": session_id,
            "original_query": session_data.get("original_query", ""),
            "timestamp": datetime.now().isoformat(),
            "confidence": state_snapshot.get("confidence", ""),
            "reasoning_note": state_snapshot.get("reasoning_note", "")
        }
        
    except Exception as e:
        print(f"Error extracting session state: {str(e)}")
        return {
            "final_plan": [],
            "final_steps": [],
            "final_answer": "",
            "tool_usage": [],
            "session_id": getattr(session, 'session_id', 'unknown'),
            "original_query": getattr(session, 'original_query', 'unknown'),
            "timestamp": datetime.now().isoformat()
        }

def get_tool_success_rate(tool_usage: List[Dict]) -> Dict[str, Tuple[int, int]]:
    """
    Calculate success rate for each tool used in the session.
    
    Args:
        tool_usage: List of tool usage dictionaries from extract_session_state
        
    Returns:
        Dict mapping tool names to (success_count, total_attempts)
    """
    tool_stats = {}
    
    for usage in tool_usage:
        tool_name = usage["tool_name"]
        if tool_name not in tool_stats:
            tool_stats[tool_name] = [0, 0]  # [successes, total]
            
        tool_stats[tool_name][1] += 1  # increment total
        if usage["status"] == "success":
            tool_stats[tool_name][0] += 1  # increment successes
            
    return tool_stats

# Example usage in test_queries.py:
"""
async def execute_query(self, query: str, tools: str, complexity: str) -> Dict:
    try:
        # Run the query through the agent
        session = await self.agent_loop.run(query)
        
        # Extract session state
        session_state = extract_session_state(session)
        
        return {
            "query": query,
            "tools": tools,
            "complexity": complexity,
            "final_plan": "\n".join(session_state["final_plan"]) if isinstance(session_state["final_plan"], list) else session_state["final_plan"],
            "output": session_state["final_answer"],
            "tool_usage": session_state["tool_usage"]
        }
        
    except Exception as e:
        logger.error(f"Error executing query '{query}': {str(e)}")
        return {
            "query": query,
            "tools": tools,
            "complexity": complexity,
            "final_plan": f"Error: {str(e)}",
            "output": "Failed to execute query",
            "tool_usage": []
        }
"""