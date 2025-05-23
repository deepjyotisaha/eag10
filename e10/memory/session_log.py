import json
from pathlib import Path
from datetime import datetime


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

def extract_session_details(session: Dict) -> Dict:
    """
    Extract key details from a session including final steps, plan, answer, and tool usage.
    
    Args:
        session: The session dictionary loaded from a session log file
        
    Returns:
        Dict containing:
        - final_plan: List of plan steps
        - final_steps: List of executed steps
        - final_answer: The final answer from the session
        - tool_usage: List of tools used with their status
    """
    try:
        # Get the state snapshot which contains final results
        state = session.get("state_snapshot", {})
        
        # Extract final plan and steps
        final_plan = state.get("final_plan", [])
        final_steps = state.get("final_steps", [])
        final_answer = state.get("final_answer", "")
        
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
                        "description": step.get("description")
                    })
        
        return {
            "final_plan": final_plan,
            "final_steps": final_steps,
            "final_answer": final_answer,
            "tool_usage": tool_usage
        }
        
    except Exception as e:
        print(f"Error extracting session details: {str(e)}")
        return {
            "final_plan": [],
            "final_steps": [],
            "final_answer": "",
            "tool_usage": []
        }

def get_tool_success_rate(tool_usage: List[Dict]) -> Dict[str, Tuple[int, int]]:
    """
    Calculate success rate for each tool used in the session.
    
    Args:
        tool_usage: List of tool usage dictionaries from extract_session_details
        
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

# Example usage:
"""
session = load_session_from_file("path/to/session.json")
details = extract_session_details(session)

print("Final Plan:", details["final_plan"])
print("Final Answer:", details["final_answer"])
print("\nTool Usage:")
for tool in details["tool_usage"]:
    print(f"- {tool['tool_name']}: {tool['status']}")

success_rates = get_tool_success_rate(details["tool_usage"])
print("\nTool Success Rates:")
for tool, (successes, total) in success_rates.items():
    print(f"- {tool}: {successes}/{total} successful")
"""