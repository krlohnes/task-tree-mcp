#!/usr/bin/env python3
"""
Task Tree MCP Server

An MCP server for hierarchical task management with AI-driven suggestions
and interactive approval workflow.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from uuid import UUID

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    ServerCapabilities,
)

from task_tree import TaskGraph, TaskNode, TaskStatus, TaskPriority, ContextInjector

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("task-tree-mcp")

# Global task graph instance
task_graph: Optional[TaskGraph] = None
context_injector: Optional[ContextInjector] = None

# Enhanced verification state tracking
verification_sessions: Dict[str, Dict[str, Any]] = {}


def get_session_id() -> str:
    """Get or create session ID for verification tracking."""
    # In a real implementation, this would use request context
    # For this demo, we'll use a simple global session
    return "current_session"


def record_verification_claim(session_id: str, task_id: str, evidence: str, verification_steps: str) -> None:
    """Record a verification claim for later validation."""
    if session_id not in verification_sessions:
        verification_sessions[session_id] = {}
    
    verification_sessions[session_id][task_id] = {
        "evidence": evidence,
        "verification_steps": verification_steps,
        "required_tools": parse_required_tools(verification_steps),
        "tool_calls_made": [],
        "validated": False
    }


def parse_required_tools(verification_steps: str) -> List[Dict[str, Any]]:
    """Parse verification steps to extract required tool calls."""
    required_tools = []
    steps_lower = verification_steps.lower()
    
    # Map verification descriptions to required tool calls
    if "ls" in steps_lower or "file exists" in steps_lower or "checked file" in steps_lower:
        required_tools.append({"tool": "LS", "purpose": "file_existence"})
    
    if "read" in steps_lower or "file contents" in steps_lower or "reviewed file" in steps_lower:
        required_tools.append({"tool": "Read", "purpose": "file_content"})
    
    if "bash" in steps_lower or "python" in steps_lower or "executed" in steps_lower or "ran" in steps_lower:
        required_tools.append({"tool": "Bash", "purpose": "execution"})
    
    if "grep" in steps_lower or "search" in steps_lower:
        required_tools.append({"tool": "Grep", "purpose": "search"})
    
    return required_tools


def validate_tool_call_evidence(session_id: str, task_id: str, tool_name: str, tool_args: Dict[str, Any], tool_result: str) -> bool:
    """Validate that a tool call result supports the claimed evidence."""
    if session_id not in verification_sessions or task_id not in verification_sessions[session_id]:
        return False
    
    session_data = verification_sessions[session_id][task_id]
    evidence = session_data["evidence"].lower()
    
    # Record this tool call
    session_data["tool_calls_made"].append({
        "tool": tool_name,
        "args": tool_args,
        "result": tool_result
    })
    
    # Validate specific claims against tool results
    if tool_name == "Read":
        # Check if evidence claims about file contents match actual contents
        if "hello, world" in evidence and "hello, world" not in tool_result.lower():
            return False
        if "print(\"hello, world!\")" in evidence and "print(\"hello, world!\")" not in tool_result.lower():
            return False
    
    if tool_name == "Bash":
        # Check if claimed output matches actual output
        if "hello, world" in evidence and "hello, world" not in tool_result.lower():
            return False
    
    if tool_name == "LS":
        # Check if claimed file existence matches actual file listing
        file_path_in_evidence = extract_file_path_from_evidence(evidence)
        if file_path_in_evidence and file_path_in_evidence not in tool_result:
            return False
    
    return True


def extract_file_path_from_evidence(evidence: str) -> Optional[str]:
    """Extract file path mentioned in evidence."""
    # Simple heuristic - look for .py files
    words = evidence.split()
    for word in words:
        if word.endswith('.py'):
            return word.replace(':', '')  # Remove any trailing colons
    return None


def check_verification_requirements(session_id: str, task_id: str) -> Dict[str, Any]:
    """Check if all required verification tools have been called."""
    if session_id not in verification_sessions or task_id not in verification_sessions[session_id]:
        return {"valid": False, "reason": "No verification claim recorded"}
    
    session_data = verification_sessions[session_id][task_id]
    required_tools = session_data["required_tools"]
    tools_made = session_data["tool_calls_made"]
    
    missing_tools = []
    contradictions = []
    
    for required_tool in required_tools:
        tool_name = required_tool["tool"]
        purpose = required_tool["purpose"]
        
        # Check if this tool was called
        matching_calls = [call for call in tools_made if call["tool"] == tool_name]
        if not matching_calls:
            missing_tools.append(f"{tool_name} (for {purpose})")
        else:
            # Validate the tool results against evidence claims
            for call in matching_calls:
                if not validate_tool_call_evidence(session_id, task_id, tool_name, call["args"], call["result"]):
                    contradictions.append(f"{tool_name} result contradicts evidence claim")
    
    if missing_tools or contradictions:
        return {
            "valid": False,
            "missing_tools": missing_tools,
            "contradictions": contradictions
        }
    
    return {"valid": True}


def get_task_graph() -> TaskGraph:
    """Get or create the global task graph instance."""
    global task_graph, context_injector
    if task_graph is None:
        db_path = Path(__file__).parent.parent / "tasks.db"
        logger.info(f"Using database path: {db_path}")
        task_graph = TaskGraph(db_path)
        context_injector = ContextInjector(task_graph, db_path)
    return task_graph


def get_context_injector() -> ContextInjector:
    """Get the global context injector instance."""
    get_task_graph()  # Ensure initialized
    return context_injector


def format_task_for_display(task: TaskNode) -> str:
    """Format a task for display in responses."""
    status_emoji = {
        "pending": "⏳",
        "in_progress": "🔄", 
        "completed": "✅",
        "blocked": "🚫",
        "cancelled": "❌"
    }
    
    priority_emoji = {
        "low": "🔵",
        "medium": "🟡", 
        "high": "🟠",
        "critical": "🔴"
    }
    
    status = task.status if isinstance(task.status, str) else task.status.value
    priority = task.priority if isinstance(task.priority, str) else task.priority.value
    
    result = f"{status_emoji.get(status, '⚪')} {priority_emoji.get(priority, '⚪')} **{task.title}**"
    
    if task.description:
        result += f"\n   {task.description}"
    
    if task.completion_criteria:
        result += f"\n   🎯 Success criteria: {task.completion_criteria}"
    
    if task.tags:
        result += f"\n   🏷️ Tags: {', '.join(sorted(task.tags))}"
    
    result += f"\n   📅 Created: {task.created_at.strftime('%Y-%m-%d %H:%M')}"
    result += f"\n   🆔 ID: `{str(task.id)}`"
    
    return result


def detect_flat_pattern(graph: TaskGraph, parent_id: Optional[UUID] = None) -> Dict[str, Any]:
    """Detect if recent task creation follows flat list patterns."""
    if parent_id:
        parent_task = graph.get_task(parent_id)
        if not parent_task:
            return {"is_flat": False, "reason": "Parent not found"}
        
        children = [graph.get_task(child_id) for child_id in parent_task.child_ids]
        children = [child for child in children if child]  # Filter None
        
        if len(children) < 2:
            return {"is_flat": False, "reason": "Too few children to analyze"}
        
        # Check for flat patterns
        flat_indicators = 0
        total_checks = 0
        
        # 1. Check for sequential naming
        sequential_names = sum(1 for child in children 
                             if any(word in child.title.lower() 
                                   for word in ['step', 'phase', 'part', '1', '2', '3']))
        if sequential_names > len(children) * 0.5:  # More than half have sequential names
            flat_indicators += 1
        total_checks += 1
        
        # 2. Check for lack of validation siblings
        validation_siblings = sum(1 for child in children 
                                if any(word in child.title.lower() 
                                      for word in ['validate', 'verify', 'test', 'check']))
        if validation_siblings < len(children) * 0.3:  # Less than 30% have validation
            flat_indicators += 1
        total_checks += 1
        
        # 3. Check for shallow depth (no grandchildren)
        has_grandchildren = sum(1 for child in children if len(child.child_ids) > 0)
        if has_grandchildren < len(children) * 0.3:  # Less than 30% have children
            flat_indicators += 1
        total_checks += 1
        
        # 4. Check for generic action words
        generic_actions = sum(1 for child in children 
                            if any(child.title.lower().startswith(word) 
                                  for word in ['do ', 'create ', 'make ', 'build ', 'implement ']))
        if generic_actions > len(children) * 0.6:  # More than 60% are generic
            flat_indicators += 1
        total_checks += 1
        
        is_flat = flat_indicators >= total_checks * 0.5  # More than half indicators triggered
        
        return {
            "is_flat": is_flat,
            "score": flat_indicators / total_checks,
            "indicators": {
                "sequential_naming": sequential_names,
                "validation_siblings": validation_siblings,
                "has_grandchildren": has_grandchildren, 
                "generic_actions": generic_actions
            },
            "children_count": len(children)
        }
    
    return {"is_flat": False, "reason": "No parent provided"}


def get_hierarchical_suggestions(detection_result: Dict[str, Any]) -> str:
    """Generate suggestions for improving flat task structures."""
    if not detection_result.get("is_flat", False):
        return ""
    
    suggestions = ["🔄 **Consider improving this task structure:**\n"]
    
    indicators = detection_result.get("indicators", {})
    children_count = detection_result.get("children_count", 0)
    
    if indicators.get("validation_siblings", 0) < children_count * 0.3:
        suggestions.append("• Add validation siblings (✅) for each action task")
    
    if indicators.get("has_grandchildren", 0) < children_count * 0.3:
        suggestions.append("• Break down tasks into deeper hierarchies (3+ levels)")
    
    if indicators.get("sequential_naming", 0) > 0:
        suggestions.append("• Use domain-specific names instead of 'Step 1', 'Step 2'")
    
    if indicators.get("generic_actions", 0) > children_count * 0.6:
        suggestions.append("• Replace generic actions with specific implementation details")
    
    suggestions.append("\n💡 **Example improvement:**")
    suggestions.append("Instead of: Step 1 → Step 2 → Step 3")
    suggestions.append("Try: Research & Design → Implementation → Validation")
    suggestions.append("  ├─ Analyze requirements → ✅ Validate approach")  
    suggestions.append("  ├─ Core logic → ✅ Unit tests")
    suggestions.append("  └─ Integration → ✅ End-to-end tests")
    
    return "\n".join(suggestions)


def get_hierarchical_planning_guidance() -> str:
    """Provide guidance on hierarchical task planning patterns."""
    return """💡 **Hierarchical Planning Tips**

Instead of flat task lists, create deep hierarchical structures:

❌ **Avoid**: Flat lists
- Implement feature
- Test feature  
- Deploy feature

✅ **Better**: Hierarchical with validation siblings
- Implement user authentication system
  ├─ Design JWT token structure
  │  └─ ✅ Validate token schema against security requirements
  ├─ Build password hashing
  │  └─ ✅ Verify hash strength meets compliance standards
  └─ Create session management
     └─ ✅ Test session timeout and renewal logic

🔥 **Best**: Deep hierarchy with checkpoints
- Implement authentication system
  ├─ Research & Design
  │  ├─ Analyze JWT vs session-based auth
  │  └─ ✅ Validate choice against requirements
  ├─ 🧪 Write tests first (TDD)
  ├─ Implementation
  │  ├─ JWT token service
  │  │  └─ ✅ Validate security standards
  │  └─ Password management
  │     └─ ✅ Test against attack patterns
  └─ 🤔 CHECKPOINT: "Ready for integration testing?"

**Key Patterns**:
• Add validation siblings (✅) to every action
• Use checkpoints (🤔) for user authorization  
• Reference requirements: (req: parent task)
• Go 3+ levels deep for complex tasks

**🔒 Immutability Pattern**:
When requirements change, create new tasks instead of editing existing ones:
❌ **Don't edit**: "Build REST API" → "Build GraphQL API"
✅ **Do create new**: "Build REST API" → cancelled
                      "Build GraphQL API" → new task

This preserves decision history and shows why pivots happened."""


# Create the MCP server
server = Server("task-tree")


@server.list_resources()
async def handle_list_resources() -> List[Resource]:
    """List available task tree resources."""
    graph = get_task_graph()
    injector = get_context_injector()
    
    resources = []
    
    # Current task context resource
    current_task_id = injector.get_current_task_id()
    if current_task_id:
        resources.append(Resource(
            uri="task://current/context",
            name="Current Task Context",
            description="The current task lineage context that would be injected",
            mimeType="text/plain"
        ))
    
    # Task tree visualization resource
    resources.append(Resource(
        uri="task://tree/visualization", 
        name="Task Tree Visualization",
        description="Visual representation of the entire task hierarchy",
        mimeType="text/plain"
    ))
    
    # Task statistics resource
    resources.append(Resource(
        uri="task://stats/summary",
        name="Task Statistics",
        description="Summary statistics of all tasks in the system", 
        mimeType="application/json"
    ))
    
    return resources


@server.read_resource()
async def handle_read_resource(uri: str) -> str:
    """Read a specific task tree resource."""
    graph = get_task_graph()
    injector = get_context_injector()
    
    if uri == "task://current/context":
        context = injector.get_context_for_injection()
        if not context:
            return "No current task set"
        return context
    
    elif uri == "task://tree/visualization":
        # Build simple text tree
        roots = graph.get_root_tasks()
        if not roots:
            return "No tasks found"
        
        def build_text_tree(task: TaskNode, indent: int = 0) -> List[str]:
            status = task.status if isinstance(task.status, str) else task.status.value
            priority = task.priority if isinstance(task.priority, str) else task.priority.value
            
            prefix = "  " * indent + ("└─ " if indent > 0 else "")
            lines = [f"{prefix}{task.title} [{status}] [{priority}]"]
            
            children = graph.get_children(task.id)
            for child in sorted(children, key=lambda t: t.created_at):
                lines.extend(build_text_tree(child, indent + 1))
            
            return lines
        
        all_lines = []
        for root in roots:
            all_lines.extend(build_text_tree(root))
            all_lines.append("")  # Empty line between trees
        
        return "\n".join(all_lines)
    
    elif uri == "task://stats/summary":
        stats = graph.get_task_stats()
        return json.dumps(stats, indent=2)
    
    else:
        raise ValueError(f"Unknown resource: {uri}")


@server.list_tools()
async def handle_list_tools() -> List[Tool]:
    """List available task management tools."""
    return [
        Tool(
            name="get_current_task",
            description="Get the currently active task and its context lineage",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="create_task",
            description="Create a new task in the hierarchy",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Task title"},
                    "description": {"type": "string", "description": "Optional task description"},
                    "parent_id": {"type": "string", "description": "Optional parent task ID"},
                    "priority": {
                        "type": "string", 
                        "enum": ["low", "medium", "high", "critical"],
                        "description": "Task priority"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional tags for the task"
                    },
                    "completion_criteria": {
                        "type": "string", 
                        "description": "Optional success criteria for task completion"
                    },
                    "set_as_current": {
                        "type": "boolean",
                        "description": "Set this task as the current active task"
                    }
                },
                "required": ["title"]
            }
        ),
        Tool(
            name="update_task_status",
            description="Update the status of an existing task",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "Task ID to update"},
                    "status": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "completed", "blocked", "cancelled"],
                        "description": "New task status"
                    },
                    "reason": {"type": "string", "description": "Optional reason for the status change"}
                },
                "required": ["task_id", "status"]
            }
        ),
        Tool(
            name="set_current_task",
            description="Set the currently active task for context injection",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "Task ID to set as current (empty to clear)"}
                },
                "required": []
            }
        ),
        Tool(
            name="search_tasks",
            description="Search for tasks by title, description, tags, or other criteria",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query text"},
                    "status": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "completed", "blocked", "cancelled"],
                        "description": "Filter by status"
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "critical"],
                        "description": "Filter by priority"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by tags (task must have all listed tags)"
                    },
                    "limit": {"type": "integer", "description": "Maximum number of results", "default": 10}
                },
                "required": []
            }
        ),
        Tool(
            name="get_task_details",
            description="Get detailed information about a specific task",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "Task ID to get details for"}
                },
                "required": ["task_id"]
            }
        ),
        Tool(
            name="suggest_task_completion",
            description="Suggest that a task should be marked as completed (requires user approval)",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "Task ID to suggest completion for"},
                    "reason": {"type": "string", "description": "Explanation of why the task should be completed"},
                    "evidence": {"type": "string", "description": "Evidence that the task completion criteria have been met"}
                },
                "required": ["task_id", "reason", "evidence"]
            }
        ),
        Tool(
            name="suggest_new_subtask",
            description="Suggest creating a new subtask (requires user approval)",
            inputSchema={
                "type": "object",
                "properties": {
                    "parent_id": {"type": "string", "description": "Parent task ID"},
                    "title": {"type": "string", "description": "Suggested subtask title"},
                    "description": {"type": "string", "description": "Suggested subtask description"},
                    "priority": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "critical"],
                        "description": "Suggested priority"
                    },
                    "reason": {"type": "string", "description": "Why this subtask is needed"}
                },
                "required": ["parent_id", "title", "reason"]
            }
        ),
        Tool(
            name="get_available_tasks",
            description="Get tasks that can be worked on next (all dependencies met)",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Maximum number of tasks to return", "default": 5}
                },
                "required": []
            }
        ),
        Tool(
            name="get_task_lineage",
            description="Get the full path from root to a specific task",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "Task ID to get lineage for"}
                },
                "required": ["task_id"]
            }
        ),
        Tool(
            name="record_verification_claim",
            description="Record a verification claim that requires actual tool calls to validate",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "Task ID to record verification claim for"},
                    "evidence": {"type": "string", "description": "The evidence you claim to have"},
                    "verification_steps": {"type": "string", "description": "Specific tool calls you will make to verify the evidence (e.g., 'LS tool, Read tool, Bash tool')"} 
                },
                "required": ["task_id", "evidence", "verification_steps"]
            }
        ),
        Tool(
            name="submit_verification_evidence", 
            description="Submit actual tool call results as evidence (required before confirm_evidence)",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "Task ID to submit evidence for"},
                    "tool_name": {"type": "string", "description": "Name of tool that was called (LS, Read, Bash, etc.)", "enum": ["LS", "Read", "Bash", "Grep", "Write", "Edit"]},
                    "tool_args": {"type": "object", "description": "Arguments passed to the tool"},
                    "tool_result": {"type": "string", "description": "Actual result/output from the tool call"}
                },
                "required": ["task_id", "tool_name", "tool_args", "tool_result"]
            }
        ),
        Tool(
            name="confirm_evidence",
            description="Confirm evidence after making required tool calls (enhanced verification)",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "Task ID to confirm completion for"},
                    "evidence": {"type": "string", "description": "The evidence being confirmed"},
                    "verification_steps": {"type": "string", "description": "Specific steps taken to verify the evidence is accurate"},
                    "criteria_mapping": {"type": "string", "description": "How each piece of evidence maps to specific completion criteria"}
                },
                "required": ["task_id", "evidence", "verification_steps", "criteria_mapping"]
            }
        )
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls for task management."""
    graph = get_task_graph()
    injector = get_context_injector()
    
    try:
        if name == "get_current_task":
            current_id = injector.get_current_task_id()
            if not current_id:
                return [TextContent(type="text", text="No current task set")]
            
            task = graph.get_task(current_id)
            if not task:
                return [TextContent(type="text", text="Current task not found (clearing)")]
            
            # Get context and lineage
            context = injector.get_context_for_injection()
            lineage = graph.get_lineage(current_id)
            
            response = f"## Current Task\n\n{format_task_for_display(task)}\n\n"
            response += f"## Task Lineage\n\n"
            for i, ancestor in enumerate(lineage):
                indent = "  " * i
                arrow = "└─ " if i > 0 else ""
                status = ancestor.status if isinstance(ancestor.status, str) else ancestor.status.value
                response += f"{indent}{arrow}**{ancestor.title}** [{status}]\n"
            
            response += f"\n## Context for Injection\n\n```\n{context}\n```"
            
            return [TextContent(type="text", text=response)]
        
        elif name == "create_task":
            title = arguments["title"]
            description = arguments.get("description")
            parent_id_str = arguments.get("parent_id")
            priority_str = arguments.get("priority", "medium")
            tags = set(arguments.get("tags", []))
            completion_criteria = arguments.get("completion_criteria")
            set_as_current = arguments.get("set_as_current", False)
            
            # Convert parent_id if provided
            parent_id = None
            if parent_id_str:
                try:
                    parent_id = UUID(parent_id_str)
                    if parent_id not in graph.nodes:
                        return [TextContent(type="text", text=f"❌ Parent task {parent_id_str} not found")]
                except ValueError:
                    return [TextContent(type="text", text=f"❌ Invalid parent task ID format")]
            
            # Validate completion criteria requirement
            if not completion_criteria and "trivial" not in tags:
                return [TextContent(type="text", text=f"""❌ **Completion Criteria Required**

📝 **Task:** {title}

This task cannot be created without completion criteria.

**To fix this, either:**
1. **Add completion criteria**: Specify what conditions must be met for this task to be considered complete
2. **Tag as trivial**: Add `#trivial` to the tags if this is a simple task that doesn't need explicit criteria

**Examples of good criteria:**
• Function returns expected output for all test cases
• All unit tests pass and coverage > 80%
• Integration tested with manual verification
• Documentation updated to reflect changes
• Error handling covers all edge cases""")]
            
            # Create the task
            task = graph.create_task(
                title=title,
                description=description,
                parent_id=parent_id,
                priority=TaskPriority(priority_str),
                tags=tags,
                completion_criteria=completion_criteria
            )
            
            if set_as_current:
                injector.set_current_task_id(task.id)
            
            response = f"✅ **Task Created Successfully** 🔧 SERVER_VERSION_2025-08-04-15:17\n\n{format_task_for_display(task)}"
            if set_as_current:
                response += "\n\n🎯 Set as current task"
            
            # DEBUG: Always show parent_id status
            response += f"\n\n🔧 DEBUG: parent_id = {parent_id}, type = {type(parent_id)}"
            
            # Check for flat patterns and provide suggestions
            if parent_id:
                response += f"\n\n🔧 DEBUG: Pattern detection running for parent {parent_id}"
                try:
                    detection_result = detect_flat_pattern(graph, parent_id)
                    response += f"\n\n🔧 DEBUG: Detection result: {detection_result}"
                    suggestions = get_hierarchical_suggestions(detection_result)
                    if suggestions:
                        response += "\n\n" + suggestions
                    else:
                        response += "\n\n🔧 DEBUG: No suggestions generated"
                except Exception as e:
                    response += f"\n\n🔧 DEBUG: Error in pattern detection: {e}"
            else:
                response += f"\n\n🔧 DEBUG: parent_id is None/False, skipping pattern detection"
            
            # Add hierarchical planning guidance for root tasks
            if not parent_id:  # Only show guidance for root tasks
                response += "\n\n" + get_hierarchical_planning_guidance()
            
            return [TextContent(type="text", text=response)]
        
        elif name == "update_task_status":
            task_id_str = arguments["task_id"]
            new_status = arguments["status"]
            reason = arguments.get("reason", "")
            
            try:
                task_id = UUID(task_id_str)
            except ValueError:
                return [TextContent(type="text", text="❌ Invalid task ID format")]
            
            task = graph.get_task(task_id)
            if not task:
                return [TextContent(type="text", text=f"❌ Task {task_id_str} not found")]
            
            old_status = task.status if isinstance(task.status, str) else task.status.value
            
            # Validate completion criteria before allowing completion
            if new_status == "completed":
                # Check if task has completion criteria or #trivial tag
                if not task.completion_criteria and "trivial" not in task.tags:
                    return [TextContent(type="text", text=f"""❌ **Cannot Complete Task Without Criteria**

📝 **{task.title}**

This task cannot be marked as completed because it has no completion criteria defined.

**To fix this, either:**
1. **Add completion criteria**: What specific conditions must be met for this task to be truly done?
2. **Tag as trivial**: Add `#trivial` tag if this is a simple task that doesn't need criteria

**Why this matters:** Completion criteria prevent premature "mission accomplished" moments and ensure work is actually finished.

**Examples of good criteria:**
• Function returns expected output for test cases
• All unit tests pass  
• Integration verified with manual testing
• Documentation updated with changes
• Error handling covers edge cases""")]
                
                task.mark_completed()
            elif new_status == "in_progress":
                task.mark_in_progress()
            elif new_status == "blocked":
                task.mark_blocked()
            else:
                task.status = TaskStatus(new_status)
            
            graph.update_task(task)
            
            response = f"✅ **Task Status Updated**\n\n"
            response += f"📝 **{task.title}**\n"
            response += f"📊 Status: {old_status} → **{new_status}**\n"
            if reason:
                response += f"💬 Reason: {reason}\n"
            response += f"🆔 ID: `{str(task.id)}`"
            
            return [TextContent(type="text", text=response)]
        
        elif name == "set_current_task":
            task_id_str = arguments.get("task_id", "")
            
            if not task_id_str:
                injector.set_current_task_id(None)
                return [TextContent(type="text", text="✅ Current task cleared")]
            
            try:
                task_id = UUID(task_id_str)
            except ValueError:
                return [TextContent(type="text", text="❌ Invalid task ID format")]
            
            task = graph.get_task(task_id)
            if not task:
                return [TextContent(type="text", text=f"❌ Task {task_id_str} not found")]
            
            injector.set_current_task_id(task_id)
            
            response = f"🎯 **Current Task Set**\n\n{format_task_for_display(task)}"
            return [TextContent(type="text", text=response)]
        
        elif name == "search_tasks":
            query = arguments.get("query", "")
            status_filter = arguments.get("status")
            priority_filter = arguments.get("priority")
            tags_filter = set(arguments.get("tags", []))
            limit = arguments.get("limit", 10)
            
            # Convert enum strings to objects if needed
            status_obj = TaskStatus(status_filter) if status_filter else None
            priority_obj = TaskPriority(priority_filter) if priority_filter else None
            
            tasks = graph.search_tasks(
                query=query,
                status=status_obj,
                priority=priority_obj,
                tags=tags_filter if tags_filter else None
            )
            
            if not tasks:
                return [TextContent(type="text", text="🔍 No tasks found matching the criteria")]
            
            response = f"🔍 **Found {len(tasks)} task(s)**\n\n"
            for i, task in enumerate(tasks[:limit]):
                if i > 0:
                    response += "\n---\n\n"
                response += format_task_for_display(task)
            
            if len(tasks) > limit:
                response += f"\n\n... and {len(tasks) - limit} more tasks"
            
            return [TextContent(type="text", text=response)]
        
        elif name == "get_task_details":
            task_id_str = arguments["task_id"]
            
            try:
                task_id = UUID(task_id_str)
            except ValueError:
                return [TextContent(type="text", text="❌ Invalid task ID format")]
            
            task = graph.get_task(task_id)
            if not task:
                return [TextContent(type="text", text=f"❌ Task {task_id_str} not found")]
            
            response = f"## Task Details\n\n{format_task_for_display(task)}\n\n"
            
            # Add lineage
            lineage = graph.get_lineage(task_id)
            if len(lineage) > 1:
                response += "## Task Lineage\n\n"
                for i, ancestor in enumerate(lineage):
                    indent = "  " * i
                    arrow = "└─ " if i > 0 else ""
                    status = ancestor.status if isinstance(ancestor.status, str) else ancestor.status.value
                    current_marker = " **(current)**" if ancestor.id == task_id else ""
                    response += f"{indent}{arrow}**{ancestor.title}** [{status}]{current_marker}\n"
                response += "\n"
            
            # Add children
            children = graph.get_children(task_id)
            if children:
                response += f"## Children ({len(children)})\n\n"
                for child in children:
                    status = child.status if isinstance(child.status, str) else child.status.value
                    priority = child.priority if isinstance(child.priority, str) else child.priority.value
                    response += f"• **{child.title}** [{status}] [{priority}]\n"
            
            return [TextContent(type="text", text=response)]
        
        elif name == "suggest_task_completion":
            task_id_str = arguments["task_id"]
            reason = arguments["reason"]
            evidence = arguments["evidence"]  # Now required field
            
            try:
                task_id = UUID(task_id_str)
            except ValueError:
                return [TextContent(type="text", text="❌ Invalid task ID format")]
            
            task = graph.get_task(task_id)
            if not task:
                return [TextContent(type="text", text=f"❌ Task {task_id_str} not found")]
            
            # Validate completion criteria before suggesting completion
            if not task.completion_criteria and "trivial" not in task.tags:
                response = f"❌ **Cannot Suggest Completion Without Criteria**\n\n"
                response += f"📝 **Task:** {task.title}\n"
                response += f"💭 **Your Reason:** {reason}\n"
                if evidence:
                    response += f"📋 **Your Evidence:** {evidence}\n"
                response += f"\n**Missing Completion Criteria**\n"
                response += f"This task cannot be completed because it lacks explicit completion criteria.\n\n"
                response += f"**To fix this, either:**\n"
                response += f"1. **Add completion criteria child task**: Create a subtask to define what 'done' means\n"
                response += f"2. **Tag as trivial**: Add `#trivial` tag if this is a simple task\n\n"
                response += f"**Suggested actions:**\n"
                response += f"• `create_task` with title \"Define completion criteria for: {task.title}\" and parent_id `{task_id_str}`\n"
                response += f"• Or `update_task_status` with task_id `{task_id_str}`, status `in_progress`, and add #trivial tag if appropriate"
                
                return [TextContent(type="text", text=response)]
            
            # Validate evidence is provided and non-empty
            if not evidence or evidence.strip() == "":
                response = f"❌ **Evidence Required for Completion**\n\n"
                response += f"📝 **Task:** {task.title}\n"
                response += f"💭 **Your Reason:** {reason}\n"
                if task.completion_criteria:
                    response += f"🎯 **Success Criteria:** {task.completion_criteria}\n"
                response += f"\n**Missing Evidence**\n"
                response += f"You must provide concrete evidence that proves the completion criteria have been met.\n\n"
                response += f"**Evidence should include:**\n"
                response += f"• Specific outputs, test results, or deliverables\n"
                response += f"• Screenshots, logs, or other concrete proof\n"
                response += f"• Verification steps taken to confirm success\n"
                response += f"• How each aspect of the criteria was addressed\n\n"
                response += f"**Example evidence formats:**\n"
                response += f"• \"Tests pass: 15/15 unit tests green, screenshot attached\"\n"
                response += f"• \"Function returns expected outputs: f(1)=2, f(2)=4, f(3)=6\"\n"
                response += f"• \"Integration verified: API endpoint returns 200 with correct JSON\"\n"
                response += f"• \"Documentation updated: README.md includes new feature section\""
                
                return [TextContent(type="text", text=response)]
            
            # Enhanced verification - always require tool-based verification
            # All evidence must go through the new verification system
            
            response = f"🔍 **Evidence Verification Required**\n\n"
            response += f"📝 **Task:** {task.title}\n"
            response += f"🎯 **Success Criteria:** {task.completion_criteria if task.completion_criteria else 'Tagged as #trivial'}\n"
            response += f"💭 **Your Reason:** {reason}\n"
            response += f"📋 **Your Evidence:** {evidence}\n\n"
            response += f"**⚠️ Tool-Based Evidence Verification Required**\n"
            response += f"Before this task can be completed, you must provide evidence using actual tool calls.\n\n"
            response += f"**Required verification approach:**\n"
            response += f"1. **Record your verification plan**: Use `record_verification_claim` to declare what you will verify\n"
            response += f"2. **Make actual tool calls**: Call LS, Read, Bash, or other tools to gather evidence\n"
            response += f"3. **Submit each tool result**: Use `submit_verification_evidence` for each tool call\n"
            response += f"4. **Confirm completion**: Use `confirm_evidence` after all required tools have been called\n\n"
            response += f"**Step 1: Record your verification plan**\n"
            response += f"Use `record_verification_claim` with:\n"
            response += f"- task_id: `{task_id_str}`\n"
            response += f"- evidence: Your evidence text\n"
            response += f"- verification_steps: Specific tools you will use (e.g., 'LS tool to check file exists, Read tool to verify contents, Bash tool to test execution')\n\n"
            response += f"**Step 2: Make tool calls and submit results**\n"
            response += f"For each tool call you make:\n"
            response += f"1. Call the tool (LS, Read, Bash, etc.)\n"
            response += f"2. Immediately use `submit_verification_evidence` with the actual tool results\n\n"
            response += f"**Step 3: Complete verification**\n"
            response += f"Use `confirm_evidence` only after all required tool results have been submitted"
            
            return [TextContent(type="text", text=response)]
        
        elif name == "suggest_new_subtask":
            parent_id_str = arguments["parent_id"]
            title = arguments["title"]
            description = arguments.get("description", "")
            priority = arguments.get("priority", "medium")
            reason = arguments["reason"]
            
            try:
                parent_id = UUID(parent_id_str)
            except ValueError:
                return [TextContent(type="text", text="❌ Invalid parent task ID format")]
            
            parent_task = graph.get_task(parent_id)
            if not parent_task:
                return [TextContent(type="text", text=f"❌ Parent task {parent_id_str} not found")]
            
            response = f"🤖 **New Subtask Suggestion**\n\n"
            response += f"👆 **Parent Task:** {parent_task.title}\n"
            response += f"📝 **Suggested Subtask:** {title}\n"
            if description:
                response += f"📄 **Description:** {description}\n"
            response += f"⚡ **Priority:** {priority}\n"
            response += f"💭 **Reason:** {reason}\n"
            response += f"\n**Should I create this subtask?**\n"
            response += f"Reply with:\n"
            response += f"• `create_task` with title `{title}` and parent_id `{parent_id_str}` to approve\n"
            response += f"• Or suggest modifications"
            
            return [TextContent(type="text", text=response)]
        
        elif name == "get_available_tasks":
            limit = arguments.get("limit", 5)
            
            available = graph.get_available_tasks()
            
            if not available:
                return [TextContent(type="text", text="📋 No tasks are currently available to work on")]
            
            response = f"📋 **Available Tasks ({len(available)})**\n\n"
            response += "These tasks can be started immediately (all dependencies met):\n\n"
            
            for i, task in enumerate(available[:limit]):
                if i > 0:
                    response += "\n---\n\n"
                response += format_task_for_display(task)
            
            if len(available) > limit:
                response += f"\n\n... and {len(available) - limit} more available tasks"
            
            return [TextContent(type="text", text=response)]
        
        elif name == "get_task_lineage":
            task_id_str = arguments["task_id"]
            
            try:
                task_id = UUID(task_id_str)
            except ValueError:
                return [TextContent(type="text", text="❌ Invalid task ID format")]
            
            task = graph.get_task(task_id)
            if not task:
                return [TextContent(type="text", text=f"❌ Task {task_id_str} not found")]
            
            lineage = graph.get_lineage(task_id)
            
            response = f"🛤️ **Task Lineage Path**\n\n"
            response += "Path from root to current task:\n\n"
            
            for i, ancestor in enumerate(lineage):
                indent = "  " * i
                arrow = "└─ " if i > 0 else ""
                status = ancestor.status if isinstance(ancestor.status, str) else ancestor.status.value
                priority = ancestor.priority if isinstance(ancestor.priority, str) else ancestor.priority.value
                current_marker = " **(target)**" if ancestor.id == task_id else ""
                response += f"{indent}{arrow}**{ancestor.title}** [{status}] [{priority}]{current_marker}\n"
            
            return [TextContent(type="text", text=response)]
        
        elif name == "record_verification_claim":
            task_id_str = arguments["task_id"]
            evidence = arguments["evidence"]
            verification_steps = arguments["verification_steps"]
            
            try:
                task_id = UUID(task_id_str)
            except ValueError:
                return [TextContent(type="text", text="❌ Invalid task ID format")]
            
            task = graph.get_task(task_id)
            if not task:
                return [TextContent(type="text", text=f"❌ Task {task_id_str} not found")]
            
            # Record the verification claim
            session_id = get_session_id()
            record_verification_claim(session_id, task_id_str, evidence, verification_steps)
            
            required_tools = parse_required_tools(verification_steps)
            
            response = f"📝 **Verification Claim Recorded**\n\n"
            response += f"🎯 **Task:** {task.title}\n"
            response += f"📋 **Evidence Claimed:** {evidence}\n"
            response += f"🔧 **Verification Steps:** {verification_steps}\n\n"
            response += f"**Required Tool Calls Detected:**\n"
            for tool in required_tools:
                response += f"• {tool['tool']} (for {tool['purpose']})\n"
            response += f"\n**Next Steps:**\n"
            response += f"1. Make the actual tool calls listed above\n"
            response += f"2. Use `confirm_evidence` after all tool calls are complete\n"
            response += f"\n**⚠️ Tool call results will be validated against your evidence claims**"
            
            return [TextContent(type="text", text=response)]
            
        elif name == "submit_verification_evidence":
            task_id_str = arguments["task_id"]
            tool_name = arguments["tool_name"]
            tool_args = arguments["tool_args"]
            tool_result = arguments["tool_result"]
            
            try:
                task_id = UUID(task_id_str)
            except ValueError:
                return [TextContent(type="text", text="❌ Invalid task ID format")]
            
            task = graph.get_task(task_id)
            if not task:
                return [TextContent(type="text", text=f"❌ Task {task_id_str} not found")]
            
            # Submit the tool result for verification
            session_id = get_session_id()
            
            if session_id not in verification_sessions or task_id_str not in verification_sessions[session_id]:
                return [TextContent(type="text", text=f"❌ No verification claim recorded for this task. Use `record_verification_claim` first.")]
            
            # Validate this tool call against the claimed evidence
            is_valid = validate_tool_call_evidence(session_id, task_id_str, tool_name, tool_args, tool_result)
            
            session_data = verification_sessions[session_id][task_id_str]
            
            response = f"📤 **Verification Evidence Submitted**\n\n"
            response += f"🎯 **Task:** {task.title}\n"
            response += f"🔧 **Tool:** {tool_name}\n"
            response += f"📋 **Tool Args:** {tool_args}\n"
            response += f"📊 **Tool Result:** {tool_result[:200]}{'...' if len(tool_result) > 200 else ''}\n\n"
            
            if is_valid:
                response += f"✅ **Evidence Validation:** Tool result supports claimed evidence\n"
            else:
                response += f"❌ **Evidence Validation:** Tool result contradicts claimed evidence\n"
            
            response += f"\n**Verification Progress:**\n"
            tools_made = session_data["tool_calls_made"]
            required_tools = session_data["required_tools"]
            
            for required_tool in required_tools:
                matching_calls = [call for call in tools_made if call["tool"] == required_tool["tool"]]
                status = "✅" if matching_calls else "⏳"
                response += f"• {status} {required_tool['tool']} (for {required_tool['purpose']})\n"
            
            response += f"\n**Next Steps:**\n"
            remaining_tools = [tool for tool in required_tools if not any(call["tool"] == tool["tool"] for call in tools_made)]
            
            if remaining_tools:
                response += f"Continue making tool calls for: {', '.join(tool['tool'] for tool in remaining_tools)}\n"
            else:
                response += f"All required tools called! Use `confirm_evidence` to complete verification.\n"
            
            return [TextContent(type="text", text=response)]
            
        elif name == "confirm_evidence":
            task_id_str = arguments["task_id"]
            evidence = arguments["evidence"]
            verification_steps = arguments["verification_steps"]
            criteria_mapping = arguments["criteria_mapping"]
            
            try:
                task_id = UUID(task_id_str)
            except ValueError:
                return [TextContent(type="text", text="❌ Invalid task ID format")]
            
            task = graph.get_task(task_id)
            if not task:
                return [TextContent(type="text", text=f"❌ Task {task_id_str} not found")]
            
            # Enhanced verification check
            session_id = get_session_id()
            verification_result = check_verification_requirements(session_id, task_id_str)
            
            if not verification_result["valid"]:
                response = f"❌ **Verification Failed - Evidence Does Not Match Tool Results**\n\n"
                response += f"📝 **Task:** {task.title}\n"
                response += f"📋 **Your Evidence:** {evidence}\n"
                response += f"🔍 **Your Verification Steps:** {verification_steps}\n\n"
                
                if "reason" in verification_result:
                    response += f"**Issue:** {verification_result['reason']}\n\n"
                
                if "missing_tools" in verification_result and verification_result["missing_tools"]:
                    response += f"**Missing Required Tool Calls:**\n"
                    for missing in verification_result["missing_tools"]:
                        response += f"• {missing}\n"
                    response += f"\n"
                
                if "contradictions" in verification_result and verification_result["contradictions"]:
                    response += f"**Evidence Contradictions Found:**\n"
                    for contradiction in verification_result["contradictions"]:
                        response += f"• {contradiction}\n"
                    response += f"\n"
                
                response += f"**To fix this:**\n"
                response += f"1. Use `record_verification_claim` again with accurate evidence\n"
                response += f"2. Make the required tool calls with correct parameters\n"
                response += f"3. Ensure tool results actually support your evidence claims\n"
                response += f"4. Try `confirm_evidence` again after making actual tool calls"
                
                return [TextContent(type="text", text=response)]
            
            response = f"✅ **Evidence Confirmed - Task Completion Approved**\n\n"
            response += f"📝 **Task:** {task.title}\n"
            if task.completion_criteria:
                response += f"🎯 **Success Criteria:** {task.completion_criteria}\n"
            else:
                response += f"🏷️ **Tagged as:** #trivial (no criteria needed)\n"
            response += f"📋 **Evidence:** {evidence}\n"
            response += f"🔍 **Verification Steps:** {verification_steps}\n"
            response += f"🗺️ **Criteria Mapping:** {criteria_mapping}\n\n"
            response += f"**✅ Evidence validated against actual tool call results.**\n\n"
            response += f"**To finalize completion:**\n"
            response += f"• `update_task_status` with task_id `{task_id_str}` and status `completed` to approve\n"
            response += f"• Or explain what still needs to be done"
            
            return [TextContent(type="text", text=response)]
        
        else:
            return [TextContent(type="text", text=f"❌ Unknown tool: {name}")]
    
    except Exception as e:
        logger.error(f"Error in tool {name}: {e}")
        return [TextContent(type="text", text=f"❌ Error: {str(e)}")]


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="task-tree",
                server_version="1.0.0",
                capabilities=ServerCapabilities(
                    tools={},
                    resources={},
                ),
            ),
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())