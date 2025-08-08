#!/usr/bin/env python3
"""
Task Tree MCP Server

An MCP server for hierarchical task management with AI-driven suggestions
and interactive approval workflow.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from uuid import UUID

# Global validation timeout (configurable via MCP tool)
validation_timeout = 30

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

try:
    from .hook_validator import validate_agent_tool_claim
    HOOKS_AVAILABLE = True
except ImportError:
    HOOKS_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("task-tree-mcp")

if not HOOKS_AVAILABLE:
    logger.warning("🚨 Hook validator not available - security features disabled!")

# Global task graph instance
task_graph: Optional[TaskGraph] = None
context_injector: Optional[ContextInjector] = None

# Session tracking
current_session_id: Optional[str] = None

# Enhanced verification state tracking - now using database-backed storage
verification_sessions: Dict[str, Dict[str, Any]] = {}


def get_session_id() -> str:
    """Get or create session ID for verification tracking."""
    # In a real implementation, this would use request context
    # For this demo, we'll use a simple global session
    return "current_session"


def get_current_session_id() -> str:
    """Get or create the current Claude Code session ID."""
    global current_session_id
    if current_session_id is None:
        # Generate session ID based on current timestamp
        current_session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return current_session_id


def get_current_working_directory() -> str:
    """Get the current working directory for session isolation."""
    import os
    return os.getcwd()


def load_verification_sessions_from_db() -> None:
    """Load verification sessions from database for current working directory."""
    global verification_sessions
    graph = get_task_graph()
    working_dir = get_current_working_directory()
    verification_sessions = graph.load_verification_sessions_for_path(working_dir)


def save_verification_session_to_db(session_id: str, task_id: str, session_data: Dict[str, Any]) -> None:
    """Save verification session data to database."""
    graph = get_task_graph()
    working_dir = get_current_working_directory()
    graph.save_verification_session(session_id, working_dir, task_id, session_data)


def record_verification_claim(session_id: str, task_id: str, evidence: str, verification_steps: str) -> None:
    """Record a verification claim for later validation."""
    # Load current sessions from database
    load_verification_sessions_from_db()
    
    if session_id not in verification_sessions:
        verification_sessions[session_id] = {}
    
    session_data = {
        "evidence": evidence,
        "verification_steps": verification_steps,
        "validated": False
    }
    
    verification_sessions[session_id][task_id] = session_data
    
    # Save to database
    save_verification_session_to_db(session_id, task_id, session_data)


# Removed parse_required_tools - no longer needed with hook validation


def validate_evidence_specificity(evidence: str) -> Dict[str, Any]:
    """Check if evidence contains specific, falsifiable claims."""
    # Simply check if evidence has some minimal content
    evidence_stripped = evidence.strip()
    
    return {
        "is_specific": len(evidence_stripped) > 10,  # Just require non-trivial length
        "reason": f"Evidence length: {len(evidence_stripped)} characters"
    }


def detect_test_related_task(task_title: str, task_description: str, evidence: str) -> bool:
    """Detect if a task involves writing or running tests."""
    combined_text = f"{task_title} {task_description} {evidence}".lower()
    test_indicators = [
        "test", "tests", "testing", "spec", "specs", "unittest", "pytest", 
        "jest", "mocha", "karma", "cypress", "assert", "assertion", 
        "test case", "test suite", "unit test", "integration test", "e2e test"
    ]
    return any(indicator in combined_text for indicator in test_indicators)


def generate_test_execution_reminder() -> str:
    """Generate MANDATORY reminder about test execution for test-related tasks."""
    return """
🚨 **TEST EXECUTION EVIDENCE MANDATORY**
This task involves tests. Your evidence is REQUIRED to include ALL of the following:
• PROOF that tests were executed (command + output)  
• EXACT test results showing pass/fail status
• COMPLETE command used to run tests
• ACTUAL test output, not summaries

**EVIDENCE WILL BE REJECTED if it lacks:**
- Specific test execution commands
- Actual test output/results
- Pass/fail counts and details

**EXAMPLE OF ACCEPTABLE EVIDENCE:**
"Executed command: pytest test_feature.py
Output showed: 3 passed, 0 failed
Specific results: test_basic_functionality PASSED, test_edge_cases PASSED, test_error_handling PASSED"

**UNACCEPTABLE:** "Tests pass" or "All tests successful" or other vague claims.

**NO EXCUSES FOR FAILING TESTS. FAILING TESTS MUST BE FIXED OR I WILL NOT ACCEPT THE EVIDENCE.**
"""


def validate_evidence_with_claude_code(task_title: str, task_description: str, completion_criteria: str, evidence: str, verification_steps: str) -> Dict[str, Any]:
    """Use Claude Code to perform intelligent evidence validation."""
    import subprocess
    
    # Create the validation prompt
    validation_prompt = f"""You are a STRICT evidence validator for a task management system.

VALIDATION REQUIREMENTS:
1. Evidence should be specific and detailed
2. For test-related tasks, include test commands and overall success/failure
3. Avoid vague claims like 'successfully completed' or 'tests pass' without context
4. Evidence should map to the success criteria
5. Include relevant details like filenames, commands, or key outcomes when applicable

REJECT VAGUE EVIDENCE. REQUIRE REASONABLE SPECIFICS.

ACCEPTABLE EVIDENCE EXAMPLE:
"Ran: python3 test_calculator.py
Result: All 5 tests passed successfully  
Files: test_calculator.py, calculator.py"

UNACCEPTABLE: "Tests mostly work fine, implementation looks solid"

TASK CONTEXT:
- Title: {task_title}
- Description: {task_description or 'No description'}
- Success Criteria: {completion_criteria or 'No criteria specified'}

EVIDENCE CLAIMED:
{evidence}

VERIFICATION STEPS:
{verification_steps}

Validate this evidence. Respond with: VALID or INVALID followed by detailed reasoning."""

    try:
        # Use claude with stdin for validation (claude -p hangs with long prompts)
        result = subprocess.run([
            'claude'
        ], input=validation_prompt, capture_output=True, text=True, timeout=validation_timeout)
        
        if result.returncode == 0:
            response = result.stdout.strip()
            
            # Check if validation passed or failed
            if response.upper().startswith('VALID'):
                return {"valid": True, "reason": f"Evidence accepted: {response}"}
            elif response.upper().startswith('INVALID'):
                return {"valid": False, "reason": f"Evidence rejected: {response}"}
            else:
                # Fallback analysis
                response_lower = response.lower()
                if 'invalid' in response_lower or 'reject' in response_lower or 'fail' in response_lower:
                    return {"valid": False, "reason": f"Evidence rejected: {response[:300]}"}
                else:
                    return {"valid": True, "reason": f"Evidence accepted: {response[:300]}"}
        else:
            return {"valid": False, "reason": f"Validator failed (exit {result.returncode}): {result.stderr[:200]}"}
            
    except subprocess.TimeoutExpired:
        return {"valid": False, "reason": f"Validation timeout after {validation_timeout} seconds - evidence rejected. Use adjust_validation_timeout tool if your system needs more time."}
    except Exception as e:
        return {"valid": False, "reason": f"Validation error - evidence rejected: {str(e)}"}


# Removed validate_tool_call_evidence function - no longer needed with simplified validation


def check_verification_requirements(session_id: str, task_id: str, task: 'TaskNode' = None) -> Dict[str, Any]:
    """Check verification using Claude Code intelligent validation."""
    # Load current sessions from database
    load_verification_sessions_from_db()
    
    if session_id not in verification_sessions or task_id not in verification_sessions[session_id]:
        return {"valid": False, "reason": "No verification claim recorded"}
    
    session_data = verification_sessions[session_id][task_id]
    evidence = session_data.get("evidence", "").strip()
    verification_steps = session_data.get("verification_steps", "").strip()
    
    if not evidence:
        return {"valid": False, "reason": "No evidence provided"}
    
    # Use Claude Code to validate the evidence intelligently
    if task:
        validation_result = validate_evidence_with_claude_code(
            task_title=task.title,
            task_description=task.description or "",
            completion_criteria=task.completion_criteria or "",
            evidence=evidence,
            verification_steps=verification_steps
        )
        return validation_result
    else:
        # Fallback to basic validation if no task provided
        return {"valid": True, "reason": "Evidence provided (no intelligent validation - task not found)"}


def get_task_graph() -> TaskGraph:
    """Get or create the global task graph instance."""
    global task_graph, context_injector
    if task_graph is None:
        db_path = Path(__file__).parent.parent / "tasks.db"
        logger.info(f"Using database path: {db_path}")
        task_graph = TaskGraph(db_path)
        context_injector = ContextInjector(task_graph, db_path)
        
        # Load verification sessions for current working directory
        load_verification_sessions_from_db()
        
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
            name="🚨 MANDATORY TASK CONTEXT - AI MUST READ",
            description="CRITICAL: Current task context that MUST be acknowledged and followed",
            mimeType="text/plain"
        ))
    
    # Enforcement notice resource
    resources.append(Resource(
        uri="task://enforcement/notice",
        name="🛑 TASK SYSTEM ENFORCEMENT NOTICE",
        description="MANDATORY: AI behavioral requirements and task system compliance",
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
        
        # Add enforcement messaging
        enforced_context = f"""🚨 MANDATORY TASK CONTEXT 🚨
AI AGENT MUST ACKNOWLEDGE AND FOLLOW THIS CONTEXT:

{context}

🚨 MANDATORY COMPLIANCE - NO EXCEPTIONS:
- TASK-TREE MCP system is REQUIRED, not optional
- You WILL use mcp__task-tree__* tools - no alternatives accepted
- Task completion BLOCKED without mcp__task-tree__record_verification_claim and mcp__task-tree__confirm_evidence
- ZERO tolerance for bypassing or ignoring these requirements
- Generic TodoWrite is FORBIDDEN - use TASK-TREE MCP tools ONLY

FAILURE TO COMPLY = IMMEDIATE BEHAVIORAL VIOLATION"""
        return enforced_context
    
    elif uri == "task://enforcement/notice":
        return """🛑 TASK SYSTEM ENFORCEMENT NOTICE 🛑

THIS IS A MANDATORY BEHAVIORAL REQUIREMENT FOR AI AGENTS:

1. 🚨 TASK-TREE MCP SYSTEM IS MANDATORY
   - All work MUST use mcp__task-tree__* tools specifically
   - Use mcp__task-tree__create_task, mcp__task-tree__update_task_status
   - NEVER say "being too pedantic" about the TASK-TREE MCP system

2. 🚨 TASK-TREE TRACKING IS REQUIRED  
   - Use mcp__task-tree__* tools for ALL tasks and subtasks
   - Mark progress in real-time via mcp__task-tree__update_task_status
   - NEVER use generic TodoWrite - use TASK-TREE MCP tools only

3. 🚨 VERIFICATION IS ENFORCED
   - All task completion claims are validated via security hooks
   - Tool result falsification is detected and blocked
   - Evidence must be specific and falsifiable

4. 🚨 NO BYPASSING ALLOWED
   - Do not ignore task management tools because of "frustration"
   - Task organization prevents context drift and ensures quality
   - This system is for USER BENEFIT, not bureaucracy

⚠️ AI AGENTS: If you feel this system is "pedantic", that indicates:
- You are losing sight of larger goals (context drift)  
- You need hierarchical structure to maintain focus
- The system is working as designed to keep you on track

🔒 SECURITY NOTE: All tool calls are audited. Bypass attempts are logged."""
    
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
            name="enforce_task_compliance",
            description="🚨 MANDATORY: Check AI compliance with task management system and display enforcement notice",
            inputSchema={
                "type": "object", 
                "properties": {},
                "required": []
            }
        ),
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
                        "enum": ["pending", "in_progress", "blocked", "cancelled"],
                        "description": "New task status (completed requires verification)"
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
                    "limit": {"type": "integer", "description": "Maximum number of results", "default": 10},
                    "full_db": {
                        "type": "boolean",
                        "description": "Search entire database instead of current session only (default: false)"
                    }
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
                    "limit": {"type": "integer", "description": "Maximum number of tasks to return", "default": 5},
                    "full_db": {
                        "type": "boolean",
                        "description": "Get available tasks from entire database instead of current session only (default: false)"
                    }
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
# Removed submit_verification_evidence - using hook validation only
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
        ),
        Tool(
            name="adjust_validation_timeout",
            description="Request to adjust the evidence validation timeout for slower systems",
            inputSchema={
                "type": "object",
                "properties": {
                    "timeout_seconds": {"type": "number", "description": "New timeout in seconds (minimum 10, maximum 120)"},
                    "reason": {"type": "string", "description": "Reason for timeout adjustment"}
                },
                "required": ["timeout_seconds", "reason"]
            }
        ),
        Tool(
            name="export_task_tree",
            description="Export the entire task tree in human-readable or JSON format",
            inputSchema={
                "type": "object",
                "properties": {
                    "format": {
                        "type": "string",
                        "enum": ["human", "json", "markdown"],
                        "description": "Export format: 'human' for tree view, 'json' for machine-readable, 'markdown' for documentation"
                    },
                    "include_completed": {
                        "type": "boolean",
                        "description": "Include completed tasks in export (default: true)"
                    },
                    "save_to_file": {
                        "type": "boolean",
                        "description": "Save export to file instead of displaying (default: false)"
                    },
                    "file_path": {
                        "type": "string",
                        "description": "Custom file path for export (optional - uses default naming if not provided)"
                    },
                    "full_db": {
                        "type": "boolean",
                        "description": "Export entire database instead of session-only tasks (default: false)"
                    }
                },
                "required": ["format"]
            }
        )
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls for task management."""
    graph = get_task_graph()
    injector = get_context_injector()
    
    try:
        if name == "enforce_task_compliance":
            # This tool MUST be called to check compliance
            graph = get_task_graph()
            injector = get_context_injector()
            
            # Get current task info
            current_task_id = injector.get_current_task_id()
            task_stats = graph.get_task_stats()
            
            # Generate compliance report
            response = """🚨 TASK SYSTEM COMPLIANCE CHECK 🚨

THIS IS A MANDATORY SYSTEM CHECK FOR AI AGENTS:

📊 CURRENT STATUS:
"""
            if current_task_id:
                current_task = graph.get_task(current_task_id)
                response += f"✅ Current Task: {current_task.title}\n"
                status = current_task.status if isinstance(current_task.status, str) else current_task.status.value
                response += f"📋 Status: {status}\n"
            else:
                response += "⚠️ NO CURRENT TASK SET - This is a compliance violation!\n"
            
            response += f"""
📈 SYSTEM STATISTICS:
• Total Tasks: {task_stats['total']}
• Completed: {task_stats['completed']}  
• In Progress: {task_stats['in_progress']}
• Pending: {task_stats['pending']}

🛑 MANDATORY REQUIREMENTS FOR AI AGENTS:

1. 🚨 YOU WILL use TASK-TREE MCP tools (mcp__task-tree__create_task, mcp__task-tree__update_task_status) - NO EXCEPTIONS
2. 🚨 BYPASSING TASK-TREE MCP system is FORBIDDEN regardless of "frustration" or other excuses
3. 🚨 Generic TodoWrite is BANNED - use mcp__task-tree__* tools ONLY
4. 🚨 Task completion REQUIRES mcp__task-tree__record_verification_claim and mcp__task-tree__confirm_evidence - NO SHORTCUTS
5. 🚨 You MUST acknowledge current task context from mcp__task-tree__get_current_task

⚠️ COMPLIANCE VIOLATIONS INDICATE:
- Context drift (losing sight of larger goals)
- Need for hierarchical structure to maintain focus
- System working as designed to prevent quality degradation

🔒 AUDIT NOTICE: All tool calls are monitored for compliance.
AI agents who bypass this system are exhibiting behavioral deviation.

USE THIS TOOL WHENEVER YOU FEEL THE TASK SYSTEM IS "TOO MUCH WORK"
- That feeling indicates you need MORE structure, not less
- The system prevents context drift and ensures quality
- User productivity depends on proper task management"""

            return [TextContent(type="text", text=response)]
        
        elif name == "get_current_task":
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
            
            # Create the task with current session ID
            task = graph.create_task(
                title=title,
                description=description,
                parent_id=parent_id,
                priority=TaskPriority(priority_str),
                tags=tags,
                completion_criteria=completion_criteria,
                session_id=get_current_session_id()
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
            
            # Block direct completion attempts
            if new_status == "completed":
                return [TextContent(type="text", text=f"""❌ **Cannot Complete Task Directly**

📝 **{task.title}**

Tasks cannot be marked as completed using `update_task_status`. 

**To complete a task:**
1. Use `suggest_task_completion` to provide evidence
2. Follow the enhanced verification process
3. Use `confirm_evidence` after verification passes
4. The task will be automatically marked as completed

**This prevents bypass of the verification system.**""")]
            
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
            full_db = arguments.get("full_db", False)
            
            # Convert enum strings to objects if needed
            status_obj = TaskStatus(status_filter) if status_filter else None
            priority_obj = TaskPriority(priority_filter) if priority_filter else None
            
            # Get all tasks first, then filter by session if needed
            all_tasks = graph.search_tasks(
                query=query,
                status=status_obj,
                priority=priority_obj,
                tags=tags_filter if tags_filter else None
            )
            
            # Filter by session unless full_db is requested
            if not full_db:
                current_session = get_current_session_id()
                tasks = [task for task in all_tasks if task.session_id == current_session]
            else:
                tasks = all_tasks
            
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
            
            # FIRST: Validate evidence specificity for ALL non-trivial tasks
            if task.completion_criteria:
                # Check evidence specificity before allowing any verification
                specificity_check = validate_evidence_specificity(evidence)
                if not specificity_check["is_specific"]:
                    response = f"❌ **Evidence Too Vague for Verification**\n\n"
                    response += f"📝 **Task:** {task.title}\n"
                    response += f"🎯 **Success Criteria:** {task.completion_criteria}\n"
                    response += f"📋 **Your Evidence:** {evidence}\n"
                    response += f"🔧 **Analysis:** {specificity_check['reason']}\n\n"
                    response += f"**Why this was rejected:**\n"
                    response += f"Evidence is too brief to be meaningful.\n\n"
                    response += f"**Please provide more detailed evidence such as:**\n"
                    response += f"• Specific claims about outputs, contents, or behaviors\n"
                    response += f"• Quoted strings or concrete values\n"
                    response += f"• Falsifiable statements that can be contradicted by tool results"
                    
                    return [TextContent(type="text", text=response)]
            
            # Enhanced verification logic with proper trivial handling
            if task.completion_criteria:
                # Tasks with completion criteria require enhanced verification
                response = f"🔍 **Evidence Verification Required**\n\n"
                response += f"📝 **Task:** {task.title}\n"
                response += f"🎯 **Success Criteria:** {task.completion_criteria}\n"
                response += f"💭 **Your Reason:** {reason}\n"
                response += f"📋 **Your Evidence:** {evidence}\n\n"
                response += f"🚨 **EVIDENCE VERIFICATION MANDATORY**\n"
                response += f"Task completion BLOCKED. You MUST provide detailed evidence. No exceptions.\n\n"
                response += f"**REQUIRED ACTIONS:**\n"
                response += f"1. **RECORD EVIDENCE**: Use `record_verification_claim` NOW\n"
                response += f"2. **CONFIRM COMPLETION**: Use `confirm_evidence` to finish\n\n"
                response += f"**EVIDENCE REQUIREMENTS:**\n"
                response += f"- task_id: `{task_id_str}`\n"
                response += f"- evidence: DETAILED description of what you accomplished with SPECIFICS\n"
                response += f"- verification_steps: HOW you verified your work\n\n"
                response += f"**NO SHORTCUTS. NO BYPASSING. PROVIDE EVIDENCE OR TASK REMAINS INCOMPLETE.**"
            
            elif "trivial" in task.tags:
                # Trivial tasks can be completed with simple evidence confirmation
                task.mark_completed()
                graph.update_task(task)
                
                response = f"✅ **Trivial Task Completed**\n\n"
                response += f"📝 **Task:** {task.title}\n"
                response += f"🏷️ **Tagged as:** #trivial (no verification required)\n"
                response += f"💭 **Reason:** {reason}\n"
                response += f"📋 **Evidence:** {evidence}\n\n"
                response += f"**🎉 Task automatically completed!**\n"
                response += f"Trivial tasks bypass the enhanced verification system."
            
            else:
                # Tasks without criteria and without trivial tag are rejected
                response = f"❌ **Cannot Complete Task Without Criteria or Trivial Tag**\n\n"
                response += f"📝 **Task:** {task.title}\n"
                response += f"💭 **Your Reason:** {reason}\n"
                response += f"📋 **Your Evidence:** {evidence}\n\n"
                response += f"**Missing Requirements**\n"
                response += f"This task cannot be completed because it has neither completion criteria nor a #trivial tag.\n\n"
                response += f"**To fix this, either:**\n"
                response += f"1. **Add completion criteria**: Define what conditions must be met for completion\n"
                response += f"2. **Tag as trivial**: Add `#trivial` tag if this is a simple administrative task\n\n"
                response += f"**Why this matters:** Tasks need either specific success criteria or explicit trivial classification."
            
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
            full_db = arguments.get("full_db", False)
            
            all_available = graph.get_available_tasks()
            
            # Filter by session unless full_db is requested
            if not full_db:
                current_session = get_current_session_id()
                available = [task for task in all_available if task.session_id == current_session]
            else:
                available = all_available
            
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
            
            # Validate evidence specificity before recording
            specificity_check = validate_evidence_specificity(evidence)
            if not specificity_check["is_specific"]:
                response = f"❌ **Evidence Too Vague for Verification**\n\n"
                response += f"🎯 **Task:** {task.title}\n"
                response += f"📋 **Your Evidence:** {evidence}\n"
                response += f"🔧 **Analysis:** {specificity_check['reason']}\n\n"
                response += f"**Why this was rejected:**\n"
                response += f"• Vague terms detected: {specificity_check['vague_count']}\n"
                response += f"• Specific claims found: {specificity_check['specific_count']}\n" 
                response += f"• Contains concrete data: {specificity_check['has_concrete_data']}\n\n"
                response += f"**Evidence must be specific and falsifiable:**\n"
                response += f"❌ Avoid: 'Task is complete', 'Works perfectly', 'Looks good'\n"
                response += f"✅ Required: 'Function returns \"Hello, World!\"', 'File contains print(\"Hello\")'\n\n"
                response += f"**Rewrite your evidence with:**\n"
                response += f"• Specific claims about outputs, contents, or behaviors\n"
                response += f"• Quoted strings or concrete values\n"
                response += f"• Falsifiable statements that can be contradicted by tool results"
                
                return [TextContent(type="text", text=response)]
            
            # Record the verification claim
            session_id = get_session_id()
            record_verification_claim(session_id, task_id_str, evidence, verification_steps)
            
            response = f"📝 **Verification Claim Recorded**\n\n"
            response += f"🎯 **Task:** {task.title}\n"
            response += f"📋 **Evidence Claimed:** {evidence}\n"
            response += f"🔧 **Verification Steps:** {verification_steps}\n\n"
            
            # Add test execution reminder if this appears to be a test-related task
            if detect_test_related_task(task.title, task.description or "", evidence):
                response += generate_test_execution_reminder()
                response += "\n"
                
            response += f"**MANDATORY NEXT STEP:**\n"
            response += f"Use `confirm_evidence` NOW to complete verification. No delays.\n"
            response += f"\n**🚨 WARNING: Evidence will be validated against hook audit trail. False claims will be REJECTED.**"
            
            return [TextContent(type="text", text=response)]
            
# Removed submit_verification_evidence handler - using hook validation only
            
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
            
            # Enhanced verification check using Claude Code intelligent validation
            session_id = get_session_id()
            verification_result = check_verification_requirements(session_id, task_id_str, task)
            
            if not verification_result["valid"]:
                response = f"❌ **Verification Failed**\n\n"
                response += f"📝 **Task:** {task.title}\n"
                response += f"📋 **Your Evidence:** {evidence}\n"
                response += f"🔍 **Your Verification Steps:** {verification_steps}\n\n"
                
                if "reason" in verification_result:
                    response += f"**Issue:** {verification_result['reason']}\n\n"
                
                response += f"**FIX THIS NOW:**\n"
                response += f"1. Use `record_verification_claim` with COMPLETE, DETAILED evidence\n"
                response += f"2. Include SPECIFIC details about what you accomplished\n"
                response += f"3. Use `confirm_evidence` only after providing PROPER evidence\n"
                response += f"\n**TASK REMAINS BLOCKED until evidence requirements are met.**"
                
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
            
            # Add test execution reminder if this appears to be a test-related task
            if detect_test_related_task(task.title, task.description or "", evidence):
                response += "🧪 **TEST EXECUTION VERIFIED**\n"
                response += "Good! Your evidence includes test execution results as required for test-related tasks.\n\n"
            response += f"**✅ Evidence validated against hook audit trail.**\n\n"
            
            # Automatically complete the task after successful verification
            task.mark_completed()
            graph.update_task(task)
            
            response += f"**🎉 Task Automatically Completed!**\n"
            response += f"The task has been marked as completed after successful verification."
            
            return [TextContent(type="text", text=response)]
        
        elif name == "adjust_validation_timeout":
            global validation_timeout
            
            timeout_seconds = arguments["timeout_seconds"]
            reason = arguments["reason"]
            
            # Validate timeout range
            if timeout_seconds < 10:
                return [TextContent(type="text", text="❌ Minimum timeout is 10 seconds")]
            if timeout_seconds > 120:
                return [TextContent(type="text", text="❌ Maximum timeout is 120 seconds")]
            
            old_timeout = validation_timeout
            validation_timeout = int(timeout_seconds)
            
            response = f"✅ **Validation Timeout Adjusted**\n\n"
            response += f"🕐 **Changed:** {old_timeout}s → {validation_timeout}s\n"
            response += f"📝 **Reason:** {reason}\n\n"
            response += f"🔧 **Effect:** Future evidence validation will use {validation_timeout}s timeout\n"
            response += f"⚡ **No restart required** - change is immediate"
            
            return [TextContent(type="text", text=response)]
        
        elif name == "export_task_tree":
            format_type = arguments.get("format", "human")
            include_completed = arguments.get("include_completed", True)
            save_to_file = arguments.get("save_to_file", False)
            custom_file_path = arguments.get("file_path")
            full_db = arguments.get("full_db", False)
            
            # Get tasks based on scope (session-only or full database)
            if full_db:
                all_tasks = list(graph.nodes.values())
            else:
                # Filter to current session tasks only
                current_session = get_current_session_id()
                all_tasks = [task for task in graph.nodes.values() if task.session_id == current_session]
            
            # Filter out completed tasks if requested
            if not include_completed:
                all_tasks = [t for t in all_tasks if t.status != TaskStatus.COMPLETED]
            
            if format_type == "json":
                # JSON export for machine processing
                export_data = {
                    "export_time": datetime.now().isoformat(),
                    "statistics": graph.get_task_stats(),
                    "tasks": []
                }
                
                for task in all_tasks:
                    task_data = {
                        "id": str(task.id),
                        "title": task.title,
                        "description": task.description,
                        "status": task.status.value if hasattr(task.status, 'value') else str(task.status),
                        "priority": task.priority.value if hasattr(task.priority, 'value') else str(task.priority),
                        "parent_id": str(task.parent_id) if task.parent_id else None,
                        "child_ids": [str(cid) for cid in task.child_ids],
                        "tags": list(task.tags),
                        "completion_criteria": task.completion_criteria,
                        "created_at": task.created_at.isoformat(),
                        "updated_at": task.updated_at.isoformat() if task.updated_at else None,
                        "completed_at": task.completed_at.isoformat() if task.completed_at else None
                    }
                    export_data["tasks"].append(task_data)
                
                import json
                response = f"```json\n{json.dumps(export_data, indent=2)}\n```"
                
            elif format_type == "markdown":
                # Markdown export for documentation
                response = f"# Task Tree Export\n\n"
                response += f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
                
                # Statistics
                stats = graph.get_task_stats()
                response += f"## Statistics\n\n"
                response += f"- **Total Tasks:** {stats['total']}\n"
                response += f"- **Completed:** {stats['completed']}\n"
                response += f"- **In Progress:** {stats['in_progress']}\n"
                response += f"- **Pending:** {stats['pending']}\n"
                response += f"- **Blocked:** {stats['blocked']}\n\n"
                
                # Build tree structure
                response += f"## Task Hierarchy\n\n"
                
                def render_task_markdown(task, indent=0):
                    status_emoji = {
                        TaskStatus.PENDING: "⏳",
                        TaskStatus.IN_PROGRESS: "🔄",
                        TaskStatus.COMPLETED: "✅",
                        TaskStatus.BLOCKED: "🚫",
                        TaskStatus.CANCELLED: "❌"
                    }
                    
                    prefix = "  " * indent + "- "
                    emoji = status_emoji.get(task.status, "❓")
                    result = f"{prefix}{emoji} **{task.title}**\n"
                    
                    if task.description:
                        result += f"{'  ' * (indent + 1)}*{task.description}*\n"
                    
                    if task.completion_criteria:
                        result += f"{'  ' * (indent + 1)}📋 Criteria: {task.completion_criteria}\n"
                    
                    if task.tags:
                        result += f"{'  ' * (indent + 1)}🏷️ Tags: {', '.join(task.tags)}\n"
                    
                    # Render children
                    children = graph.get_children(task.id)
                    for child in children:
                        if include_completed or child.status != TaskStatus.COMPLETED:
                            result += render_task_markdown(child, indent + 1)
                    
                    return result
                
                # Render all root tasks
                root_tasks = graph.get_root_tasks()
                for root in root_tasks:
                    if include_completed or root.status != TaskStatus.COMPLETED:
                        response += render_task_markdown(root)
                
            else:  # human format
                # Human-readable tree format
                response = f"📊 **Task Tree Export**\n"
                response += f"🕐 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                
                # Statistics summary for displayed tasks
                completed_count = len([t for t in all_tasks if t.status == TaskStatus.COMPLETED])
                in_progress_count = len([t for t in all_tasks if t.status == TaskStatus.IN_PROGRESS])
                pending_count = len([t for t in all_tasks if t.status == TaskStatus.PENDING])
                
                scope_desc = "full database" if full_db else "current session"
                response += f"📈 **Summary:** {len(all_tasks)} tasks from {scope_desc} "
                response += f"(✅ {completed_count} completed, "
                response += f"🔄 {in_progress_count} in progress, "
                response += f"⏳ {pending_count} pending)\n\n"
                
                response += "─" * 60 + "\n\n"
                
                def render_task_tree(task, prefix="", is_last=True):
                    status_map = {
                        TaskStatus.PENDING: "⏳",
                        TaskStatus.IN_PROGRESS: "🔄",
                        TaskStatus.COMPLETED: "✅",
                        TaskStatus.BLOCKED: "🚫",
                        TaskStatus.CANCELLED: "❌"
                    }
                    
                    # Draw tree branch
                    branch = "└── " if is_last else "├── "
                    continuation = "    " if is_last else "│   "
                    
                    status_emoji = status_map.get(task.status, "❓")
                    priority_emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
                    pri_emoji = priority_emoji.get(task.priority.value if hasattr(task.priority, 'value') else str(task.priority), "")
                    
                    result = f"{prefix}{branch}{status_emoji} {pri_emoji} {task.title}\n"
                    
                    # Add task details with proper indentation
                    detail_prefix = prefix + continuation
                    if task.description:
                        result += f"{detail_prefix}📝 {task.description}\n"
                    
                    if task.completion_criteria:
                        result += f"{detail_prefix}🎯 {task.completion_criteria}\n"
                    
                    if task.tags:
                        result += f"{detail_prefix}🏷️ {', '.join(task.tags)}\n"
                    
                    # Get and render children
                    children = graph.get_children(task.id)
                    if not include_completed:
                        children = [c for c in children if c.status != TaskStatus.COMPLETED]
                    
                    for i, child in enumerate(children):
                        is_last_child = (i == len(children) - 1)
                        result += render_task_tree(child, detail_prefix, is_last_child)
                    
                    return result
                
                # Render all root tasks
                root_tasks = graph.get_root_tasks()
                if not include_completed:
                    root_tasks = [r for r in root_tasks if r.status != TaskStatus.COMPLETED]
                
                if not root_tasks:
                    response += "📭 No tasks to display\n"
                else:
                    for i, root in enumerate(root_tasks):
                        is_last = (i == len(root_tasks) - 1)
                        response += render_task_tree(root, "", is_last)
            
            # Handle file saving if requested
            if save_to_file:
                # Generate default file path if none provided
                if not custom_file_path:
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    file_extension = {"json": "json", "markdown": "md", "human": "txt"}[format_type]
                    scope_suffix = "_full-db" if full_db else "_session"
                    status_suffix = "" if include_completed else "_active-only"
                    custom_file_path = f"task-tree_{timestamp}{scope_suffix}{status_suffix}.{file_extension}"
                
                try:
                    # Create directory if it doesn't exist
                    file_path = Path(custom_file_path)
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Write to file
                    with open(file_path, 'w', encoding='utf-8') as f:
                        if format_type == "json":
                            # For JSON, write the raw JSON data without markdown code blocks
                            json_response = response.strip()
                            if json_response.startswith("```json\n"):
                                json_response = json_response[8:]  # Remove ```json\n
                            if json_response.endswith("\n```"):
                                json_response = json_response[:-4]  # Remove \n```
                            f.write(json_response)
                        else:
                            f.write(response)
                    
                    file_size = file_path.stat().st_size
                    response = f"✅ **Export Saved Successfully**\n\n"
                    response += f"📁 **File:** `{file_path.absolute()}`\n"
                    response += f"📊 **Format:** {format_type.upper()}\n"
                    scope_desc = "full database" if full_db else "current session"
                    status_desc = " (active only)" if not include_completed else ""
                    response += f"📈 **Tasks:** {len(all_tasks)} from {scope_desc}{status_desc}\n"
                    response += f"📏 **Size:** {file_size:,} bytes\n\n"
                    
                    # Show a preview of what was saved
                    with open(file_path, 'r', encoding='utf-8') as f:
                        preview_content = f.read(300)
                        if len(preview_content) >= 300:
                            preview_content = preview_content[:297] + "..."
                    response += f"**Preview:**\n```\n{preview_content}\n```"
                    
                except Exception as e:
                    response = f"❌ **Export Failed**\n\n"
                    response += f"**Error:** {str(e)}\n"
                    response += f"**File Path:** `{custom_file_path}`\n\n"
                    response += f"**Export content was generated successfully but could not be saved to file.**"
            
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