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
• Go 3+ levels deep for complex tasks"""


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
                "required": ["task_id", "reason"]
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
            
            response = f"✅ **Task Created Successfully**\n\n{format_task_for_display(task)}"
            if set_as_current:
                response += "\n\n🎯 Set as current task"
            
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
            
            # Update status
            if new_status == "completed":
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
            evidence = arguments.get("evidence", "")
            
            try:
                task_id = UUID(task_id_str)
            except ValueError:
                return [TextContent(type="text", text="❌ Invalid task ID format")]
            
            task = graph.get_task(task_id)
            if not task:
                return [TextContent(type="text", text=f"❌ Task {task_id_str} not found")]
            
            response = f"🤖 **Task Completion Suggestion**\n\n"
            response += f"📝 **Task:** {task.title}\n"
            response += f"💭 **Reason:** {reason}\n"
            if evidence:
                response += f"📋 **Evidence:** {evidence}\n"
            if task.completion_criteria:
                response += f"🎯 **Success Criteria:** {task.completion_criteria}\n"
            response += f"\n**Should I mark this task as completed?**\n"
            response += f"Reply with:\n"
            response += f"• `update_task_status` with task_id `{task_id_str}` and status `completed` to approve\n"
            response += f"• Or explain what still needs to be done"
            
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