#!/usr/bin/env python3
"""
Simple Task Tree MCP Server for Claude Code compatibility.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ServerCapabilities

# Set up path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent))

from task_tree import TaskGraph, TaskNode, TaskStatus, TaskPriority, ContextInjector

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("task-tree-mcp")

# Global instances
task_graph: Optional[TaskGraph] = None
context_injector: Optional[ContextInjector] = None


def init_task_system():
    """Initialize the task management system."""
    global task_graph, context_injector
    if task_graph is None:
        db_path = Path(__file__).parent.parent / "tasks.db"
        task_graph = TaskGraph(db_path)
        context_injector = ContextInjector(task_graph, db_path)
    return task_graph, context_injector


# Create the server
server = Server("task-tree")


@server.list_tools()
async def list_tools() -> List[Tool]:
    """List available tools."""
    return [
        Tool(
            name="create_task",
            description="Create a new task",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "priority": {"type": "string", "enum": ["low", "medium", "high", "critical"], "default": "medium"},
                    "set_current": {"type": "boolean", "default": False}
                },
                "required": ["title"]
            }
        ),
        Tool(
            name="get_current_task",
            description="Get the current task and context",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="list_tasks",
            description="List all tasks",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="complete_task",
            description="Mark a task as completed",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"}
                },
                "required": ["task_id"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls."""
    try:
        graph, injector = init_task_system()
        
        if name == "create_task":
            title = arguments["title"]
            description = arguments.get("description", "")
            priority = TaskPriority(arguments.get("priority", "medium"))
            set_current = arguments.get("set_current", False)
            
            task = graph.create_task(
                title=title,
                description=description,
                priority=priority
            )
            
            if set_current:
                injector.set_current_task_id(task.id)
            
            response = f"✅ Created task: {task.title}\n"
            response += f"ID: {str(task.id)[:8]}...\n"
            response += f"Priority: {task.priority}\n"
            if set_current:
                response += "🎯 Set as current task"
            
            return [TextContent(type="text", text=response)]
        
        elif name == "get_current_task":
            current_id = injector.get_current_task_id()
            if not current_id:
                return [TextContent(type="text", text="No current task set")]
            
            task = graph.get_task(current_id)
            if not task:
                return [TextContent(type="text", text="Current task not found")]
            
            context = injector.get_context_for_injection()
            
            response = f"🎯 Current Task: {task.title}\n"
            response += f"Status: {task.status}\n"
            response += f"Priority: {task.priority}\n"
            if task.description:
                response += f"Description: {task.description}\n"
            response += f"\nContext:\n{context}"
            
            return [TextContent(type="text", text=response)]
        
        elif name == "list_tasks":
            tasks = list(graph.nodes.values())
            if not tasks:
                return [TextContent(type="text", text="No tasks found")]
            
            response = "📋 All Tasks:\n\n"
            for task in tasks[:10]:  # Limit to 10 tasks
                status_emoji = {"pending": "⏳", "in_progress": "🔄", "completed": "✅"}.get(task.status, "⚪")
                response += f"{status_emoji} {task.title} [{task.priority}]\n"
                if task.description:
                    response += f"   {task.description}\n"
                response += f"   ID: {str(task.id)[:8]}...\n\n"
            
            return [TextContent(type="text", text=response)]
        
        elif name == "complete_task":
            task_id_str = arguments["task_id"]
            try:
                task_id = UUID(task_id_str)
            except ValueError:
                return [TextContent(type="text", text="Invalid task ID format")]
            
            task = graph.get_task(task_id)
            if not task:
                return [TextContent(type="text", text="Task not found")]
            
            task.mark_completed()
            graph.update_task(task)
            
            return [TextContent(type="text", text=f"✅ Completed task: {task.title}")]
        
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    
    except Exception as e:
        logger.error(f"Error in tool {name}: {e}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def main():
    """Run the server."""
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
                )
            )
        )


if __name__ == "__main__":
    asyncio.run(main())