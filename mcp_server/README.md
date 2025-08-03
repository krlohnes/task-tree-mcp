# Task Tree MCP Server

A Model Context Protocol (MCP) server for hierarchical task management with AI-driven suggestions and interactive approval workflow.

## Features

### Core Task Management
- ✅ **Hierarchical Tasks**: Parent-child relationships with unlimited depth
- ✅ **Status Tracking**: pending, in_progress, completed, blocked, cancelled
- ✅ **Priority Levels**: low, medium, high, critical with visual indicators
- ✅ **Rich Metadata**: descriptions, tags, completion criteria, timestamps
- ✅ **Persistent Storage**: SQLite database maintains state across sessions

### AI-Collaborative Features
- 🤖 **Task Suggestions**: Claude can suggest task completions and new subtasks
- 🔄 **Interactive Approval**: User maintains control over all task changes
- 📋 **Context Awareness**: Current task lineage available to Claude
- 🎯 **Smart Recommendations**: Based on completion criteria and progress

### MCP Integration
- 📡 **10 MCP Tools**: Complete task management API
- 📚 **3 Resources**: Current context, tree visualization, statistics
- 🔍 **Search & Filter**: Find tasks by title, status, priority, tags
- 📊 **Rich Responses**: Formatted with emojis and structure

## Installation

```bash
cd mcp_server
pip install -e .
```

## Configuration

Add to your Claude Desktop `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "task-tree": {
      "command": "python",
      "args": ["/path/to/ai-on-the-prize/mcp_server/server.py"]
    }
  }
}
```

## Available Tools

### Core Operations
- `get_current_task` - Get active task and lineage context
- `create_task` - Create new tasks with full metadata
- `update_task_status` - Change task status with reason
- `set_current_task` - Set active task for context
- `get_task_details` - Detailed task information with hierarchy

### Search & Discovery  
- `search_tasks` - Find tasks by criteria
- `get_available_tasks` - Tasks ready to work on
- `get_task_lineage` - Path from root to specific task

### AI Collaboration
- `suggest_task_completion` - Claude suggests marking task complete
- `suggest_new_subtask` - Claude suggests breaking down work

## Example Workflow

### 1. Create Root Task
```
create_task(
  title="Implement user authentication",
  priority="high", 
  completion_criteria="Users can register, login, and logout securely",
  set_as_current=true
)
```

### 2. Claude Suggests Subtasks
Claude: "I notice this authentication task is complex. Should I create a subtask for JWT token handling?"

```
suggest_new_subtask(
  parent_id="auth-task-id",
  title="Implement JWT token handling", 
  reason="Authentication requires secure token management"
)
```

### 3. Work on Subtask
```
create_task(
  title="Implement JWT token handling",
  parent_id="auth-task-id",
  set_as_current=true
)
```

Now Claude has context:
```
=== TASK CONTEXT ===
[HIGH] Implement user authentication
  └─ [HIGH] Implement JWT token handling
=== END CONTEXT ===
```

### 4. Claude Suggests Completion
Claude: "I've implemented JWT generation, validation, and refresh logic. This meets the success criteria. Should I mark this task as completed?"

```
suggest_task_completion(
  task_id="jwt-task-id",
  reason="JWT functionality is complete and tested",
  evidence="All token operations working correctly"
)
```

### 5. User Approves
```
update_task_status(
  task_id="jwt-task-id", 
  status="completed"
)
```

## Interactive Suggestions

The MCP server enables a collaborative workflow:

1. **Claude works** on current task with full context
2. **Claude detects** completion or need for subtasks  
3. **Claude suggests** actions with reasoning
4. **User approves** or modifies suggestions
5. **Tasks update** automatically

This keeps you in control while letting Claude proactively manage the task hierarchy.

## Resources

- `task://current/context` - Current task lineage for context injection
- `task://tree/visualization` - ASCII tree of all tasks
- `task://stats/summary` - JSON statistics about task distribution

## Benefits Over Hooks

- ✅ **Two-way communication**: Claude can suggest AND execute changes
- ✅ **Interactive approval**: User maintains control with natural conversation
- ✅ **Rich formatting**: Emojis, structure, and clear presentation
- ✅ **Standardized protocol**: Works across all Claude interfaces
- ✅ **Extensible**: Easy to add new collaborative features

The MCP approach transforms task trees from passive context injection into active AI collaboration while preserving human oversight.