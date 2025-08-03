# Task Tree MCP Server - Installation Demo

## 🚀 Quick Installation Guide

### Step 1: Clone and Setup
```bash
# Navigate to your projects directory
cd ~/projects

# Clone the repository (or navigate to existing)
cd ai-on-the-prize

# Activate virtual environment and install dependencies
source venv/bin/activate
pip install mcp networkx pydantic rich
```

### Step 2: Install MCP Server
```bash
# Add to Claude Desktop configuration
claude mcp install task-tree \
  --command python \
  --args "/Users/$(whoami)/projects/ai-on-the-prize/mcp_server/server.py" \
  --env PYTHONPATH="/Users/$(whoami)/projects/ai-on-the-prize/mcp_server"
```

*Or manually edit `~/Library/Application Support/Claude/claude_desktop_config.json`:*

```json
{
  "mcpServers": {
    "task-tree": {
      "command": "python",
      "args": [
        "/Users/YOUR_USERNAME/projects/ai-on-the-prize/mcp_server/server.py"
      ],
      "env": {
        "PYTHONPATH": "/Users/YOUR_USERNAME/projects/ai-on-the-prize/mcp_server"
      }
    }
  }
}
```

### Step 3: Restart Claude Desktop
Close and reopen Claude Desktop to load the MCP server.

## 🧪 Testing the Installation

### Test 1: Check MCP Server Connection
In Claude Desktop, you should see a small connection indicator showing the task-tree server is connected.

### Test 2: Create Your First Task
```
Let's test the task tree system. Can you create a root task for "Learn the task tree system" with high priority?
```

**Expected Claude Response:**
✅ **Task Created Successfully**

🔴 🟠 **Learn the task tree system**
   Learn how to use the hierarchical task management system
   📅 Created: 2025-01-30 14:25
   🆔 ID: `abc12345...`

### Test 3: Test AI Suggestions
```
I think we should break this learning task into smaller subtasks. What do you suggest?
```

**Expected Claude Response:**
🤖 **New Subtask Suggestions**

I suggest breaking this down into:
1. "Understand basic task creation and management"
2. "Learn the AI collaboration features"  
3. "Practice with complex hierarchical tasks"

**Should I create these subtasks?**

### Test 4: Test Context Awareness
```
Set the first subtask as current and show me the context that would be available.
```

**Expected Response:**
🎯 **Current Task Set**
...current task details...

## Context for Injection
```
=== TASK CONTEXT ===
[HIGH] Learn the task tree system | (Learn how to use the hierarchical task management system)
  └─ [MEDIUM] Understand basic task creation and management
=== END CONTEXT ===
```

### Test 5: Test Completion Suggestions
```
I've read through the documentation and understand how to create and manage tasks. I think the first subtask is complete.
```

**Expected Response:**
🤖 **Task Completion Suggestion**

📝 **Task:** Understand basic task creation and management
💭 **Reason:** You've demonstrated understanding of task creation and management
📋 **Evidence:** Successfully created tasks and navigated the system

**Should I mark this task as completed?**

## 🎉 Success Indicators

✅ **Connection**: MCP server shows as connected in Claude Desktop  
✅ **Task Creation**: Can create tasks with metadata  
✅ **AI Suggestions**: Claude proactively suggests subtasks and completions  
✅ **Context Flow**: Task lineage appears automatically in responses  
✅ **Interactive Approval**: Changes require your explicit approval  

## 🐛 Troubleshooting

### Server Not Connecting
- Check Python path in config is correct
- Ensure virtual environment has all dependencies
- Check Claude Desktop logs for errors

### Tools Not Available
- Restart Claude Desktop completely
- Verify config JSON syntax is valid
- Check server.py file permissions

### Database Issues
- Ensure write permissions in project directory
- Check if tasks.db is being created
- Verify SQLite is available

## 🚀 Ready to Use!

Once installed, the task tree system provides:

- **Hierarchical task management** with unlimited depth
- **AI-driven suggestions** for task completion and breakdown
- **Interactive approval workflow** keeping you in control
- **Automatic context injection** maintaining focus on goals
- **Persistent storage** across sessions

Your AI assistant can now actively collaborate on managing complex projects while ensuring you stay "on the prize" with your original objectives!