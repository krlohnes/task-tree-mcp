# Task Tree MCP

A hierarchical task management system for Claude Code that maintains context and prioritization through lineage injection. Instead of flat task lists, tasks are organized as nodes in a tree where AI agents maintain awareness of high-level goals while working on implementation details.

## 🎯 Core Problem Statement

**Problem**: AI agents lose sight of original goals when diving deep into implementation details, leading to context drift and forgotten priorities.

**Solution**: Lineage-based context injection ensures AI agents always see the path from root objective to current task, keeping them "on the prize."

## ✅ Features

### **Hierarchical Task Management**
- **Tree Structure**: Tasks organized with unlimited parent-child depth
- **Status Tracking**: pending, in_progress, completed, blocked, cancelled
- **Priority System**: low, medium, high with contextual weighting
- **Rich Metadata**: descriptions, tags, completion criteria, timestamps
- **Persistent Storage**: SQLite database maintains state across sessions

### **Claude Code Integration (MCP Server)**
- **Global Installation**: Available across all Claude Code projects
- **Context Injection**: Task lineage flows into every AI interaction
- **Interactive Tools**: Create, search, complete, and navigate tasks
- **Real-time Updates**: Changes reflect immediately across sessions

### **Intelligence Layer**
- **Hierarchical Planning Guidance**: Teaches proper task decomposition vs flat lists
- **Pattern Detection**: Identifies and suggests improvements for flat task patterns
- **Completion Criteria Validation**: Prevents premature task completion without explicit success criteria
- **Immutability Pattern**: Encourages creating new tasks vs editing existing ones for audit trail

## 🛠️ Technical Architecture

### **Core Components**
- **TaskNode**: Pydantic model with status, priority, relationships, completion criteria
- **TaskGraph**: NetworkX-based graph operations with SQLite persistence
- **ContextInjector**: Lineage extraction and context formatting for prompts
- **MCP Server**: Claude Code integration providing interactive task tools

### **Key Design Decisions**
- **Direct Lineage Only**: Simple root→current path (no siblings/branches to avoid noise)
- **Immutable Tasks**: No edit capabilities - create new tasks when requirements change
- **Safety-First Validation**: Completion criteria required by default with #trivial tag override

## 🚀 Getting Started

### **Installation**

**Prerequisites:**
```bash
# 1. Install Python dependencies
pip install pydantic networkx

# 2. Clone or download this repository
git clone https://github.com/your-username/ai-on-the-prize.git
cd ai-on-the-prize
```

**Add to Claude Code:**
```bash
# Option 1: Using absolute paths (recommended)
claude mcp add task-tree "python" "/absolute/path/to/ai-on-the-prize/mcp_server/server.py" -s user

# Option 2: Using virtual environment
claude mcp add task-tree "/path/to/venv/bin/python" "/absolute/path/to/ai-on-the-prize/mcp_server/server.py" -s user

# Verify installation
claude mcp list
```

**Restart Claude Code** after installation to activate the MCP server.

### **Basic Usage**
```bash
# Create hierarchical tasks
create_task(title="Build authentication system", priority="high", 
           completion_criteria="Users can register, login, and access protected routes")

# Create subtasks with validation siblings
create_task(title="JWT token implementation", parent_id="...", 
           completion_criteria="Tokens generate, validate, and refresh correctly")
create_task(title="Validate JWT security standards", parent_id="...",
           completion_criteria="OWASP compliance verified, no vulnerabilities found")

# Work with full context awareness
get_current_task()  # Shows: Root → Auth → JWT → Current context
```

### **Best Practices**
- **Start with completion criteria**: Define success before beginning work
- **Use validation siblings**: Every action should have a verification step
- **Go 3+ levels deep**: Break complex tasks into hierarchical components
- **Create vs Edit**: Make new tasks when requirements change (preserves history)
- **Add checkpoints**: Include user authorization points for autonomous work

## 🎉 Success Metrics Achieved

- ✅ **Context Retention**: Root goals always visible regardless of task depth
- ✅ **Quality Enforcement**: Completion criteria prevent premature claims
- ✅ **Pattern Teaching**: AI learns hierarchical vs flat task organization
- ✅ **Audit Trail**: Immutable tasks preserve decision history

## 🔧 System Requirements

- **Python 3.8+** with pydantic, networkx
- **Claude Code** with MCP support
- **SQLite** (included with Python)

The task tree system transforms AI task management from passive checklists to active context-aware hierarchical planning, ensuring AI agents stay focused on high-priority objectives while maintaining awareness of implementation details.
