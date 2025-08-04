# AI on the Prize - Task Tree System

A production-ready hierarchical task management system for Claude Code that maintains context and prioritization through lineage injection. Instead of flat task lists, tasks are organized as nodes in a tree where AI agents maintain awareness of high-level goals while working on implementation details.

## 🎯 Core Mission Accomplished

**Problem Solved**: AI agents lose sight of original goals when diving deep into implementation details, leading to context drift and forgotten priorities.

**Solution Delivered**: Lineage-based context injection ensures AI agents always see the path from root objective to current task, keeping them "on the prize."

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
- **Removed Complexity**: No dependency management (tree structure handles ordering naturally)

## 📊 Implementation Status

### ✅ **Phase 3: Intelligence Layer (Complete)**
- **Enhanced LLM Usage Guidance**: Hierarchical planning examples and patterns
- **Completion Criteria Validation**: Prevents premature completion claims
- **Pattern Detection**: Identifies flat task lists and suggests hierarchical alternatives
- **Simplified Architecture**: Removed unused dependency fields

### ✅ **Phase 4: Advanced Features (Complete by Design)**
- **Task Mutation**: Handled naturally through immutable create/cancel pattern
- **Multi-Agent Support**: Left as user exercise (proper approach)
- **Quality Enforcement**: Completion criteria prevent "mission accomplished" moments

## 🚀 Getting Started

### **Installation**
```bash
# Add to Claude Code globally
claude mcp add task-tree "/path/to/venv/bin/python" "/path/to/mcp_server/server.py" -s user
```

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
