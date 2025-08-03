# AI on the Prize - Task Tree System

A hierarchical task management system for Claude Code that maintains context and prioritization through lineage injection. Instead of simple checkboxes, tasks are organized as nodes in a tree where completion creates new branches of work while preserving the original goal hierarchy.

## Core Concept

The system injects task lineage (root → current node path) into every UserPromptSubmit event in Claude Code. This ensures AI agents always maintain visibility into:
- The original high-priority goal (root node)
- The path of decisions that led to the current task
- The context needed to make informed decisions about subtasks

## Architecture Overview

### Data Models
- **TaskNode**: Individual task with ID, content, status, parent/children relationships, metadata
- **TaskGraph**: Graph operations, lineage extraction, persistence
- **ContextInjector**: Hook integration for Claude Code prompt augmentation
- **TaskManager**: CRUD operations, completion logic, validation

### Technical Stack
- **Python** for core implementation
- **networkx** for graph operations and traversal
- **pydantic** for data validation and serialization
- **sqlite3** for persistent storage
- **rich** for CLI visualization and debugging

## Implementation Phases

### Phase 1: Core Architecture
1. Design TaskNode and TaskGraph data models using pydantic
2. Implement graph operations with networkx (add nodes, traverse lineage, mark completion)
3. Create persistence layer with sqlite for task storage
4. Build basic CLI interface for task management

### Phase 2: Claude Code Integration
1. Create UserPromptSubmit hook that injects task lineage
2. Implement context truncation strategies for token management
3. Add slash commands for task tree manipulation (/task create, /task complete, etc.)
4. Build visualization tools for understanding task hierarchy

### Phase 3: Intelligence Layer
1. Add semantic task clustering to reduce noise
2. Implement dynamic context weighting based on relevance
3. Create task completion criteria and validation
4. Add circular dependency detection and resolution

### Phase 4: Advanced Features
1. Multi-agent task distribution
2. Task mutation handling (scope changes, pivots)
3. Analytics and reporting on task patterns
4. Integration with existing project management tools

## Key Design Challenges

### Context Window Management
- **Problem**: Deep task trees could consume significant tokens with full lineage injection
- **Solutions**: 
  - Smart truncation based on relevance scores
  - Dynamic summarization of completed branches
  - Configurable maximum lineage depth

### Task Completion Ambiguity
- **Problem**: Unlike checkboxes, "completing a node" can be subjective
- **Solutions**:
  - Explicit completion criteria in task definitions
  - Validation hooks before marking complete
  - Support for partial completion states

### Circular Dependencies
- **Problem**: Task B depends on A, but completing A reveals B was wrong approach
- **Solutions**:
  - Dependency graph validation
  - Task mutation and pivot support
  - Rollback mechanisms for invalid task paths

### Over-Constraining vs Context Drift
- **Problem**: Balance between maintaining focus and allowing beneficial pivots
- **Solutions**:
  - Weighted context injection (recent tasks get more weight)
  - AI-generated relevance scoring
  - Escape hatches for major direction changes

## Integration with Claude Code

Following the pattern established by code-personas project:

```
.claude/
├── commands/
│   └── task.md              # /task slash command implementation
├── hooks/
│   └── user-prompt-submit   # Hook that injects task lineage
├── lib/
│   ├── task_graph.py       # Core graph operations
│   ├── task_node.py        # Task data model
│   └── context_injector.py # Prompt augmentation logic
└── data/
    └── tasks.db            # SQLite task storage
```

## Example Usage Flow

1. **Initial Task Creation**:
   ```
   /task create "Implement user authentication system"
   ```

2. **Subtask Generation** (AI discovers complexity):
   ```
   Current context: Root → "Implement user authentication system"
   AI creates subtasks: JWT handling, password hashing, session management
   ```

3. **Deep Nesting** (working on specific implementation):
   ```
   Current context: Root → "Implement user authentication" → "JWT handling" → "Token refresh logic"
   AI maintains awareness of authentication goal while working on specific refresh implementation
   ```

4. **Task Completion and Branching**:
   ```
   Completed: "Token refresh logic"
   New context: Root → "Implement user authentication" → "JWT handling" → [next task]
   ```

## Success Metrics

- **Context Retention**: AI agents maintain awareness of original goals even in deep task nests
- **Priority Preservation**: High-priority tasks remain visible and influence decision-making
- **Efficient Navigation**: Easy movement between task levels without losing progress
- **Completion Accuracy**: Clear criteria for when tasks are truly finished vs partially complete

## Future Enhancements

- Integration with existing project management tools (Jira, Linear, etc.)
- Multi-agent coordination where different agents work at different tree levels
- Machine learning for automatic task prioritization and relevance scoring
- Visual task tree browser with interactive navigation
- Team collaboration features for shared task trees