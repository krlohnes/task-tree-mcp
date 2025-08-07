"""Task graph operations and management."""

import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from uuid import UUID

import networkx as nx

from .task_node import TaskNode, TaskStatus, TaskPriority


class TaskGraph:
    """
    Manages the task tree using NetworkX for graph operations.
    
    Provides methods for task creation, traversal, lineage extraction,
    and persistence to SQLite database.
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        """Initialize task graph with optional database path."""
        self.graph = nx.DiGraph()  # Directed graph for parent->child relationships
        self.nodes: Dict[UUID, TaskNode] = {}  # Fast node lookup
        
        # Database setup
        self.db_path = db_path or Path("tasks.db")
        self._init_database()
        self._load_from_database()
    
    def _init_database(self) -> None:
        """Initialize SQLite database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_tasks_updated 
                ON tasks(updated_at DESC)
            """)
    
    def _load_from_database(self) -> None:
        """Load tasks from database into memory."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT id, data FROM tasks")
            
            for task_id_str, task_data in cursor.fetchall():
                try:
                    task_node = TaskNode.model_validate_json(task_data)
                    # Use database ID as source of truth
                    db_id = UUID(task_id_str)
                    task_node.id = db_id
                    
                    self.nodes[db_id] = task_node
                    self.graph.add_node(db_id)
                    
                    # Add edges for parent-child relationships
                    if task_node.parent_id:
                        self.graph.add_edge(task_node.parent_id, db_id)
                
                except Exception as e:
                    print(f"Error loading task {task_id_str}: {e}")
    
    def _save_to_database(self, task_node: TaskNode) -> None:
        """Save a single task to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO tasks (id, data, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            """, (str(task_node.id), task_node.model_dump_json()))
    
    def create_task(
        self,
        title: str,
        description: Optional[str] = None,
        parent_id: Optional[UUID] = None,
        priority: TaskPriority = TaskPriority.MEDIUM,
        tags: Optional[Set[str]] = None,
        completion_criteria: Optional[str] = None
    ) -> TaskNode:
        """Create a new task node."""
        task = TaskNode(
            title=title,
            description=description,
            parent_id=parent_id,
            priority=priority,
            tags=tags or set(),
            completion_criteria=completion_criteria
        )
        
        # Add to graph and lookup dict
        self.nodes[task.id] = task
        self.graph.add_node(task.id)
        
        # Handle parent-child relationship
        if parent_id:
            if parent_id not in self.nodes:
                raise ValueError(f"Parent task {parent_id} not found")
            
            # Add edge in graph
            self.graph.add_edge(parent_id, task.id)
            
            # Update parent's child list
            parent = self.nodes[parent_id]
            parent.add_child(task.id)
            self._save_to_database(parent)
        
        # Save to database
        self._save_to_database(task)
        
        return task
    
    def get_task(self, task_id: UUID) -> Optional[TaskNode]:
        """Get a task by ID."""
        return self.nodes.get(task_id)
    
    def update_task(self, task: TaskNode) -> None:
        """Update an existing task."""
        if task.id not in self.nodes:
            raise ValueError(f"Task {task.id} not found")
        
        self.nodes[task.id] = task
        self._save_to_database(task)
    
    def delete_task(self, task_id: UUID, cascade: bool = False) -> None:
        """Delete a task, optionally cascading to children."""
        if task_id not in self.nodes:
            raise ValueError(f"Task {task_id} not found")
        
        task = self.nodes[task_id]
        
        if cascade:
            # Delete all descendants
            descendants = list(nx.descendants(self.graph, task_id))
            for desc_id in descendants:
                if desc_id in self.nodes:
                    del self.nodes[desc_id]
            
            # Remove from graph
            self.graph.remove_nodes_from([task_id] + descendants)
            
            # Delete from database
            with sqlite3.connect(self.db_path) as conn:
                placeholders = ",".join("?" * (len(descendants) + 1))
                conn.execute(f"DELETE FROM tasks WHERE id IN ({placeholders})", 
                           [str(task_id)] + [str(d) for d in descendants])
        
        else:
            # Check for children
            if task.child_ids:
                raise ValueError(f"Task {task_id} has children. Use cascade=True to delete them.")
            
            # Remove from parent's child list
            if task.parent_id and task.parent_id in self.nodes:
                parent = self.nodes[task.parent_id]
                parent.remove_child(task_id)
                self._save_to_database(parent)
            
            # Remove from graph and nodes
            self.graph.remove_node(task_id)
            del self.nodes[task_id]
            
            # Delete from database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM tasks WHERE id = ?", (str(task_id),))
    
    def get_lineage(self, task_id: UUID) -> List[TaskNode]:
        """Get the path from root to the specified task."""
        if task_id not in self.nodes:
            raise ValueError(f"Task {task_id} not found")
        
        # Find path to root
        path = []
        current_id = task_id
        
        while current_id is not None:
            if current_id not in self.nodes:
                break
            
            task = self.nodes[current_id]
            path.append(task)
            current_id = task.parent_id
        
        # Reverse to get root-to-task order
        return list(reversed(path))
    
    def get_lineage_context(self, task_id: UUID, max_length: int = 1000) -> str:
        """Get formatted lineage context for prompt injection."""
        lineage = self.get_lineage(task_id)
        
        if not lineage:
            return ""
        
        # Build context string
        context_parts = []
        context_parts.append("=== TASK CONTEXT ===")
        
        for i, task in enumerate(lineage):
            indent = "  " * i
            arrow = "└─ " if i > 0 else ""
            summary = task.get_context_summary()
            context_parts.append(f"{indent}{arrow}{summary}")
        
        context_parts.append("=== END CONTEXT ===")
        full_context = "\n".join(context_parts)
        
        # Truncate if too long
        if len(full_context) > max_length:
            lines = context_parts[:-1]  # Remove end marker
            while len("\n".join(lines)) > max_length - 50 and len(lines) > 2:
                lines.pop()  # Remove middle items, keep root and current
            
            lines.append("... (context truncated)")
            lines.append("=== END CONTEXT ===")
            full_context = "\n".join(lines)
        
        return full_context
    
    def get_root_tasks(self) -> List[TaskNode]:
        """Get all tasks that have no parent."""
        return [task for task in self.nodes.values() if task.is_root()]
    
    def get_children(self, task_id: UUID) -> List[TaskNode]:
        """Get direct children of a task."""
        if task_id not in self.nodes:
            return []
        
        task = self.nodes[task_id]
        return [self.nodes[child_id] for child_id in task.child_ids if child_id in self.nodes]
    
    def get_descendants(self, task_id: UUID) -> List[TaskNode]:
        """Get all descendants of a task."""
        if task_id not in self.graph:
            return []
        
        descendant_ids = nx.descendants(self.graph, task_id)
        return [self.nodes[desc_id] for desc_id in descendant_ids if desc_id in self.nodes]
    
    def get_available_tasks(self) -> List[TaskNode]:
        """Get tasks that can be started (all dependencies met)."""
        completed_task_ids = {
            task_id for task_id, task in self.nodes.items() 
            if task.status == TaskStatus.COMPLETED
        }
        
        available = []
        for task in self.nodes.values():
            if task.status == TaskStatus.PENDING and task.can_start(completed_task_ids):
                available.append(task)
        
        return sorted(available, key=lambda t: (t.priority, t.created_at))
    
    def get_blocked_tasks(self) -> List[TaskNode]:
        """Get tasks that are blocked by dependencies."""
        completed_task_ids = {
            task_id for task_id, task in self.nodes.items() 
            if task.status == TaskStatus.COMPLETED
        }
        
        blocked = []
        for task in self.nodes.values():
            if task.status == TaskStatus.PENDING and not task.can_start(completed_task_ids):
                blocked.append(task)
        
        return blocked
    
    def detect_cycles(self) -> List[List[UUID]]:
        """Detect circular dependencies in the task graph."""
        try:
            # NetworkX will raise NetworkXError if cycles exist
            cycles = list(nx.simple_cycles(self.graph))
            return cycles
        except nx.NetworkXError:
            return []
    
    def get_task_stats(self) -> Dict[str, int]:
        """Get statistics about tasks in the graph."""
        stats = {
            "total": len(self.nodes),
            "pending": 0,
            "in_progress": 0,
            "completed": 0,
            "blocked": 0,
            "cancelled": 0,
            "root_tasks": 0,
            "leaf_tasks": 0
        }
        
        for task in self.nodes.values():
            status_key = task.status if isinstance(task.status, str) else task.status.value
            stats[status_key] += 1
            if task.is_root():
                stats["root_tasks"] += 1
            if task.is_leaf():
                stats["leaf_tasks"] += 1
        
        return stats
    
    def search_tasks(
        self,
        query: str = "",
        status: Optional[TaskStatus] = None,
        priority: Optional[TaskPriority] = None,
        tags: Optional[Set[str]] = None
    ) -> List[TaskNode]:
        """Search tasks by various criteria."""
        results = []
        query_lower = query.lower()
        
        for task in self.nodes.values():
            # Filter by status
            if status and task.status != status:
                continue
            
            # Filter by priority
            if priority and task.priority != priority:
                continue
            
            # Filter by tags
            if tags and not tags.issubset(task.tags):
                continue
            
            # Filter by query text
            if query and not (
                query_lower in task.title.lower() or
                (task.description and query_lower in task.description.lower())
            ):
                continue
            
            results.append(task)
        
        return sorted(results, key=lambda t: (t.priority, t.created_at))
    
    def export_to_dict(self) -> Dict:
        """Export the entire task graph to a dictionary."""
        return {
            "nodes": {str(task_id): task.model_dump() for task_id, task in self.nodes.items()},
            "edges": list(self.graph.edges()),
            "stats": self.get_task_stats()
        }