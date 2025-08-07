"""Task node data model and core functionality."""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """Task completion status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class TaskPriority(str, Enum):
    """Task priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TaskNode(BaseModel):
    """
    Individual task node in the task tree.
    
    Represents a single unit of work with hierarchical relationships,
    status tracking, and metadata for context injection.
    """
    
    id: UUID = Field(default_factory=uuid4, description="Unique task identifier")
    title: str = Field(..., min_length=1, max_length=200, description="Brief task title")
    description: Optional[str] = Field(None, max_length=2000, description="Detailed task description")
    
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Current task status")
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM, description="Task priority level")
    
    # Hierarchical relationships
    parent_id: Optional[UUID] = Field(None, description="Parent task ID")
    child_ids: Set[UUID] = Field(default_factory=set, description="Set of child task IDs")
    
    # Metadata
    session_id: Optional[str] = Field(None, description="Claude Code session that created this task")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Task creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    completed_at: Optional[datetime] = Field(None, description="Task completion timestamp")
    
    # Context and tracking
    tags: Set[str] = Field(default_factory=set, description="Task tags for categorization")
    estimated_effort: Optional[int] = Field(None, ge=1, description="Estimated effort in minutes")
    actual_effort: Optional[int] = Field(None, ge=0, description="Actual effort spent in minutes")
    
    # AI agent context
    completion_criteria: Optional[str] = Field(None, description="Explicit criteria for task completion")
    context_weight: float = Field(default=1.0, ge=0.0, le=10.0, description="Weight for context injection")
    
    
    class Config:
        """Pydantic model configuration."""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
            set: lambda v: list(v)
        }
    
    def add_child(self, child_id: UUID) -> None:
        """Add a child task ID to this node."""
        self.child_ids.add(child_id)
        self.updated_at = datetime.utcnow()
    
    def remove_child(self, child_id: UUID) -> None:
        """Remove a child task ID from this node."""
        self.child_ids.discard(child_id)
        self.updated_at = datetime.utcnow()
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to this task."""
        self.tags.add(tag.lower().strip())
        self.updated_at = datetime.utcnow()
    
    def remove_tag(self, tag: str) -> None:
        """Remove a tag from this task."""
        self.tags.discard(tag.lower().strip())
        self.updated_at = datetime.utcnow()
    
    
    def mark_in_progress(self) -> None:
        """Mark task as in progress."""
        if self.status == TaskStatus.PENDING:
            self.status = TaskStatus.IN_PROGRESS
            self.updated_at = datetime.utcnow()
    
    def mark_completed(self) -> None:
        """Mark task as completed."""
        if self.status in [TaskStatus.PENDING, TaskStatus.IN_PROGRESS, TaskStatus.BLOCKED]:
            self.status = TaskStatus.COMPLETED
            self.completed_at = datetime.utcnow()
            self.updated_at = datetime.utcnow()
    
    def mark_blocked(self) -> None:
        """Mark task as blocked."""
        if self.status in [TaskStatus.PENDING, TaskStatus.IN_PROGRESS]:
            self.status = TaskStatus.BLOCKED
            self.updated_at = datetime.utcnow()
    
    def is_root(self) -> bool:
        """Check if this is a root task (no parent)."""
        return self.parent_id is None
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf task (no children)."""
        return len(self.child_ids) == 0
    
    def can_start(self, completed_tasks: Set[UUID]) -> bool:
        """Check if this task can start (always true with hierarchical structure)."""
        return True
    
    def get_context_summary(self) -> str:
        """Get a summary string for context injection."""
        parts = [f"[{self.priority.upper()}] {self.title}"]
        
        if self.description:
            # Truncate description for context injection
            desc = self.description[:100] + "..." if len(self.description) > 100 else self.description
            parts.append(f"({desc})")
        
        if self.tags:
            parts.append(f"Tags: {', '.join(sorted(self.tags))}")
        
        if self.completion_criteria:
            criteria = self.completion_criteria[:150] + "..." if len(self.completion_criteria) > 150 else self.completion_criteria
            parts.append(f"Success criteria: {criteria}")
        
        return " | ".join(parts)
    
    def __str__(self) -> str:
        """String representation for debugging."""
        return f"TaskNode({self.title}, {self.status}, priority={self.priority})"
    
    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return f"TaskNode(id={self.id}, title='{self.title}', status={self.status}, parent_id={self.parent_id})"