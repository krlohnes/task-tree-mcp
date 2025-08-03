"""Context injection for Claude Code integration."""

import os
from pathlib import Path
from typing import Optional
from uuid import UUID

from .task_graph import TaskGraph


class ContextInjector:
    """
    Handles context injection for Claude Code hooks.
    
    Integrates with Claude Code's UserPromptSubmit hook to inject
    task lineage context into every user interaction.
    """
    
    def __init__(self, task_graph: Optional[TaskGraph] = None, db_path: Optional[Path] = None):
        """Initialize context injector with task graph."""
        self.task_graph = task_graph or TaskGraph(db_path)
        self.current_task_file = Path.home() / ".claude" / "state" / "current-task"
        self.current_task_file.parent.mkdir(parents=True, exist_ok=True)
    
    def get_current_task_id(self) -> Optional[UUID]:
        """Get the currently active task ID from state file."""
        try:
            if self.current_task_file.exists():
                task_id_str = self.current_task_file.read_text().strip()
                if task_id_str:
                    return UUID(task_id_str)
        except (ValueError, FileNotFoundError):
            pass
        return None
    
    def set_current_task_id(self, task_id: Optional[UUID]) -> None:
        """Set the currently active task ID in state file."""
        if task_id is None:
            if self.current_task_file.exists():
                self.current_task_file.unlink()
        else:
            self.current_task_file.write_text(str(task_id))
    
    def get_context_for_injection(self, max_length: int = 1000) -> str:
        """
        Get task context for injection into user prompts.
        
        Args:
            max_length: Maximum length of context to inject
            
        Returns:
            Formatted context string for prompt injection
        """
        current_task_id = self.get_current_task_id()
        
        if not current_task_id:
            return ""
        
        try:
            return self.task_graph.get_lineage_context(current_task_id, max_length)
        except ValueError:
            # Current task not found, clear the state
            self.set_current_task_id(None)
            return ""
    
    def inject_context(self, user_prompt: str, max_length: int = 1000) -> str:
        """
        Inject task context into a user prompt.
        
        Args:
            user_prompt: Original user prompt
            max_length: Maximum length of context to inject
            
        Returns:
            Augmented prompt with task context
        """
        context = self.get_context_for_injection(max_length)
        
        if not context:
            return user_prompt
        
        # Inject context at the beginning of the prompt
        return f"{context}\n\n{user_prompt}"
    
    def auto_advance_task(self) -> Optional[UUID]:
        """
        Automatically advance to the next available task.
        
        Returns:
            ID of the new current task, or None if no tasks available
        """
        available_tasks = self.task_graph.get_available_tasks()
        
        if available_tasks:
            # Pick highest priority available task
            next_task = available_tasks[0]
            self.set_current_task_id(next_task.id)
            return next_task.id
        
        return None
    
    def should_inject_context(self, user_prompt: str) -> bool:
        """
        Determine if context should be injected for this prompt.
        
        Some prompts (like meta-commands) might not need task context.
        """
        # Skip injection for certain command patterns
        skip_patterns = [
            "/task",  # Task management commands
            "/persona",  # Persona switching
            "claude code",  # Meta discussions about Claude Code
            "what is",  # General questions
            "how to",  # General how-to questions
        ]
        
        prompt_lower = user_prompt.lower().strip()
        
        for pattern in skip_patterns:
            if prompt_lower.startswith(pattern):
                return False
        
        return True
    
    def create_hook_script(self, hook_path: Path) -> None:
        """
        Create the actual hook script for Claude Code integration.
        
        Args:
            hook_path: Path where the hook script should be created
        """
        hook_script = f'''#!/usr/bin/env python3
"""
Claude Code UserPromptSubmit hook for task tree context injection.
This script is automatically called by Claude Code before each user prompt.
"""

import sys
from pathlib import Path

# Add the task_tree module to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

try:
    from task_tree.context_injector import ContextInjector
    
    def main():
        """Main hook function called by Claude Code."""
        # Read the user prompt from stdin
        user_prompt = sys.stdin.read().strip()
        
        # Initialize context injector
        injector = ContextInjector()
        
        # Check if we should inject context
        if not injector.should_inject_context(user_prompt):
            print(user_prompt)
            return
        
        # Inject task context
        augmented_prompt = injector.inject_context(user_prompt)
        
        # Output the augmented prompt
        print(augmented_prompt)
    
    if __name__ == "__main__":
        main()

except ImportError as e:
    # If task_tree module is not available, just pass through the prompt
    print(sys.stdin.read().strip())
except Exception as e:
    # Log error but don't break the user experience
    import logging
    logging.error(f"Task tree hook error: {{e}}")
    print(sys.stdin.read().strip())
'''
        
        # Write hook script
        hook_path.parent.mkdir(parents=True, exist_ok=True)
        hook_path.write_text(hook_script)
        
        # Make executable
        hook_path.chmod(0o755)
    
    def setup_claude_integration(self, claude_dir: Optional[Path] = None) -> None:
        """
        Set up Claude Code integration by creating necessary files.
        
        Args:
            claude_dir: Claude configuration directory (defaults to ~/.claude)
        """
        if claude_dir is None:
            claude_dir = Path.home() / ".claude"
        
        # Create hook script
        hook_path = claude_dir / "hooks" / "user-prompt-submit"
        self.create_hook_script(hook_path)
        
        # Create lib directory and copy task_tree module
        lib_dir = claude_dir / "lib"
        lib_dir.mkdir(parents=True, exist_ok=True)
        
        # Note: In a real implementation, you'd copy the task_tree module here
        # For now, we'll assume it's installed via pip or similar
        
        print(f"Task tree integration set up in {claude_dir}")
        print("To enable, add this to your Claude Code settings.json:")
        print(f'''
{{
  "hooks": {{
    "UserPromptSubmit": [
      {{
        "hooks": [
          {{
            "type": "command",
            "command": "{hook_path}"
          }}
        ]
      }}
    ]
  }}
}}
''')
    
    def get_context_stats(self) -> dict:
        """Get statistics about context injection."""
        current_task_id = self.get_current_task_id()
        stats = {
            "has_current_task": current_task_id is not None,
            "current_task_id": str(current_task_id) if current_task_id else None,
            "task_graph_stats": self.task_graph.get_task_stats()
        }
        
        if current_task_id:
            try:
                lineage = self.task_graph.get_lineage(current_task_id)
                stats["lineage_depth"] = len(lineage)
                stats["root_task"] = lineage[0].title if lineage else None
            except ValueError:
                stats["lineage_depth"] = 0
                stats["root_task"] = None
        
        return stats