#!/usr/bin/env python3
"""
Example usage of the task tree system.

This script demonstrates how to create and manage hierarchical tasks
with the AI on the Prize task tree system.
"""

from pathlib import Path
from src.task_tree import TaskGraph, TaskNode, TaskStatus, TaskPriority, ContextInjector


def main():
    """Demonstrate task tree functionality."""
    print("🎯 AI on the Prize - Task Tree Example")
    print("=" * 50)
    
    # Initialize task graph with example database
    db_path = Path("example_tasks.db")
    if db_path.exists():
        db_path.unlink()  # Start fresh
    
    graph = TaskGraph(db_path)
    injector = ContextInjector(graph, db_path)
    
    print("\n1. Creating a root task...")
    root_task = graph.create_task(
        title="Implement user authentication system",
        description="Build complete authentication system with JWT, password hashing, and session management",
        priority=TaskPriority.HIGH,
        tags={"backend", "security", "authentication"},
        completion_criteria="Users can register, login, logout, and maintain sessions securely"
    )
    print(f"✅ Created root task: {root_task.title}")
    
    print("\n2. Adding subtasks...")
    jwt_task = graph.create_task(
        title="Implement JWT token handling",
        description="Create JWT token generation, validation, and refresh logic",
        parent_id=root_task.id,
        priority=TaskPriority.HIGH,
        tags={"jwt", "tokens"},
        completion_criteria="JWT tokens can be generated, validated, and refreshed"
    )
    
    password_task = graph.create_task(
        title="Implement password hashing and validation",
        description="Secure password storage using bcrypt and validation logic",
        parent_id=root_task.id,
        priority=TaskPriority.HIGH,
        tags={"passwords", "security"},
        completion_criteria="Passwords are securely hashed and can be validated"
    )
    
    session_task = graph.create_task(
        title="Build session management",
        description="Handle user sessions, logout, and session cleanup",
        parent_id=root_task.id,
        priority=TaskPriority.MEDIUM,
        tags={"sessions", "cleanup"},
        completion_criteria="User sessions are properly managed and cleaned up"
    )
    
    print(f"✅ Created JWT task: {jwt_task.title}")
    print(f"✅ Created password task: {password_task.title}")
    print(f"✅ Created session task: {session_task.title}")
    
    print("\n3. Adding deeper subtasks to JWT handling...")
    token_gen_task = graph.create_task(
        title="Token generation logic",
        description="Generate JWT tokens with proper claims and expiration",
        parent_id=jwt_task.id,
        priority=TaskPriority.HIGH,
        completion_criteria="Tokens generated with user ID, expiration, and proper signing"
    )
    
    token_refresh_task = graph.create_task(
        title="Token refresh mechanism",
        description="Implement refresh token logic for seamless user experience",
        parent_id=jwt_task.id,
        priority=TaskPriority.MEDIUM,
        completion_criteria="Expired tokens can be refreshed without full re-authentication"
    )
    
    print(f"✅ Created token generation task: {token_gen_task.title}")
    print(f"✅ Created token refresh task: {token_refresh_task.title}")
    
    print("\n4. Demonstrating task tree structure...")
    print("\nTask Tree:")
    print(f"└─ {root_task.title}")
    for child in graph.get_children(root_task.id):
        print(f"   ├─ {child.title}")
        grandchildren = graph.get_children(child.id)
        for j, grandchild in enumerate(grandchildren):
            connector = "└─" if j == len(grandchildren) - 1 else "├─"
            print(f"   │  {connector} {grandchild.title}")
    
    print("\n5. Setting current task and showing context injection...")
    injector.set_current_task_id(token_refresh_task.id)
    
    print("\nCurrent task lineage:")
    lineage = graph.get_lineage(token_refresh_task.id)
    for i, task in enumerate(lineage):
        indent = "  " * i
        arrow = "└─ " if i > 0 else ""
        print(f"{indent}{arrow}{task.title} [{task.status}]")
    
    print("\n6. Context that would be injected into Claude Code prompts:")
    context = injector.get_context_for_injection()
    print("─" * 60)
    print(context)
    print("─" * 60)
    
    print("\n7. Simulating task completion...")
    # Mark token generation as in progress
    token_gen_task.mark_in_progress()
    graph.update_task(token_gen_task)
    print(f"🔄 Started: {token_gen_task.title}")
    
    # Complete token generation
    token_gen_task.mark_completed()
    graph.update_task(token_gen_task)
    print(f"✅ Completed: {token_gen_task.title}")
    
    # Work on token refresh
    injector.set_current_task_id(token_refresh_task.id)
    token_refresh_task.mark_in_progress()
    graph.update_task(token_refresh_task)
    print(f"🔄 Started: {token_refresh_task.title}")
    
    print("\n8. Updated context after progress:")
    context = injector.get_context_for_injection()
    print("─" * 60)
    print(context)
    print("─" * 60)
    
    print("\n9. Task statistics:")
    stats = graph.get_task_stats()
    for key, value in stats.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    
    print("\n10. Available tasks (what can be worked on next):")
    available = graph.get_available_tasks()
    for task in available:
        print(f"  • [{task.priority}] {task.title}")
    
    print("\n11. Demonstrating search functionality...")
    security_tasks = graph.search_tasks(query="security")
    print(f"Tasks related to 'security': {len(security_tasks)}")
    for task in security_tasks:
        print(f"  • {task.title}")
    
    high_priority_tasks = graph.search_tasks(priority=TaskPriority.HIGH)
    print(f"\nHigh priority tasks: {len(high_priority_tasks)}")
    for task in high_priority_tasks:
        print(f"  • {task.title} [{task.status}]")
    
    print("\n12. Context injection simulation...")
    example_prompts = [
        "Help me implement the token refresh logic",
        "What's the best way to handle JWT expiration?",
        "Add error handling for invalid tokens"
    ]
    
    for prompt in example_prompts:
        print(f"\nOriginal prompt: '{prompt}'")
        augmented = injector.inject_context(prompt)
        print("Augmented prompt would include task context above the user's request")
        print(f"User request appears after context: '{prompt}'")
    
    print(f"\n🎉 Task tree example completed!")
    print(f"Database saved to: {db_path.absolute()}")
    print("\nNext steps:")
    print("1. Install the package: pip install -e .")
    print("2. Use CLI: task-tree list")
    print("3. Set up Claude Code integration: task-tree setup")


if __name__ == "__main__":
    main()