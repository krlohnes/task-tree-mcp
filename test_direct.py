#!/usr/bin/env python3
"""
Direct test of the task tree system without MCP.
"""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "mcp_server"))

from task_tree import TaskGraph, TaskNode, TaskStatus, TaskPriority, ContextInjector

def test_task_system():
    """Test the task system directly."""
    print("🧪 Testing Task Tree System Directly")
    print("=" * 50)
    
    # Initialize
    db_path = Path("test_tasks.db")
    if db_path.exists():
        db_path.unlink()
    
    graph = TaskGraph(db_path)
    injector = ContextInjector(graph, db_path)
    
    # Test 1: Create task
    print("\n1. Creating test task...")
    task = graph.create_task(
        title="Test Claude Code MCP Integration",
        description="Verify that the task tree system works properly",
        priority=TaskPriority.HIGH,
        completion_criteria="Task system responds correctly"
    )
    print(f"✅ Created task: {task.title}")
    print(f"   ID: {str(task.id)[:8]}...")
    print(f"   Priority: {task.priority}")
    
    # Test 2: Set as current
    print("\n2. Setting as current task...")
    injector.set_current_task_id(task.id)
    print("✅ Set as current task")
    
    # Test 3: Get context
    print("\n3. Getting context...")
    context = injector.get_context_for_injection()
    print("✅ Context retrieved:")
    print(context)
    
    # Test 4: Create subtask
    print("\n4. Creating subtask...")
    subtask = graph.create_task(
        title="Test MCP tools functionality",
        description="Verify MCP tools work correctly",
        parent_id=task.id,
        priority=TaskPriority.MEDIUM
    )
    print(f"✅ Created subtask: {subtask.title}")
    
    # Test 5: Update context
    print("\n5. Setting subtask as current...")
    injector.set_current_task_id(subtask.id)
    context = injector.get_context_for_injection()
    print("✅ Updated context:")
    print(context)
    
    # Test 6: Complete subtask
    print("\n6. Completing subtask...")
    subtask.mark_completed()
    graph.update_task(subtask)
    print(f"✅ Completed: {subtask.title}")
    
    # Test 7: Statistics
    print("\n7. Task statistics...")
    stats = graph.get_task_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print(f"\n🎉 All tests passed!")
    print(f"Database saved to: {db_path.absolute()}")
    
    return task, subtask

if __name__ == "__main__":
    test_task_system()