"""Command-line interface for task tree management."""

import json
from pathlib import Path
from typing import List, Optional
from uuid import UUID

import typer
from rich.console import Console
from rich.table import Table
from rich.tree import Tree
from rich.panel import Panel
from rich.text import Text

from .task_graph import TaskGraph
from .task_node import TaskNode, TaskStatus, TaskPriority
from .context_injector import ContextInjector

app = typer.Typer(help="Task Tree - Hierarchical task management for AI agents")
console = Console()


def get_task_graph(db_path: Optional[str] = None) -> TaskGraph:
    """Get task graph instance with optional database path."""
    if db_path:
        return TaskGraph(Path(db_path))
    return TaskGraph()


def format_task_status(status: TaskStatus) -> Text:
    """Format task status with colors."""
    colors = {
        TaskStatus.PENDING: "yellow",
        TaskStatus.IN_PROGRESS: "blue",
        TaskStatus.COMPLETED: "green",
        TaskStatus.BLOCKED: "red",
        TaskStatus.CANCELLED: "dim"
    }
    return Text(status.value.upper(), style=colors.get(status, "white"))


def format_task_priority(priority: TaskPriority) -> Text:
    """Format task priority with colors."""
    colors = {
        TaskPriority.LOW: "dim",
        TaskPriority.MEDIUM: "white",
        TaskPriority.HIGH: "yellow",
        TaskPriority.CRITICAL: "red bold"
    }
    return Text(priority.value.upper(), style=colors.get(priority, "white"))


@app.command()
def create(
    title: str = typer.Argument(..., help="Task title"),
    description: Optional[str] = typer.Option(None, "--desc", "-d", help="Task description"),
    parent: Optional[str] = typer.Option(None, "--parent", "-p", help="Parent task ID"),
    priority: TaskPriority = typer.Option(TaskPriority.MEDIUM, "--priority", help="Task priority"),
    tags: Optional[str] = typer.Option(None, "--tags", "-t", help="Comma-separated tags"),
    criteria: Optional[str] = typer.Option(None, "--criteria", "-c", help="Completion criteria"),
    db_path: Optional[str] = typer.Option(None, "--db", help="Database path"),
    set_current: bool = typer.Option(False, "--current", help="Set as current task")
):
    """Create a new task."""
    graph = get_task_graph(db_path)
    
    parent_id = None
    if parent:
        try:
            parent_id = UUID(parent)
            if parent_id not in graph.nodes:
                console.print(f"[red]Error: Parent task {parent} not found[/red]")
                raise typer.Exit(1)
        except ValueError:
            console.print(f"[red]Error: Invalid parent task ID format[/red]")
            raise typer.Exit(1)
    
    tag_set = set()
    if tags:
        tag_set = {tag.strip() for tag in tags.split(",")}
    
    try:
        task = graph.create_task(
            title=title,
            description=description,
            parent_id=parent_id,
            priority=priority,
            tags=tag_set,
            completion_criteria=criteria
        )
        
        console.print(f"[green]Created task: {task.id}[/green]")
        console.print(f"Title: {task.title}")
        
        if set_current:
            injector = ContextInjector(graph)
            injector.set_current_task_id(task.id)
            console.print("[blue]Set as current task[/blue]")
        
    except Exception as e:
        console.print(f"[red]Error creating task: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def list_tasks(
    status: Optional[TaskStatus] = typer.Option(None, "--status", "-s", help="Filter by status"),
    priority: Optional[TaskPriority] = typer.Option(None, "--priority", "-p", help="Filter by priority"),
    query: Optional[str] = typer.Option(None, "--query", "-q", help="Search query"),
    db_path: Optional[str] = typer.Option(None, "--db", help="Database path"),
    show_details: bool = typer.Option(False, "--details", help="Show detailed information")
):
    """List tasks with optional filtering."""
    graph = get_task_graph(db_path)
    
    tasks = graph.search_tasks(
        query=query or "",
        status=status,
        priority=priority
    )
    
    if not tasks:
        console.print("[yellow]No tasks found[/yellow]")
        return
    
    if show_details:
        for task in tasks:
            panel_content = []
            panel_content.append(f"ID: {task.id}")
            if task.description:
                panel_content.append(f"Description: {task.description}")
            if task.tags:
                panel_content.append(f"Tags: {', '.join(sorted(task.tags))}")
            if task.completion_criteria:
                panel_content.append(f"Success Criteria: {task.completion_criteria}")
            panel_content.append(f"Created: {task.created_at.strftime('%Y-%m-%d %H:%M')}")
            
            console.print(Panel(
                "\n".join(panel_content),
                title=f"{task.title} [{task.status.value}]",
                border_style="blue" if task.status == TaskStatus.IN_PROGRESS else "white"
            ))
    else:
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("ID", style="dim", width=8)
        table.add_column("Title", style="bold")
        table.add_column("Status", justify="center")
        table.add_column("Priority", justify="center")
        table.add_column("Tags")
        
        for task in tasks:
            table.add_row(
                str(task.id),
                task.title,
                format_task_status(task.status),
                format_task_priority(task.priority),
                ", ".join(sorted(task.tags)) if task.tags else ""
            )
        
        console.print(table)


@app.command()
def show(
    task_id: str = typer.Argument(..., help="Task ID"),
    db_path: Optional[str] = typer.Option(None, "--db", help="Database path")
):
    """Show detailed information about a task."""
    graph = get_task_graph(db_path)
    
    try:
        uuid_id = UUID(task_id)
        task = graph.get_task(uuid_id)
        
        if not task:
            console.print(f"[red]Task {task_id} not found[/red]")
            raise typer.Exit(1)
        
        # Main task info
        info_lines = [
            f"ID: {task.id}",
            f"Title: {task.title}",
            f"Status: {task.status.value}",
            f"Priority: {task.priority.value}",
            f"Created: {task.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Updated: {task.updated_at.strftime('%Y-%m-%d %H:%M:%S')}"
        ]
        
        if task.description:
            info_lines.append(f"Description: {task.description}")
        
        if task.tags:
            info_lines.append(f"Tags: {', '.join(sorted(task.tags))}")
        
        if task.completion_criteria:
            info_lines.append(f"Success Criteria: {task.completion_criteria}")
        
        if task.completed_at:
            info_lines.append(f"Completed: {task.completed_at.strftime('%Y-%m-%d %H:%M:%S')}")
        
        console.print(Panel("\n".join(info_lines), title="Task Information"))
        
        # Lineage
        lineage = graph.get_lineage(uuid_id)
        if len(lineage) > 1:
            console.print("\n[bold]Task Lineage:[/bold]")
            tree = Tree("Root")
            current_tree = tree
            
            for i, ancestor in enumerate(lineage[:-1]):
                if i == 0:
                    current_tree.label = f"{ancestor.title} [{ancestor.status.value}]"
                else:
                    current_tree = current_tree.add(f"{ancestor.title} [{ancestor.status.value}]")
            
            current_tree.add(f"[bold]{task.title} [{task.status.value}][/bold] (current)")
            console.print(tree)
        
        # Children
        children = graph.get_children(uuid_id)
        if children:
            console.print(f"\n[bold]Children ({len(children)}):[/bold]")
            for child in children:
                status_color = "green" if child.status == TaskStatus.COMPLETED else "yellow"
                console.print(f"  • {child.title} [{child.status.value}]", style=status_color)
        
    except ValueError:
        console.print(f"[red]Invalid task ID format[/red]")
        raise typer.Exit(1)


@app.command()
def update(
    task_id: str = typer.Argument(..., help="Task ID"),
    status: Optional[TaskStatus] = typer.Option(None, "--status", help="Update status"),
    priority: Optional[TaskPriority] = typer.Option(None, "--priority", help="Update priority"),
    title: Optional[str] = typer.Option(None, "--title", help="Update title"),
    description: Optional[str] = typer.Option(None, "--desc", help="Update description"),
    db_path: Optional[str] = typer.Option(None, "--db", help="Database path")
):
    """Update a task."""
    graph = get_task_graph(db_path)
    
    try:
        uuid_id = UUID(task_id)
        task = graph.get_task(uuid_id)
        
        if not task:
            console.print(f"[red]Task {task_id} not found[/red]")
            raise typer.Exit(1)
        
        # Update fields
        if status:
            if status == TaskStatus.COMPLETED:
                task.mark_completed()
            elif status == TaskStatus.IN_PROGRESS:
                task.mark_in_progress()
            elif status == TaskStatus.BLOCKED:
                task.mark_blocked()
            else:
                task.status = status
        
        if priority:
            task.priority = priority
        
        if title:
            task.title = title
        
        if description:
            task.description = description
        
        graph.update_task(task)
        console.print(f"[green]Updated task {task_id}[/green]")
        
    except ValueError:
        console.print(f"[red]Invalid task ID format[/red]")
        raise typer.Exit(1)


@app.command()
def current(
    task_id: Optional[str] = typer.Argument(None, help="Task ID to set as current (or show current if not provided)"),
    db_path: Optional[str] = typer.Option(None, "--db", help="Database path"),
    clear: bool = typer.Option(False, "--clear", help="Clear current task")
):
    """Get or set the current task."""
    graph = get_task_graph(db_path)
    injector = ContextInjector(graph)
    
    if clear:
        injector.set_current_task_id(None)
        console.print("[yellow]Cleared current task[/yellow]")
        return
    
    if task_id:
        try:
            uuid_id = UUID(task_id)
            task = graph.get_task(uuid_id)
            
            if not task:
                console.print(f"[red]Task {task_id} not found[/red]")
                raise typer.Exit(1)
            
            injector.set_current_task_id(uuid_id)
            console.print(f"[green]Set current task: {task.title}[/green]")
            
        except ValueError:
            console.print(f"[red]Invalid task ID format[/red]")
            raise typer.Exit(1)
    else:
        current_id = injector.get_current_task_id()
        if current_id:
            task = graph.get_task(current_id)
            if task:
                console.print(f"Current task: {task.title} ({current_id})")
                
                # Show context
                context = injector.get_context_for_injection()
                if context:
                    console.print("\n[dim]Context that will be injected:[/dim]")
                    console.print(Panel(context, border_style="dim"))
            else:
                console.print("[yellow]Current task not found (clearing)[/yellow]")
                injector.set_current_task_id(None)
        else:
            console.print("[yellow]No current task set[/yellow]")


@app.command()
def stats(
    db_path: Optional[str] = typer.Option(None, "--db", help="Database path")
):
    """Show task statistics."""
    graph = get_task_graph(db_path)
    stats = graph.get_task_stats()
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="bold")
    table.add_column("Count", justify="right")
    
    for key, value in stats.items():
        table.add_row(key.replace("_", " ").title(), str(value))
    
    console.print(table)
    
    # Show available tasks
    available = graph.get_available_tasks()
    if available:
        console.print(f"\n[bold]Available Tasks ({len(available)}):[/bold]")
        for task in available[:5]:  # Show first 5
            console.print(f"  • [{task.priority.value}] {task.title}")
        if len(available) > 5:
            console.print(f"  ... and {len(available) - 5} more")
    
    # Show blocked tasks
    blocked = graph.get_blocked_tasks()
    if blocked:
        console.print(f"\n[bold red]Blocked Tasks ({len(blocked)}):[/bold red]")
        for task in blocked[:3]:  # Show first 3
            console.print(f"  • {task.title}")
        if len(blocked) > 3:
            console.print(f"  ... and {len(blocked) - 3} more")


@app.command()
def tree(
    root_id: Optional[str] = typer.Option(None, "--root", help="Root task ID (shows all roots if not provided)"),
    db_path: Optional[str] = typer.Option(None, "--db", help="Database path")
):
    """Show task tree visualization."""
    graph = get_task_graph(db_path)
    
    def build_tree(task: TaskNode, tree_node: Tree) -> None:
        """Recursively build tree visualization."""
        children = graph.get_children(task.id)
        for child in sorted(children, key=lambda t: t.created_at):
            status_style = {
                TaskStatus.COMPLETED: "green",
                TaskStatus.IN_PROGRESS: "blue",
                TaskStatus.BLOCKED: "red",
                TaskStatus.CANCELLED: "dim"
            }.get(child.status, "white")
            
            child_label = f"{child.title} [{child.status.value}]"
            child_tree = tree_node.add(Text(child_label, style=status_style))
            build_tree(child, child_tree)
    
    if root_id:
        try:
            uuid_id = UUID(root_id)
            root_task = graph.get_task(uuid_id)
            
            if not root_task:
                console.print(f"[red]Task {root_id} not found[/red]")
                raise typer.Exit(1)
            
            tree = Tree(f"{root_task.title} [{root_task.status.value}]")
            build_tree(root_task, tree)
            console.print(tree)
            
        except ValueError:
            console.print(f"[red]Invalid task ID format[/red]")
            raise typer.Exit(1)
    else:
        roots = graph.get_root_tasks()
        if not roots:
            console.print("[yellow]No tasks found[/yellow]")
            return
        
        for i, root in enumerate(roots):
            if i > 0:
                console.print()
            
            tree = Tree(f"{root.title} [{root.status.value}]")
            build_tree(root, tree)
            console.print(tree)


@app.command()
def setup(
    claude_dir: Optional[str] = typer.Option(None, "--claude-dir", help="Claude configuration directory")
):
    """Set up Claude Code integration."""
    injector = ContextInjector()
    claude_path = Path(claude_dir) if claude_dir else None
    
    try:
        injector.setup_claude_integration(claude_path)
        console.print("[green]Task tree integration set up successfully![/green]")
    except Exception as e:
        console.print(f"[red]Error setting up integration: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()