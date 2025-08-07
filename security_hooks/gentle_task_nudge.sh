#!/bin/bash
# Contextual Task Management Nudge Hook
# Provides reminders only when TodoWrite is used in isolation (without adjacent MCP task-tree tools)

AUDIT_DIR="$HOME/.claude/task_tree_audit"
DEBUG_LOG="$AUDIT_DIR/gentle_nudge.log"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Only process TodoWrite tool calls
if [[ "$TOOL_NAME" != "TodoWrite" ]]; then
    exit 0  # Silent exit for non-TodoWrite tools
fi

echo "[$TIMESTAMP] TodoWrite detected, checking context" >> "$DEBUG_LOG"

# Get recent audit log
RECENT_AUDIT=$(find "$AUDIT_DIR" -name "tool_audit_*.log" -mtime -1 2>/dev/null | head -1)
if [[ -z "$RECENT_AUDIT" ]]; then
    exit 0  # No audit log available
fi

# Get last 5 tool calls to check context around this TodoWrite
LAST_TOOLS=$(grep '"tool_name":' "$RECENT_AUDIT" | tail -5 | sed 's/.*"tool_name": *"\([^"]*\)".*/\1/')

echo "[$TIMESTAMP] Last 5 tools: $(echo "$LAST_TOOLS" | tr '\n' ' ')" >> "$DEBUG_LOG"

# Check if TodoWrite is paired with MCP task-tree tools
MCP_ADJACENT=false
PREV_TOOL=""
while IFS= read -r tool; do
    if [[ "$tool" == "TodoWrite" ]]; then
        # Check if previous or next tool is MCP task-tree
        if [[ "$PREV_TOOL" == mcp__task-tree__* ]]; then
            MCP_ADJACENT=true
            break
        fi
    elif [[ "$PREV_TOOL" == "TodoWrite" && "$tool" == mcp__task-tree__* ]]; then
        MCP_ADJACENT=true
        break
    fi
    PREV_TOOL="$tool"
done <<< "$LAST_TOOLS"

if [[ "$MCP_ADJACENT" == "true" ]]; then
    echo "[$TIMESTAMP] TodoWrite has adjacent MCP tool, no nudge needed" >> "$DEBUG_LOG"
    exit 0
fi

# Count isolated TodoWrite usage for throttling
TODOWRITE_COUNT=$(grep -c "TodoWrite" "$RECENT_AUDIT" 2>/dev/null || echo "0")
ISOLATED_NUDGES=$(grep -c "Isolated TodoWrite detected" "$DEBUG_LOG" 2>/dev/null || echo "0")

# Show nudge every 10 isolated TodoWrite uses to avoid spam
if [[ $((TODOWRITE_COUNT % 10)) -eq 0 ]] && [[ $TODOWRITE_COUNT -gt $((ISOLATED_NUDGES * 10)) ]]; then
    echo "[$TIMESTAMP] Isolated TodoWrite detected, showing nudge" >> "$DEBUG_LOG"
    
    {
        echo
        echo "💡 Task Management Tip: Consider pairing TodoWrite with MCP task-tree tools!"
        echo "   Try: mcp__task-tree__create_task → TodoWrite → mcp__task-tree__update_task_status"
        echo "   Benefits: Better organization, verification, and progress tracking"
        echo
        echo "   (Reminder: Use MCP tools before or after TodoWrite for enhanced workflow)"
        echo
    } >&2
fi

exit 0  # Always allow the operation to continue