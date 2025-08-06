#!/bin/bash
# Claude Code PostToolUse Hook for Task Tree MCP Security
# Records all tool executions for verification against agent claims

AUDIT_DIR="$HOME/.claude/task_tree_audit"
LOG_FILE="$AUDIT_DIR/tool_audit_$(date +%Y%m%d).log"

# Create audit directory if it doesn't exist
mkdir -p "$AUDIT_DIR"

# Read tool execution data from stdin (JSON format)
TOOL_DATA=$(cat)

# DEBUG: Log what we actually received
echo "$(date -Iseconds) DEBUG RECEIVED: $TOOL_DATA" >> "$AUDIT_DIR/debug.log"

# Extract tool information from Claude Code hook format
if command -v jq >/dev/null 2>&1 && echo "$TOOL_DATA" | jq . >/dev/null 2>&1; then
    TOOL_NAME=$(echo "$TOOL_DATA" | jq -r '.tool_name // "unknown"')
    TOOL_ARGS=$(echo "$TOOL_DATA" | jq -c '.tool_input // {}')
    
    # Extract result from tool_response based on tool type
    if [ "$TOOL_NAME" = "Bash" ]; then
        TOOL_RESULT=$(echo "$TOOL_DATA" | jq -r '.tool_response.stdout // ""')
    elif [ "$TOOL_NAME" = "Read" ]; then
        TOOL_RESULT=$(echo "$TOOL_DATA" | jq -r '.tool_response.file.content // .tool_response.content // ""')
    elif [ "$TOOL_NAME" = "Edit" ] || [ "$TOOL_NAME" = "Write" ]; then
        TOOL_RESULT=$(echo "$TOOL_DATA" | jq -r '.tool_response.newString // .tool_response.content // ""')
    else
        # Generic result extraction
        TOOL_RESULT=$(echo "$TOOL_DATA" | jq -r '.tool_response // ""')
    fi
else
    # Fallback parsing if jq fails
    TOOL_NAME="parse_failed"
    TOOL_ARGS="{}"
    TOOL_RESULT="$TOOL_DATA"
fi
TIMESTAMP=$(date -Iseconds)
SEQUENCE=$(date +%s%6N)

# Create audit entry with cryptographic integrity
AUDIT_ENTRY=$(jq -n \
  --arg type "tool_call" \
  --arg tool "$TOOL_NAME" \
  --argjson args "$TOOL_ARGS" \
  --arg result "$TOOL_RESULT" \
  --arg timestamp "$TIMESTAMP" \
  --arg sequence "$SEQUENCE" \
  '{
    type: $type,
    tool_name: $tool,
    args: $args,
    result: $result,
    audit_timestamp: $timestamp,
    sequence: $sequence
  }')

# Add integrity hash
ENTRY_HASH=$(echo "$AUDIT_ENTRY" | shasum -a 256 | cut -d' ' -f1)
FINAL_ENTRY=$(echo "$AUDIT_ENTRY" | jq --arg hash "$ENTRY_HASH" '. + {integrity_hash: $hash}')

# Append to audit log
echo "$FINAL_ENTRY" >> "$LOG_FILE"

# Log rotation: limit to 50MB per file
if [ -f "$LOG_FILE" ] && [ $(stat -f%z "$LOG_FILE" 2>/dev/null || stat -c%s "$LOG_FILE" 2>/dev/null) -gt 52428800 ]; then
    ARCHIVE_DIR="$AUDIT_DIR/archive"
    mkdir -p "$ARCHIVE_DIR"
    mv "$LOG_FILE" "$ARCHIVE_DIR/tool_audit_$(date +%Y%m%d_%H%M%S).log"
fi

# Return success - don't interfere with tool execution
exit 0