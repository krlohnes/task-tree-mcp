#!/usr/bin/env python3
"""
Tool Audit Hook for Task Tree MCP Security

This hook records all tool calls made by Claude Code to provide
immutable audit trail for verification against agent claims.

CRITICAL SECURITY COMPONENT - DO NOT MODIFY
"""

import json
import hashlib
import time
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path

class ToolAuditHook:
    def __init__(self):
        # Create secure audit directory
        self.audit_dir = Path.home() / ".claude" / "task_tree_audit"
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        
        # Daily log rotation
        today = datetime.now().strftime("%Y%m%d")
        self.log_file = self.audit_dir / f"tool_audit_{today}.log"
        
        # Initialize log with header if new
        if not self.log_file.exists():
            self._init_log()
    
    def _init_log(self):
        """Initialize log file with security header."""
        header = {
            "type": "audit_log_init",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "security_notice": "This file contains cryptographic hashes for security verification"
        }
        self._write_entry(header)
    
    def _write_entry(self, entry):
        """Write entry to audit log with cryptographic integrity."""
        # Add timestamp and sequence
        entry["audit_timestamp"] = datetime.now().isoformat()
        entry["sequence"] = int(time.time() * 1000000)  # microsecond precision
        
        # Serialize entry
        entry_json = json.dumps(entry, sort_keys=True)
        
        # Add cryptographic hash for integrity
        entry_hash = hashlib.sha256(entry_json.encode()).hexdigest()
        entry["integrity_hash"] = entry_hash
        
        # Check for log rotation (daily rotation + size limit)
        self._rotate_log_if_needed()
        
        # Write to file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry) + "\\n")
    
    def _rotate_log_if_needed(self):
        """Rotate log if it becomes too large or it's a new day."""
        MAX_LOG_SIZE = 50 * 1024 * 1024  # 50MB limit
        
        # Check if new day (already handled in __init__)
        today = datetime.now().strftime("%Y%m%d")
        expected_file = self.audit_dir / f"tool_audit_{today}.log"
        
        if expected_file != self.log_file:
            self.log_file = expected_file
            if not self.log_file.exists():
                self._init_log()
        
        # Check size limit
        if self.log_file.exists() and self.log_file.stat().st_size > MAX_LOG_SIZE:
            # Archive current log with timestamp
            timestamp = datetime.now().strftime("%H%M%S")
            archive_name = f"tool_audit_{today}_{timestamp}.log"
            archive_path = self.audit_dir / "archive" / archive_name
            
            # Create archive directory
            archive_path.parent.mkdir(exist_ok=True)
            
            # Move current log to archive
            shutil.move(self.log_file, archive_path)
            
            # Start fresh log
            self._init_log()
    
    def clean_old_logs(self, days_to_keep=30):
        """Clean up old audit logs beyond retention period."""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        for log_file in self.audit_dir.glob("tool_audit_*.log"):
            try:
                # Extract date from filename
                date_str = log_file.stem.split("_")[-1][:8]  # YYYYMMDD
                log_date = datetime.strptime(date_str, "%Y%m%d")
                
                if log_date < cutoff_date:
                    # Archive old log before deletion
                    archive_dir = self.audit_dir / "archive"
                    archive_dir.mkdir(exist_ok=True)
                    shutil.move(log_file, archive_dir / log_file.name)
                    
            except (ValueError, IndexError):
                continue  # Skip files with unexpected names
    
    def record_tool_call(self, tool_name, args, result):
        """Record a tool call with full details."""
        entry = {
            "type": "tool_call",
            "tool_name": tool_name,
            "args": args,
            "result": result,
            "result_hash": hashlib.md5(str(result).encode()).hexdigest()
        }
        self._write_entry(entry)
        
        # Return hash for later verification
        return entry["integrity_hash"]

# Global hook instance
_audit_hook = ToolAuditHook()

def on_tool_call(tool_name, args, result):
    """Claude Code hook - called after every tool execution."""
    return _audit_hook.record_tool_call(tool_name, args, result)

def get_audit_log_path():
    """Get current audit log path for MCP server validation."""
    return str(_audit_hook.log_file)