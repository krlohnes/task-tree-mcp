#!/usr/bin/env python3
"""
Hook-based Tool Result Validator

Validates agent-submitted tool results against the immutable
audit trail created by Claude Code hooks.

CRITICAL SECURITY COMPONENT
"""

import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional

class HookValidator:
    def __init__(self):
        self.audit_dir = Path.home() / ".claude" / "task_tree_audit"
        self._audit_cache = {}  # Cache recent entries for performance
    
    def _get_recent_audit_entries(self, hours=24):
        """Get recent audit entries from log files."""
        entries = []
        
        # Check today and yesterday's logs
        for days_back in range(2):
            date = (datetime.now() - timedelta(days=days_back))
            log_file = self.audit_dir / f"tool_audit_{date.strftime('%Y%m%d')}.log"
            
            if log_file.exists():
                with open(log_file, 'r') as f:
                    for line in f:
                        try:
                            entry = json.loads(line.strip())
                            if entry.get("type") == "tool_call":
                                entries.append(entry)
                        except json.JSONDecodeError:
                            continue
        
        # Sort by timestamp (most recent first)
        entries.sort(key=lambda x: x.get("sequence", 0), reverse=True)
        return entries
    
    def validate_tool_result(self, tool_name: str, tool_args: Dict[str, Any], 
                           claimed_result: str, tolerance_minutes: int = 5) -> Dict[str, Any]:
        """
        Validate claimed tool result against audit trail.
        
        Returns validation result with details.
        """
        # Get recent audit entries
        recent_entries = self._get_recent_audit_entries()
        
        # Find matching tool calls within time tolerance
        now = datetime.now()
        tolerance = timedelta(minutes=tolerance_minutes)
        
        matching_calls = []
        
        for entry in recent_entries:
            # Check if entry is within time tolerance
            entry_time = datetime.fromisoformat(entry.get("audit_timestamp", ""))
            if now - entry_time > tolerance:
                continue  # Too old
            
            # Check tool name match
            if entry.get("tool_name") != tool_name:
                continue
            
            # Check args match (basic comparison)
            entry_args = entry.get("args", {})
            if self._args_match(entry_args, tool_args):
                matching_calls.append(entry)
        
        if not matching_calls:
            return {
                "valid": False,
                "reason": f"No matching {tool_name} call found in audit trail within {tolerance_minutes} minutes",
                "claimed_result": claimed_result,
                "actual_result": None
            }
        
        # Use most recent matching call
        actual_entry = matching_calls[0]
        actual_result = actual_entry.get("result", "")
        
        # Compare results
        if self._results_match(actual_result, claimed_result):
            return {
                "valid": True,
                "reason": "Tool result matches audit trail",
                "claimed_result": claimed_result,
                "actual_result": actual_result,
                "audit_hash": actual_entry.get("integrity_hash")
            }
        else:
            return {
                "valid": False,
                "reason": "Tool result does NOT match audit trail",
                "claimed_result": claimed_result,
                "actual_result": actual_result,
                "audit_hash": actual_entry.get("integrity_hash"),
                "discrepancy": self._analyze_discrepancy(actual_result, claimed_result)
            }
    
    def _args_match(self, audit_args: Dict[str, Any], claimed_args: Dict[str, Any]) -> bool:
        """Check if tool arguments match (with some tolerance)."""
        # Basic key comparison
        audit_keys = set(audit_args.keys())
        claimed_keys = set(claimed_args.keys())
        
        # Must have same keys
        if audit_keys != claimed_keys:
            return False
        
        # Check values match
        for key in audit_keys:
            if str(audit_args[key]).strip() != str(claimed_args[key]).strip():
                return False
        
        return True
    
    def _results_match(self, actual_result: str, claimed_result: str) -> bool:
        """Check if actual and claimed results match."""
        # Normalize whitespace and compare
        actual_normalized = " ".join(str(actual_result).split())
        claimed_normalized = " ".join(str(claimed_result).split())
        
        return actual_normalized == claimed_normalized
    
    def _analyze_discrepancy(self, actual: str, claimed: str) -> Dict[str, Any]:
        """Analyze discrepancy between actual and claimed results."""
        actual_len = len(str(actual))
        claimed_len = len(str(claimed))
        
        return {
            "actual_length": actual_len,
            "claimed_length": claimed_len,
            "length_diff": claimed_len - actual_len,
            "actual_preview": str(actual)[:100],
            "claimed_preview": str(claimed)[:100]
        }

# Global validator instance
_hook_validator = HookValidator()

def validate_agent_tool_claim(tool_name: str, tool_args: Dict[str, Any], 
                            claimed_result: str) -> Dict[str, Any]:
    """Validate an agent's tool result claim against hook audit trail."""
    return _hook_validator.validate_tool_result(tool_name, tool_args, claimed_result)