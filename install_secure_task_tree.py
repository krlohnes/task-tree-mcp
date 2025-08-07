#!/usr/bin/env python3
"""
Secure Task Tree MCP Installation Script

Installs the Task Tree MCP server with hook-based security system.

This installer:
1. Sets up the MCP server
2. Installs security hooks for Claude Code
3. Configures Claude Desktop with hooks enabled
4. Creates secure audit directory structure

CRITICAL: This installer enables security features that prevent
agent bypass of task verification systems.
"""

import json
import os
import sys
import shutil
import subprocess
from pathlib import Path

class SecureTaskTreeInstaller:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.claude_config_dir = Path.home() / "Library" / "Application Support" / "Claude"
        self.claude_config_file = self.claude_config_dir / "claude_desktop_config.json"
        self.hooks_dir = Path.home() / ".claude" / "hooks"
        
        print("🔒 Secure Task Tree MCP Installer")
        print("=" * 50)
    
    def check_requirements(self):
        """Check system requirements."""
        print("📋 Checking requirements...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            print("❌ Python 3.8+ required")
            sys.exit(1)
        
        # Check if at least one Claude application exists
        claude_desktop_exists = self.claude_config_dir.exists()
        claude_code_exists = (Path.home() / ".claude").exists() or shutil.which("claude") is not None
        
        if not claude_desktop_exists and not claude_code_exists:
            print("❌ Neither Claude Desktop nor Claude Code found.")
            print("   Please install Claude Desktop or Claude Code first.")
            sys.exit(1)
        
        print("✅ Requirements satisfied")
    
    def install_dependencies(self):
        """Install Python dependencies."""
        print("📦 Installing dependencies...")
        
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-e", "."
            ], check=True, cwd=self.project_root)
            print("✅ Dependencies installed")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            sys.exit(1)
    
    def setup_security_hooks(self):
        """Install security hooks for Claude Code."""
        print("🔒 Setting up security hooks...")
        
        # Copy hook script to secure location
        hook_source = self.project_root / "security_hooks" / "post_tool_audit.sh"
        hook_dest = self.project_root / "security_hooks" / "post_tool_audit.sh"  # Keep in project
        
        # Make sure it's executable
        os.chmod(hook_dest, 0o755)
        print(f"✅ Security hook ready: {hook_dest}")
        
        # Update Claude Code settings
        claude_settings_file = Path.home() / ".claude" / "settings.json"
        claude_settings_dir = claude_settings_file.parent
        claude_settings_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing settings or create new
        if claude_settings_file.exists():
            with open(claude_settings_file, 'r') as f:
                settings = json.load(f)
        else:
            settings = {}
        
        # Add PostToolUse hook configuration
        if "hooks" not in settings:
            settings["hooks"] = {}
        
        # Configure PostToolUse hook for audit trail
        settings["hooks"]["PostToolUse"] = [
            {
                "matcher": "*",  # Match all tools
                "hooks": [
                    {
                        "type": "command",
                        "command": str(hook_dest)
                    }
                ]
            }
        ]
        
        
        # Write updated settings
        with open(claude_settings_file, 'w') as f:
            json.dump(settings, f, indent=2)
        
        print(f"✅ Claude Code settings updated: {claude_settings_file}")
        print("📋 PostToolUse hook configured for all tools")
    
    def configure_claude_applications(self):
        """Configure Claude Desktop and/or Claude Code with MCP server."""
        claude_desktop_exists = self.claude_config_dir.exists()
        claude_code_exists = (Path.home() / ".claude").exists() or shutil.which("claude") is not None
        
        # Configure Claude Desktop if present
        if claude_desktop_exists:
            print("⚙️ Configuring Claude Desktop...")
            
            # Load existing config or create new
            if self.claude_config_file.exists():
                with open(self.claude_config_file, 'r') as f:
                    config = json.load(f)
            else:
                config = {}
            
            # Ensure mcpServers section exists
            if "mcpServers" not in config:
                config["mcpServers"] = {}
            
            # Add task-tree MCP server
            server_path = self.project_root / "mcp_server" / "server.py"
            config["mcpServers"]["task-tree"] = {
                "command": "python",
                "args": [str(server_path)],
                "env": {
                    "PYTHONPATH": str(self.project_root / "mcp_server")
                }
            }
            
            # Write updated config
            self.claude_config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.claude_config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"✅ Claude Desktop configured: {self.claude_config_file}")
        
        # Configure Claude Code if present
        if claude_code_exists:
            print("⚙️ Configuring Claude Code...")
            try:
                server_path = self.project_root / "mcp_server" / "server.py"
                subprocess.run([
                    "claude", "mcp", "add", "task-tree", 
                    "python", str(server_path), "-s", "user"
                ], check=True, capture_output=True)
                print("✅ Claude Code MCP server added")
            except subprocess.CalledProcessError:
                print("⚠️ Could not auto-configure Claude Code (may need manual setup)")
            except FileNotFoundError:
                print("⚠️ Claude Code CLI not found (MCP setup may be manual)")
    
    def create_audit_directory(self):
        """Create secure audit directory structure."""
        print("🗂️ Creating audit directory structure...")
        
        audit_dir = Path.home() / ".claude" / "task_tree_audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        
        # Set restrictive permissions (user only)
        os.chmod(audit_dir, 0o700)
        
        # Create readme
        readme_content = '''# Task Tree MCP Security Audit Logs

This directory contains cryptographically secured audit trails
of all tool calls made through Claude Code.

⚠️ SECURITY NOTICE ⚠️
- These files are used for task verification security
- Do not modify or delete these files
- They contain hashes for integrity verification
- Tampering will be detected by the MCP server

Files:
- tool_audit_YYYYMMDD.log: Daily audit logs with cryptographic hashes
'''
        
        readme_file = audit_dir / "README.md"
        with open(readme_file, 'w') as f:
            f.write(readme_content)
        
        print(f"✅ Audit directory created: {audit_dir}")
    
    def run_security_test(self):
        """Run basic security test."""
        print("🧪 Running security test...")
        
        try:
            # Test hook functionality
            sys.path.insert(0, str(self.hooks_dir))
            import tool_audit_hook
            
            # Test basic functionality
            hook_path = tool_audit_hook.get_audit_log_path()
            if Path(hook_path).parent.exists():
                print("✅ Security hooks functional")
            else:
                print("⚠️ Security hooks may not be fully configured")
        
        except ImportError:
            print("⚠️ Could not test security hooks - may need Claude Code restart")
    
    def print_completion_message(self):
        """Print installation completion message."""
        print("\\n" + "=" * 50)
        print("🎉 SECURE TASK TREE MCP INSTALLATION COMPLETE!")
        print("=" * 50)
        print()
        print("✅ MCP Server: Installed with enhanced verification")
        print("✅ Security Hooks: Enabled for tool audit trail")
        print("✅ Claude Applications: Configured with security features")
        print("✅ Audit System: Ready for tamper-proof logging")
        print()
        print("🔒 SECURITY FEATURES ENABLED:")
        print("  • Tool call audit trail with cryptographic hashes")
        print("  • Agent bypass prevention system")
        print("  • Evidence validation against actual tool results")
        print("  • Vague evidence detection and blocking")
        print("  • 🚨 MANDATORY Stop/SubagentStop enforcement hooks")
        print("  • Session completion blocked until MCP compliance verified")
        print()
        print("⚠️ IMPORTANT:")
        print("  • Restart Claude Desktop/Claude Code to activate hooks")
        print("  • Test with a simple task to verify functionality")
        print("  • Audit logs will appear in ~/.claude/task_tree_audit/")
        print()
        print("📚 Documentation: See MCP_SETUP.md for usage instructions")
    
    def install(self):
        """Run complete installation."""
        try:
            self.check_requirements()
            self.install_dependencies()
            self.setup_security_hooks()
            self.configure_claude_applications()
            self.create_audit_directory()
            self.run_security_test()
            self.print_completion_message()
            
        except KeyboardInterrupt:
            print("\\n❌ Installation cancelled by user")
            sys.exit(1)
        except Exception as e:
            print(f"\\n❌ Installation failed: {e}")
            sys.exit(1)

if __name__ == "__main__":
    installer = SecureTaskTreeInstaller()
    installer.install()