#!/usr/bin/env python3
"""
Database Cleanup Script for Task Tree MCP
Safely clears all task data from SQLite databases
"""

import sqlite3
import os
import sys
from pathlib import Path
import argparse


def clear_database(db_path: Path):
    """Clear all task data from a SQLite database."""
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        return False
    
    try:
        # Connect to the database
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Get table info first to see what we're working with
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        if not tables:
            print(f"ℹ️ Database {db_path.name} is already empty")
            conn.close()
            return True
        
        print(f"🗂️ Found tables in {db_path.name}: {[t[0] for t in tables]}")
        
        # Count records before clearing
        total_records = 0
        for table_name in [t[0] for t in tables]:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            if count > 0:
                print(f"  - {table_name}: {count} records")
                total_records += count
        
        if total_records == 0:
            print(f"ℹ️ Database {db_path.name} has no records to clear")
            conn.close()
            return True
        
        # Clear all tables
        cleared_tables = 0
        for table_name in [t[0] for t in tables]:
            try:
                cursor.execute(f"DELETE FROM {table_name}")
                cleared_tables += 1
            except sqlite3.Error as e:
                print(f"⚠️ Warning: Could not clear table {table_name}: {e}")
        
        # Reset auto-increment counters (only if the table exists)
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sqlite_sequence'")
        if cursor.fetchone():
            cursor.execute("DELETE FROM sqlite_sequence")
        
        # Commit changes
        conn.commit()
        conn.close()
        
        print(f"✅ Cleared {cleared_tables} tables from {db_path.name} ({total_records} total records)")
        return True
        
    except sqlite3.Error as e:
        print(f"❌ Database error with {db_path.name}: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error with {db_path.name}: {e}")
        return False


def find_task_databases(base_dir: Path):
    """Find all task database files in the project."""
    databases = []
    
    # Common database file patterns
    patterns = ['tasks.db', 'test_tasks.db', 'example_tasks.db', '*.db']
    
    for pattern in patterns:
        if '*' in pattern:
            databases.extend(base_dir.glob(pattern))
            databases.extend(base_dir.glob(f"**/{pattern}"))
        else:
            # Direct file lookup
            for db_file in base_dir.rglob(pattern):
                databases.append(db_file)
    
    # Remove duplicates and filter out non-task databases
    unique_dbs = []
    for db in set(databases):
        if db.is_file() and db.suffix == '.db':
            unique_dbs.append(db)
    
    return sorted(unique_dbs)


def main():
    """Main function to handle command line arguments and execute cleanup."""
    parser = argparse.ArgumentParser(
        description="Clear task data from SQLite databases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python clear_database.py                    # Find and clear all task databases
  python clear_database.py --file tasks.db   # Clear specific database
  python clear_database.py --dry-run         # Show what would be cleared
        """
    )
    
    parser.add_argument(
        '--file', '-f',
        type=str,
        help='Specific database file to clear'
    )
    
    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Show what would be cleared without actually clearing'
    )
    
    parser.add_argument(
        '--yes', '-y',
        action='store_true',
        help='Skip confirmation prompt'
    )
    
    args = parser.parse_args()
    
    # Get the project directory
    project_dir = Path(__file__).parent
    
    # Find databases to clear
    if args.file:
        # Specific file
        db_path = Path(args.file)
        if not db_path.is_absolute():
            db_path = project_dir / db_path
        databases = [db_path]
    else:
        # Auto-discover databases
        databases = find_task_databases(project_dir)
    
    if not databases:
        print("ℹ️ No task databases found")
        return 0
    
    print("🔍 Task databases found:")
    for i, db in enumerate(databases, 1):
        rel_path = db.relative_to(project_dir) if db.is_relative_to(project_dir) else db
        size = db.stat().st_size if db.exists() else 0
        print(f"  {i}. {rel_path} ({size:,} bytes)")
    
    if args.dry_run:
        print("\n🚫 Dry run mode - no changes will be made")
        return 0
    
    # Confirmation prompt
    if not args.yes:
        print(f"\n⚠️ This will permanently delete all task data from {len(databases)} database(s)")
        response = input("Continue? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("❌ Operation cancelled")
            return 1
    
    # Clear databases
    print("\n🧹 Clearing databases...")
    success_count = 0
    for db in databases:
        if clear_database(db):
            success_count += 1
    
    print(f"\n🎉 Completed: {success_count}/{len(databases)} databases cleared successfully")
    return 0 if success_count == len(databases) else 1


if __name__ == "__main__":
    sys.exit(main())