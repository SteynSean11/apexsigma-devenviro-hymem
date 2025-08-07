#!/usr/bin/env python3
"""
Simple project cleanup script - organize log files and clean temp files
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
import json


def create_log_structure():
    """Create organized log directory structure"""
    print("Creating organized log structure...")
    
    log_dirs = [
        "logs/github",      # GitHub Actions logs
        "logs/system",      # System status reports  
        "logs/archived",    # Old logs
        "temp",             # Temporary files
        "backup"            # Backup files
    ]
    
    for dir_path in log_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"  Created: {dir_path}")


def organize_github_logs():
    """Organize GitHub Actions log directories"""
    print("\nOrganizing GitHub Actions logs...")
    
    # Find test directories (GitHub Actions logs)
    test_dirs = list(Path(".").glob("*test (3.*)"))
    test_dirs.extend(list(Path(".").glob("test (3.*)")))
    
    moved_count = 0
    for test_dir in test_dirs:
        if test_dir.is_dir():
            target = Path("logs/github") / test_dir.name
            if not target.exists():
                shutil.move(str(test_dir), str(target))
                print(f"  Moved: {test_dir.name} -> logs/github/")
                moved_count += 1
    
    print(f"  Moved {moved_count} GitHub log directories")


def organize_log_files():
    """Organize individual log files"""
    print("\nOrganizing log files...")
    
    # Find log files
    log_patterns = ["*.txt", "*.log", "logs_*.zip", "system_status_*.json"]
    
    moved_count = 0
    for pattern in log_patterns:
        for log_file in Path(".").glob(pattern):
            if log_file.is_file():
                # Determine target directory
                if "system_status" in log_file.name:
                    target_dir = Path("logs/system")
                elif any(x in log_file.name.lower() for x in ["test", "github", "ci"]):
                    target_dir = Path("logs/github")
                else:
                    target_dir = Path("logs")
                
                target_file = target_dir / log_file.name
                if not target_file.exists():
                    shutil.move(str(log_file), str(target_file))
                    print(f"  Moved: {log_file.name} -> {target_dir.name}/")
                    moved_count += 1
    
    print(f"  Moved {moved_count} log files")


def clean_temp_files():
    """Clean temporary files and cache directories"""
    print("\nCleaning temporary files...")
    
    temp_patterns = [
        "__pycache__",
        "*.pyc",
        ".pytest_cache", 
        ".mypy_cache",
        "*.tmp"
    ]
    
    cleaned_count = 0
    space_freed = 0
    
    for pattern in temp_patterns:
        for temp_item in Path(".").rglob(pattern):
            if temp_item.exists():
                # Calculate size before deletion
                if temp_item.is_file():
                    space_freed += temp_item.stat().st_size
                    temp_item.unlink()
                    cleaned_count += 1
                elif temp_item.is_dir():
                    # Calculate directory size
                    for file in temp_item.rglob("*"):
                        if file.is_file():
                            space_freed += file.stat().st_size
                    shutil.rmtree(temp_item)
                    cleaned_count += 1
                
                print(f"  Cleaned: {temp_item.name}")
    
    print(f"  Cleaned {cleaned_count} temporary items")
    print(f"  Space freed: {space_freed:,} bytes ({space_freed/1024:.1f} KB)")


def organize_existing_logs():
    """Organize files already in logs directory"""
    print("\nOrganizing existing logs directory...")
    
    logs_dir = Path("logs")
    if not logs_dir.exists():
        return
    
    moved_count = 0
    for item in logs_dir.iterdir():
        if item.is_file():
            if "system_status" in item.name:
                target = Path("logs/system") / item.name
                if not target.exists():
                    shutil.move(str(item), str(target))
                    print(f"  Moved: {item.name} -> system/")
                    moved_count += 1
    
    print(f"  Organized {moved_count} existing log files")


def create_project_index():
    """Create an index of the organized project structure"""
    print("\nCreating project index...")
    
    index = {
        "cleanup_date": datetime.now().isoformat(),
        "structure": {},
        "logs": {}
    }
    
    # Index main project structure
    for item in Path(".").iterdir():
        if item.name.startswith("."):
            continue
        if item.name in ["venv", "__pycache__"]:
            continue
            
        if item.is_dir():
            file_count = len(list(item.rglob("*")))
            index["structure"][item.name] = {
                "type": "directory",
                "files": file_count
            }
        else:
            index["structure"][item.name] = {
                "type": "file",
                "size": item.stat().st_size
            }
    
    # Index logs organization
    logs_dir = Path("logs")
    if logs_dir.exists():
        for subdir in logs_dir.iterdir():
            if subdir.is_dir():
                files = list(subdir.glob("*"))
                index["logs"][subdir.name] = {
                    "file_count": len(files),
                    "files": [f.name for f in files[:5]]  # First 5 files
                }
    
    # Save index
    index_file = Path("logs/project_index.json")
    with open(index_file, 'w') as f:
        json.dump(index, f, indent=2)
    
    print(f"  Created: logs/project_index.json")
    return index


def display_summary(index):
    """Display cleanup summary"""
    print("\n" + "="*60)
    print("PROJECT CLEANUP COMPLETE!")
    print("="*60)
    
    print("\nORGANIZED STRUCTURE:")
    for name, info in index["structure"].items():
        if info["type"] == "directory":
            print(f"  {name}/ ({info['files']} files)")
        else:
            print(f"  {name} ({info['size']} bytes)")
    
    print("\nLOG ORGANIZATION:")
    for name, info in index["logs"].items():
        print(f"  logs/{name}/ ({info['file_count']} files)")
        for file in info["files"]:
            print(f"    - {file}")
    
    print("\nPROJECT IS NOW CLEAN AND ORGANIZED!")
    print("All logs are categorized and temporary files removed.")


def main():
    """Main cleanup function"""
    print("APEXSIGMA DEVENVIRO PROJECT CLEANUP")
    print("="*60)
    
    try:
        create_log_structure()
        organize_github_logs()
        organize_log_files() 
        organize_existing_logs()
        clean_temp_files()
        index = create_project_index()
        display_summary(index)
        
        print("\nSUCCESS: Project cleanup completed successfully!")
        return True
        
    except Exception as e:
        print(f"\nERROR: Cleanup failed: {e}")
        return False


if __name__ == "__main__":
    main()