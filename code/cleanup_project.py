#!/usr/bin/env python3
"""
Project cleanup and organization script
Tidies up log files, temporary artifacts, and organizes project structure
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
import json


class ProjectCleanup:
    """Comprehensive project cleanup and organization"""
    
    def __init__(self, project_root=None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.cleanup_report = {
            "timestamp": datetime.now().isoformat(),
            "actions": [],
            "files_moved": 0,
            "files_deleted": 0,
            "directories_created": 0,
            "space_freed": 0
        }
    
    def log_action(self, action, details=""):
        """Log cleanup action"""
        self.cleanup_report["actions"].append({
            "action": action,
            "details": details,
            "timestamp": datetime.now().isoformat()
        })
        print(f"✓ {action}: {details}")
    
    def get_file_size(self, file_path):
        """Get file size in bytes"""
        try:
            return file_path.stat().st_size
        except:
            return 0
    
    def create_organized_structure(self):
        """Create organized directory structure"""
        print("CREATING ORGANIZED STRUCTURE:")
        print("=" * 50)
        
        # Define organized structure
        structure = {
            "logs": "All log files and system reports",
            "logs/github": "GitHub Actions logs and CI/CD artifacts", 
            "logs/system": "System status reports and monitoring",
            "logs/archived": "Archived and historical logs",
            "temp": "Temporary files and build artifacts",
            "backup": "Backup files and recovery data",
            "archive": "Archived project artifacts"
        }
        
        for dir_path, description in structure.items():
            full_path = self.project_root / dir_path
            if not full_path.exists():
                full_path.mkdir(parents=True, exist_ok=True)
                self.cleanup_report["directories_created"] += 1
                self.log_action("Created directory", f"{dir_path} - {description}")
    
    def organize_log_files(self):
        """Organize all log files"""
        print("\nORGANIZING LOG FILES:")
        print("=" * 50)
        
        # Find all log-related files
        log_patterns = [
            "*.log", "*.txt", "*test*.txt", "*_docs.txt", 
            "logs_*.zip", "system_status_*.json"
        ]
        
        # GitHub Actions logs (numbered test files)
        github_logs = list(self.project_root.glob("*test (3.*)"))
        
        for log_dir in github_logs:
            if log_dir.is_dir():
                target = self.project_root / "logs" / "github" / log_dir.name
                if not target.exists():
                    shutil.move(str(log_dir), str(target))
                    self.cleanup_report["files_moved"] += 1
                    self.log_action("Moved GitHub logs", f"{log_dir.name} → logs/github/")
        
        # Individual log files
        log_files = []
        for pattern in log_patterns:
            log_files.extend(self.project_root.glob(pattern))
        
        for log_file in log_files:
            if log_file.is_file():
                # Determine target directory
                if "system_status" in log_file.name:
                    target_dir = self.project_root / "logs" / "system"
                elif any(x in log_file.name.lower() for x in ["test", "github", "ci", "action"]):
                    target_dir = self.project_root / "logs" / "github"
                else:
                    target_dir = self.project_root / "logs"
                
                target_file = target_dir / log_file.name
                if not target_file.exists():
                    shutil.move(str(log_file), str(target_file))
                    self.cleanup_report["files_moved"] += 1
                    self.log_action("Moved log file", f"{log_file.name} → {target_dir.name}/")
        
        # Handle existing logs directory content
        existing_logs = self.project_root / "logs"
        if existing_logs.exists():
            for item in existing_logs.iterdir():
                if item.is_file() and item.suffix in [".json", ".log"]:
                    if "system_status" in item.name:
                        target = self.project_root / "logs" / "system" / item.name
                        if not target.exists():
                            shutil.move(str(item), str(target))
                            self.log_action("Organized system log", item.name)
    
    def clean_temporary_files(self):
        """Clean up temporary and build files"""
        print("\nCLEANING TEMPORARY FILES:")
        print("=" * 50)
        
        # Patterns for temporary files
        temp_patterns = [
            "__pycache__",
            "*.pyc", 
            "*.pyo",
            ".pytest_cache",
            "*.tmp",
            "*.temp",
            ".mypy_cache",
            "*.coverage"
        ]
        
        temp_files = []
        for pattern in temp_patterns:
            temp_files.extend(self.project_root.rglob(pattern))
        
        for temp_item in temp_files:
            size = 0
            if temp_item.is_file():
                size = self.get_file_size(temp_item)
                temp_item.unlink()
                self.cleanup_report["files_deleted"] += 1
            elif temp_item.is_dir():
                # Calculate directory size
                for file in temp_item.rglob("*"):
                    if file.is_file():
                        size += self.get_file_size(file)
                shutil.rmtree(temp_item)
                self.cleanup_report["files_deleted"] += len(list(temp_item.rglob("*")))
            
            self.cleanup_report["space_freed"] += size
            self.log_action("Removed temp file/dir", f"{temp_item.name} ({size} bytes)")
    
    def organize_project_structure(self):
        """Ensure proper project structure"""
        print("\nVERIFYING PROJECT STRUCTURE:")
        print("=" * 50)
        
        # Expected project directories
        expected_dirs = [
            "code",
            "docs", 
            "tests",
            ".github",
            "config",
            "logs",
            "temp"
        ]
        
        for dir_name in expected_dirs:
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                dir_path.mkdir(exist_ok=True)
                self.cleanup_report["directories_created"] += 1
                self.log_action("Created missing directory", dir_name)
            else:
                self.log_action("Verified directory", f"{dir_name} exists")
    
    def archive_old_artifacts(self):
        """Archive old build and deployment artifacts"""
        print("\nARCHIVING OLD ARTIFACTS:")
        print("=" * 50)
        
        # Create archive with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir = self.project_root / "archive" / f"day2_completion_{timestamp}"
        
        # Files to archive (if they exist)
        artifacts_to_archive = [
            "build/",
            "dist/", 
            "*.egg-info/",
            ".coverage",
            "coverage.xml",
            "htmlcov/"
        ]
        
        archive_created = False
        for pattern in artifacts_to_archive:
            artifacts = list(self.project_root.glob(pattern))
            for artifact in artifacts:
                if not archive_created:
                    archive_dir.mkdir(parents=True, exist_ok=True)
                    archive_created = True
                
                target = archive_dir / artifact.name
                if artifact.is_file():
                    shutil.copy2(artifact, target)
                elif artifact.is_dir():
                    shutil.copytree(artifact, target, dirs_exist_ok=True)
                
                self.log_action("Archived artifact", f"{artifact.name} → archive/")
        
        if archive_created:
            self.log_action("Created archive", f"day2_completion_{timestamp}")
    
    def create_cleanup_index(self):
        """Create index of organized files"""
        print("\nCREATING CLEANUP INDEX:")
        print("=" * 50)
        
        index = {
            "cleanup_date": datetime.now().isoformat(),
            "project_structure": {},
            "log_organization": {},
            "cleanup_stats": self.cleanup_report
        }
        
        # Index project structure
        for item in self.project_root.iterdir():
            if item.is_dir():
                file_count = len(list(item.rglob("*"))) if item.name != "venv" else "excluded"
                index["project_structure"][item.name] = {
                    "type": "directory",
                    "file_count": file_count
                }
            else:
                index["project_structure"][item.name] = {
                    "type": "file",
                    "size": self.get_file_size(item)
                }
        
        # Index log organization
        logs_dir = self.project_root / "logs"
        if logs_dir.exists():
            for subdir in logs_dir.iterdir():
                if subdir.is_dir():
                    files = list(subdir.glob("*"))
                    index["log_organization"][subdir.name] = {
                        "file_count": len(files),
                        "files": [f.name for f in files[:10]]  # First 10 files
                    }
        
        # Save index
        index_file = self.project_root / "logs" / "cleanup_index.json"
        with open(index_file, 'w') as f:
            json.dump(index, f, indent=2)
        
        self.log_action("Created cleanup index", "logs/cleanup_index.json")
        return index
    
    def generate_cleanup_report(self):
        """Generate final cleanup report"""
        print("\nCLEANUP SUMMARY:")
        print("=" * 50)
        
        stats = self.cleanup_report
        
        print(f"Files moved: {stats['files_moved']}")
        print(f"Files deleted: {stats['files_deleted']}")
        print(f"Directories created: {stats['directories_created']}")
        print(f"Space freed: {stats['space_freed']:,} bytes ({stats['space_freed']/1024:.1f} KB)")
        print(f"Total actions: {len(stats['actions'])}")
        print()
        
        print("PROJECT ORGANIZATION COMPLETE!")
        print("✓ Log files organized by category")
        print("✓ Temporary files cleaned")
        print("✓ Project structure verified")
        print("✓ Build artifacts archived")
        print("✓ Cleanup index created")
        print()
        
        # Save detailed report
        report_file = self.project_root / "logs" / f"cleanup_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(self.cleanup_report, f, indent=2)
        
        print(f"Detailed report saved: {report_file}")
        return self.cleanup_report
    
    def run_full_cleanup(self):
        """Run complete project cleanup"""
        print("APEXSIGMA DEVENVIRO PROJECT CLEANUP")
        print("=" * 60)
        print(f"Project: {self.project_root}")
        print(f"Started: {self.cleanup_report['timestamp']}")
        print()
        
        # Run all cleanup tasks
        self.create_organized_structure()
        self.organize_log_files()
        self.clean_temporary_files()
        self.organize_project_structure()
        self.archive_old_artifacts()
        index = self.create_cleanup_index()
        report = self.generate_cleanup_report()
        
        return {
            "cleanup_report": report,
            "project_index": index
        }


def main():
    """Main cleanup function"""
    cleanup = ProjectCleanup()
    results = cleanup.run_full_cleanup()
    
    print("\n" + "=" * 60)
    print("🎉 PROJECT CLEANUP COMPLETE! 🎉")
    print("Your ApexSigma DevEnviro project is now perfectly organized!")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    main()