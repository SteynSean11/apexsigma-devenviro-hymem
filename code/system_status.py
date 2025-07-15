#!/usr/bin/env python3
"""
ApexSigma DevEnviro System Status Dashboard
Comprehensive monitoring of all services and infrastructure
"""

import subprocess
import requests
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional


class SystemStatus:
    """Comprehensive system status monitoring"""
    
    def __init__(self):
        self.status_data = {
            "timestamp": datetime.now().isoformat(),
            "docker": {},
            "memory_services": {},
            "global_structure": {},
            "project": {},
            "ci_cd": {},
            "summary": {}
        }
    
    def check_docker_services(self) -> Dict[str, Any]:
        """Check Docker services status"""
        print("DOCKER SERVICES:")
        print("=" * 50)
        
        docker_status = {
            "docker_available": False,
            "apexsigma_services": [],
            "total_containers": 0,
            "running_containers": 0
        }
        
        try:
            # Check if Docker is available
            result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                docker_status["docker_available"] = True
                print(f"SUCCESS: Docker version: {result.stdout.strip()}")
            
            # Check all containers
            result = subprocess.run(['docker', 'ps', '-a', '--format', 'json'], capture_output=True, text=True)
            if result.returncode == 0:
                containers = []
                for line in result.stdout.strip().split('\n'):
                    if line:
                        try:
                            container = json.loads(line)
                            containers.append(container)
                            docker_status["total_containers"] += 1
                            if container.get("State") == "running":
                                docker_status["running_containers"] += 1
                        except json.JSONDecodeError:
                            continue
                
                # Filter ApexSigma services
                apexsigma_services = [c for c in containers if "apexsigma" in c.get("Names", "").lower()]
                docker_status["apexsigma_services"] = apexsigma_services
                
                print(f"Total containers: {docker_status['total_containers']}")
                print(f"Running containers: {docker_status['running_containers']}")
                print(f"ApexSigma services: {len(apexsigma_services)}")
                print()
                
                for service in apexsigma_services:
                    name = service.get("Names", "Unknown")
                    state = service.get("State", "Unknown")
                    status = service.get("Status", "Unknown")
                    ports = service.get("Ports", "No ports")
                    
                    status_icon = "SUCCESS" if state == "running" else "ERROR"
                    print(f"{status_icon}: {name}")
                    print(f"   State: {state}")
                    print(f"   Status: {status}")
                    print(f"   Ports: {ports}")
                    print()
            
        except FileNotFoundError:
            print("ERROR: Docker not found - is Docker installed?")
        except Exception as e:
            print(f"ERROR: Docker check failed: {e}")
        
        self.status_data["docker"] = docker_status
        return docker_status
    
    def check_memory_services(self) -> Dict[str, Any]:
        """Check memory services health"""
        print("MEMORY SERVICES:")
        print("=" * 50)
        
        memory_status = {
            "qdrant": {"available": False, "healthy": False, "details": {}},
            "mem0": {"available": False, "healthy": False, "details": {}}
        }
        
        # Check Qdrant (try root endpoint as health check)
        try:
            response = requests.get("http://localhost:6333/", timeout=5)
            memory_status["qdrant"]["available"] = True
            if response.status_code == 200:
                qdrant_info = response.json()
                memory_status["qdrant"]["healthy"] = True
                memory_status["qdrant"]["details"] = qdrant_info
                print("SUCCESS: Qdrant - Healthy and responding")
                print(f"   Version: {qdrant_info.get('version', 'Unknown')}")
                print(f"   Title: {qdrant_info.get('title', 'Unknown')}")
                
                # Get Qdrant collections info
                try:
                    collections_response = requests.get("http://localhost:6333/collections", timeout=5)
                    if collections_response.status_code == 200:
                        collections_data = collections_response.json()
                        collections = collections_data.get("result", {}).get("collections", [])
                        memory_status["qdrant"]["details"]["collections_count"] = len(collections)
                        print(f"   Collections: {len(collections)}")
                        
                        for collection in collections[:3]:  # Show first 3
                            print(f"   - {collection.get('name', 'Unknown')}")
                except:
                    pass
            else:
                print(f"ERROR: Qdrant - Unhealthy (status: {response.status_code})")
        except requests.exceptions.ConnectionError:
            print("ERROR: Qdrant - Not responding (connection refused)")
        except Exception as e:
            print(f"ERROR: Qdrant - {e}")
        
        # Check Mem0 Service
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            memory_status["mem0"]["available"] = True
            if response.status_code == 200:
                health_data = response.json()
                memory_status["mem0"]["healthy"] = True
                memory_status["mem0"]["details"] = health_data
                
                backend_status = health_data.get("memory_backend", "unknown")
                print("SUCCESS: Mem0 Service - Healthy and responding")
                print(f"   Memory Backend: {backend_status}")
                print(f"   Service Version: {health_data.get('service', 'Unknown')}")
                
                # Test API endpoints
                try:
                    info_response = requests.get("http://localhost:8000/", timeout=5)
                    if info_response.status_code == 200:
                        info_data = info_response.json()
                        print(f"   API Version: {info_data.get('version', 'Unknown')}")
                        endpoints = info_data.get('endpoints', {})
                        print(f"   Available Endpoints: {len(endpoints)}")
                except:
                    pass
            else:
                print(f"ERROR: Mem0 Service - Unhealthy (status: {response.status_code})")
        except requests.exceptions.ConnectionError:
            print("ERROR: Mem0 Service - Not responding (connection refused)")
        except Exception as e:
            print(f"ERROR: Mem0 Service - {e}")
        
        print()
        self.status_data["memory_services"] = memory_status
        return memory_status
    
    def check_global_structure(self) -> Dict[str, Any]:
        """Check global ApexSigma structure"""
        print("GLOBAL STRUCTURE:")
        print("=" * 50)
        
        global_status = {
            "apexsigma_exists": False,
            "directories": {},
            "context_files": {},
            "config_files": {}
        }
        
        apexsigma_path = Path.home() / ".apexsigma"
        if apexsigma_path.exists():
            global_status["apexsigma_exists"] = True
            print("SUCCESS: ~/.apexsigma/ directory exists")
            
            # Check directories
            required_dirs = ["config", "context", "memory", "tools"]
            for dir_name in required_dirs:
                dir_path = apexsigma_path / dir_name
                exists = dir_path.exists()
                global_status["directories"][dir_name] = exists
                status = "SUCCESS" if exists else "ERROR"
                print(f"{status}: {dir_name}/ directory {'exists' if exists else 'missing'}")
            
            # Check context files
            context_files = ["security.md", "globalrules.md", "brand.md"]
            for file_name in context_files:
                file_path = apexsigma_path / "context" / file_name
                exists = file_path.exists()
                global_status["context_files"][file_name] = exists
                status = "SUCCESS" if exists else "ERROR"
                print(f"{status}: context/{file_name} {'exists' if exists else 'missing'}")
            
            # Check config files
            config_file = apexsigma_path / "config" / "infrastructure.yml"
            exists = config_file.exists()
            global_status["config_files"]["infrastructure.yml"] = exists
            status = "SUCCESS" if exists else "ERROR"
            print(f"{status}: config/infrastructure.yml {'exists' if exists else 'missing'}")
            
        else:
            print("ERROR: ~/.apexsigma/ directory missing")
        
        # Check Gemini structure
        gemini_path = Path.home() / ".gemini"
        if gemini_path.exists():
            print("SUCCESS: ~/.gemini/ directory exists")
            global_status["gemini_exists"] = True
        else:
            print("WARNING: ~/.gemini/ directory missing")
            global_status["gemini_exists"] = False
        
        print()
        self.status_data["global_structure"] = global_status
        return global_status
    
    def check_project_status(self) -> Dict[str, Any]:
        """Check current project status"""
        print("PROJECT STATUS:")
        print("=" * 50)
        
        project_status = {
            "project_exists": False,
            "virtual_env": False,
            "git_status": "unknown",
            "dependencies": {},
            "structure": {}
        }
        
        project_path = Path.cwd()
        print(f"Project location: {project_path}")
        
        if project_path.exists():
            project_status["project_exists"] = True
            print("SUCCESS: Project directory exists")
            
            # Check virtual environment
            if 'VIRTUAL_ENV' in os.environ:
                project_status["virtual_env"] = True
                print(f"SUCCESS: Virtual environment active: {os.environ['VIRTUAL_ENV']}")
            else:
                print("WARNING: Virtual environment not active")
            
            # Check key project files
            key_files = ["pyproject.toml", "README.md", ".gitignore", ".secrets.baseline"]
            for file_name in key_files:
                exists = (project_path / file_name).exists()
                project_status["structure"][file_name] = exists
                status = "SUCCESS" if exists else "WARNING"
                print(f"{status}: {file_name} {'exists' if exists else 'missing'}")
            
            # Check directories
            key_dirs = ["code", "docs", "tests", ".github"]
            for dir_name in key_dirs:
                exists = (project_path / dir_name).exists()
                project_status["structure"][dir_name] = exists
                status = "SUCCESS" if exists else "WARNING"
                print(f"{status}: {dir_name}/ {'exists' if exists else 'missing'}")
            
            # Check git status
            try:
                result = subprocess.run(['git', 'status', '--porcelain'], 
                                      capture_output=True, text=True, cwd=project_path)
                if result.returncode == 0:
                    if result.stdout.strip():
                        project_status["git_status"] = "uncommitted_changes"
                        print("WARNING: Uncommitted changes exist")
                    else:
                        project_status["git_status"] = "clean"
                        print("SUCCESS: Git working directory clean")
                        
                    # Get current branch
                    branch_result = subprocess.run(['git', 'branch', '--show-current'], 
                                                 capture_output=True, text=True, cwd=project_path)
                    if branch_result.returncode == 0:
                        branch = branch_result.stdout.strip()
                        project_status["current_branch"] = branch
                        print(f"   Current branch: {branch}")
            except:
                project_status["git_status"] = "error"
                print("ERROR: Git status check failed")
        
        print()
        self.status_data["project"] = project_status
        return project_status
    
    def check_ci_cd_status(self) -> Dict[str, Any]:
        """Check CI/CD pipeline status"""
        print("CI/CD PIPELINE:")
        print("=" * 50)
        
        cicd_status = {
            "github_actions": False,
            "workflows": [],
            "last_run": "unknown"
        }
        
        # Check GitHub Actions workflows
        workflows_path = Path.cwd() / ".github" / "workflows"
        if workflows_path.exists():
            cicd_status["github_actions"] = True
            print("SUCCESS: GitHub Actions workflows directory exists")
            
            workflow_files = list(workflows_path.glob("*.yml")) + list(workflows_path.glob("*.yaml"))
            cicd_status["workflows"] = [f.name for f in workflow_files]
            
            for workflow in workflow_files:
                print(f"   - {workflow.name}")
            
            print(f"   Total workflows: {len(workflow_files)}")
        else:
            print("WARNING: No GitHub Actions workflows found")
        
        # Check for CI/CD indicators
        if (Path.cwd() / "pyproject.toml").exists():
            print("SUCCESS: pyproject.toml found (modern Python packaging)")
            
        if (Path.cwd() / ".secrets.baseline").exists():
            print("SUCCESS: Secret scanning baseline configured")
            
        if (Path.cwd() / "pytest.ini").exists():
            print("SUCCESS: pytest configuration found")
        
        print()
        self.status_data["ci_cd"] = cicd_status
        return cicd_status
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate overall system summary"""
        print("SYSTEM SUMMARY:")
        print("=" * 50)
        
        summary = {
            "overall_health": "unknown",
            "critical_services": 0,
            "warnings": 0,
            "successes": 0,
            "recommendations": []
        }
        
        # Count critical services
        docker_data = self.status_data.get("docker", {})
        memory_data = self.status_data.get("memory_services", {})
        
        if docker_data.get("docker_available"):
            summary["successes"] += 1
        else:
            summary["warnings"] += 1
            summary["recommendations"].append("Install or start Docker")
        
        if memory_data.get("qdrant", {}).get("healthy"):
            summary["critical_services"] += 1
            summary["successes"] += 1
        else:
            summary["recommendations"].append("Start Qdrant vector database")
        
        if memory_data.get("mem0", {}).get("healthy"):
            summary["critical_services"] += 1
            summary["successes"] += 1
        else:
            summary["recommendations"].append("Start Mem0 service")
        
        # Determine overall health
        if summary["critical_services"] >= 2:
            summary["overall_health"] = "healthy"
        elif summary["critical_services"] >= 1:
            summary["overall_health"] = "partial"
        else:
            summary["overall_health"] = "critical"
        
        # Print summary
        health_color = "SUCCESS" if summary["overall_health"] == "healthy" else "WARNING" if summary["overall_health"] == "partial" else "ERROR"
        print(f"{health_color}: Overall system health: {summary['overall_health'].upper()}")
        print(f"   Critical services running: {summary['critical_services']}/2")
        print(f"   Total successes: {summary['successes']}")
        print(f"   Warnings: {summary['warnings']}")
        print()
        
        if summary["recommendations"]:
            print("RECOMMENDATIONS:")
            for i, rec in enumerate(summary["recommendations"], 1):
                print(f"   {i}. {rec}")
            print()
        
        self.status_data["summary"] = summary
        return summary
    
    def display_quick_commands(self):
        """Display useful quick commands"""
        print("QUICK COMMANDS:")
        print("=" * 50)
        print("Service Management:")
        print("   Start services:     cd ~/.apexsigma/config && docker-compose -f infrastructure.yml up -d")
        print("   Stop services:      cd ~/.apexsigma/config && docker-compose -f infrastructure.yml down")
        print("   View Qdrant logs:   docker logs apexsigma-qdrant --tail 20")
        print("   View Mem0 logs:     docker logs apexsigma-mem0 --tail 20")
        print()
        print("Testing:")
        print("   Test Qdrant:        curl http://localhost:6333/health")
        print("   Test Mem0:          curl http://localhost:8000/health")
        print("   Test memory setup:  python code/test_mem0.py")
        print()
        print("Development:")
        print("   Run status check:   python code/system_status.py")
        print("   Check CI/CD:        git status && git log --oneline -5")
        print("   Activate venv:      source venv/bin/activate  # or venv\\Scripts\\activate on Windows")
        print()
    
    def save_status_report(self):
        """Save detailed status report to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = Path("logs") / f"system_status_{timestamp}.json"
        
        # Ensure logs directory exists
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(self.status_data, f, indent=2)
        
        print(f"Detailed status report saved to: {report_file}")
        return report_file
    
    def run_full_check(self):
        """Run comprehensive system status check"""
        print("APEXSIGMA DEVENVIRO SYSTEM STATUS")
        print("=" * 60)
        print(f"Generated: {self.status_data['timestamp']}")
        print()
        
        # Run all checks
        self.check_docker_services()
        self.check_memory_services()
        self.check_global_structure()
        self.check_project_status()
        self.check_ci_cd_status()
        self.generate_summary()
        self.display_quick_commands()
        
        # Save report
        report_file = self.save_status_report()
        
        return self.status_data


def main():
    """Main function"""
    status_checker = SystemStatus()
    status_data = status_checker.run_full_check()
    
    # Return success/failure for automation
    overall_health = status_data.get("summary", {}).get("overall_health", "critical")
    return 0 if overall_health in ["healthy", "partial"] else 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)