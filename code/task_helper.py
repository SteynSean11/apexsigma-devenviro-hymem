#!/usr/bin/env python3
"""
Task Execution Helper
Simple interface for working through Day 3 tasks step by step
"""

import json
import requests
from datetime import datetime
from typing import Dict, List, Any


class TaskHelper:
    """Helper for executing Day 3 tasks with cognitive support"""
    
    def __init__(self):
        self.tasks = [
            {
                "id": "task_1",
                "title": "Memory bridge connecting all cognitive services",
                "status": "pending",
                "priority": "high",
                "estimated_time": "90 minutes",
                "description": "Create the core memory bridge that connects Qdrant, Mem0, and workflow systems"
            },
            {
                "id": "task_2", 
                "title": "Cross-project knowledge sharing system",
                "status": "pending",
                "priority": "high",
                "estimated_time": "90 minutes",
                "description": "Implement system for sharing knowledge across different projects"
            },
            {
                "id": "task_3",
                "title": "AI agent memory persistence and retrieval",
                "status": "pending",
                "priority": "medium",
                "estimated_time": "75 minutes",
                "description": "Create persistent memory interface for AI agents"
            },
            {
                "id": "task_4",
                "title": "Context-aware development assistance",
                "status": "pending",
                "priority": "medium", 
                "estimated_time": "75 minutes",
                "description": "Implement context-aware development assistant"
            },
            {
                "id": "task_5",
                "title": "Pattern recognition and learning capabilities",
                "status": "pending",
                "priority": "medium",
                "estimated_time": "60 minutes",
                "description": "Add pattern recognition for development workflows"
            },
            {
                "id": "task_6",
                "title": "Advanced cognitive workflow automation",
                "status": "pending",
                "priority": "medium",
                "estimated_time": "60 minutes",
                "description": "Create automated workflow orchestration"
            }
        ]
        self.current_task = None
        self.completed_tasks = []
    
    def show_all_tasks(self):
        """Display all tasks with their status"""
        print("DAY 3 TASK OVERVIEW")
        print("=" * 60)
        
        for task in self.tasks:
            status_icon = {
                "pending": "[ ]",
                "in_progress": "[>]", 
                "completed": "[✓]"
            }.get(task["status"], "[ ]")
            
            print(f"{status_icon} {task['title']}")
            print(f"    Priority: {task['priority']} | Time: {task['estimated_time']}")
            print(f"    {task['description']}")
            print()
    
    def get_next_task(self):
        """Get the next pending task"""
        pending = [t for t in self.tasks if t["status"] == "pending"]
        
        if not pending:
            print("SUCCESS: All tasks completed!")
            return None
        
        # Sort by priority (high first)
        priority_order = {"high": 0, "medium": 1, "low": 2}
        pending.sort(key=lambda t: priority_order.get(t["priority"], 1))
        
        return pending[0]
    
    def start_task(self, task_id: str):
        """Start working on a specific task"""
        task = next((t for t in self.tasks if t["id"] == task_id), None)
        
        if not task:
            print(f"ERROR: Task {task_id} not found")
            return None
        
        if task["status"] != "pending":
            print(f"ERROR: Task {task_id} is not pending (current status: {task['status']})")
            return None
        
        # Update status
        task["status"] = "in_progress"
        task["start_time"] = datetime.now().isoformat()
        self.current_task = task
        
        print(f"STARTING TASK: {task['title']}")
        print("=" * 60)
        print(f"Description: {task['description']}")
        print(f"Priority: {task['priority']}")
        print(f"Estimated Time: {task['estimated_time']}")
        print()
        print("SUCCESS CRITERIA:")
        print("- Implementation completed")
        print("- Tests passing")
        print("- Documentation updated") 
        print("- Ready for next task")
        print()
        print(f"FOCUS: Work on implementing {task['title']}")
        print(f"When done, use: helper.complete_task('{task_id}', 'success', ['lesson1', 'lesson2'])")
        
        # Store in memory
        self.store_memory(f"Task Started: {task['title']}", {
            "task_id": task_id,
            "start_time": task["start_time"]
        })
        
        return task
    
    def complete_task(self, task_id: str, outcome: str, lessons_learned: List[str] = None):
        """Mark a task as complete"""
        if not self.current_task or self.current_task["id"] != task_id:
            print(f"ERROR: Task {task_id} is not currently active")
            return False
        
        # Update task
        self.current_task["status"] = "completed"
        self.current_task["outcome"] = outcome
        self.current_task["lessons_learned"] = lessons_learned or []
        self.current_task["end_time"] = datetime.now().isoformat()
        
        # Move to completed
        self.completed_tasks.append(self.current_task)
        completed_task = self.current_task
        self.current_task = None
        
        print(f"TASK COMPLETED: {completed_task['title']}")
        print("=" * 60)
        print(f"Outcome: {outcome}")
        if lessons_learned:
            print(f"Lessons Learned:")
            for lesson in lessons_learned:
                print(f"  - {lesson}")
        print()
        
        # Store in memory
        self.store_memory(f"Task Completed: {completed_task['title']}", {
            "task_id": task_id,
            "outcome": outcome,
            "lessons": lessons_learned or [],
            "completion_time": completed_task["end_time"]
        })
        
        # Show next task
        next_task = self.get_next_task()
        if next_task:
            print(f"NEXT TASK AVAILABLE: {next_task['title']}")
            print(f"Use: helper.start_task('{next_task['id']}') to begin")
        else:
            print("SUCCESS: All Day 3 tasks completed!")
            self.show_completion_summary()
        
        return True
    
    def store_memory(self, context: str, details: Dict[str, Any]):
        """Store context in memory system"""
        try:
            memory_data = {
                "message": f"Day 3 Task: {context}. Details: {json.dumps(details)}",
                "user_id": "task_helper",
                "metadata": {
                    "workflow_step": context,
                    "timestamp": datetime.now().isoformat(),
                    "project": "apexsigma-devenviro",
                    "day": 3
                }
            }
            
            response = requests.post("http://localhost:8001/memory/add", json=memory_data, timeout=5)
            if response.status_code == 200:
                print(f"SUCCESS: Stored in cognitive memory")
            return response.status_code == 200
        except Exception as e:
            print(f"WARNING: Could not store in memory: {e}")
            return False
    
    def show_progress(self):
        """Show current progress"""
        total_tasks = len(self.tasks)
        completed = len([t for t in self.tasks if t["status"] == "completed"])
        in_progress = len([t for t in self.tasks if t["status"] == "in_progress"])
        pending = len([t for t in self.tasks if t["status"] == "pending"])
        
        print("DAY 3 PROGRESS")
        print("=" * 60)
        print(f"Total Tasks: {total_tasks}")
        print(f"Completed: {completed}")
        print(f"In Progress: {in_progress}")
        print(f"Pending: {pending}")
        print(f"Progress: {(completed/total_tasks)*100:.1f}%")
        print()
        
        if self.current_task:
            print(f"CURRENT TASK: {self.current_task['title']}")
            print(f"Status: {self.current_task['status']}")
        elif pending > 0:
            next_task = self.get_next_task()
            print(f"NEXT TASK: {next_task['title']}")
        else:
            print("ALL TASKS COMPLETED!")
    
    def show_completion_summary(self):
        """Show completion summary"""
        print("\nDAY 3 COMPLETION SUMMARY")
        print("=" * 60)
        
        for task in self.completed_tasks:
            print(f"✓ {task['title']}")
            print(f"   Outcome: {task.get('outcome', 'Unknown')}")
            if task.get('lessons_learned'):
                print(f"   Lessons: {', '.join(task['lessons_learned'])}")
        
        print(f"\nTOTAL COMPLETED: {len(self.completed_tasks)}/{len(self.tasks)}")
        print("\nREADY FOR DAY 4: Advanced Pattern Recognition!")


def main():
    """Main helper function"""
    helper = TaskHelper()
    
    print("DAY 3 TASK EXECUTION HELPER")
    print("=" * 60)
    print("Your cognitive development workflow is ready!")
    print()
    
    # Show all tasks
    helper.show_all_tasks()
    
    # Show next task
    next_task = helper.get_next_task()
    if next_task:
        print(f"READY TO START: {next_task['title']}")
        print(f"Use: helper.start_task('{next_task['id']}') to begin")
    
    print("\nAVAILABLE COMMANDS:")
    print("  helper.show_all_tasks()     - Show all tasks")
    print("  helper.show_progress()      - Show current progress") 
    print("  helper.start_task('task_1') - Start a specific task")
    print("  helper.complete_task('task_1', 'success', ['lesson']) - Complete task")
    print()
    print("START WITH THE FIRST TASK AND WORK SYSTEMATICALLY!")
    
    return helper


if __name__ == "__main__":
    helper = main()