#!/usr/bin/env python3
"""
Check Linear project status and tasks
"""

import os
import requests
from pathlib import Path
from dotenv import load_dotenv
import json


def get_linear_project_status():
    """Fetch current Linear project status and tasks"""
    print("Fetching Linear Project Status...")
    
    # Load environment variables
    project_root = Path(__file__).resolve().parent.parent
    env_file = project_root / "config" / "secrets" / ".env"
    load_dotenv(env_file)
    
    api_key = os.getenv("LINEAR_API_KEY")
    
    if not api_key or api_key == "YOUR_ACTUAL_KEY_HERE":
        print("ERROR: Linear API key not configured")
        return None
    
    headers = {
        "Authorization": api_key,
        "Content-Type": "application/json",
        "User-Agent": "ApexSigma-DevEnviro/1.0",
    }
    
    # Query to get projects, teams, and issues
    query = """
    query {
        viewer {
            name
            organization {
                name
                teams {
                    nodes {
                        id
                        name
                        key
                        projects {
                            nodes {
                                id
                                name
                                description
                                state
                                progress
                                startDate
                                targetDate
                                issues {
                                    nodes {
                                        id
                                        title
                                        description
                                        state {
                                            name
                                            type
                                        }
                                        priority
                                        assignee {
                                            name
                                        }
                                        labels {
                                            nodes {
                                                name
                                            }
                                        }
                                        createdAt
                                        updatedAt
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    """
    
    try:
        response = requests.post(
            "https://api.linear.app/graphql",
            json={"query": query},
            headers=headers,
            timeout=30,
        )
        
        if response.status_code == 200:
            data = response.json()
            if "data" in data and "viewer" in data["data"]:
                return analyze_project_data(data["data"])
            else:
                print("❌ Unexpected response format")
                return None
        else:
            print(f"❌ API request failed: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error fetching Linear data: {e}")
        return None


def analyze_project_data(data):
    """Analyze and display Linear project data"""
    print("✅ Linear connection successful!")
    print()
    
    viewer = data["viewer"]
    org = viewer["organization"]
    
    print(f"📊 Organization: {org['name']}")
    print(f"👤 User: {viewer['name']}")
    print()
    
    # Analyze teams and projects
    teams = org["teams"]["nodes"]
    
    if not teams:
        print("❌ No teams found")
        return None
    
    total_projects = 0
    total_issues = 0
    active_issues = 0
    completed_issues = 0
    
    project_summary = []
    
    for team in teams:
        print(f"🏢 Team: {team['name']} ({team['key']})")
        
        projects = team["projects"]["nodes"]
        if not projects:
            print("   No projects found")
            continue
            
        for project in projects:
            total_projects += 1
            issues = project["issues"]["nodes"]
            total_issues += len(issues)
            
            # Count issue states
            project_active = 0
            project_completed = 0
            
            for issue in issues:
                if issue["state"]["type"] in ["completed", "canceled"]:
                    project_completed += 1
                    completed_issues += 1
                else:
                    project_active += 1
                    active_issues += 1
            
            progress = project.get("progress", 0) or 0
            
            print(f"   📁 Project: {project['name']}")
            print(f"      State: {project['state']}")
            print(f"      Progress: {progress:.1f}%")
            print(f"      Issues: {len(issues)} total, {project_active} active, {project_completed} completed")
            
            if project.get("description"):
                print(f"      Description: {project['description'][:100]}...")
            
            project_summary.append({
                "name": project["name"],
                "state": project["state"],
                "progress": progress,
                "total_issues": len(issues),
                "active_issues": project_active,
                "completed_issues": project_completed,
                "issues": issues
            })
            
            # Show recent active issues
            active_issues_list = [i for i in issues if i["state"]["type"] not in ["completed", "canceled"]]
            if active_issues_list:
                print(f"      🔥 Active Issues:")
                for issue in active_issues_list[:3]:  # Show top 3
                    priority = issue.get("priority", 0)
                    priority_str = "🔴" if priority >= 3 else "🟡" if priority >= 2 else "🟢"
                    assignee = issue["assignee"]["name"] if issue.get("assignee") else "Unassigned"
                    print(f"         {priority_str} {issue['title']} ({assignee})")
                if len(active_issues_list) > 3:
                    print(f"         ... and {len(active_issues_list) - 3} more")
            print()
    
    # Summary
    print("="*60)
    print("📈 PROJECT SUMMARY")
    print("="*60)
    print(f"Total Projects: {total_projects}")
    print(f"Total Issues: {total_issues}")
    print(f"Active Issues: {active_issues}")
    print(f"Completed Issues: {completed_issues}")
    
    if total_issues > 0:
        completion_rate = (completed_issues / total_issues) * 100
        print(f"Completion Rate: {completion_rate:.1f}%")
    
    return {
        "organization": org["name"],
        "user": viewer["name"],
        "total_projects": total_projects,
        "total_issues": total_issues,
        "active_issues": active_issues,
        "completed_issues": completed_issues,
        "projects": project_summary
    }


def suggest_next_actions(status_data):
    """Suggest next actions based on Linear project status"""
    if not status_data:
        return
    
    print("\n🎯 SUGGESTED NEXT ACTIONS")
    print("="*60)
    
    # Find high priority active issues
    high_priority_issues = []
    for project in status_data["projects"]:
        for issue in project["issues"]:
            if (issue["state"]["type"] not in ["completed", "canceled"] and 
                issue.get("priority", 0) >= 3):
                high_priority_issues.append({
                    "title": issue["title"],
                    "project": project["name"],
                    "assignee": issue["assignee"]["name"] if issue.get("assignee") else "Unassigned"
                })
    
    if high_priority_issues:
        print("🔥 HIGH PRIORITY ISSUES:")
        for issue in high_priority_issues[:5]:
            print(f"   • {issue['title']} ({issue['project']}) - {issue['assignee']}")
        print()
    
    # Suggest based on completion rate
    total_issues = status_data["total_issues"]
    active_issues = status_data["active_issues"]
    
    if total_issues == 0:
        print("📝 RECOMMENDATION: Set up initial project tasks in Linear")
    elif active_issues == 0:
        print("🎉 GREAT! All issues completed. Consider planning next phase.")
    elif active_issues > total_issues * 0.7:
        print("⚡ FOCUS: High number of active issues. Consider prioritizing.")
    else:
        print("🚀 MOMENTUM: Good progress! Keep working on active issues.")
    
    print("\n💡 DEVELOPMENT PRIORITIES:")
    print("   1. Review high-priority issues in Linear")
    print("   2. Update issue status as work progresses") 
    print("   3. Create new issues for upcoming features")
    print("   4. Use Linear integration to track development progress")


if __name__ == "__main__":
    status = get_linear_project_status()
    if status:
        suggest_next_actions(status)
    else:
        print("\n🔧 Setup Required:")
        print("1. Configure Linear API key in config/secrets/.env")
        print("2. Ensure LINEAR_API_KEY is set correctly")
        print("3. Run: python code/test_linear_wsl2.py to test connection")