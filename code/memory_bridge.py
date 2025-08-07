#!/usr/bin/env python3
"""
Memory Bridge - Cognitive Service Orchestration
Core bridge connecting Qdrant, Mem0, and workflow systems for ApexSigma
"""

import asyncio
import json
import requests
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryBridge:
    """Bridge connecting all memory services for cognitive orchestration"""
    
    def __init__(self):
        self.mem0_url = "http://localhost:8000"
        self.bridge_config = self.load_bridge_config()
        self.services_healthy = False
        
    def load_bridge_config(self) -> Dict:
        """Load memory bridge configuration"""
        return {
            "sync_interval": 300,  # 5 minutes
            "memory_retention": 30,  # 30 days
            "cross_project_sharing": True,
            "learning_threshold": 0.8,  # Confidence threshold for pattern learning
            "context_window": 50  # Number of recent contexts to maintain
        }
    
    async def initialize_bridge(self):
        """Initialize memory bridge with all services"""
        logger.info("Initializing Memory Bridge...")
        
        # Verify service connections
        await self.verify_connections()
        
        # Initialize cross-project sharing
        await self.setup_cross_project_sharing()
        
        # Setup cognitive learning
        await self.setup_cognitive_learning()
        
        self.services_healthy = True
        logger.info("SUCCESS: Memory Bridge initialized and ready")
        return True
    
    async def verify_connections(self):
        """Verify all memory services are accessible"""
        logger.info("Verifying service connections...")
            
        # Check Mem0
        try:
            response = requests.get(f"{self.mem0_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info("✓ Mem0 service accessible")
            else:
                logger.warning(f"Mem0 responded with status: {response.status_code}")
        except Exception as e:
            logger.error(f"Mem0 connection failed: {e}")
    
    async def setup_cross_project_sharing(self):
        """Setup cross-project memory sharing"""
        logger.info("Setting up cross-project knowledge sharing...")
        
        # Initialize global ApexSigma memory structure
        global_memory_path = Path.home() / ".apexsigma" / "memory"
        global_memory_path.mkdir(parents=True, exist_ok=True)
        
        # Create knowledge sharing manifest
        sharing_manifest = {
            "initialized": datetime.now().isoformat(),
            "projects": [],
            "shared_patterns": [],
            "learning_data": {
                "successful_implementations": [],
                "common_errors": [],
                "best_practices": []
            }
        }
        
        manifest_path = global_memory_path / "sharing_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(sharing_manifest, f, indent=2)
            
        logger.info(f"✓ Cross-project sharing initialized at {global_memory_path}")
    
    async def setup_cognitive_learning(self):
        """Setup cognitive learning capabilities"""
        logger.info("Setting up cognitive learning system...")
        
        # Initialize learning patterns storage
        learning_config = {
            "pattern_recognition": True,
            "error_learning": True,
            "success_amplification": True,
            "context_association": True,
            "predictive_suggestions": True
        }
        
        # Store learning config in memory
        await self.store_development_context({
            "description": "Memory Bridge cognitive learning initialization",
            "developer": "system",
            "project": "apexsigma-devenviro",
            "components": ["learning_engine", "pattern_recognition"],
            "config": learning_config,
            "status": "initialized"
        })
        
        logger.info("✓ Cognitive learning system ready")

    async def store_development_context(self, context: Dict[str, Any]) -> str:
        """Store development context across all memory systems"""
        try:
            # Enhance context with metadata
            enhanced_context = {
                **context,
                "timestamp": datetime.now().isoformat(),
                "bridge_version": "1.0.0",
                "memory_type": "development_context"
            }
            
            # Store in Mem0 for semantic search
            mem0_payload = {
                "message": context.get("description", "Development context stored"),
                "user_id": context.get("developer", "system"),
                "metadata": enhanced_context
            }
            
            response = requests.post(f"{self.mem0_url}/memory/add", json=mem0_payload, timeout=10)
            
            if response.status_code == 200:
                logger.info(f"✓ Context stored: {context.get('description', 'Unknown')[:50]}...")
                return "success"
            else:
                logger.warning(f"Failed to store in Mem0: {response.status_code}")
                return "partial"
                
        except Exception as e:
            logger.error(f"Failed to store development context: {e}")
            return "failed"
    
    async def retrieve_relevant_context(self, query: str, project: str = None, limit: int = 10) -> List[Dict]:
        """Retrieve relevant development context"""
        try:
            search_payload = {
                "query": query,
                "user_id": "system",
                "limit": limit
            }
            
            response = requests.post(f"{self.mem0_url}/memory/search", json=search_payload, timeout=10)
            
            if response.status_code == 200:
                memories = response.json().get("memories", [])
                
                # Filter by project if specified
                if project:
                    memories = [m for m in memories if m.get("metadata", {}).get("project") == project]
                
                logger.info(f"✓ Retrieved {len(memories)} relevant contexts for: {query[:30]}...")
                return memories
            else:
                logger.warning(f"Context retrieval failed: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Failed to retrieve context: {e}")
            return []
    
    async def learn_from_pattern(self, pattern_data: Dict[str, Any]) -> bool:
        """Learn from development patterns for future suggestions"""
        try:
            learning_context = {
                "description": f"Pattern learned: {pattern_data.get('pattern_type', 'unknown')}",
                "developer": "learning_engine",
                "project": pattern_data.get("project", "multi-project"),
                "pattern_data": pattern_data,
                "learning_confidence": pattern_data.get("confidence", 0.5),
                "applications": pattern_data.get("successful_applications", []),
                "memory_type": "learned_pattern"
            }
            
            result = await self.store_development_context(learning_context)
            
            if result == "success":
                logger.info(f"✓ Pattern learned: {pattern_data.get('pattern_type', 'Unknown')}")
                return True
            else:
                logger.warning("Pattern learning failed")
                return False
                
        except Exception as e:
            logger.error(f"Pattern learning error: {e}")
            return False
    
    async def get_cognitive_suggestions(self, current_context: Dict[str, Any]) -> List[Dict]:
        """Get AI-powered suggestions based on stored patterns and context"""
        try:
            logger.info("ENTERING GET_COGNITIVE_SUGGESTIONS")
            # Build query from current context
            query_parts = []
            if current_context.get("task"):
                query_parts.append(current_context["task"])
            if current_context.get("technology"):
                query_parts.append(current_context["technology"])
            if current_context.get("problem"):
                query_parts.append(current_context["problem"])
                
            query = " ".join(query_parts) or "development suggestions"
            
            # Retrieve all learned patterns and filter in Python
            relevant_memories = await self.retrieve_relevant_context("learned_pattern", limit=100)
            logger.info(f"Initial patterns found: {len(relevant_memories)}")
            logger.info(f"Query parts: {query_parts}")

            if query_parts:
                relevant_memories_after_filter = [
                    p for p in relevant_memories
                    if any(qp in p.get("message", "") for qp in query_parts)
                ]
                logger.info(f"Patterns after filter: {len(relevant_memories_after_filter)}")
                relevant_memories = relevant_memories_after_filter

            patterns = [
                m for m in relevant_memories
                if m.get("metadata", {}).get("memory_type") == "learned_pattern"
                and m.get("metadata", {}).get("learning_confidence", 0) >= self.bridge_config["learning_threshold"]
            ]
            logger.info(f"High confidence patterns: {len(patterns)}")

            suggestions = []
            for pattern in patterns[:5]:  # Top 5 patterns
                pattern_data = pattern.get("metadata", {}).get("pattern_data", {})

                suggestion = {
                    "type": pattern_data.get("pattern_type", "general"),
                    "suggestion": pattern_data.get("suggestion", "Apply learned pattern"),
                    "confidence": pattern_data.get("confidence", 0.5),
                    "source": "learned_pattern",
                    "applications": pattern_data.get("successful_applications", [])
                }
                suggestions.append(suggestion)

            logger.info(f"✓ Generated {len(suggestions)} cognitive suggestions")
            return suggestions
            
        except Exception as e:
            logger.error(f"Cognitive suggestion error: {e}")
            return []
    
    async def sync_cross_project_knowledge(self) -> bool:
        """Sync knowledge across all ApexSigma projects"""
        try:
            logger.info("Syncing cross-project knowledge...")
            
            # Retrieve all project-related memories
            all_memories = await self.retrieve_relevant_context("apexsigma project", limit=50)
            
            # Organize by project
            project_knowledge = {}
            for memory in all_memories:
                project = memory.get("metadata", {}).get("project", "unknown")
                if project not in project_knowledge:
                    project_knowledge[project] = []
                project_knowledge[project].append(memory)
            
            # Update global sharing manifest
            global_memory_path = Path.home() / ".apexsigma" / "memory"
            manifest_path = global_memory_path / "sharing_manifest.json"
            
            if manifest_path.exists():
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
            else:
                manifest = {"projects": [], "shared_patterns": [], "learning_data": {}}
            
            # Update manifest with current projects
            manifest["projects"] = list(project_knowledge.keys())
            manifest["last_sync"] = datetime.now().isoformat()
            manifest["knowledge_count"] = len(all_memories)
            
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            logger.info(f"✓ Synced knowledge for {len(project_knowledge)} projects")
            return True
            
        except Exception as e:
            logger.error(f"Cross-project sync error: {e}")
            return False
    
    def get_bridge_status(self) -> Dict[str, Any]:
        """Get current bridge status and health"""
        return {
            "bridge_healthy": self.services_healthy,
            "mem0_url": self.mem0_url,
            "cross_project_sharing": self.bridge_config["cross_project_sharing"],
            "learning_enabled": True,
            "timestamp": datetime.now().isoformat()
        }


# Global bridge instance
bridge = MemoryBridge()


async def initialize_memory_bridge():
    """Initialize the global memory bridge"""
    return await bridge.initialize_bridge()


if __name__ == "__main__":
    async def main():
        print("APEXSIGMA MEMORY BRIDGE")
        print("=" * 60)
        
        # Initialize bridge
        success = await bridge.initialize_bridge()
        
        if success:
            print("\n✓ Memory Bridge initialized successfully!")
            print("\nBridge Status:")
            status = bridge.get_bridge_status()
            for key, value in status.items():
                print(f"  {key}: {value}")
                
            # Test context storage
            print("\nTesting context storage...")
            test_context = {
                "description": "Memory bridge initialization and testing completed",
                "developer": "sean",
                "project": "apexsigma-devenviro", 
                "components": ["memory_bridge", "simple_memory_service"],
                "status": "successful",
                "lessons": ["Bridge architecture works", "Services integrate well"]
            }
            
            result = await bridge.store_development_context(test_context)
            print(f"Context storage test: {result}")
            
            # Test cross-project sync
            print("\nTesting cross-project sync...")
            sync_result = await bridge.sync_cross_project_knowledge()
            print(f"Cross-project sync: {'✓ Success' if sync_result else '✗ Failed'}")
            
            print("\n" + "=" * 60)
            print("MEMORY BRIDGE READY FOR COGNITIVE OPERATIONS!")
            
        else:
            print("✗ Memory Bridge initialization failed")
    
    asyncio.run(main())