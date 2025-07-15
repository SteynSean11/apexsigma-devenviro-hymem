#!/usr/bin/env python3
"""
Test Memory Bridge Functionality
Comprehensive testing of the memory bridge cognitive services
"""

import asyncio
import json
import requests
from memory_bridge import bridge
from datetime import datetime


async def test_bridge_initialization():
    """Test memory bridge initialization"""
    print("Testing Memory Bridge Initialization...")
    print("-" * 50)
    
    try:
        result = await bridge.initialize_bridge()
        
        if result:
            print("SUCCESS: Bridge initialization: SUCCESS")
            
            # Check bridge status
            status = bridge.get_bridge_status()
            print(f"SUCCESS: Bridge healthy: {status['bridge_healthy']}")
            print(f"SUCCESS: Collections configured: {len(status['collections'])}")
            print(f"SUCCESS: Cross-project sharing: {status['cross_project_sharing']}")
            
            return True
        else:
            print("ERROR: Bridge initialization: FAILED")
            return False
            
    except Exception as e:
        print(f"ERROR: Bridge initialization error: {e}")
        return False


async def test_context_storage_and_retrieval():
    """Test context storage and retrieval functionality"""
    print("\nTesting Context Storage & Retrieval...")
    print("-" * 50)
    
    try:
        # Test context storage
        test_contexts = [
            {
                "description": "Memory bridge development completed successfully",
                "developer": "sean",
                "project": "apexsigma-devenviro",
                "components": ["memory_bridge", "qdrant", "mem0"],
                "status": "completed",
                "lessons": ["Bridge architecture effective", "Async operations work well"]
            },
            {
                "description": "Cross-project knowledge sharing system implemented",
                "developer": "sean", 
                "project": "apexsigma-devenviro",
                "components": ["knowledge_sharing", "global_memory"],
                "status": "implemented",
                "lessons": ["Global structure helps", "JSON manifest works"]
            },
            {
                "description": "Cognitive learning patterns established",
                "developer": "system",
                "project": "apexsigma-devenviro", 
                "components": ["pattern_learning", "ai_suggestions"],
                "status": "active",
                "lessons": ["Pattern recognition working", "Confidence thresholds helpful"]
            }
        ]
        
        storage_results = []
        for i, context in enumerate(test_contexts, 1):
            result = await bridge.store_development_context(context)
            storage_results.append(result)
            print(f"SUCCESS: Context {i} storage: {result}")
        
        # Test context retrieval
        print("\nTesting context retrieval...")
        queries = [
            "memory bridge development",
            "cross-project knowledge",
            "cognitive learning patterns"
        ]
        
        retrieval_success = 0
        for query in queries:
            memories = await bridge.retrieve_relevant_context(query, limit=5)
            if memories:
                print(f"SUCCESS: Query '{query}': {len(memories)} results")
                retrieval_success += 1
            else:
                print(f"ERROR: Query '{query}': No results")
        
        if storage_results.count("success") >= 2 and retrieval_success >= 2:
            print("SUCCESS: Context storage & retrieval: SUCCESS")
            return True
        else:
            print("ERROR: Context storage & retrieval: PARTIAL")
            return False
            
    except Exception as e:
        print(f"ERROR: Context testing error: {e}")
        return False


async def test_pattern_learning():
    """Test pattern learning functionality"""
    print("\nTesting Pattern Learning...")
    print("-" * 50)
    
    try:
        # Test pattern learning
        test_patterns = [
            {
                "pattern_type": "async_initialization",
                "confidence": 0.9,
                "suggestion": "Use async/await for service initialization",
                "successful_applications": ["memory_bridge", "service_setup"],
                "project": "apexsigma-devenviro"
            },
            {
                "pattern_type": "error_handling",
                "confidence": 0.85,
                "suggestion": "Implement try/catch with detailed logging",
                "successful_applications": ["api_calls", "service_connections"],
                "project": "apexsigma-devenviro"
            }
        ]
        
        learning_results = []
        for pattern in test_patterns:
            result = await bridge.learn_from_pattern(pattern)
            learning_results.append(result)
            print(f"SUCCESS: Pattern '{pattern['pattern_type']}' learned: {result}")
        
        # Test cognitive suggestions
        print("\nTesting cognitive suggestions...")
        context = {
            "task": "service initialization",
            "technology": "python asyncio",
            "problem": "connection setup"
        }
        
        suggestions = await bridge.get_cognitive_suggestions(context)
        print(f"SUCCESS: Generated {len(suggestions)} suggestions")
        
        for i, suggestion in enumerate(suggestions[:3], 1):
            print(f"  {i}. {suggestion.get('type', 'unknown')}: {suggestion.get('suggestion', 'N/A')}")
        
        if learning_results.count(True) >= 1 and len(suggestions) > 0:
            print("SUCCESS: Pattern learning: SUCCESS")
            return True
        else:
            print("ERROR: Pattern learning: FAILED")
            return False
            
    except Exception as e:
        print(f"ERROR: Pattern learning error: {e}")
        return False


async def test_cross_project_sync():
    """Test cross-project knowledge synchronization"""
    print("\nTesting Cross-Project Sync...")
    print("-" * 50)
    
    try:
        result = await bridge.sync_cross_project_knowledge()
        
        if result:
            print("SUCCESS: Cross-project sync: SUCCESS")
            
            # Check if manifest was created/updated
            from pathlib import Path
            manifest_path = Path.home() / ".apexsigma" / "memory" / "sharing_manifest.json"
            
            if manifest_path.exists():
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
                
                print(f"SUCCESS: Projects in manifest: {len(manifest.get('projects', []))}")
                print(f"SUCCESS: Knowledge entries: {manifest.get('knowledge_count', 0)}")
                print(f"SUCCESS: Last sync: {manifest.get('last_sync', 'Unknown')}")
                
                return True
            else:
                print("ERROR: Manifest file not created")
                return False
        else:
            print("ERROR: Cross-project sync: FAILED")
            return False
            
    except Exception as e:
        print(f"ERROR: Cross-project sync error: {e}")
        return False


async def test_service_connections():
    """Test connections to underlying services"""
    print("\nTesting Service Connections...")
    print("-" * 50)
    
    # Test Qdrant connection
    try:
        response = requests.get("http://localhost:6333/", timeout=5)
        if response.status_code == 200:
            print("SUCCESS: Qdrant service: ACCESSIBLE")
            qdrant_ok = True
        else:
            print(f"ERROR: Qdrant service: ERROR {response.status_code}")
            qdrant_ok = False
    except Exception as e:
        print(f"ERROR: Qdrant service: UNREACHABLE ({e})")
        qdrant_ok = False
    
    # Test Mem0 connection
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("SUCCESS: Mem0 service: ACCESSIBLE")
            mem0_ok = True
        else:
            print(f"ERROR: Mem0 service: ERROR {response.status_code}")
            mem0_ok = False
    except Exception as e:
        print(f"ERROR: Mem0 service: UNREACHABLE ({e})")
        mem0_ok = False
    
    return qdrant_ok and mem0_ok


async def run_comprehensive_test():
    """Run comprehensive memory bridge test suite"""
    print("MEMORY BRIDGE COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    test_results = []
    
    # Test 1: Service connections
    service_test = await test_service_connections()
    test_results.append(("Service Connections", service_test))
    
    # Test 2: Bridge initialization
    init_test = await test_bridge_initialization()
    test_results.append(("Bridge Initialization", init_test))
    
    # Test 3: Context storage and retrieval
    context_test = await test_context_storage_and_retrieval()
    test_results.append(("Context Storage & Retrieval", context_test))
    
    # Test 4: Pattern learning
    pattern_test = await test_pattern_learning()
    test_results.append(("Pattern Learning", pattern_test))
    
    # Test 5: Cross-project sync
    sync_test = await test_cross_project_sync()
    test_results.append(("Cross-Project Sync", sync_test))
    
    # Test summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "PASSED" if result else "FAILED"
        print(f"{test_name:.<30} {status}")
        if result:
            passed += 1
    
    print()
    print(f"TOTAL: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
    
    if passed == total:
        print("SUCCESS: ALL TESTS PASSED - Memory Bridge is fully operational!")
    elif passed >= total * 0.8:
        print("WARNING: MOSTLY WORKING - Minor issues detected")
    else:
        print("ERROR: MAJOR ISSUES - Memory Bridge needs attention")
    
    print()
    print("Memory Bridge testing completed!")
    
    return passed == total


if __name__ == "__main__":
    asyncio.run(run_comprehensive_test())