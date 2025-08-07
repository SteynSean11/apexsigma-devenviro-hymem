#!/usr/bin/env python3
"""
Test Memory Bridge Functionality
Comprehensive testing of the memory bridge cognitive services
"""

import asyncio
import json
import requests
import pytest
from memory_bridge import bridge
from datetime import datetime
from pathlib import Path

@pytest.mark.asyncio
async def test_service_connections():
    """Test connections to underlying services"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        assert response.status_code == 200
        print("SUCCESS: Mem0 service: ACCESSIBLE")
    except Exception as e:
        pytest.fail(f"Mem0 service: UNREACHABLE ({e})")

@pytest.mark.asyncio
async def test_bridge_initialization():
    """Test memory bridge initialization"""
    try:
        result = await bridge.initialize_bridge()
        assert result is True, "Bridge initialization should return True"
        
        status = bridge.get_bridge_status()
        assert status['bridge_healthy'] is True
        assert status['cross_project_sharing'] is True
        print("SUCCESS: Bridge initialization and status are valid")
            
    except Exception as e:
        pytest.fail(f"Bridge initialization error: {e}")


@pytest.mark.asyncio
async def test_context_storage_and_retrieval():
    """Test context storage and retrieval functionality"""
    try:
        # Test context storage
        test_contexts = [
            {
                "description": "Memory bridge development completed successfully",
                "developer": "sean",
                "project": "apexsigma-devenviro",
                "components": ["memory_bridge", "simple_memory_service"],
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
        
        for context in test_contexts:
            result = await bridge.store_development_context(context)
            assert result == "success"
        
        # Test context retrieval
        queries = [
            "Memory bridge development",
            "knowledge sharing",
            "Cognitive learning"
        ]
        
        for query in queries:
            memories = await bridge.retrieve_relevant_context(query, limit=5)
            assert len(memories) > 0, f"Query '{query}' should return at least one result"
            print(f"SUCCESS: Query '{query}': {len(memories)} results")

        # Test retrieval with no results
        memories = await bridge.retrieve_relevant_context("non_existent_query_xyz", limit=5)
        assert len(memories) == 0, "Query for non-existent term should return 0 results"

    except Exception as e:
        pytest.fail(f"Context testing error: {e}")


@pytest.mark.asyncio
async def test_pattern_learning():
    """Test pattern learning functionality"""
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
        
        for pattern in test_patterns:
            result = await bridge.learn_from_pattern(pattern)
            assert result is True

        # Cognitive suggestions are not tested in the simplified service
        pass
        
    except Exception as e:
        pytest.fail(f"Pattern learning error: {e}")


@pytest.mark.asyncio
async def test_cross_project_sync():
    """Test cross-project knowledge synchronization"""
    try:
        result = await bridge.sync_cross_project_knowledge()
        assert result is True

        # Check if manifest was created/updated
        manifest_path = Path.home() / ".apexsigma" / "memory" / "sharing_manifest.json"
        assert manifest_path.exists()

        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        assert "last_sync" in manifest
        
    except Exception as e:
        pytest.fail(f"Cross-project sync error: {e}")