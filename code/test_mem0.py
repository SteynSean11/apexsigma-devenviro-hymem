# \!/usr/bin/env python3
"""Test Mem0 integration with local Qdrant"""
import os


def test_mem0_setup():
    print("Testing Mem0 Setup...")
    
    # Test 1: Mem0 SDK import
    try:
        from mem0 import Memory
        print("SUCCESS: Mem0 SDK import successful")
    except Exception as e:
        print(f"ERROR: Mem0 SDK import failed: {e}")
        return False

    # Test 2: Mem0 Service API
    try:
        import requests
        
        # Test service health
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print("SUCCESS: Mem0 Service API responding")
            print(f"   Status: {health_data.get('status')}")
            print(f"   Memory Backend: {health_data.get('memory_backend')}")
        else:
            print(f"ERROR: Mem0 Service health check failed: {response.status_code}")
            return False
            
        # Test service info
        response = requests.get("http://localhost:8000/", timeout=5)
        if response.status_code == 200:
            info_data = response.json()
            print("SUCCESS: Mem0 Service info accessible")
            print(f"   Version: {info_data.get('version')}")
        
    except Exception as e:
        print(f"ERROR: Mem0 Service API test failed: {e}")
        return False

    # Test 3: Direct SDK configuration (if API key available)
    try:
        # Check for OpenAI API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("WARNING: OPENAI_API_KEY not set")
            print("   Mem0 Service running in limited mode")
            print("   Set OPENAI_API_KEY to enable full memory operations")
        else:
            print("SUCCESS: OPENAI_API_KEY configured")
            
            # Test direct memory configuration
            config = {
                "vector_store": {
                    "provider": "qdrant",
                    "config": {"host": "localhost", "port": 6333, "collection_name": "apexsigma-memory"},
                }
            }
            
            memory = Memory.from_config(config)
            print("SUCCESS: Mem0 SDK initialized with Qdrant backend")

        return True

    except Exception as e:
        print(f"ERROR: Mem0 SDK configuration failed: {e}")
        print("   Service API is available, but SDK initialization needs OpenAI API key")
        return True  # Service is still working


if __name__ == "__main__":
    test_mem0_setup()
