#!/usr/bin/env python3
"""
Mem0 Service - Autonomous Memory API Server
Runs as a standalone service to provide memory capabilities
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from mem0 import Memory
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ApexSigma Mem0 Service",
    description="Autonomous Memory API for ApexSigma DevEnviro",
    version="1.0.0"
)

# Global memory instance
memory_instance: Optional[Memory] = None

# Pydantic models for API
class MemoryRequest(BaseModel):
    message: str
    user_id: str = "default"
    metadata: Dict[str, Any] = {}

class MemoryResponse(BaseModel):
    success: bool
    message: str
    memory_id: Optional[str] = None
    data: Optional[Dict[str, Any]] = None

class SearchRequest(BaseModel):
    query: str
    user_id: str = "default"
    limit: int = 10

class SearchResponse(BaseModel):
    success: bool
    memories: List[Dict[str, Any]]
    count: int

def initialize_memory():
    """Initialize Mem0 with Qdrant backend"""
    global memory_instance

    # Set a dummy API key if not present
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "dummy_key"

    try:
        # Use in-memory Qdrant
        config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "location": ":memory:",
                    "collection_name": "apexsigma-memory"
                }
            }
        }
        
        memory_instance = Memory.from_config(config)
        logger.info("✅ Mem0 initialized successfully with in-memory Qdrant backend")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize Mem0: {e}")
        memory_instance = None
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize memory on startup"""
    logger.info("🚀 Starting ApexSigma Mem0 Service...")
    success = initialize_memory()
    if not success:
        logger.warning("⚠️ Memory initialization failed - service will run in limited mode")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    memory_status = "connected" if memory_instance else "disconnected"
    
    return {
        "status": "healthy",
        "service": "ApexSigma Mem0 Service",
        "memory_backend": memory_status,
        "qdrant_host": os.getenv("QDRANT_HOST", "localhost"),
        "qdrant_port": os.getenv("QDRANT_PORT", "6333")
    }

@app.post("/memory/add", response_model=MemoryResponse)
async def add_memory(request: MemoryRequest):
    """Add a new memory"""
    if not memory_instance:
        raise HTTPException(status_code=503, detail="Memory service not initialized")
    
    try:
        result = memory_instance.add(
            message=request.message,
            user_id=request.user_id,
            metadata=request.metadata
        )
        
        return MemoryResponse(
            success=True,
            message="Memory added successfully",
            memory_id=result.get("id"),
            data=result
        )
        
    except Exception as e:
        logger.error(f"Error adding memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory/search", response_model=SearchResponse)
async def search_memories(request: SearchRequest):
    """Search memories"""
    if not memory_instance:
        raise HTTPException(status_code=503, detail="Memory service not initialized")
    
    try:
        results = memory_instance.search(
            query=request.query,
            user_id=request.user_id,
            limit=request.limit
        )
        
        return SearchResponse(
            success=True,
            memories=results,
            count=len(results)
        )
        
    except Exception as e:
        logger.error(f"Error searching memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory/user/{user_id}")
async def get_user_memories(user_id: str, limit: int = 50):
    """Get all memories for a user"""
    if not memory_instance:
        raise HTTPException(status_code=503, detail="Memory service not initialized")
    
    try:
        results = memory_instance.get_all(user_id=user_id, limit=limit)
        
        return {
            "success": True,
            "user_id": user_id,
            "memories": results,
            "count": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error getting user memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/memory/{memory_id}")
async def delete_memory(memory_id: str):
    """Delete a specific memory"""
    if not memory_instance:
        raise HTTPException(status_code=503, detail="Memory service not initialized")
    
    try:
        memory_instance.delete(memory_id=memory_id)
        
        return {
            "success": True,
            "message": f"Memory {memory_id} deleted successfully"
        }
        
    except Exception as e:
        logger.error(f"Error deleting memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint with service info"""
    return {
        "service": "ApexSigma Mem0 Service",
        "version": "1.0.0",
        "description": "Autonomous Memory API for cognitive collaboration",
        "endpoints": {
            "health": "/health",
            "add_memory": "/memory/add",
            "search_memories": "/memory/search",
            "get_user_memories": "/memory/user/{user_id}",
            "delete_memory": "/memory/{memory_id}"
        }
    }

if __name__ == "__main__":
    # Run the service
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"🚀 Starting Mem0 service on {host}:{port}")
    uvicorn.run(app, host=host, port=port)