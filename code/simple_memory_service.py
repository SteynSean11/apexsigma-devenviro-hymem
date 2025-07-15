#!/usr/bin/env python3
"""
Simple Memory Service - No External Dependencies
Basic memory storage and retrieval without complex configurations
"""

import json
import uuid
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ApexSigma Simple Memory Service",
    description="Basic Memory API for ApexSigma DevEnviro",
    version="1.0.0"
)

# Database path
DB_PATH = Path("~/.apexsigma/memory/simple_memory.db").expanduser()

# Pydantic models
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


def initialize_database():
    """Initialize SQLite database for memory storage"""
    try:
        # Ensure directory exists
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        # Create database and table
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                message TEXT NOT NULL,
                user_id TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info(f"✓ Database initialized at {DB_PATH}")
        return True
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return False


def add_memory_to_db(message: str, user_id: str, metadata: Dict[str, Any]) -> str:
    """Add memory to database"""
    memory_id = str(uuid.uuid4())
    
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO memories (id, message, user_id, metadata, created_at)
        VALUES (?, ?, ?, ?, ?)
    ''', (
        memory_id,
        message,
        user_id,
        json.dumps(metadata),
        datetime.now().isoformat()
    ))
    
    conn.commit()
    conn.close()
    
    return memory_id


def search_memories_in_db(query: str, user_id: str, limit: int) -> List[Dict[str, Any]]:
    """Search memories in database"""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    # Simple text search
    cursor.execute('''
        SELECT id, message, user_id, metadata, created_at
        FROM memories
        WHERE (message LIKE ? OR metadata LIKE ?)
        AND (user_id = ? OR user_id = 'system')
        ORDER BY created_at DESC
        LIMIT ?
    ''', (f'%{query}%', f'%{query}%', user_id, limit))
    
    results = []
    for row in cursor.fetchall():
        try:
            metadata = json.loads(row[3]) if row[3] else {}
        except:
            metadata = {}
            
        results.append({
            "id": row[0],
            "message": row[1],
            "user_id": row[2],
            "metadata": metadata,
            "created_at": row[4]
        })
    
    conn.close()
    return results


@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    logger.info("🚀 Starting Simple Memory Service...")
    success = initialize_database()
    if success:
        logger.info("✓ Simple Memory Service ready")
    else:
        logger.error("Database initialization failed")


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "ApexSigma Simple Memory Service", "status": "running"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    db_status = "connected" if DB_PATH.exists() else "disconnected"
    
    return {
        "status": "healthy",
        "service": "ApexSigma Simple Memory Service",
        "memory_backend": db_status,
        "database_path": str(DB_PATH),
        "version": "1.0.0"
    }


@app.post("/memory/add", response_model=MemoryResponse)
async def add_memory(request: MemoryRequest):
    """Add a new memory"""
    try:
        memory_id = add_memory_to_db(
            message=request.message,
            user_id=request.user_id,
            metadata=request.metadata
        )
        
        return MemoryResponse(
            success=True,
            message="Memory added successfully",
            memory_id=memory_id,
            data={
                "id": memory_id,
                "message": request.message,
                "user_id": request.user_id,
                "metadata": request.metadata
            }
        )
        
    except Exception as e:
        logger.error(f"Error adding memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/search", response_model=SearchResponse)
async def search_memories(request: SearchRequest):
    """Search memories"""
    try:
        results = search_memories_in_db(
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


@app.delete("/memory/{memory_id}")
async def delete_memory(memory_id: str):
    """Delete a memory"""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM memories WHERE id = ?', (memory_id,))
        deleted_count = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        if deleted_count > 0:
            return {"success": True, "message": "Memory deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Memory not found")
            
    except Exception as e:
        logger.error(f"Error deleting memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory/stats")
async def get_memory_stats():
    """Get memory statistics"""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM memories')
        total_memories = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(DISTINCT user_id) FROM memories')
        unique_users = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "total_memories": total_memories,
            "unique_users": unique_users,
            "database_path": str(DB_PATH)
        }
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("APEXSIGMA SIMPLE MEMORY SERVICE")
    print("=" * 60)
    print("Starting simple memory service on http://localhost:8000")
    print("No external API keys required!")
    print()
    
    uvicorn.run(app, host="0.0.0.0", port=8000)