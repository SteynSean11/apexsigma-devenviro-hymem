"""
ApexSigma Hybrid Memory System
RAG + Vector Embeddings + Knowledge Graph + Multi-Level Intelligence

Date: July 15, 2025
Goal: Superior memory system combining best approaches for <25ms responses
"""

import asyncio
import sqlite3
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import networkx as nx
from dataclasses import dataclass, asdict
import logging
import time
from collections import deque, defaultdict
import pickle
import threading
from functools import lru_cache
import hashlib

# Local AI engine import
import sys
sys.path.append(str(Path(__file__).parent.parent / "ai_training"))
from local_ai_engine import ApexSigmaLocalAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Memory:
    """Base memory structure"""
    id: str
    content: str
    memory_type: str
    importance: float
    created_at: datetime
    last_accessed: datetime
    access_count: int
    tags: List[str]
    metadata: Dict[str, Any]
    embeddings: Optional[List[float]] = None
    relationships: Optional[List[str]] = None

@dataclass
class DevelopmentContext:
    """Development-specific context structure"""
    project_id: str
    file_path: str
    function_name: str
    line_number: int
    code_snippet: str
    error_message: str = ""
    task_description: str = ""
    related_files: List[str] = None

class WorkingMemory:
    """Short-term, session-aware memory for immediate context"""
    
    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self.temporal_events = deque(maxlen=max_size)
        self.session_context = {}
        self.active_files = set()
        self.current_task = None
        
    async def store_session_event(self, event: Dict, ai_engine: ApexSigmaLocalAI):
        """Store immediate session context with AI analysis"""
        
        # Generate embeddings using local AI
        embedding_result = await ai_engine.ask(
            event.get("content", ""), 
            {"type": "embedding_generation"}
        )
        
        embeddings = embedding_result.get("insights", {}).get("embeddings", [])
        
        # Extract entities and relationships
        entities = await self.extract_entities(event)
        
        # Calculate relevance score
        relevance_score = await self.calculate_relevance(event)
        
        session_memory = {
            'event': event,
            'embeddings': embeddings,
            'entities': entities,
            'relevance': relevance_score,
            'timestamp': datetime.now(),
            'session_id': self.session_context.get('session_id', 'default')
        }
        
        self.temporal_events.append(session_memory)
        
        # Update session context
        if event.get("type") == "file_opened":
            self.active_files.add(event.get("file_path", ""))
        elif event.get("type") == "task_started":
            self.current_task = event.get("task_description", "")
    
    async def extract_entities(self, event: Dict) -> List[str]:
        """Extract development entities from event"""
        entities = []
        
        content = event.get("content", "")
        
        # Simple entity extraction (in production, use NER models)
        if "def " in content:
            entities.append("function_definition")
        if "class " in content:
            entities.append("class_definition")
        if "import " in content:
            entities.append("import_statement")
        if "error" in content.lower():
            entities.append("error_event")
        
        return entities
    
    async def calculate_relevance(self, event: Dict) -> float:
        """Calculate relevance score for working memory event"""
        
        base_score = 0.5
        
        # Higher relevance for errors and important events
        if event.get("type") == "error":
            base_score += 0.4
        elif event.get("type") == "task_completed":
            base_score += 0.3
        elif event.get("type") == "file_modified":
            base_score += 0.2
        
        # Boost relevance for current task context
        if self.current_task and self.current_task in event.get("content", ""):
            base_score += 0.2
        
        return min(base_score, 1.0)
    
    def get_recent_context(self, limit: int = 10) -> List[Dict]:
        """Get recent session context"""
        return list(self.temporal_events)[-limit:]

class FactualMemory:
    """Long-term structured knowledge storage"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.knowledge_graph = nx.MultiDiGraph()
        self.vector_index = {}
        self.ai_engine = None
        
        # Initialize database
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for factual storage"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS factual_memory (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    importance REAL NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    last_accessed TIMESTAMP NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    tags TEXT,
                    metadata TEXT,
                    embeddings BLOB,
                    relationships TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memory_type ON factual_memory(memory_type)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_importance ON factual_memory(importance)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_last_accessed ON factual_memory(last_accessed)
            """)
    
    async def store_factual_knowledge(self, fact: Memory, ai_engine: ApexSigmaLocalAI):
        """Store structured development knowledge"""
        
        # Generate embeddings if not provided
        if not fact.embeddings:
            embedding_result = await ai_engine.ask(
                fact.content, 
                {"type": "factual_embedding"}
            )
            fact.embeddings = embedding_result.get("insights", {}).get("embeddings", [])
        
        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO factual_memory 
                (id, content, memory_type, importance, created_at, last_accessed, 
                 access_count, tags, metadata, embeddings, relationships)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                fact.id,
                fact.content,
                fact.memory_type,
                fact.importance,
                fact.created_at,
                fact.last_accessed,
                fact.access_count,
                json.dumps(fact.tags),
                json.dumps(fact.metadata),
                pickle.dumps(fact.embeddings) if fact.embeddings else None,
                json.dumps(fact.relationships) if fact.relationships else None
            ))
        
        # Add to knowledge graph
        await self.add_to_knowledge_graph(fact)
        
        # Update vector index
        if fact.embeddings:
            self.vector_index[fact.id] = np.array(fact.embeddings)
    
    async def add_to_knowledge_graph(self, fact: Memory):
        """Add factual knowledge to knowledge graph"""
        
        # Add node to graph
        self.knowledge_graph.add_node(
            fact.id,
            content=fact.content,
            memory_type=fact.memory_type,
            importance=fact.importance,
            tags=fact.tags
        )
        
        # Add relationships to related facts
        if fact.relationships:
            for related_id in fact.relationships:
                if self.knowledge_graph.has_node(related_id):
                    self.knowledge_graph.add_edge(
                        fact.id, 
                        related_id,
                        relationship_type="related_to",
                        weight=0.7
                    )
    
    async def search_factual_memory(self, query: str, ai_engine: ApexSigmaLocalAI, top_k: int = 10) -> List[Memory]:
        """Search factual memory using semantic similarity"""
        
        # Generate query embedding
        query_result = await ai_engine.ask(query, {"type": "search_embedding"})
        query_embedding = np.array(query_result.get("insights", {}).get("embeddings", []))
        
        if len(query_embedding) == 0:
            return []
        
        # Calculate similarities with stored embeddings
        similarities = []
        for memory_id, memory_embedding in self.vector_index.items():
            similarity = np.dot(query_embedding, memory_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(memory_embedding)
            )
            similarities.append((memory_id, similarity))
        
        # Sort by similarity and get top results
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_results = similarities[:top_k]
        
        # Retrieve full memory objects
        memories = []
        for memory_id, score in top_results:
            memory = await self.get_memory_by_id(memory_id)
            if memory:
                memory.metadata["similarity_score"] = score
                memories.append(memory)
        
        return memories
    
    async def get_memory_by_id(self, memory_id: str) -> Optional[Memory]:
        """Retrieve memory by ID"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM factual_memory WHERE id = ?
            """, (memory_id,))
            
            row = cursor.fetchone()
            if row:
                return Memory(
                    id=row[0],
                    content=row[1],
                    memory_type=row[2],
                    importance=row[3],
                    created_at=datetime.fromisoformat(row[4]),
                    last_accessed=datetime.fromisoformat(row[5]),
                    access_count=row[6],
                    tags=json.loads(row[7]) if row[7] else [],
                    metadata=json.loads(row[8]) if row[8] else {},
                    embeddings=pickle.loads(row[9]) if row[9] else None,
                    relationships=json.loads(row[10]) if row[10] else None
                )
        
        return None

class EpisodicMemory:
    """Memory of specific development interactions and experiences"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.episode_graph = nx.DiGraph()
        self.sequence_patterns = defaultdict(list)
        self.pattern_detector = PatternDetector()
        
        self.init_database()
    
    def init_database(self):
        """Initialize database for episodic memory"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS episodic_memory (
                    id TEXT PRIMARY KEY,
                    episode_type TEXT NOT NULL,
                    start_time TIMESTAMP NOT NULL,
                    end_time TIMESTAMP,
                    context TEXT NOT NULL,
                    outcome TEXT,
                    patterns TEXT,
                    sequence_data BLOB,
                    success_rating REAL DEFAULT 0.5
                )
            """)
    
    async def store_development_episode(self, episode: Dict, ai_engine: ApexSigmaLocalAI):
        """Store complete development interaction episode"""
        
        episode_id = f"episode_{int(time.time())}"
        
        # Analyze episode patterns using AI
        pattern_analysis = await ai_engine.ask(
            f"Analyze development episode: {episode.get('description', '')}",
            {"type": "pattern_analysis", "episode_data": episode}
        )
        
        patterns = pattern_analysis.get("insights", {}).get("related_concepts", [])
        
        # Store episode in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO episodic_memory 
                (id, episode_type, start_time, end_time, context, outcome, patterns, sequence_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                episode_id,
                episode.get("type", "general"),
                episode.get("start_time", datetime.now()),
                episode.get("end_time"),
                json.dumps(episode.get("context", {})),
                episode.get("outcome", ""),
                json.dumps(patterns),
                pickle.dumps(episode.get("sequence", []))
            ))
        
        # Add to episode graph
        await self.add_episode_to_graph(episode_id, episode)
        
        # Learn patterns
        await self.learn_episode_patterns(episode_id, episode, patterns)
    
    async def add_episode_to_graph(self, episode_id: str, episode: Dict):
        """Add episode to the episode graph"""
        
        self.episode_graph.add_node(
            episode_id,
            episode_type=episode.get("type", "general"),
            start_time=episode.get("start_time", datetime.now()),
            outcome_success=episode.get("success", 0.5)
        )
        
        # Link to previous related episodes
        related_episodes = await self.find_related_episodes(episode)
        for related_id in related_episodes[:3]:  # Limit to 3 most related
            self.episode_graph.add_edge(related_id, episode_id, relationship="temporal_sequence")
    
    async def find_related_episodes(self, episode: Dict) -> List[str]:
        """Find episodes related to current episode"""
        
        # Simple similarity based on episode type and context
        related = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT id, context FROM episodic_memory 
                WHERE episode_type = ? 
                ORDER BY start_time DESC LIMIT 10
            """, (episode.get("type", "general"),))
            
            for row in cursor.fetchall():
                episode_id, context_json = row
                try:
                    stored_context = json.loads(context_json)
                    # Simple similarity check
                    if self.calculate_context_similarity(episode.get("context", {}), stored_context) > 0.5:
                        related.append(episode_id)
                except:
                    continue
        
        return related
    
    def calculate_context_similarity(self, context1: Dict, context2: Dict) -> float:
        """Calculate similarity between two episode contexts"""
        
        # Simple similarity based on shared keys and values
        shared_keys = set(context1.keys()) & set(context2.keys())
        if not shared_keys:
            return 0.0
        
        similarity_sum = 0.0
        for key in shared_keys:
            if context1[key] == context2[key]:
                similarity_sum += 1.0
            elif isinstance(context1[key], str) and isinstance(context2[key], str):
                # Simple string similarity
                common_words = set(context1[key].lower().split()) & set(context2[key].lower().split())
                similarity_sum += len(common_words) / max(len(context1[key].split()), len(context2[key].split()))
        
        return similarity_sum / len(shared_keys)
    
    async def learn_episode_patterns(self, episode_id: str, episode: Dict, patterns: List[str]):
        """Learn patterns from episode"""
        
        episode_type = episode.get("type", "general")
        
        # Store sequence patterns
        if "sequence" in episode:
            self.sequence_patterns[episode_type].append(episode["sequence"])
        
        # Update pattern detector
        await self.pattern_detector.learn_pattern(episode_type, patterns, episode.get("success", 0.5))

class PatternDetector:
    """Detects and learns development patterns"""
    
    def __init__(self):
        self.pattern_database = defaultdict(lambda: {"count": 0, "success_rate": 0.5, "examples": []})
    
    async def learn_pattern(self, pattern_type: str, patterns: List[str], success_rate: float):
        """Learn new pattern"""
        
        for pattern in patterns:
            pattern_key = f"{pattern_type}:{pattern}"
            entry = self.pattern_database[pattern_key]
            
            # Update pattern statistics
            total_count = entry["count"] + 1
            current_success = entry["success_rate"] * entry["count"]
            new_success_rate = (current_success + success_rate) / total_count
            
            entry["count"] = total_count
            entry["success_rate"] = new_success_rate
            entry["examples"].append({
                "timestamp": datetime.now().isoformat(),
                "success_rate": success_rate
            })
            
            # Keep only recent examples
            entry["examples"] = entry["examples"][-10:]
    
    async def get_successful_patterns(self, pattern_type: str, min_success_rate: float = 0.7) -> List[Dict]:
        """Get patterns with high success rates"""
        
        successful_patterns = []
        
        for pattern_key, entry in self.pattern_database.items():
            if pattern_key.startswith(f"{pattern_type}:") and entry["success_rate"] >= min_success_rate:
                successful_patterns.append({
                    "pattern": pattern_key.split(":", 1)[1],
                    "success_rate": entry["success_rate"],
                    "usage_count": entry["count"]
                })
        
        # Sort by success rate and usage count
        successful_patterns.sort(key=lambda x: (x["success_rate"], x["usage_count"]), reverse=True)
        
        return successful_patterns

class SemanticMemory:
    """Abstract conceptual knowledge and learned patterns"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.concept_graph = nx.Graph()
        self.abstraction_engine = AbstractionEngine()
        
        self.init_database()
    
    def init_database(self):
        """Initialize semantic memory database"""
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS semantic_memory (
                    id TEXT PRIMARY KEY,
                    concept_name TEXT NOT NULL,
                    abstraction_level INTEGER NOT NULL,
                    concept_data TEXT NOT NULL,
                    related_concepts TEXT,
                    confidence REAL DEFAULT 0.5,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL
                )
            """)
    
    async def learn_semantic_concept(self, concept: Dict, ai_engine: ApexSigmaLocalAI):
        """Learn abstract development concept"""
        
        concept_id = f"concept_{concept.get('name', 'unknown')}_{int(time.time())}"
        
        # Generate concept embeddings and abstractions
        abstraction_result = await ai_engine.ask(
            f"Abstract this development concept: {concept.get('description', '')}",
            {"type": "concept_abstraction", "concept_data": concept}
        )
        
        # Store concept
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO semantic_memory 
                (id, concept_name, abstraction_level, concept_data, related_concepts, confidence, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                concept_id,
                concept.get("name", "unknown"),
                concept.get("abstraction_level", 1),
                json.dumps(concept),
                json.dumps(concept.get("related_concepts", [])),
                concept.get("confidence", 0.5),
                datetime.now(),
                datetime.now()
            ))
        
        # Add to concept graph
        await self.add_concept_to_graph(concept_id, concept)
    
    async def add_concept_to_graph(self, concept_id: str, concept: Dict):
        """Add concept to semantic graph"""
        
        self.concept_graph.add_node(
            concept_id,
            name=concept.get("name", "unknown"),
            abstraction_level=concept.get("abstraction_level", 1),
            confidence=concept.get("confidence", 0.5)
        )
        
        # Link to related concepts
        for related_concept in concept.get("related_concepts", []):
            # Find existing concepts with similar names
            for node_id, node_data in self.concept_graph.nodes(data=True):
                if node_data.get("name", "").lower() == related_concept.lower():
                    self.concept_graph.add_edge(concept_id, node_id, relationship="related")
                    break

class AbstractionEngine:
    """Engine for creating abstractions and generalizations"""
    
    async def abstract_concept(self, concept: Dict) -> List[Dict]:
        """Create abstractions from concrete concept"""
        
        abstractions = []
        
        # Simple abstraction rules (in production, use ML models)
        if "function" in concept.get("name", "").lower():
            abstractions.append({
                "name": "code_structure",
                "abstraction_level": 2,
                "description": "Abstract code organization principle"
            })
        
        if "error" in concept.get("name", "").lower():
            abstractions.append({
                "name": "problem_solving",
                "abstraction_level": 3,
                "description": "General problem-solving approach"
            })
        
        return abstractions

class HybridMemorySystem:
    """Main hybrid memory system combining all memory types"""
    
    def __init__(self, storage_path: str = "memory_data"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Initialize memory layers
        self.working_memory = WorkingMemory()
        self.factual_memory = FactualMemory(str(self.storage_path / "factual.db"))
        self.episodic_memory = EpisodicMemory(str(self.storage_path / "episodic.db"))
        self.semantic_memory = SemanticMemory(str(self.storage_path / "semantic.db"))
        
        # Initialize AI engine
        self.ai_engine = ApexSigmaLocalAI()
        self.initialized = False
        
        # Performance optimization
        self.query_cache = {}
        self.cache_lock = threading.RLock()
        
        logger.info("HybridMemorySystem initialized")
    
    async def initialize(self):
        """Initialize the hybrid memory system"""
        
        if not self.initialized:
            # Initialize AI engine
            success = await self.ai_engine.initialize()
            if not success:
                logger.error("Failed to initialize AI engine")
                return False
            
            self.initialized = True
            logger.info("HybridMemorySystem ready")
        
        return True
    
    async def store_memory(self, content: str, memory_type: str, context: Dict = None) -> str:
        """Store memory in appropriate layer"""
        
        if not self.initialized:
            await self.initialize()
        
        memory_id = hashlib.sha256(f"{content}_{datetime.now().isoformat()}".encode()).hexdigest()[:16]
        
        # Determine importance using AI
        importance_result = await self.ai_engine.ask(
            f"Rate the importance of this development memory: {content}",
            context or {}
        )
        importance = min(importance_result.get("insights", {}).get("primary_suggestion", "0.5"), 1.0)
        
        # Create memory object
        memory = Memory(
            id=memory_id,
            content=content,
            memory_type=memory_type,
            importance=float(importance) if isinstance(importance, (int, float)) else 0.5,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            access_count=0,
            tags=context.get("tags", []) if context else [],
            metadata=context or {}
        )
        
        # Store in appropriate layer based on type
        if memory_type == "session":
            await self.working_memory.store_session_event(asdict(memory), self.ai_engine)
        
        elif memory_type in ["fact", "knowledge", "code"]:
            await self.factual_memory.store_factual_knowledge(memory, self.ai_engine)
        
        elif memory_type == "episode":
            episode_data = {
                "type": "development_episode",
                "description": content,
                "context": context or {},
                "start_time": datetime.now()
            }
            await self.episodic_memory.store_development_episode(episode_data, self.ai_engine)
        
        elif memory_type == "concept":
            concept_data = {
                "name": content.split()[0] if content.split() else "unknown",
                "description": content,
                "abstraction_level": 1
            }
            await self.semantic_memory.learn_semantic_concept(concept_data, self.ai_engine)
        
        logger.info(f"Stored {memory_type} memory: {memory_id}")
        return memory_id
    
    @lru_cache(maxsize=100)
    async def intelligent_search(self, query: str, context: Dict = None, max_results: int = 10) -> List[Dict]:
        """Intelligent search across all memory layers"""
        
        start_time = time.perf_counter()
        
        if not self.initialized:
            await self.initialize()
        
        # Generate cache key
        cache_key = hashlib.sha256(f"{query}_{json.dumps(context, sort_keys=True) if context else ''}".encode()).hexdigest()
        
        with self.cache_lock:
            if cache_key in self.query_cache:
                cached_result = self.query_cache[cache_key]
                if datetime.now() - cached_result["timestamp"] < timedelta(minutes=5):
                    logger.info(f"Cache hit for query: {query[:50]}...")
                    return cached_result["results"]
        
        # Parallel search across memory layers
        search_tasks = [
            self.search_working_memory(query),
            self.search_factual_memory(query),
            self.search_episodic_memory(query),
            self.search_semantic_memory(query)
        ]
        
        layer_results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # Combine and rank results
        all_results = []
        
        for i, results in enumerate(layer_results):
            if isinstance(results, Exception):
                logger.warning(f"Search failed in layer {i}: {results}")
                continue
            
            for result in results:
                result["memory_layer"] = ["working", "factual", "episodic", "semantic"][i]
                all_results.append(result)
        
        # Rank results using AI
        ranked_results = await self.rank_search_results(all_results, query, context)
        
        # Cache results
        with self.cache_lock:
            self.query_cache[cache_key] = {
                "results": ranked_results[:max_results],
                "timestamp": datetime.now()
            }
        
        elapsed_time = (time.perf_counter() - start_time) * 1000
        logger.info(f"Intelligent search completed in {elapsed_time:.1f}ms")
        
        return ranked_results[:max_results]
    
    async def search_working_memory(self, query: str) -> List[Dict]:
        """Search working memory"""
        
        recent_context = self.working_memory.get_recent_context()
        
        # Simple relevance matching
        relevant_memories = []
        for memory in recent_context:
            event = memory.get("event", {})
            content = event.get("content", "")
            
            # Basic keyword matching (in production, use embeddings)
            if any(word.lower() in content.lower() for word in query.split()):
                relevant_memories.append({
                    "id": f"working_{memory.get('timestamp', datetime.now()).isoformat()}",
                    "content": content,
                    "relevance_score": memory.get("relevance", 0.5),
                    "memory_type": "working",
                    "timestamp": memory.get("timestamp", datetime.now()).isoformat()
                })
        
        return relevant_memories
    
    async def search_factual_memory(self, query: str) -> List[Dict]:
        """Search factual memory"""
        
        try:
            memories = await self.factual_memory.search_factual_memory(query, self.ai_engine)
            
            results = []
            for memory in memories:
                results.append({
                    "id": memory.id,
                    "content": memory.content,
                    "relevance_score": memory.metadata.get("similarity_score", 0.5),
                    "memory_type": memory.memory_type,
                    "importance": memory.importance,
                    "last_accessed": memory.last_accessed.isoformat()
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Factual memory search failed: {e}")
            return []
    
    async def search_episodic_memory(self, query: str) -> List[Dict]:
        """Search episodic memory"""
        
        try:
            # Simple episodic search (in production, use more sophisticated methods)
            with sqlite3.connect(self.episodic_memory.db_path) as conn:
                cursor = conn.execute("""
                    SELECT id, episode_type, context, outcome, success_rating 
                    FROM episodic_memory 
                    WHERE context LIKE ? OR outcome LIKE ?
                    ORDER BY success_rating DESC, start_time DESC
                    LIMIT 10
                """, (f"%{query}%", f"%{query}%"))
                
                results = []
                for row in cursor.fetchall():
                    episode_id, episode_type, context, outcome, success_rating = row
                    results.append({
                        "id": episode_id,
                        "content": f"Episode: {episode_type} - {outcome}",
                        "relevance_score": success_rating or 0.5,
                        "memory_type": "episodic",
                        "episode_type": episode_type,
                        "context": context
                    })
                
                return results
                
        except Exception as e:
            logger.error(f"Episodic memory search failed: {e}")
            return []
    
    async def search_semantic_memory(self, query: str) -> List[Dict]:
        """Search semantic memory"""
        
        try:
            # Simple semantic search
            with sqlite3.connect(self.semantic_memory.db_path) as conn:
                cursor = conn.execute("""
                    SELECT id, concept_name, concept_data, confidence 
                    FROM semantic_memory 
                    WHERE concept_name LIKE ? OR concept_data LIKE ?
                    ORDER BY confidence DESC, updated_at DESC
                    LIMIT 10
                """, (f"%{query}%", f"%{query}%"))
                
                results = []
                for row in cursor.fetchall():
                    concept_id, concept_name, concept_data, confidence = row
                    results.append({
                        "id": concept_id,
                        "content": f"Concept: {concept_name}",
                        "relevance_score": confidence or 0.5,
                        "memory_type": "semantic",
                        "concept_name": concept_name,
                        "concept_data": concept_data
                    })
                
                return results
                
        except Exception as e:
            logger.error(f"Semantic memory search failed: {e}")
            return []
    
    async def rank_search_results(self, results: List[Dict], query: str, context: Dict = None) -> List[Dict]:
        """Rank search results using AI"""
        
        if not results:
            return []
        
        # Use AI to rank results
        ranking_context = {
            "query": query,
            "context": context or {},
            "results_count": len(results)
        }
        
        ranking_result = await self.ai_engine.ask(
            f"Rank these search results for relevance to: {query}",
            ranking_context
        )
        
        # Apply AI-suggested ranking (simplified)
        # In production, this would use sophisticated ranking models
        
        # For now, sort by relevance score and memory layer priority
        layer_weights = {
            "working": 1.2,  # Recent context is important
            "factual": 1.0,
            "episodic": 0.8,
            "semantic": 0.6
        }
        
        for result in results:
            layer_weight = layer_weights.get(result.get("memory_layer", "factual"), 1.0)
            result["final_score"] = result.get("relevance_score", 0.5) * layer_weight
        
        # Sort by final score
        results.sort(key=lambda x: x.get("final_score", 0), reverse=True)
        
        return results
    
    async def get_system_stats(self) -> Dict:
        """Get hybrid memory system statistics"""
        
        stats = {
            "initialized": self.initialized,
            "working_memory": {
                "events_count": len(self.working_memory.temporal_events),
                "active_files": len(self.working_memory.active_files),
                "current_task": self.working_memory.current_task
            },
            "cache": {
                "query_cache_size": len(self.query_cache),
                "cache_hit_rate": 0.0  # Would track this in production
            }
        }
        
        # Get AI engine stats
        if self.initialized:
            ai_stats = await self.ai_engine.get_status()
            stats["ai_engine"] = ai_stats
        
        return stats

# Example usage and testing
async def main():
    """Test the hybrid memory system"""
    
    memory_system = HybridMemorySystem()
    
    print("🔄 Initializing Hybrid Memory System...")
    success = await memory_system.initialize()
    
    if not success:
        print("❌ Failed to initialize memory system")
        return
    
    print("✅ Hybrid Memory System ready!")
    
    # Test storing different types of memories
    test_memories = [
        {
            "content": "Fixed database connection timeout issue by increasing connection pool size",
            "type": "fact",
            "context": {"project": "api_service", "component": "database", "tags": ["fix", "performance"]}
        },
        {
            "content": "Debugging session for authentication middleware error",
            "type": "episode", 
            "context": {"project": "web_app", "duration": "30_minutes", "success": True}
        },
        {
            "content": "Error handling pattern for async operations",
            "type": "concept",
            "context": {"abstraction_level": 2, "related": ["async", "exceptions"]}
        }
    ]
    
    # Store memories
    for memory in test_memories:
        memory_id = await memory_system.store_memory(
            memory["content"],
            memory["type"], 
            memory["context"]
        )
        print(f"💾 Stored {memory['type']} memory: {memory_id}")
    
    # Test intelligent search
    search_queries = [
        "database connection issues",
        "authentication problems", 
        "async error handling patterns"
    ]
    
    for query in search_queries:
        print(f"\n🔍 Searching: {query}")
        results = await memory_system.intelligent_search(query, max_results=3)
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. [{result['memory_layer']}] {result['content'][:80]}...")
            print(f"     Score: {result.get('final_score', 0):.2f}")
    
    # Get system stats
    stats = await memory_system.get_system_stats()
    print(f"\n📊 System Stats:")
    print(f"  Working Memory Events: {stats['working_memory']['events_count']}")
    print(f"  Query Cache Size: {stats['cache']['query_cache_size']}")
    print(f"  AI Engine Status: {stats.get('ai_engine', {}).get('status', 'unknown')}")

if __name__ == "__main__":
    asyncio.run(main())