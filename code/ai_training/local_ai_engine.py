"""
ApexSigma Local AI Inference Engine
Serves custom-trained models for development intelligence

Date: July 15, 2025
Goal: Local AI inference for privacy-first development assistance
"""

import asyncio
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
import time
from functools import lru_cache
import sqlite3
from datetime import datetime
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalAIEngine:
    """High-performance local AI inference engine for development intelligence"""
    
    def __init__(self, models_dir: str = "models/apex_ai"):
        self.models_dir = Path(models_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.tokenizers = {}
        self.model_configs = {}
        
        # Performance optimization
        self.inference_cache = {}
        self.cache_size = 1000
        self.performance_stats = {
            "total_inferences": 0,
            "cache_hits": 0,
            "average_inference_time": 0.0
        }
        
        # Thread safety
        self.model_lock = threading.RLock()
        
    async def initialize_local_models(self):
        """Load all trained ApexSigma models for local inference"""
        
        logger.info("Initializing ApexSigma local AI models...")
        
        if not self.models_dir.exists():
            logger.error(f"Models directory not found: {self.models_dir}")
            return False
        
        # Load training summary to identify available models
        summary_path = self.models_dir / "training_summary.json"
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                training_summary = json.load(f)
            
            for model_info in training_summary.get("trained_models", []):
                await self.load_model(model_info)
        
        logger.info(f"Loaded {len(self.models)} local AI models")
        return len(self.models) > 0
    
    async def load_model(self, model_info: Dict):
        """Load individual model for inference"""
        
        model_name = model_info["name"]
        
        # Prefer quantized models for local deployment
        model_path = Path(model_info.get("quantized_model", model_info["full_model"]))
        
        if not model_path.exists():
            logger.warning(f"Model path not found: {model_path}")
            return False
        
        try:
            logger.info(f"Loading model: {model_name}")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.tokenizers[model_name] = tokenizer
            
            # Load model
            if "quantized" in str(model_path):
                # Load quantized model
                model = await self.load_quantized_model(model_path)
            else:
                # Load full model
                model = AutoModel.from_pretrained(model_path)
            
            if model is not None:
                model.to(self.device)
                model.eval()  # Set to evaluation mode
                self.models[model_name] = model
                
                # Load model configuration
                config_path = model_path / "config.json"
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        self.model_configs[model_name] = json.load(f)
                
                logger.info(f"Successfully loaded {model_name}")
                return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return False
    
    async def load_quantized_model(self, model_path: Path):
        """Load quantized model for efficient local inference"""
        
        try:
            # Load quantized state dict
            state_dict_path = model_path / "pytorch_model.bin"
            if state_dict_path.exists():
                state_dict = torch.load(state_dict_path, map_location=self.device)
                
                # Create model architecture and load quantized weights
                config_path = model_path / "config.json"
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    
                    # Create base model and apply quantization
                    model = AutoModel.from_config(config)
                    model.load_state_dict(state_dict)
                    
                    # Apply dynamic quantization if not already quantized
                    model = torch.quantization.quantize_dynamic(
                        model, {torch.nn.Linear}, dtype=torch.qint8
                    )
                    
                    return model
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to load quantized model: {e}")
            return None
    
    @lru_cache(maxsize=1000)
    async def cached_inference(self, model_name: str, input_text: str, inference_type: str) -> Dict:
        """Cached inference for frequently used queries"""
        return await self.raw_inference(model_name, input_text, inference_type)
    
    async def raw_inference(self, model_name: str, input_text: str, inference_type: str) -> Dict:
        """Perform raw AI inference without caching"""
        
        start_time = time.perf_counter()
        
        if model_name not in self.models:
            return {"error": f"Model {model_name} not available"}
        
        try:
            with self.model_lock:
                model = self.models[model_name]
                tokenizer = self.tokenizers[model_name]
            
            # Tokenize input
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.device)
            
            # Perform inference
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Process outputs based on inference type
            result = await self.process_inference_output(outputs, inference_type, tokenizer)
            
            # Update performance stats
            inference_time = (time.perf_counter() - start_time) * 1000  # ms
            self.performance_stats["total_inferences"] += 1
            self.performance_stats["average_inference_time"] = (
                (self.performance_stats["average_inference_time"] * (self.performance_stats["total_inferences"] - 1) + inference_time) /
                self.performance_stats["total_inferences"]
            )
            
            result["inference_time_ms"] = inference_time
            return result
            
        except Exception as e:
            logger.error(f"Inference failed for {model_name}: {e}")
            return {"error": str(e)}
    
    async def process_inference_output(self, outputs, inference_type: str, tokenizer) -> Dict:
        """Process model outputs based on inference type"""
        
        if inference_type == "embedding":
            # Extract embeddings from last hidden state
            embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
            return {
                "embeddings": embeddings.cpu().numpy().tolist(),
                "embedding_dim": embeddings.shape[-1]
            }
            
        elif inference_type == "similarity":
            # Calculate similarity scores (for search ranking)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs.last_hidden_state.mean(dim=1)
            scores = torch.softmax(logits, dim=-1)
            return {
                "similarity_scores": scores.cpu().numpy().tolist(),
                "confidence": float(torch.max(scores))
            }
            
        elif inference_type == "classification":
            # Classification outputs (for autonomous planning)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs.last_hidden_state[:, 0]
            predictions = torch.argmax(logits, dim=-1)
            confidences = torch.softmax(logits, dim=-1)
            
            return {
                "predictions": predictions.cpu().numpy().tolist(),
                "confidences": confidences.cpu().numpy().tolist(),
                "predicted_class": int(predictions[0]) if len(predictions) > 0 else 0
            }
            
        elif inference_type == "generation":
            # Text generation (for planning and suggestions)
            if hasattr(outputs, 'logits'):
                # Get most likely next tokens
                logits = outputs.logits[0, -1, :]  # Last token logits
                top_tokens = torch.topk(logits, k=5)
                
                generated_tokens = []
                for token_id, score in zip(top_tokens.indices, top_tokens.values):
                    token = tokenizer.decode([token_id])
                    generated_tokens.append({
                        "token": token,
                        "score": float(score)
                    })
                
                return {
                    "generated_tokens": generated_tokens,
                    "next_token_predictions": generated_tokens
                }
            
        # Default: return raw outputs
        return {
            "raw_output": str(outputs),
            "output_shape": str(outputs.last_hidden_state.shape) if hasattr(outputs, 'last_hidden_state') else None
        }
    
    async def development_intelligence(self, query: str, context: Dict = None) -> Dict:
        """Main development intelligence interface"""
        
        # Determine best model for query type
        model_name = await self.select_best_model(query, context)
        
        if not model_name:
            return {"error": "No suitable model found for query"}
        
        # Prepare enhanced input with context
        enhanced_input = await self.prepare_contextual_input(query, context)
        
        # Perform inference
        result = await self.cached_inference(model_name, enhanced_input, "embedding")
        
        # Post-process for development-specific insights
        insights = await self.extract_development_insights(result, query, context)
        
        return {
            "model_used": model_name,
            "query": query,
            "insights": insights,
            "performance": {
                "inference_time_ms": result.get("inference_time_ms", 0),
                "model_confidence": result.get("confidence", 0.0)
            }
        }
    
    async def select_best_model(self, query: str, context: Dict = None) -> Optional[str]:
        """Select the best model for a given query"""
        
        query_lower = query.lower()
        
        # Model selection logic based on query content
        if any(keyword in query_lower for keyword in ["code", "function", "class", "import", "def"]):
            # Code understanding tasks
            for model_name in self.models.keys():
                if "code_understanding" in model_name:
                    return model_name
        
        elif any(keyword in query_lower for keyword in ["search", "find", "look for", "similar"]):
            # Search and ranking tasks
            for model_name in self.models.keys():
                if "search_ranking" in model_name:
                    return model_name
        
        elif any(keyword in query_lower for keyword in ["plan", "roadmap", "strategy", "steps", "how to"]):
            # Autonomous planning tasks
            for model_name in self.models.keys():
                if "autonomous_planning" in model_name:
                    return model_name
        
        elif any(keyword in query_lower for keyword in ["workflow", "process", "pattern", "best practice"]):
            # Development workflow tasks
            for model_name in self.models.keys():
                if "development_workflows" in model_name:
                    return model_name
        
        # Default: use first available model
        return list(self.models.keys())[0] if self.models else None
    
    async def prepare_contextual_input(self, query: str, context: Dict = None) -> str:
        """Prepare enhanced input with development context"""
        
        enhanced_input = query
        
        if context:
            context_parts = []
            
            # Add file context
            if "current_file" in context:
                context_parts.append(f"Current file: {context['current_file']}")
            
            # Add project context
            if "project_type" in context:
                context_parts.append(f"Project type: {context['project_type']}")
            
            # Add error context
            if "error_message" in context:
                context_parts.append(f"Error: {context['error_message']}")
            
            # Add code context
            if "code_snippet" in context:
                context_parts.append(f"Code:\n{context['code_snippet']}")
            
            if context_parts:
                enhanced_input = f"Context: {' | '.join(context_parts)}\n\nQuery: {query}"
        
        return enhanced_input
    
    async def extract_development_insights(self, ai_result: Dict, query: str, context: Dict = None) -> Dict:
        """Extract development-specific insights from AI results"""
        
        insights = {
            "primary_suggestion": None,
            "alternative_approaches": [],
            "potential_issues": [],
            "best_practices": [],
            "related_concepts": []
        }
        
        # Analyze embeddings for semantic insights
        if "embeddings" in ai_result:
            embeddings = np.array(ai_result["embeddings"])
            
            # Simple insight extraction based on embedding patterns
            # (In production, this would use more sophisticated analysis)
            
            embedding_magnitude = np.linalg.norm(embeddings)
            
            if embedding_magnitude > 10:
                insights["primary_suggestion"] = "High complexity detected - consider breaking down the task"
            elif embedding_magnitude < 2:
                insights["primary_suggestion"] = "Simple task - direct implementation recommended"
            else:
                insights["primary_suggestion"] = "Moderate complexity - standard approach suitable"
            
            # Extract related concepts based on embedding similarity patterns
            insights["related_concepts"] = await self.find_related_concepts(embeddings)
        
        # Query-specific insights
        query_lower = query.lower()
        
        if "error" in query_lower or "bug" in query_lower:
            insights["best_practices"].extend([
                "Add comprehensive logging",
                "Write unit tests to prevent regression",
                "Document the solution for future reference"
            ])
        
        elif "optimize" in query_lower or "performance" in query_lower:
            insights["best_practices"].extend([
                "Profile before optimizing",
                "Measure performance impact",
                "Consider trade-offs between readability and performance"
            ])
        
        elif "api" in query_lower or "service" in query_lower:
            insights["best_practices"].extend([
                "Implement proper error handling",
                "Add input validation",
                "Consider rate limiting and authentication"
            ])
        
        return insights
    
    async def find_related_concepts(self, embeddings: np.ndarray) -> List[str]:
        """Find concepts related to the embedding vector"""
        
        # Simplified concept mapping based on embedding patterns
        # In production, this would use a trained concept database
        
        concepts = []
        
        # Analyze embedding dimensions for concept hints
        embedding_sum = np.sum(embeddings)
        
        if embedding_sum > 0:
            concepts.extend(["positive_sentiment", "constructive_approach"])
        
        if np.std(embeddings) > 1:
            concepts.extend(["complex_problem", "multiple_factors"])
        
        if np.max(embeddings) > 5:
            concepts.extend(["high_importance", "critical_component"])
        
        return concepts[:5]  # Limit to top 5 concepts
    
    async def get_performance_stats(self) -> Dict:
        """Get performance statistics for the local AI engine"""
        
        return {
            "models_loaded": len(self.models),
            "cache_size": len(self.inference_cache),
            "performance": self.performance_stats,
            "device": str(self.device),
            "cache_hit_rate": (
                self.performance_stats["cache_hits"] / 
                max(self.performance_stats["total_inferences"], 1)
            )
        }
    
    async def warm_up_models(self):
        """Warm up models with sample queries for faster initial responses"""
        
        logger.info("Warming up AI models...")
        
        sample_queries = [
            "How to optimize database queries",
            "Best practices for error handling",
            "Code review checklist",
            "API design patterns",
            "Unit testing strategies"
        ]
        
        for query in sample_queries:
            for model_name in self.models.keys():
                try:
                    await self.raw_inference(model_name, query, "embedding")
                except Exception as e:
                    logger.warning(f"Warm-up failed for {model_name}: {e}")
        
        logger.info("Model warm-up completed")

class ApexSigmaLocalAI:
    """Main interface for ApexSigma local AI functionality"""
    
    def __init__(self):
        self.engine = LocalAIEngine()
        self.initialized = False
    
    async def initialize(self):
        """Initialize the local AI system"""
        if not self.initialized:
            success = await self.engine.initialize_local_models()
            if success:
                await self.engine.warm_up_models()
                self.initialized = True
                logger.info("ApexSigma Local AI initialized successfully")
            else:
                logger.error("Failed to initialize ApexSigma Local AI")
            return success
        return True
    
    async def ask(self, question: str, context: Dict = None) -> Dict:
        """Ask the local AI a development-related question"""
        if not self.initialized:
            await self.initialize()
        
        return await self.engine.development_intelligence(question, context)
    
    async def analyze_code(self, code: str, context: Dict = None) -> Dict:
        """Analyze code using local AI models"""
        enhanced_context = context or {}
        enhanced_context["code_snippet"] = code
        
        return await self.ask("Analyze this code for improvements and issues", enhanced_context)
    
    async def suggest_next_steps(self, current_task: str, context: Dict = None) -> Dict:
        """Get AI suggestions for next development steps"""
        question = f"What are the next steps for: {current_task}"
        return await self.ask(question, context)
    
    async def debug_assistance(self, error_message: str, code: str = None, context: Dict = None) -> Dict:
        """Get AI assistance for debugging"""
        enhanced_context = context or {}
        enhanced_context["error_message"] = error_message
        if code:
            enhanced_context["code_snippet"] = code
        
        return await self.ask("Help debug this error", enhanced_context)
    
    async def get_status(self) -> Dict:
        """Get status of the local AI system"""
        if not self.initialized:
            return {"status": "not_initialized"}
        
        stats = await self.engine.get_performance_stats()
        return {
            "status": "ready",
            "stats": stats,
            "capabilities": [
                "code_analysis",
                "debug_assistance", 
                "next_step_suggestions",
                "development_planning",
                "best_practice_recommendations"
            ]
        }

# Example usage and testing
async def main():
    """Test the local AI engine"""
    
    ai = ApexSigmaLocalAI()
    
    # Initialize the system
    print("🔄 Initializing ApexSigma Local AI...")
    success = await ai.initialize()
    
    if not success:
        print("❌ Failed to initialize AI system")
        return
    
    print("✅ ApexSigma Local AI ready!")
    
    # Test queries
    test_queries = [
        {
            "question": "How to optimize a slow database query?",
            "context": {"project_type": "web_api", "database": "postgresql"}
        },
        {
            "question": "Best practices for error handling in Python?",
            "context": {"language": "python", "framework": "fastapi"}
        }
    ]
    
    for test in test_queries:
        print(f"\n🤔 Question: {test['question']}")
        result = await ai.ask(test["question"], test["context"])
        
        print(f"🧠 Model: {result.get('model_used', 'unknown')}")
        print(f"⚡ Time: {result.get('performance', {}).get('inference_time_ms', 0):.1f}ms")
        print(f"💡 Suggestion: {result.get('insights', {}).get('primary_suggestion', 'No suggestion')}")
    
    # Get system status
    status = await ai.get_status()
    print(f"\n📊 System Status: {status['status']}")
    print(f"🎯 Total Inferences: {status['stats']['performance']['total_inferences']}")
    print(f"⏱️ Average Time: {status['stats']['performance']['average_inference_time']:.1f}ms")

if __name__ == "__main__":
    asyncio.run(main())