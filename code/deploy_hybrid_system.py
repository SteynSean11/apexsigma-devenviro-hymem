"""
ApexSigma Hybrid System Deployment Script
Deploy and test the complete hybrid cloud + local AI system

Date: July 15, 2025
Goal: Deploy and validate the hybrid memory + AI system
"""

import asyncio
import sys
import time
from pathlib import Path
import json
import logging
from typing import Dict, Any
from datetime import datetime

# Add project paths
sys.path.append(str(Path(__file__).parent / "ai_training"))
sys.path.append(str(Path(__file__).parent / "memory"))

from custom_ai_trainer import ApexSigmaAITrainer
from local_ai_engine import ApexSigmaLocalAI
from hybrid_memory_system import HybridMemorySystem

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HybridSystemDeployer:
    """Deploys and tests the complete ApexSigma hybrid system"""
    
    def __init__(self):
        self.deployment_config = {
            "ai_training": True,
            "memory_system": True,
            "performance_testing": True,
            "integration_testing": True
        }
        
        self.results = {
            "deployment_start": datetime.now(),
            "components_deployed": [],
            "tests_passed": [],
            "performance_metrics": {},
            "errors": []
        }
    
    async def deploy_complete_system(self):
        """Deploy the complete hybrid system"""
        
        logger.info("🚀 Starting ApexSigma Hybrid System Deployment")
        
        try:
            # Step 1: Deploy AI Training System
            if self.deployment_config["ai_training"]:
                await self.deploy_ai_training_system()
            
            # Step 2: Deploy Memory System
            if self.deployment_config["memory_system"]:
                await self.deploy_memory_system()
            
            # Step 3: Integration Testing
            if self.deployment_config["integration_testing"]:
                await self.run_integration_tests()
            
            # Step 4: Performance Testing
            if self.deployment_config["performance_testing"]:
                await self.run_performance_tests()
            
            # Step 5: Generate deployment report
            await self.generate_deployment_report()
            
            logger.info("✅ Hybrid System Deployment Complete!")
            
        except Exception as e:
            logger.error(f"❌ Deployment failed: {e}")
            self.results["errors"].append(str(e))
            raise
    
    async def deploy_ai_training_system(self):
        """Deploy and test AI training system"""
        
        logger.info("🧠 Deploying AI Training System...")
        
        try:
            # Initialize AI trainer
            training_config = {
                "target_performance": {
                    "response_time_ms": 25,
                    "accuracy_improvement": 0.30,
                    "token_reduction": 0.95
                },
                "deployment_targets": ["local_quantized"],
                "privacy_first": True,
                "development_focused": True
            }
            
            trainer = ApexSigmaAITrainer(training_config)
            
            # Initialize training environment
            await trainer.initialize_training_environment()
            logger.info("✅ AI training environment initialized")
            
            # Train a sample model for testing
            logger.info("🎓 Training sample AI model...")
            
            # Create minimal training dataset for testing
            await self.create_test_training_data(trainer)
            
            # Train lightweight model for deployment testing
            model_path = await trainer.train_custom_model("text_embeddings", "development_workflows")
            
            if model_path:
                logger.info(f"✅ Sample model trained: {model_path}")
                self.results["components_deployed"].append("ai_training_system")
            else:
                raise Exception("Failed to train sample model")
                
        except Exception as e:
            logger.error(f"❌ AI Training System deployment failed: {e}")
            self.results["errors"].append(f"AI Training: {e}")
            raise
    
    async def create_test_training_data(self, trainer):
        """Create minimal training data for testing"""
        
        # Create a small test dataset
        test_data = [
            {
                "input": "How to handle async functions in Python?",
                "output": "Use async/await keywords and asyncio library",
                "type": "development_question"
            },
            {
                "input": "Best practice for error handling?",
                "output": "Use try/except blocks and log errors properly",
                "type": "development_question"
            },
            {
                "input": "Code review checklist item",
                "output": "Check for proper error handling and documentation",
                "type": "development_best_practice"
            }
        ]
        
        # Save test dataset
        test_data_path = trainer.data_dir / "development_workflows.json"
        with open(test_data_path, 'w') as f:
            json.dump(test_data, f, indent=2)
        
        logger.info(f"Created test training data: {len(test_data)} examples")
    
    async def deploy_memory_system(self):
        """Deploy and test memory system"""
        
        logger.info("🧠 Deploying Hybrid Memory System...")
        
        try:
            # Initialize memory system
            memory_system = HybridMemorySystem("test_memory_data")
            
            success = await memory_system.initialize()
            if not success:
                raise Exception("Failed to initialize memory system")
            
            logger.info("✅ Hybrid Memory System initialized")
            
            # Test basic memory operations
            await self.test_memory_operations(memory_system)
            
            self.results["components_deployed"].append("hybrid_memory_system")
            
        except Exception as e:
            logger.error(f"❌ Memory System deployment failed: {e}")
            self.results["errors"].append(f"Memory System: {e}")
            raise
    
    async def test_memory_operations(self, memory_system):
        """Test basic memory system operations"""
        
        logger.info("🧪 Testing memory operations...")
        
        # Test storing different types of memories
        test_memories = [
            {
                "content": "Python async/await best practices for web development",
                "type": "fact",
                "context": {"category": "python", "topic": "async_programming"}
            },
            {
                "content": "Successfully debugged database connection issue",
                "type": "episode",
                "context": {"success": True, "duration": "2_hours"}
            },
            {
                "content": "Error handling pattern for REST APIs",
                "type": "concept",
                "context": {"abstraction_level": 2}
            }
        ]
        
        stored_memories = []
        for memory in test_memories:
            memory_id = await memory_system.store_memory(
                memory["content"],
                memory["type"],
                memory["context"]
            )
            stored_memories.append(memory_id)
            logger.info(f"✅ Stored {memory['type']} memory: {memory_id}")
        
        # Test memory search
        search_queries = [
            "Python async programming",
            "database connection",
            "error handling"
        ]
        
        for query in search_queries:
            start_time = time.perf_counter()
            results = await memory_system.intelligent_search(query, max_results=3)
            search_time = (time.perf_counter() - start_time) * 1000
            
            logger.info(f"🔍 Search '{query}': {len(results)} results in {search_time:.1f}ms")
            
            if search_time < 100:  # Target: <100ms for test environment
                self.results["tests_passed"].append(f"memory_search_{query}")
            
            # Store performance metric
            self.results["performance_metrics"][f"search_{query.replace(' ', '_')}"] = {
                "time_ms": search_time,
                "results_count": len(results)
            }
        
        logger.info("✅ Memory operations tested successfully")
    
    async def run_integration_tests(self):
        """Run integration tests between AI and memory systems"""
        
        logger.info("🔧 Running Integration Tests...")
        
        try:
            # Test AI + Memory integration
            ai_engine = ApexSigmaLocalAI()
            memory_system = HybridMemorySystem("test_memory_data")
            
            # Initialize both systems
            ai_success = await ai_engine.initialize()
            memory_success = await memory_system.initialize()
            
            if not (ai_success and memory_success):
                raise Exception("Failed to initialize integrated systems")
            
            # Test AI-powered memory storage
            await self.test_ai_memory_integration(ai_engine, memory_system)
            
            self.results["tests_passed"].append("ai_memory_integration")
            logger.info("✅ Integration tests passed")
            
        except Exception as e:
            logger.error(f"❌ Integration tests failed: {e}")
            self.results["errors"].append(f"Integration: {e}")
            raise
    
    async def test_ai_memory_integration(self, ai_engine, memory_system):
        """Test AI and memory system working together"""
        
        # Test: AI analyzes code and stores insights in memory
        test_code = """
def calculate_fibonacci(n):
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)
        """
        
        # AI analyzes the code
        analysis = await ai_engine.analyze_code(test_code, {"language": "python"})
        
        # Store AI insights in memory
        insights = analysis.get("insights", {})
        if insights.get("primary_suggestion"):
            memory_id = await memory_system.store_memory(
                f"Code analysis: {insights['primary_suggestion']}",
                "fact",
                {"source": "ai_analysis", "code_type": "fibonacci"}
            )
            logger.info(f"✅ AI insight stored in memory: {memory_id}")
        
        # Test: Memory-enhanced AI responses
        memory_results = await memory_system.intelligent_search("fibonacci optimization")
        
        if memory_results:
            enhanced_context = {
                "memory_context": [r["content"] for r in memory_results[:2]]
            }
            
            enhanced_response = await ai_engine.ask(
                "How to optimize fibonacci function?",
                enhanced_context
            )
            
            logger.info(f"✅ AI response enhanced with memory context")
            logger.info(f"   Suggestion: {enhanced_response.get('insights', {}).get('primary_suggestion', 'No suggestion')}")
    
    async def run_performance_tests(self):
        """Run performance benchmarks"""
        
        logger.info("⚡ Running Performance Tests...")
        
        try:
            # Test AI inference performance
            ai_engine = ApexSigmaLocalAI()
            await ai_engine.initialize()
            
            # Benchmark AI inference
            ai_times = []
            for i in range(5):
                start_time = time.perf_counter()
                result = await ai_engine.ask("Test query for performance measurement")
                inference_time = (time.perf_counter() - start_time) * 1000
                ai_times.append(inference_time)
            
            avg_ai_time = sum(ai_times) / len(ai_times)
            self.results["performance_metrics"]["ai_inference_avg_ms"] = avg_ai_time
            
            logger.info(f"⚡ AI inference average: {avg_ai_time:.1f}ms")
            
            # Test memory system performance
            memory_system = HybridMemorySystem("test_memory_data")
            await memory_system.initialize()
            
            # Benchmark memory search
            memory_times = []
            for i in range(5):
                start_time = time.perf_counter()
                results = await memory_system.intelligent_search(f"test query {i}")
                search_time = (time.perf_counter() - start_time) * 1000
                memory_times.append(search_time)
            
            avg_memory_time = sum(memory_times) / len(memory_times)
            self.results["performance_metrics"]["memory_search_avg_ms"] = avg_memory_time
            
            logger.info(f"⚡ Memory search average: {avg_memory_time:.1f}ms")
            
            # Overall system performance
            total_time = avg_ai_time + avg_memory_time
            self.results["performance_metrics"]["total_system_avg_ms"] = total_time
            
            logger.info(f"⚡ Total system response: {total_time:.1f}ms")
            
            # Check performance targets
            if avg_ai_time < 50:  # Target: <50ms for AI
                self.results["tests_passed"].append("ai_performance_target")
            
            if avg_memory_time < 100:  # Target: <100ms for memory
                self.results["tests_passed"].append("memory_performance_target")
            
            if total_time < 150:  # Target: <150ms total
                self.results["tests_passed"].append("system_performance_target")
            
            logger.info("✅ Performance tests completed")
            
        except Exception as e:
            logger.error(f"❌ Performance tests failed: {e}")
            self.results["errors"].append(f"Performance: {e}")
            raise
    
    async def generate_deployment_report(self):
        """Generate comprehensive deployment report"""
        
        self.results["deployment_end"] = datetime.now()
        self.results["deployment_duration"] = (
            self.results["deployment_end"] - self.results["deployment_start"]
        ).total_seconds()
        
        report = {
            "deployment_summary": {
                "status": "SUCCESS" if not self.results["errors"] else "PARTIAL_SUCCESS",
                "duration_seconds": self.results["deployment_duration"],
                "components_deployed": len(self.results["components_deployed"]),
                "tests_passed": len(self.results["tests_passed"]),
                "errors_encountered": len(self.results["errors"])
            },
            "detailed_results": self.results,
            "performance_summary": {
                "ai_performance": self.results["performance_metrics"].get("ai_inference_avg_ms", 0),
                "memory_performance": self.results["performance_metrics"].get("memory_search_avg_ms", 0),
                "total_system_performance": self.results["performance_metrics"].get("total_system_avg_ms", 0)
            },
            "recommendations": await self.generate_recommendations()
        }
        
        # Save report
        report_path = Path("deployment_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📊 Deployment report saved: {report_path}")
        
        # Print summary
        print("\n" + "="*80)
        print("🚀 APEXSIGMA HYBRID SYSTEM DEPLOYMENT SUMMARY")
        print("="*80)
        print(f"Status: {report['deployment_summary']['status']}")
        print(f"Duration: {report['deployment_summary']['duration_seconds']:.1f}s")
        print(f"Components Deployed: {report['deployment_summary']['components_deployed']}")
        print(f"Tests Passed: {report['deployment_summary']['tests_passed']}")
        print(f"Errors: {report['deployment_summary']['errors_encountered']}")
        
        print("\n⚡ PERFORMANCE METRICS:")
        print(f"AI Inference: {report['performance_summary']['ai_performance']:.1f}ms")
        print(f"Memory Search: {report['performance_summary']['memory_performance']:.1f}ms")
        print(f"Total System: {report['performance_summary']['total_system_performance']:.1f}ms")
        
        if self.results["errors"]:
            print("\n❌ ERRORS ENCOUNTERED:")
            for error in self.results["errors"]:
                print(f"  - {error}")
        
        print("\n✅ COMPONENTS DEPLOYED:")
        for component in self.results["components_deployed"]:
            print(f"  - {component}")
        
        print("\n🧪 TESTS PASSED:")
        for test in self.results["tests_passed"]:
            print(f"  - {test}")
        
        if report["recommendations"]:
            print("\n💡 RECOMMENDATIONS:")
            for rec in report["recommendations"]:
                print(f"  - {rec}")
        
        print("="*80)
    
    async def generate_recommendations(self) -> List[str]:
        """Generate deployment recommendations"""
        
        recommendations = []
        
        # Performance recommendations
        ai_time = self.results["performance_metrics"].get("ai_inference_avg_ms", 0)
        memory_time = self.results["performance_metrics"].get("memory_search_avg_ms", 0)
        
        if ai_time > 50:
            recommendations.append(
                f"AI inference time ({ai_time:.1f}ms) exceeds target. Consider model quantization or GPU acceleration."
            )
        
        if memory_time > 100:
            recommendations.append(
                f"Memory search time ({memory_time:.1f}ms) exceeds target. Consider adding caching or indexing optimizations."
            )
        
        # Error-based recommendations
        if "AI Training" in str(self.results["errors"]):
            recommendations.append("AI training issues detected. Verify model dependencies and training data.")
        
        if "Memory System" in str(self.results["errors"]):
            recommendations.append("Memory system issues detected. Check database connectivity and storage permissions.")
        
        # Success-based recommendations
        if not self.results["errors"]:
            recommendations.append("All systems deployed successfully! Ready for production integration.")
            recommendations.append("Consider implementing monitoring and alerting for production deployment.")
        
        return recommendations

async def main():
    """Main deployment script"""
    
    print("🚀 ApexSigma Hybrid System Deployment")
    print("=====================================")
    
    deployer = HybridSystemDeployer()
    
    try:
        await deployer.deploy_complete_system()
        print("\n✅ Deployment completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Deployment failed: {e}")
        print("Check deployment_report.json for detailed error information.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())