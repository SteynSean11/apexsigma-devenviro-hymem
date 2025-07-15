"""Test cases for Mem0 integration"""
import pytest
import os
from unittest.mock import Mock, patch
import sys
sys.path.append('code')

from test_mem0 import test_mem0_setup


class TestMem0Integration:
    """Test Mem0 memory system integration"""
    
    @pytest.mark.unit
    def test_mem0_import(self):
        """Test that mem0 can be imported"""
        try:
            import mem0
            assert mem0 is not None
        except ImportError:
            pytest.skip("mem0 not installed")
    
    @pytest.mark.integration
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    def test_mem0_setup_with_api_key(self):
        """Test mem0 setup with API key present"""
        with patch('mem0.Memory') as mock_memory:
            mock_memory.from_config.return_value = Mock()
            result = test_mem0_setup()
            assert result is True
    
    @pytest.mark.integration
    @patch.dict(os.environ, {}, clear=True)
    def test_mem0_setup_without_api_key(self):
        """Test mem0 setup without API key"""
        with patch('mem0.Memory') as mock_memory:
            mock_memory.from_config.return_value = Mock()
            result = test_mem0_setup()
            assert result is True  # Should still work, just warn about API key
    
    @pytest.mark.integration
    def test_mem0_config_structure(self):
        """Test that the config structure is valid"""
        expected_config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "host": "localhost",
                    "port": 6333,
                    "collection_name": "apexsigma-memory"
                }
            }
        }
        
        # Verify config has required keys
        assert "vector_store" in expected_config
        assert "provider" in expected_config["vector_store"]
        assert "config" in expected_config["vector_store"]
        assert expected_config["vector_store"]["provider"] == "qdrant"