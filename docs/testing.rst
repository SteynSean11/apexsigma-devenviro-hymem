Testing Guide
=============

This guide covers testing strategies, running tests, and writing new tests for the ApexSigma DevEnviro project.

Running Tests
-------------

**Run all tests:**

.. code-block:: bash

   pytest

**Run with coverage:**

.. code-block:: bash

   pytest --cov=code --cov-report=html

**Run specific test categories:**

.. code-block:: bash

   # Unit tests only
   pytest -m unit
   
   # Integration tests only
   pytest -m integration
   
   # Skip slow tests
   pytest -m "not slow"

**Run specific test files:**

.. code-block:: bash

   pytest tests/test_mem0_integration.py -v

Test Structure
--------------

The project uses pytest with the following structure:

.. code-block:: text

   tests/
   ├── __init__.py
   ├── conftest.py              # Shared fixtures
   ├── test_mem0_integration.py # Memory system tests
   └── test_monitoring.py       # Error tracking tests

**Test Categories:**

- ``@pytest.mark.unit`` - Fast, isolated unit tests
- ``@pytest.mark.integration`` - Tests with external dependencies
- ``@pytest.mark.slow`` - Tests that take longer to run

Writing Tests
-------------

**Basic Test Structure:**

.. code-block:: python

   import pytest
   from unittest.mock import Mock, patch
   
   class TestMemorySystem:
       """Test the memory integration system"""
       
       @pytest.mark.unit
       def test_memory_initialization(self):
           """Test memory system can be initialized"""
           # Test implementation
           assert True
       
       @pytest.mark.integration
       def test_memory_with_qdrant(self, sample_config):
           """Test memory system with Qdrant backend"""
           # Test implementation
           pass

**Using Fixtures:**

.. code-block:: python

   @pytest.fixture
   def sample_config():
       """Provide test configuration"""
       return {
           "vector_store": {
               "provider": "qdrant",
               "config": {
                   "host": "localhost",
                   "port": 6333,
                   "collection_name": "test-memory"
               }
           }
       }

**Mocking External Dependencies:**

.. code-block:: python

   @patch('mem0.Memory')
   def test_memory_setup_with_mock(self, mock_memory):
       """Test memory setup with mocked dependencies"""
       mock_memory.from_config.return_value = Mock()
       
       result = test_mem0_setup()
       
       assert result is True
       mock_memory.from_config.assert_called_once()

Test Configuration
------------------

**pytest.ini Configuration:**

.. code-block:: ini

   [tool:pytest]
   testpaths = code tests
   python_files = test_*.py *_test.py
   python_classes = Test*
   python_functions = test_*
   addopts = 
       --verbose
       --tb=short
       --strict-markers
       --cov=code
       --cov-report=term-missing
   markers =
       unit: Unit tests
       integration: Integration tests
       slow: Slow tests

**Coverage Configuration:**

Coverage settings are defined in ``pyproject.toml``:

.. code-block:: toml

   [tool.coverage.run]
   source = ["code"]
   omit = [
       "*/tests/*",
       "*/venv/*",
   ]
   
   [tool.coverage.report]
   exclude_lines = [
       "pragma: no cover",
       "def __repr__",
       "raise NotImplementedError",
   ]

Continuous Integration
----------------------

**GitHub Actions automatically:**

1. Runs tests on Python 3.10 and 3.11
2. Generates coverage reports
3. Uploads results to Codecov
4. Fails the build if tests don't pass

**CI Configuration:**

.. code-block:: yaml

   - name: Test with pytest
     run: |
       pytest code/ -v --cov=code --cov-report=xml --cov-report=html

Testing Best Practices
-----------------------

**1. Test Naming:**

.. code-block:: python

   def test_should_return_true_when_valid_input():
       """Clear, descriptive test names"""
       pass

**2. Test Independence:**

.. code-block:: python

   def test_memory_initialization(self):
       """Each test should be independent"""
       # Setup
       config = create_test_config()
       
       # Action
       result = initialize_memory(config)
       
       # Assert
       assert result is not None

**3. Use Fixtures for Setup:**

.. code-block:: python

   @pytest.fixture
   def temp_dir():
       """Create temporary directory for tests"""
       temp_path = tempfile.mkdtemp()
       yield Path(temp_path)
       shutil.rmtree(temp_path)

**4. Mock External Services:**

.. code-block:: python

   @patch('openai.ChatCompletion.create')
   def test_ai_integration(self, mock_openai):
       """Mock external API calls"""
       mock_openai.return_value = {"choices": [{"text": "response"}]}
       # Test implementation

**5. Test Error Conditions:**

.. code-block:: python

   def test_should_raise_error_when_invalid_config():
       """Test error handling"""
       with pytest.raises(ValueError, match="Invalid configuration"):
           create_memory_system(invalid_config)

Performance Testing
-------------------

**Measure Test Performance:**

.. code-block:: python

   import time
   from code.monitoring import error_tracker
   
   def test_performance_tracking():
       """Test performance monitoring"""
       start_time = time.time()
       
       # Operation to test
       perform_operation()
       
       duration = time.time() - start_time
       error_tracker.log_performance("test_operation", duration)
       
       assert duration < 1.0  # Should complete in under 1 second

**Load Testing:**

.. code-block:: python

   @pytest.mark.slow
   def test_memory_system_load():
       """Test system under load"""
       for i in range(100):
           result = process_memory_request(f"test_data_{i}")
           assert result is not None

Test Data Management
--------------------

**Test Data Location:**

.. code-block:: text

   tests/
   ├── data/
   │   ├── sample_configs.json
   │   ├── test_memories.json
   │   └── mock_responses.json
   └── fixtures/
       ├── memory_fixtures.py
       └── api_fixtures.py

**Loading Test Data:**

.. code-block:: python

   import json
   from pathlib import Path
   
   def load_test_data(filename):
       """Load test data from JSON file"""
       data_path = Path(__file__).parent / "data" / filename
       with open(data_path) as f:
           return json.load(f)

Debugging Tests
---------------

**Run tests with debugging:**

.. code-block:: bash

   # Run with pdb on failures
   pytest --pdb
   
   # Run specific test with output
   pytest tests/test_memory.py::test_specific_function -s -vv
   
   # Run last failed tests
   pytest --lf

**Debug with VS Code:**

Add to ``.vscode/launch.json``:

.. code-block:: json

   {
       "name": "Debug Tests",
       "type": "python",
       "request": "launch",
       "module": "pytest",
       "args": ["tests/", "-v"],
       "console": "integratedTerminal"
   }

Testing Checklist
------------------

Before submitting code:

- [ ] All tests pass locally
- [ ] New functionality has corresponding tests
- [ ] Tests cover both success and error cases
- [ ] Test names are descriptive
- [ ] No tests marked as ``@pytest.mark.skip`` without reason
- [ ] Coverage remains above 80%
- [ ] Tests run in reasonable time (< 5 minutes total)

**Test Coverage Goals:**

- Unit tests: > 90% coverage
- Integration tests: Cover all major workflows
- End-to-end tests: Cover critical user journeys

Running tests is part of the development workflow - make sure they pass before committing!