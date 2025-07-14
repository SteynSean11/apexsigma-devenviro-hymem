Quick Start Guide
=================

Get up and running with ApexSigma DevEnviro in just a few minutes.

Basic Usage
-----------

1. **Test Memory Integration:**

.. code-block:: python

   from code.test_mem0 import test_mem0_setup
   
   # Test the memory system
   result = test_mem0_setup()
   print(f"Memory system status: {'✅ Working' if result else '❌ Failed'}")

2. **Run All Tests:**

.. code-block:: bash

   pytest tests/ -v

3. **Check Code Quality:**

.. code-block:: bash

   # Format code
   black code/
   
   # Check linting
   flake8 code/
   
   # Type checking
   mypy code/

Core Components
---------------

**Memory System:**
The project uses Mem0 with Qdrant for long-term memory storage:

.. code-block:: python

   from mem0 import Memory
   
   config = {
       "vector_store": {
           "provider": "qdrant",
           "config": {
               "host": "localhost",
               "port": 6333,
               "collection_name": "apexsigma-memory"
           }
       }
   }
   
   memory = Memory.from_config(config)

**Error Tracking:**
Monitor application health and performance:

.. code-block:: python

   from code.monitoring import error_tracker, track_errors
   
   # Decorator for automatic error tracking
   @track_errors
   def your_function():
       # Your code here
       pass
   
   # Manual error logging
   try:
       risky_operation()
   except Exception as e:
       error_tracker.log_error(e, {"context": "additional_info"})

**Security Features:**
The project includes comprehensive security measures:

.. code-block:: bash

   # Scan for secrets
   detect-secrets scan --baseline .secrets.baseline
   
   # Run pre-commit hooks
   pre-commit run --all-files

Development Workflow
--------------------

1. **Make changes to your code**

2. **Run tests locally:**

.. code-block:: bash

   pytest tests/ -v --cov=code

3. **Check code quality:**

.. code-block:: bash

   pre-commit run --all-files

4. **Commit and push:**

.. code-block:: bash

   git add .
   git commit -m "Your descriptive commit message"
   git push origin main

5. **CI/CD Pipeline automatically:**
   
   - Runs tests across Python 3.10 and 3.11
   - Checks code quality and security
   - Builds and deploys documentation
   - Generates coverage reports

Configuration
-------------

**Environment Variables:**

- ``OPENAI_API_KEY``: Required for AI functionality
- ``LINEAR_API_KEY``: For Linear integration
- ``DATABASE_URL``: Database connection string

**Project Structure:**

.. code-block:: text

   apexsigma-devenviro/
   ├── code/                 # Main application code
   ├── tests/                # Test files
   ├── docs/                 # Documentation
   ├── .github/workflows/    # CI/CD configuration
   ├── requirements.txt      # Production dependencies
   ├── requirements-dev.txt  # Development dependencies
   └── pyproject.toml        # Project configuration

Next Steps
----------

- :doc:`installation` - Detailed installation guide
- :doc:`api` - API reference documentation
- :doc:`security` - Security guidelines and best practices
- :doc:`testing` - Testing strategies and examples

**Ready to contribute?**

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

The CI/CD pipeline will automatically validate your changes!