Security Guidelines
==================

This document outlines security best practices for the ApexSigma DevEnviro project.

Secret Management
-----------------

Critical Security Rules
~~~~~~~~~~~~~~~~~~~~~~~~

**NEVER commit secrets to git repositories:**

* API keys, tokens, passwords, certificates
* Database connection strings with credentials
* Private keys or sensitive configuration files

**Use .gitignore for sensitive files:**

.. code-block:: bash

   # Secrets and environment files
   .env
   .env.*
   config/secrets/
   *.key
   *.pem
   *.p12

**Environment Variables for Production:**

* Store secrets in environment variables
* Use secure secret management services
* Never hardcode secrets in source code

Local Development
~~~~~~~~~~~~~~~~~

**Safe Practices:**

.. code-block:: python

   # Create local .env file (already in .gitignore)
   # .env file content:
   OPENAI_API_KEY=your_key_here
   LINEAR_API_KEY=your_linear_key
   DATABASE_URL=your_db_connection

   # Use python-dotenv to load secrets
   from dotenv import load_dotenv
   import os
   
   load_dotenv()
   api_key = os.getenv("OPENAI_API_KEY")

**Dangerous Practices (NEVER DO THIS):**

.. code-block:: python

   # NEVER hardcode secrets
   API_KEY = "sk-1234567890abcdef"
   DATABASE_PASSWORD = "mypassword123"

Production Deployment
~~~~~~~~~~~~~~~~~~~~~

**GitHub Actions Secrets:**

1. Go to Repository Settings → Secrets and variables → Actions
2. Add secrets as repository secrets:
   
   * ``OPENAI_API_KEY``
   * ``DATABASE_URL``
   * ``LINEAR_API_KEY``

**Environment Variable Usage in CI/CD:**

.. code-block:: yaml

   # In .github/workflows/ci.yml
   env:
     OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
     DATABASE_URL: ${{ secrets.DATABASE_URL }}

Secret Scanning & Protection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**GitHub Advanced Security:**

* Enable secret scanning in repository settings
* Configure push protection to block commits with secrets
* Review and remediate any detected secrets

**Pre-commit Hooks:**

.. code-block:: bash

   # Install secret detection tools
   pip install detect-secrets
   pre-commit install
   
   # Scan for secrets
   detect-secrets scan --baseline .secrets.baseline

Incident Response
~~~~~~~~~~~~~~~~~

**If you accidentally commit a secret:**

1. **Immediately revoke the compromised secret**
   
   * Generate new API keys
   * Rotate passwords
   * Update access tokens

2. **Remove from git history:**

.. code-block:: bash

   # Create clean branch without secret history
   git checkout --orphan clean-branch
   git add .
   git commit -m "Clean commit without secrets"
   git push -u origin clean-branch

3. **For advanced history rewriting (use with caution):**

.. code-block:: bash

   # Remove specific file from entire git history
   git filter-branch --force --index-filter \
     'git rm --cached --ignore-unmatch path/to/secret/file' \
     --prune-empty --tag-name-filter cat -- --all

Security Tools
~~~~~~~~~~~~~~

**Recommended Tools:**

* **detect-secrets**: Pre-commit hook for secret detection
* **git-secrets**: AWS tool for preventing secrets in git
* **truffleHog**: Search git repos for high entropy strings
* **GitGuardian**: Commercial secret scanning service

**Installation:**

.. code-block:: bash

   pip install detect-secrets
   detect-secrets scan --baseline .secrets.baseline

Code Review Security
~~~~~~~~~~~~~~~~~~~~

**Security Checklist for Pull Requests:**

* ✅ No hardcoded secrets or API keys
* ✅ Environment variables used for configuration
* ✅ Sensitive files in .gitignore
* ✅ No database credentials in code
* ✅ No private keys or certificates committed

Monitoring & Alerts
~~~~~~~~~~~~~~~~~~~

**Set up monitoring for:**

* Failed authentication attempts
* Unusual API usage patterns
* Unauthorized access attempts
* Secret rotation schedules

**Implementation in monitoring.py:**

.. code-block:: python

   from code.monitoring import error_tracker
   
   # Log security events
   error_tracker.log_error(
       security_event,
       context={'event_type': 'unauthorized_access', 'ip': request.ip}
   )

Compliance & Auditing
~~~~~~~~~~~~~~~~~~~~~

**Documentation Requirements:**

* Log all secret rotations with timestamps
* Maintain access control lists
* Document security incidents thoroughly
* Conduct regular security reviews

**Audit Trail:**

.. code-block:: python

   # Example audit logging
   audit_log = {
       'timestamp': datetime.now().isoformat(),
       'action': 'secret_rotation',
       'resource': 'openai_api_key',
       'user': 'admin',
       'status': 'success'
   }

Security Checklist
------------------

Before Each Commit
~~~~~~~~~~~~~~~~~~

* ✅ Check for secrets with ``git diff``
* ✅ Verify .env files are not staged
* ✅ Run secret detection tools
* ✅ Review file changes for sensitive data

.. code-block:: bash

   # Pre-commit security check
   git diff --cached | grep -i "api_key\|password\|secret\|token"

Before Each Release
~~~~~~~~~~~~~~~~~~~

* ✅ Audit all environment variables
* ✅ Verify secret rotation schedule
* ✅ Update security documentation
* ✅ Run comprehensive security scans

Regular Maintenance
~~~~~~~~~~~~~~~~~~

* ✅ Rotate API keys monthly
* ✅ Update security tools and dependencies
* ✅ Review access permissions quarterly
* ✅ Audit secret usage and access logs

Contact & Reporting
-------------------

**Security Issues:**

* Report security vulnerabilities immediately to security@apexsigma.com
* Use encrypted communication for sensitive reports
* Follow responsible disclosure practices

**Emergency Response:**

* Immediate secret revocation: Contact DevOps team
* Security incident reporting: Use internal incident response procedures
* External security research: Follow coordinated disclosure timeline

Implementation Examples
-----------------------

**Secure Configuration Loading:**

.. code-block:: python

   import os
   from pathlib import Path
   from dotenv import load_dotenv
   
   class SecureConfig:
       def __init__(self):
           # Load from .env file in development
           env_path = Path('.env')
           if env_path.exists():
               load_dotenv(env_path)
           
           # Required secrets
           self.openai_api_key = self._get_required_env('OPENAI_API_KEY')
           self.linear_api_key = self._get_required_env('LINEAR_API_KEY')
           
       def _get_required_env(self, key: str) -> str:
           value = os.getenv(key)
           if not value:
               raise ValueError(f"Required environment variable {key} not set")
           return value

**Secret Validation:**

.. code-block:: python

   import re
   
   def validate_api_key(key: str, key_type: str) -> bool:
       """Validate API key format without logging the key value."""
       patterns = {
           'openai': r'^sk-[a-zA-Z0-9]{48}$',
           'linear': r'^lin_api_[a-zA-Z0-9]{40}$'
       }
       
       pattern = patterns.get(key_type)
       if not pattern:
           return False
           
       is_valid = bool(re.match(pattern, key))
       
       # Log validation attempt without exposing key
       logger.info(f"API key validation for {key_type}: {'valid' if is_valid else 'invalid'}")
       
       return is_valid

Notes
-----

* Last updated: 2025-07-14
* Next review: 2025-08-14
* Security contact: security@apexsigma.com
* This document should be reviewed monthly and updated as needed