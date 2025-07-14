# Security Guidelines - ApexSigma DevEnviro

## 🔒 Secret Management

### **Critical Security Rules**

1. **NEVER commit secrets to git repositories**
   - API keys, tokens, passwords, certificates
   - Database connection strings with credentials
   - Private keys or sensitive configuration

2. **Use .gitignore for sensitive files**
   ```
   # Secrets and environment files
   .env
   .env.*
   config/secrets/
   *.key
   *.pem
   *.p12
   ```

3. **Environment Variables for Production**
   - Store secrets in environment variables
   - Use secure secret management services
   - Never hardcode secrets in source code

### **Local Development**

#### ✅ **Safe Practices:**
```bash
# Create local .env file (already in .gitignore)
echo "OPENAI_API_KEY=your_key_here" > .env

# Use python-dotenv to load secrets
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
```

#### ❌ **Dangerous Practices:**
```python
# NEVER do this
API_KEY = "sk-1234567890abcdef"  # Hardcoded secret
```

### **Production Deployment**

#### **GitHub Actions Secrets:**
1. Go to Repository Settings → Secrets and variables → Actions
2. Add secrets as repository secrets:
   - `OPENAI_API_KEY`
   - `DATABASE_URL`
   - `LINEAR_API_KEY`

#### **Environment Variable Usage:**
```yaml
# In GitHub Actions
env:
  OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
  DATABASE_URL: ${{ secrets.DATABASE_URL }}
```

### **Secret Scanning & Protection**

#### **GitHub Advanced Security:**
- Enable secret scanning in repository settings
- Configure push protection to block commits with secrets
- Review and remediate any detected secrets

#### **Pre-commit Hooks:**
```bash
# Install pre-commit hooks that check for secrets
pip install detect-secrets
pre-commit install
```

### **Incident Response**

#### **If you accidentally commit a secret:**

1. **Immediately revoke the compromised secret**
   - Generate new API keys
   - Rotate passwords
   - Update access tokens

2. **Remove from git history:**
   ```bash
   # Create clean branch without secret history
   git checkout --orphan clean-branch
   git add .
   git commit -m "Clean commit without secrets"
   git push -u origin clean-branch
   ```

3. **Force push to overwrite history (if no other collaborators):**
   ```bash
   # Use with extreme caution
   git filter-branch --force --index-filter \
     'git rm --cached --ignore-unmatch path/to/secret/file' \
     --prune-empty --tag-name-filter cat -- --all
   ```

### **Security Tools**

#### **Recommended Tools:**
- **detect-secrets**: Pre-commit hook for secret detection
- **git-secrets**: AWS tool for preventing secrets in git
- **truffleHog**: Search git repos for high entropy strings
- **GitGuardian**: Commercial secret scanning service

#### **Installation:**
```bash
pip install detect-secrets
detect-secrets scan --baseline .secrets.baseline
```

### **Code Review Security**

#### **Security Checklist for PRs:**
- [ ] No hardcoded secrets or API keys
- [ ] Environment variables used for configuration
- [ ] Sensitive files in .gitignore
- [ ] No database credentials in code
- [ ] No private keys or certificates committed

### **Monitoring & Alerts**

#### **Set up monitoring for:**
- Failed authentication attempts
- Unusual API usage patterns
- Unauthorized access attempts
- Secret rotation schedules

### **Compliance & Auditing**

#### **Documentation Requirements:**
- Log all secret rotations
- Maintain access control lists
- Document security incidents
- Regular security reviews

### **Contact & Reporting**

#### **Security Issues:**
- Report security vulnerabilities immediately
- Use encrypted communication for sensitive reports
- Follow responsible disclosure practices

---

## 📋 Security Checklist

### **Before Each Commit:**
- [ ] Check for secrets with `git diff`
- [ ] Verify .env files are not staged
- [ ] Run secret detection tools
- [ ] Review file changes for sensitive data

### **Before Each Release:**
- [ ] Audit all environment variables
- [ ] Verify secret rotation schedule
- [ ] Update security documentation
- [ ] Run security scans

### **Regular Maintenance:**
- [ ] Rotate API keys monthly
- [ ] Update security tools
- [ ] Review access permissions
- [ ] Audit secret usage

---

*Last updated: 2025-07-14*
*Review scheduled: Monthly*