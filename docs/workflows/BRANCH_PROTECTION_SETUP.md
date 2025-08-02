# Branch Protection Configuration Guide

Comprehensive guide for setting up branch protection rules to ensure code quality and security for the tiny-llm-edge-profiler repository.

## Overview

Branch protection rules are essential for maintaining code quality, enforcing review processes, and preventing accidental or malicious changes to critical branches.

## Required Branch Protection Rules

### Main Branch Protection

Configure the following protection rules for the `main` branch:

#### Basic Protection Settings

```yaml
# Branch: main
protection_rules:
  required_status_checks:
    strict: true  # Require branches to be up to date before merging
    contexts:
      - "Build and Test (ubuntu-latest, 3.11)"
      - "Build and Test (ubuntu-latest, 3.10)"
      - "Build and Test (ubuntu-latest, 3.9)"
      - "Lint and Format"
      - "Security Scan / Static Code Analysis"
      - "Security Scan / Dependency Scanning"
      - "Security Scan / Container Scanning"
      
  enforce_admins: true  # Include administrators in restrictions
  
  required_pull_request_reviews:
    required_approving_review_count: 2
    dismiss_stale_reviews: true
    require_code_owner_reviews: true
    require_last_push_approval: false
    
  restrictions:
    users: []
    teams: []
    apps: []
    
  required_linear_history: false
  allow_force_pushes: false
  allow_deletions: false
  
  required_conversation_resolution: true
```

#### Advanced Protection Settings

```yaml
# Additional security settings
security_settings:
  require_signed_commits: true  # Require GPG signature verification
  
  lock_branch: false  # Allow pushes (with PR requirements)
  
  allow_fork_syncing: true  # Allow fork synchronization
  
  block_creations: false  # Allow branch/tag creation from this branch
```

## Setup Instructions

### 1. Via GitHub Web Interface

1. **Navigate to Repository Settings**
   - Go to `Settings > Branches`
   - Click "Add rule" for branch protection

2. **Configure Branch Name Pattern**
   ```
   Branch name pattern: main
   ```

3. **Enable Required Status Checks**
   - ✅ Require status checks to pass before merging
   - ✅ Require branches to be up to date before merging
   - Select all relevant status checks from your CI workflows

4. **Configure Pull Request Requirements**
   - ✅ Require a pull request before merging
   - Set "Required approving reviews" to `2`
   - ✅ Dismiss stale PR approvals when new commits are pushed
   - ✅ Require review from code owners

5. **Advanced Settings**
   - ✅ Require conversation resolution before merging
   - ✅ Include administrators
   - ❌ Allow force pushes (disabled for security)
   - ❌ Allow deletions (disabled for security)

### 2. Via GitHub CLI

```bash
# Create branch protection rule using GitHub CLI
gh api repos/$OWNER/$REPO/branches/main/protection \
  --method PUT \
  --field required_status_checks='{"strict":true,"contexts":["Build and Test (ubuntu-latest, 3.11)","Lint and Format","Security Scan / Static Code Analysis"]}' \
  --field enforce_admins=true \
  --field required_pull_request_reviews='{"required_approving_review_count":2,"dismiss_stale_reviews":true,"require_code_owner_reviews":true}' \
  --field restrictions=null \
  --field allow_force_pushes=false \
  --field allow_deletions=false \
  --field required_conversation_resolution=true
```

### 3. Via Terraform (Infrastructure as Code)

```hcl
# terraform/github_branch_protection.tf
resource "github_branch_protection" "main" {
  repository_id = github_repository.tiny_llm_profiler.node_id
  pattern       = "main"

  required_status_checks {
    strict   = true
    contexts = [
      "Build and Test (ubuntu-latest, 3.11)",
      "Build and Test (ubuntu-latest, 3.10)", 
      "Build and Test (ubuntu-latest, 3.9)",
      "Lint and Format",
      "Security Scan / Static Code Analysis",
      "Security Scan / Dependency Scanning",
      "Security Scan / Container Scanning"
    ]
  }

  required_pull_request_reviews {
    required_approving_review_count = 2
    dismiss_stale_reviews          = true
    require_code_owner_reviews     = true
    require_last_push_approval     = false
  }

  enforce_admins         = true
  allows_deletions       = false
  allows_force_pushes    = false
  require_conversation_resolution = true
}
```

## Development Branch Protection

### Develop Branch (Optional)

For repositories using GitFlow or similar branching strategies:

```yaml
# Branch: develop
protection_rules:
  required_status_checks:
    strict: true
    contexts:
      - "Build and Test (ubuntu-latest, 3.11)"
      - "Lint and Format"
      - "Security Scan / Static Code Analysis"
      
  required_pull_request_reviews:
    required_approving_review_count: 1
    dismiss_stale_reviews: true
    require_code_owner_reviews: false
    
  enforce_admins: false  # More flexible for development
  allow_force_pushes: false
  allow_deletions: false
```

## Release Branch Protection

### Release Branches (release/*)

```yaml
# Branch pattern: release/*
protection_rules:
  required_status_checks:
    strict: true
    contexts:
      - "Build and Test (ubuntu-latest, 3.11)"
      - "Security Scan / Comprehensive"
      - "Integration Tests"
      
  required_pull_request_reviews:
    required_approving_review_count: 2
    dismiss_stale_reviews: true
    require_code_owner_reviews: true
    
  enforce_admins: true
  allow_force_pushes: false
  allow_deletions: false
  
  # Additional release-specific settings
  restrict_pushes: true  # Only allow merges via PR
```

## Status Check Configuration

### Required Status Checks Mapping

Map your workflow job names to required status checks:

| Workflow | Job Name | Status Check Context |
|----------|----------|---------------------|
| CI | `test` | `Build and Test (ubuntu-latest, 3.11)` |
| CI | `lint` | `Lint and Format` |
| Security | `static-analysis` | `Security Scan / Static Code Analysis` |
| Security | `dependency-scanning` | `Security Scan / Dependency Scanning` |
| Security | `container-scanning` | `Security Scan / Container Scanning` |

### Custom Status Checks

Create custom status checks for specific requirements:

```python
# scripts/custom_status_check.py
import requests
import sys

def create_status_check(repo, sha, state, context, description):
    """Create a custom status check"""
    url = f"https://api.github.com/repos/{repo}/statuses/{sha}"
    
    data = {
        "state": state,  # success, error, failure, pending
        "context": context,
        "description": description,
        "target_url": f"https://github.com/{repo}/actions"
    }
    
    headers = {
        "Authorization": f"token {os.environ['GITHUB_TOKEN']}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    response = requests.post(url, json=data, headers=headers)
    return response.status_code == 201

# Usage in workflow
if __name__ == "__main__":
    repo = sys.argv[1]  # "owner/repo"
    sha = sys.argv[2]   # commit SHA
    
    # Check custom requirements
    if check_custom_requirements():
        create_status_check(repo, sha, "success", "Custom Check", "All custom requirements met")
    else:
        create_status_check(repo, sha, "failure", "Custom Check", "Custom requirements not met")
```

## Code Owners Configuration

Create a `CODEOWNERS` file to enforce review requirements:

```bash
# .github/CODEOWNERS

# Global owners
* @team-leads @senior-developers

# Source code
/src/ @backend-team @security-team

# Tests
/tests/ @qa-team @backend-team

# Infrastructure and deployment
/docker/ @devops-team @security-team
/scripts/ @devops-team
/.github/workflows/ @devops-team @security-team

# Documentation
/docs/ @tech-writers @team-leads
README.md @team-leads
*.md @tech-writers

# Security sensitive files
/SECURITY.md @security-team
/requirements*.txt @security-team @backend-team
/Dockerfile @security-team @devops-team

# Configuration files
/*.yml @devops-team
/*.yaml @devops-team
/*.json @backend-team
/pyproject.toml @backend-team

# CI/CD
/.github/ @devops-team @security-team
```

## Emergency Procedures

### Temporary Protection Bypass

For critical hotfixes, establish an emergency procedure:

1. **Create Emergency Branch**
   ```bash
   git checkout -b hotfix/critical-fix
   git push origin hotfix/critical-fix
   ```

2. **Apply Temporary Protection Override**
   - Temporarily disable "Include administrators" if needed
   - Merge critical fix with reduced review requirements
   - Document the emergency procedure

3. **Restore Protection Rules**
   - Re-enable all protection rules
   - Create follow-up PR with proper review process
   - Update emergency runbook

### Protection Rule Recovery

Script to restore protection rules after emergency changes:

```bash
#!/bin/bash
# scripts/restore_branch_protection.sh

REPO="owner/repo"
BRANCH="main"

echo "Restoring branch protection for $BRANCH..."

gh api repos/$REPO/branches/$BRANCH/protection \
  --method PUT \
  --field required_status_checks='{"strict":true,"contexts":["Build and Test (ubuntu-latest, 3.11)","Lint and Format","Security Scan / Static Code Analysis"]}' \
  --field enforce_admins=true \
  --field required_pull_request_reviews='{"required_approving_review_count":2,"dismiss_stale_reviews":true,"require_code_owner_reviews":true}' \
  --field restrictions=null \
  --field allow_force_pushes=false \
  --field allow_deletions=false \
  --field required_conversation_resolution=true

echo "✅ Branch protection restored"
```

## Monitoring and Compliance

### Branch Protection Monitoring

Create alerts for branch protection rule changes:

```yaml
# .github/workflows/branch-protection-monitor.yml
name: Branch Protection Monitor

on:
  repository_dispatch:
    types: [branch_protection_rule]

jobs:
  monitor:
    runs-on: ubuntu-latest
    steps:
      - name: Log protection rule changes
        run: |
          echo "Branch protection rule modified: ${{ github.event.client_payload.action }}"
          # Send notification to security team
```

### Compliance Reporting

Generate compliance reports for branch protection:

```python
# scripts/compliance_report.py
import requests
import json

def generate_branch_protection_report(repo):
    """Generate branch protection compliance report"""
    url = f"https://api.github.com/repos/{repo}/branches/main/protection"
    headers = {"Authorization": f"token {os.environ['GITHUB_TOKEN']}"}
    
    response = requests.get(url, headers=headers)
    protection = response.json()
    
    compliance = {
        "repository": repo,
        "branch": "main",
        "compliant": True,
        "issues": []
    }
    
    # Check required reviews
    if protection["required_pull_request_reviews"]["required_approving_review_count"] < 2:
        compliance["compliant"] = False
        compliance["issues"].append("Insufficient required reviews")
    
    # Check admin enforcement
    if not protection["enforce_admins"]["enabled"]:
        compliance["compliant"] = False
        compliance["issues"].append("Admin enforcement disabled")
    
    return compliance
```

This comprehensive branch protection setup ensures code quality, security, and compliance while maintaining development velocity.