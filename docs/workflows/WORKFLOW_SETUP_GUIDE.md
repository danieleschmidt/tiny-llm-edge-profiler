# GitHub Workflows Setup Guide

This guide provides step-by-step instructions for setting up GitHub Actions workflows from the provided templates.

## 🚨 IMPORTANT: Manual Setup Required

Due to GitHub App permission limitations, workflow files cannot be automatically created in the `.github/workflows/` directory. Repository maintainers must manually copy the example workflows from `docs/workflows/examples/` to `.github/workflows/`.

## Prerequisites

Before setting up workflows, ensure you have:

1. **Repository Admin Access**: Required to create workflows and manage settings
2. **Required Secrets**: Configure necessary secrets in repository settings
3. **Branch Protection**: Set up branch protection rules for `main` branch
4. **Environment Configuration**: Create staging and production environments

## Workflow Setup Steps

### Step 1: Create Workflow Directory

```bash
mkdir -p .github/workflows
```

### Step 2: Copy Example Workflows

Copy the desired workflow files from `docs/workflows/examples/` to `.github/workflows/`:

```bash
# Core CI/CD workflows
cp docs/workflows/examples/ci.yml .github/workflows/
cp docs/workflows/examples/cd.yml .github/workflows/

# Security and maintenance workflows
cp docs/workflows/examples/security-scan.yml .github/workflows/
cp docs/workflows/examples/dependency-update.yml .github/workflows/
```

### Step 3: Configure Repository Secrets

Add the following secrets in your repository settings (`Settings > Secrets and variables > Actions`):

#### Required Secrets

| Secret Name | Description | Example Value |
|-------------|-------------|---------------|
| `AWS_ACCESS_KEY_ID` | AWS access key for deployments | `AKIA...` |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key for deployments | `xyz123...` |
| `STAGING_URL` | Staging environment URL | `https://staging.example.com` |
| `STAGING_API_KEY` | API key for staging tests | `sk_test_...` |
| `SLACK_WEBHOOK` | Slack webhook for notifications | `https://hooks.slack.com/...` |

#### Optional Secrets

| Secret Name | Description | Usage |
|-------------|-------------|-------|
| `SONAR_TOKEN` | SonarCloud authentication | Code quality analysis |
| `DOCKER_HUB_TOKEN` | Docker Hub access token | Container registry |
| `NPM_TOKEN` | NPM registry token | Package publishing |

### Step 4: Configure Environments

Create the following environments in repository settings (`Settings > Environments`):

#### Staging Environment
- **Name**: `staging`
- **Protection Rules**: None (auto-deploy from main)
- **Environment Secrets**: `STAGING_API_KEY`, `STAGING_URL`

#### Production Environment
- **Name**: `production`
- **Protection Rules**: 
  - Required reviewers (2+ team members)
  - Deployment branches: `main` and `v*` tags only
- **Environment Secrets**: Production-specific credentials

### Step 5: Branch Protection Configuration

Configure branch protection for `main` branch (`Settings > Branches`):

```yaml
# Branch protection settings
Protect matching branches: true
Restrictions:
  - Require a pull request before merging
  - Require approvals: 2
  - Dismiss stale PR approvals when new commits are pushed
  - Require review from code owners
  - Require status checks to pass before merging
    - Required status checks:
      - "Build and Test"
      - "Security Scan"
      - "Lint and Format"
  - Require branches to be up to date before merging
  - Require conversation resolution before merging
  - Include administrators
```

## Available Workflows

### 1. Continuous Integration (`ci.yml`)

**Purpose**: Automated testing, linting, and validation on pull requests and pushes.

**Features**:
- Multi-platform testing (Linux, macOS, Windows)
- Python version matrix testing (3.9, 3.10, 3.11)
- Code quality checks (linting, formatting)
- Security scanning
- Test coverage reporting

**Triggers**:
- Pull requests to `main`
- Pushes to `main` and `develop`

### 2. Continuous Deployment (`cd.yml`)

**Purpose**: Automated deployment pipeline for staging and production environments.

**Features**:
- Container building and pushing
- SBOM generation
- Blue-green deployments
- Integration testing
- Automated rollback on failure

**Triggers**:
- Pushes to `main` (staging deployment)
- Tagged releases `v*` (production deployment)

### 3. Security Scanning (`security-scan.yml`)

**Purpose**: Comprehensive security analysis of code, dependencies, and infrastructure.

**Features**:
- Secret scanning with TruffleHog
- Dependency vulnerability analysis
- Container image scanning
- Static code analysis with CodeQL
- Infrastructure security checks
- License compliance verification

**Triggers**:
- Pull requests to `main`
- Pushes to `main` and `develop`
- Weekly scheduled scans

### 4. Dependency Updates (`dependency-update.yml`)

**Purpose**: Automated dependency management and security monitoring.

**Features**:
- Python package updates
- Docker base image updates
- GitHub Actions version updates
- Security advisory monitoring
- Automated pull request creation

**Triggers**:
- Weekly scheduled runs
- Manual workflow dispatch

## Workflow Customization

### Environment-Specific Configuration

Modify workflow files to match your infrastructure:

```yaml
# Example: Update deployment targets
- name: Deploy to ECS
  run: |
    aws ecs update-service \
      --cluster YOUR-CLUSTER-NAME \
      --service YOUR-SERVICE-NAME \
      --force-new-deployment
```

### Notification Channels

Configure notification preferences:

```yaml
# Slack notifications
- name: Slack notification
  uses: 8398a7/action-slack@v3
  with:
    channel: '#your-channel'
    webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

### Test Configuration

Adjust test commands for your project:

```yaml
# Custom test execution
- name: Run tests
  run: |
    pytest tests/ \
      --cov=src/ \
      --cov-report=xml \
      --junit-xml=test-results.xml
```

## Security Considerations

### Secret Management
- Use GitHub Secrets for all sensitive data
- Rotate secrets regularly
- Use environment-specific secrets
- Never commit secrets to repository

### Access Control
- Limit workflow permissions using `permissions:` blocks
- Use least-privilege access for tokens
- Regular audit of repository access

### Supply Chain Security
- Pin action versions to specific commits
- Use official actions from verified publishers
- Enable Dependabot for GitHub Actions updates

## Monitoring and Maintenance

### Workflow Health Monitoring

Monitor workflow performance and reliability:

1. **Success Rates**: Track workflow success/failure rates
2. **Execution Time**: Monitor for performance degradation
3. **Resource Usage**: Track runner usage and costs
4. **Security Alerts**: Monitor for security notifications

### Regular Maintenance Tasks

- **Monthly**: Review workflow performance metrics
- **Quarterly**: Update workflow dependencies
- **Annually**: Audit security configurations and access

## Troubleshooting

### Common Issues

#### 1. Workflow Not Triggering
```yaml
# Check trigger conditions
on:
  push:
    branches: [main]  # Ensure branch name matches
  pull_request:
    branches: [main]
```

#### 2. Secret Access Issues
```yaml
# Verify secret names and environment context
env:
  SECRET_VALUE: ${{ secrets.SECRET_NAME }}
```

#### 3. Permission Errors
```yaml
# Add required permissions
permissions:
  contents: read
  packages: write
  security-events: write
```

### Debug Mode

Enable debug logging for troubleshooting:

```yaml
- name: Enable debug logging
  run: echo "ACTIONS_STEP_DEBUG=true" >> $GITHUB_ENV
```

## Support and Resources

- **GitHub Actions Documentation**: https://docs.github.com/en/actions
- **Action Marketplace**: https://github.com/marketplace?type=actions
- **Security Best Practices**: https://docs.github.com/en/actions/security-guides
- **Repository Issues**: Use repository issues for workflow-specific questions

---

**Next Steps**: After completing the workflow setup, proceed with configuring monitoring and alerting systems as described in the monitoring documentation.