# Manual Setup Requirements

This document lists items that require manual setup due to permission limitations.

## GitHub Actions Workflows

**Location**: `.github/workflows/`

### Required Workflow Files

1. **ci.yml** - Continuous Integration
   - Python testing and linting
   - Security scanning
   - Code quality checks

2. **dependabot.yml** - Dependency Updates
   - Automated dependency management
   - Security vulnerability fixes

3. **release.yml** - Release Automation
   - Automated versioning
   - Changelog generation

## Repository Settings

### Branch Protection (Admin Required)

Navigate to: Settings → Branches → Add rule

- **Branch name pattern**: `main`
- **Require pull request reviews**: ✓ (minimum 1)
- **Require status checks**: ✓
- **Require up-to-date branches**: ✓
- **Dismiss stale reviews**: ✓

### Security Settings

Navigate to: Settings → Security & analysis

- **Dependency graph**: ✓
- **Dependabot alerts**: ✓
- **Dependabot security updates**: ✓
- **Secret scanning**: ✓

### General Settings

- **Merge button**: Configure merge options
- **Topics**: Add relevant repository topics
- **Description**: Update repository description

## External Integrations

### Recommended Tools
- Code coverage reporting (Codecov)
- Security scanning (Snyk, CodeQL)
- Documentation hosting (Read the Docs)

## References

- [GitHub Repository Settings](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features)
- [GitHub Actions Setup](https://docs.github.com/en/actions/quickstart)