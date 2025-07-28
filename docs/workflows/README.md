# Workflow Requirements

This document outlines the workflow requirements for this repository.

## Required Workflows (Manual Setup)

### 1. Continuous Integration (CI)
- **Purpose**: Run tests, linting, and type checking on PRs
- **Triggers**: Pull requests, pushes to main
- **Required Actions**:
  - Python code quality checks (black, flake8, mypy)
  - Unit test execution
  - Security scanning

### 2. Dependency Updates
- **Purpose**: Automated dependency management
- **Frequency**: Weekly
- **Tools**: Dependabot or Renovate Bot

### 3. Release Automation
- **Purpose**: Automated versioning and releases
- **Triggers**: Manual dispatch or tag creation
- **Components**: Changelog generation, version bumping

## Branch Protection Rules

### Main Branch Protection
- Require pull request reviews (minimum 1)
- Require status checks to pass
- Require branches to be up to date
- Dismiss stale reviews on new commits

## Repository Settings

### Required Settings
- Enable security alerts
- Enable automated security fixes
- Configure merge options (squash merge recommended)

## Manual Setup Steps

See [SETUP_REQUIRED.md](../SETUP_REQUIRED.md) for detailed manual setup instructions.

## References

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Branch Protection Rules](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches)