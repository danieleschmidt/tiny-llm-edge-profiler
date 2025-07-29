# CI/CD Setup Instructions

This document provides detailed instructions for setting up GitHub Actions workflows for the tiny-llm-edge-profiler project.

## Quick Setup Checklist

- [ ] Copy workflow templates to `.github/workflows/`
- [ ] Configure repository secrets
- [ ] Set up branch protection rules
- [ ] Configure self-hosted runners (for hardware tests)
- [ ] Enable security scanning
- [ ] Configure deployment settings

## 1. Workflow Setup

### Copy Templates

```bash
# Create workflows directory
mkdir -p .github/workflows

# Copy CI/CD workflow
cp docs/workflows/github-actions-templates/ci.yml .github/workflows/

# Optional: Copy additional workflows
cp docs/workflows/github-actions-templates/release.yml .github/workflows/
cp docs/workflows/github-actions-templates/security.yml .github/workflows/
```

### Enable Workflows

1. Push the workflow files to your repository
2. Go to **Actions** tab in GitHub
3. Enable workflows if prompted
4. Verify workflows appear in the Actions tab

## 2. Repository Secrets Configuration

### Required Secrets

Configure these secrets in **Settings → Secrets and variables → Actions**:

#### For PyPI Publishing
```
PYPI_TOKEN = <your-pypi-api-token>
```

#### For Hardware Testing (if using)
```
HARDWARE_TEST_CONFIG = <base64-encoded-config>
ESP32_DEVICE_PATH = /dev/ttyUSB0
POWER_SENSOR_CONFIG = <sensor-configuration>
```

#### For Documentation (if using external hosting)
```
DOCS_DEPLOY_KEY = <ssh-private-key>
DOCS_HOST = <documentation-host>
```

### Creating Secrets

1. Navigate to repository **Settings → Secrets and variables → Actions**
2. Click **New repository secret**
3. Add each required secret with appropriate values

## 3. Branch Protection Rules

### Main Branch Protection

Configure protection for the `main` branch:

1. Go to **Settings → Branches**
2. Click **Add rule** or edit existing rule for `main`
3. Configure the following settings:

#### Required Settings
- [x] **Require a pull request before merging**
  - [x] Require approvals: `1`
  - [x] Dismiss stale reviews when new commits are pushed
  - [x] Require review from code owners

- [x] **Require status checks to pass before merging**
  - [x] Require branches to be up to date before merging
  - **Required status checks:**
    - `Test Suite (3.8)`
    - `Test Suite (3.9)`
    - `Test Suite (3.10)`
    - `Test Suite (3.11)`
    - `Test Suite (3.12)`
    - `Type Checking`
    - `Security Scan`
    - `Build Package`

- [x] **Require conversation resolution before merging**
- [x] **Include administrators**

## 4. Self-Hosted Runners (Hardware Testing)

### When You Need Self-Hosted Runners

Self-hosted runners are required for:
- Hardware-in-the-loop testing with ESP32, STM32, etc.
- Power consumption measurements
- Real device profiling tests

### Setting Up Self-Hosted Runners

#### Prerequisites
- Linux machine with USB access to development boards
- Python 3.8+ installed
- Hardware devices (ESP32, STM32F4, RP2040, etc.)

#### Installation Steps

1. **Create runner in GitHub:**
   - Go to **Settings → Actions → Runners**
   - Click **New self-hosted runner**
   - Follow instructions to download and configure runner

2. **Install hardware dependencies:**
   ```bash
   # Install system dependencies
   sudo apt update
   sudo apt install -y build-essential libusb-1.0-0-dev
   
   # Install Python dependencies
   pip install pyserial pyftdi esptool
   
   # Add user to dialout group for USB access
   sudo usermod -a -G dialout $USER
   ```

3. **Configure hardware devices:**
   ```bash
   # Create udev rules for consistent device naming
   sudo tee /etc/udev/rules.d/99-embedded-devices.rules << EOF
   # ESP32 devices
   SUBSYSTEM=="tty", ATTRS{idVendor}=="10c4", ATTRS{idProduct}=="ea60", SYMLINK+="esp32_%n"
   
   # STM32 devices
   SUBSYSTEM=="tty", ATTRS{idVendor}=="0483", ATTRS{idProduct}=="5740", SYMLINK+="stm32_%n"
   EOF
   
   sudo udevadm control --reload-rules
   ```

4. **Test hardware connectivity:**
   ```bash
   # Test Python hardware access
   python -c "
   from tiny_llm_profiler import EdgeProfiler
   profiler = EdgeProfiler(platform='esp32')
   devices = profiler.discover_devices()
   print(f'Found {len(devices)} devices')
   "
   ```

### Runner Configuration

Create `runner-config.yaml`:
```yaml
hardware_config:
  esp32:
    device_path: "/dev/esp32_0"
    baudrate: 921600
  stm32f4:
    device_path: "/dev/stm32_0"
    baudrate: 115200
  power_sensor:
    type: "ina219"
    i2c_address: 0x40
```

## 5. Security Scanning Setup

### Enable Dependabot

1. Create `.github/dependabot.yml`:
   ```yaml
   version: 2
   updates:
     - package-ecosystem: "pip"
       directory: "/"
       schedule:
         interval: "weekly"
       open-pull-requests-limit: 5
   ```

### Enable Security Advisories

1. Go to **Settings → Security & analysis**
2. Enable:
   - [x] Dependency graph
   - [x] Dependabot alerts
   - [x] Dependabot security updates
   - [x] Code scanning alerts

### Configure CodeQL

1. Go to **Security → Code scanning alerts**
2. Click **Set up code scanning**
3. Choose **GitHub Actions**
4. Commit the generated workflow

## 6. Performance Monitoring

### Benchmark Tracking

The CI workflow includes performance benchmarks that track:
- Model quantization speed
- Profiler initialization time
- Memory usage patterns
- Hardware communication latency

### Viewing Performance Trends

1. Go to **Actions** tab
2. Select a completed workflow run
3. View **Performance Benchmarks** job
4. Check artifacts for detailed reports

## 7. Documentation Deployment

### MkDocs Configuration

If using MkDocs for documentation:

1. Create `mkdocs.yml` in repository root
2. Configure deployment in workflow
3. Set up GitHub Pages or external hosting

### Auto-deployment Setup

```yaml
# Add to workflow for documentation deployment
- name: Deploy documentation
  if: github.ref == 'refs/heads/main'
  run: |
    mkdocs gh-deploy --force
```

## 8. Troubleshooting

### Common Issues

#### Hardware Tests Failing
```bash
# Check device permissions
ls -la /dev/tty*
groups $USER

# Test device connectivity
python -c "import serial; print(serial.tools.list_ports.comports())"
```

#### Build Failures
```bash
# Check Python version compatibility
python --version
pip list | grep -E "(setuptools|wheel|build)"

# Validate package configuration
python -m build --wheel
twine check dist/*
```

#### Security Scan Failures
```bash
# Run security tools locally
safety check
bandit -r src/
semgrep --config=auto src/
```

### Getting Help

1. Check [GitHub Actions documentation](https://docs.github.com/en/actions)
2. Review workflow logs in the Actions tab
3. Open an issue in the repository for project-specific problems

## 9. Advanced Configuration

### Matrix Testing

Expand the test matrix for more comprehensive coverage:

```yaml
strategy:
  matrix:
    python-version: ['3.8', '3.9', '3.10', '3.11', '3.12']
    os: [ubuntu-latest, windows-latest, macos-latest]
    exclude:
      - os: windows-latest
        python-version: '3.8'  # Skip older Python on Windows
```

### Conditional Workflows

Use conditions to control when workflows run:

```yaml
# Only run on specific file changes
on:
  push:
    paths:
      - 'src/**'
      - 'tests/**'
      - 'pyproject.toml'

# Skip CI for documentation-only changes
jobs:
  test:
    if: "!contains(github.event.head_commit.message, '[skip ci]')"
```

### Custom Actions

Create reusable actions for common tasks:

```yaml
# .github/actions/setup-hardware/action.yml
name: 'Setup Hardware Environment'
description: 'Configure hardware testing environment'
runs:
  using: 'composite'
  steps:
    - run: |
        echo "Setting up hardware environment..."
        # Hardware setup commands
      shell: bash
```

This comprehensive setup ensures robust CI/CD for your embedded systems profiling project!