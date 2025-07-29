# Security Policy

## Supported Versions

The following versions of tiny-llm-edge-profiler are currently supported with security updates:

| Version | Supported          | Notes |
| ------- | ------------------ | ----- |
| 0.1.x   | :white_check_mark: | Current development version |
| < 0.1   | :x:                | Pre-release, not supported |

## Security Considerations

### Hardware Security
- **Firmware Verification**: All profiling firmware is cryptographically signed
- **Device Authentication**: Secure handshake protocol for device communication
- **Isolation**: Models run in sandboxed environments on target devices
- **Memory Protection**: Sensitive data is cleared from device memory after profiling

### Data Protection
- **Model Privacy**: Optional encryption for proprietary models during transfer
- **Results Confidentiality**: Profiling data can be encrypted at rest
- **Communication Security**: TLS/SSL for network-based device communication
- **Access Control**: Role-based access to profiling infrastructure

### Common Security Risks

#### Hardware Access
- **Risk**: Unauthorized access to connected development boards
- **Mitigation**: Use device-specific authentication tokens

#### Model Extraction
- **Risk**: Proprietary models could be extracted from devices
- **Mitigation**: Models are automatically cleared after profiling

#### Communication Interception
- **Risk**: Profiling data could be intercepted during transmission  
- **Mitigation**: Use encrypted communication channels

## Reporting a Vulnerability

### Where to Report
**Please do not report security vulnerabilities through public GitHub issues.**

Report security vulnerabilities through one of these channels:

1. **GitHub Security Advisories** (Preferred)
   - Go to the repository's Security tab
   - Click "Report a vulnerability"
   - Fill out the private vulnerability report

2. **Email** (For sensitive issues)
   - Send details to: security@terragonlabs.com
   - Use PGP key if available for sensitive information

### What to Include

Please include the following information in your vulnerability report:

1. **Description**: Clear description of the vulnerability
2. **Impact**: Potential impact and severity assessment
3. **Reproduction**: Steps to reproduce the issue
4. **Environment**: Platform, version, and hardware details
5. **Evidence**: Screenshots, logs, or proof-of-concept code
6. **Suggested Fix**: If you have ideas for remediation

## Response Timeline

We are committed to addressing security issues promptly:

- **Initial response**: Within 48 hours of report
- **Confirmation**: Within 7 days if vulnerability is valid
- **Fix Development**: Target resolution within 30 days for critical issues
- **Disclosure**: Coordinated disclosure after fix is available

## Security Best Practices

### For Users

#### Secure Development Environment
```bash
# Use virtual environments
python -m venv profiler-env
source profiler-env/bin/activate

# Verify package integrity
pip install tiny-llm-edge-profiler --require-hashes

# Use secure device connections
tiny-profiler --device /dev/ttyUSB0 --secure-mode
```

#### Hardware Security
- Use dedicated development boards for profiling
- Isolate profiling network from production systems
- Regularly update device firmware
- Monitor for unauthorized device access

#### Model Protection
```python
from tiny_llm_profiler import EdgeProfiler

# Enable model encryption
profiler = EdgeProfiler(
    platform="esp32",
    encrypt_models=True,
    auto_cleanup=True  # Clear models after profiling
)
```

### For Contributors

#### Code Security
- Follow secure coding practices
- Use static analysis tools (bandit, semgrep)
- Implement input validation for all user inputs
- Handle secrets securely (never hardcode)

#### Testing Security
```bash
# Run security tests
pytest tests/security/ -v

# Security linting
bandit -r src/
semgrep --config=auto src/

# Dependency vulnerability scanning  
safety check
```

## Security Resources

### Standards Adherence
- **NIST Cybersecurity Framework**: Core security functions
- **OWASP Top 10**: Application security risks
- **CWE/SANS Top 25**: Common weakness enumeration

For additional security guidelines, see [GitHub Security Documentation](https://docs.github.com/en/code-security).