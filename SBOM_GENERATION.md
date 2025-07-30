# Software Bill of Materials (SBOM) Generation Guide

This document outlines the process for generating and maintaining Software Bill of Materials (SBOM) for the tiny-llm-edge-profiler project to ensure supply chain security and compliance.

## Overview

The SBOM provides a comprehensive inventory of all software components, dependencies, and their relationships within the project. This is critical for:

- **Supply Chain Security**: Track all third-party components and their vulnerabilities
- **Compliance**: Meet regulatory requirements (NIST, EU Cyber Resilience Act)
- **Risk Management**: Assess and mitigate security risks from dependencies
- **Incident Response**: Quickly identify affected components during security incidents

## SBOM Generation Process

### 1. Automated Generation via Scripts

Use the provided script to generate SBOM in multiple formats:

```bash
# Generate SBOM for current environment
python scripts/generate_sbom.py

# Generate SBOM for production build
python scripts/generate_sbom.py --environment production

# Generate SBOM with vulnerability data
python scripts/generate_sbom.py --include-vulnerabilities

# Generate SBOM in specific format
python scripts/generate_sbom.py --format spdx-json
```

### 2. Manual Generation Commands

For direct control over SBOM generation:

```bash
# Install SBOM generation tools
pip install cyclonedx-bom pip-audit syft

# Generate CycloneDX SBOM (JSON format)
cyclonedx-py --format json --output-file sbom/sbom-cyclonedx.json .

# Generate SPDX SBOM
pip-audit --format=cyclonedx-json --output=sbom/sbom-spdx.json

# Generate detailed component analysis
syft . -o spdx-json=sbom/sbom-syft.json
```

### 3. CI/CD Integration

SBOM generation is integrated into the CI/CD pipeline:

```yaml
# Example GitHub Actions workflow step
- name: Generate SBOM
  run: |
    python scripts/generate_sbom.py --format cyclonedx-json
    python scripts/generate_sbom.py --format spdx-json
    
- name: Upload SBOM Artifacts
  uses: actions/upload-artifact@v3
  with:
    name: sbom-files
    path: sbom/
```

## SBOM Formats Supported

### 1. CycloneDX Format
- **File**: `sbom/sbom-cyclonedx.json`
- **Standard**: OWASP CycloneDX v1.4+
- **Use Case**: Industry standard for dependency tracking

### 2. SPDX Format  
- **File**: `sbom/sbom-spdx.json`
- **Standard**: SPDX v2.3+
- **Use Case**: Legal compliance and license tracking

### 3. SWID Format
- **File**: `sbom/sbom-swid.json`  
- **Standard**: ISO/IEC 19770-2
- **Use Case**: Software identification for asset management

## SBOM Components Tracked

### Python Dependencies
- **Runtime Dependencies**: All packages from requirements.txt
- **Development Dependencies**: Packages from requirements-dev.txt
- **Optional Dependencies**: Hardware-specific packages
- **Transitive Dependencies**: All sub-dependencies recursively

### System Components
- **Base Images**: Docker base images and their components
- **System Libraries**: OS-level libraries used by the application
- **Build Tools**: Compilers, build systems, and toolchains

### Embedded Firmware
- **Toolchains**: ARM, RISC-V, ESP32 development toolchains
- **SDK Components**: Platform-specific SDKs and libraries
- **Bootloaders**: Firmware components for target devices

## Vulnerability Tracking

### Automated Vulnerability Scanning
```bash
# Scan for known vulnerabilities
pip-audit --format json --output sbom/vulnerabilities.json

# Security scanning with safety
safety check --json --output sbom/safety-report.json

# Container vulnerability scanning
grype sbom/sbom-cyclonedx.json -o json > sbom/container-vulnerabilities.json
```

### Vulnerability Database Sources
- **National Vulnerability Database (NVD)**
- **OSV Database (Open Source Vulnerabilities)**
- **GitHub Security Advisories**
- **PyPI Security Advisories**

## Compliance Requirements

### NIST Guidelines
- **NIST SP 800-161**: Supply Chain Risk Management
- **NIST SP 800-218**: SSDF (Secure Software Development Framework)
- **Executive Order 14028**: Improving Cybersecurity

### EU Cyber Resilience Act
- **Article 10**: Security requirements for products with digital elements
- **Annex I**: Essential cybersecurity requirements
- **Vulnerability handling**: Coordinated disclosure process

### Industry Standards
- **NTIA Minimum Elements**: Required SBOM data fields
- **CISA Guidelines**: Federal software supply chain security
- **ISO 27001**: Information security management

## SBOM Maintenance

### Regular Updates
- **Dependency Updates**: Regenerate SBOM when dependencies change  
- **Vulnerability Refresh**: Update vulnerability data weekly
- **Release SBOM**: Generate clean SBOM for each release

### Version Control
- **SBOM History**: Track SBOM changes over time
- **Baseline Comparison**: Compare SBOMs between versions
- **Change Documentation**: Document significant component changes

### Automated Monitoring
```bash
# Set up automated vulnerability monitoring
pip-audit --format json --output - | jq '.vulnerabilities | length'

# Monitor for new vulnerabilities daily
echo "0 6 * * * cd /project && python scripts/check_vulnerabilities.py" | crontab -
```

## SBOM Distribution

### Internal Distribution
- **Development Teams**: Access to complete SBOM with vulnerability data
- **Security Team**: Full SBOM for risk assessment
- **Compliance Team**: SBOM for audit and regulatory purposes

### External Distribution  
- **Customers**: Sanitized SBOM without internal details
- **Partners**: Component compatibility information
- **Regulators**: Compliance-focused SBOM subset

### Distribution Formats
```bash
# Customer-facing SBOM (sanitized)
python scripts/generate_sbom.py --customer-format --output sbom/customer-sbom.json

# Partner SBOM (compatibility focus)  
python scripts/generate_sbom.py --partner-format --output sbom/partner-sbom.json

# Regulatory SBOM (compliance focus)
python scripts/generate_sbom.py --compliance-format --output sbom/compliance-sbom.json
```

## Integration with Development Workflow

### Pre-commit Hooks
```yaml
# .pre-commit-config.yaml addition
- repo: local
  hooks:
    - id: sbom-generation
      name: Generate SBOM
      entry: python scripts/generate_sbom.py --quick
      language: python
      files: requirements.*\.txt$
      pass_filenames: false
```

### IDE Integration
- **VS Code Extension**: SBOM viewer and vulnerability highlighting
- **PyCharm Plugin**: Dependency security analysis
- **Vim Plugin**: Inline vulnerability warnings

### Documentation Integration
- **Automatic Updates**: SBOM summary in README.md
- **Security Documentation**: Link SBOM to security policies
- **Release Notes**: Include SBOM changes in release documentation

## Troubleshooting

### Common Issues
1. **Missing Dependencies**: Ensure all requirements files are up to date
2. **Format Errors**: Validate SBOM files with official schemas
3. **Large File Sizes**: Use compression for distribution
4. **Performance**: Use incremental updates for large projects

### Validation Commands
```bash
# Validate CycloneDX SBOM
cyclonedx-bom validate --input-file sbom/sbom-cyclonedx.json

# Validate SPDX SBOM  
spdx-tools --validate sbom/sbom-spdx.json

# Check SBOM completeness
python scripts/validate_sbom.py sbom/sbom-cyclonedx.json
```

## References

- [NIST SBOM Guidance](https://www.nist.gov/itl/executive-order-improving-nations-cybersecurity/software-supply-chain-security-guidance)
- [CycloneDX Specification](https://cyclonedx.org/specification/overview/)
- [SPDX Specification](https://spdx.github.io/spdx-spec/)
- [NTIA SBOM Minimum Elements](https://www.ntia.doc.gov/files/ntia/publications/sbom_minimum_elements_report.pdf)
- [CISA SBOM Guide](https://www.cisa.gov/sites/default/files/publications/SBOM_Sharing_Primer_Intl_20210818.pdf)