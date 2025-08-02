#!/bin/bash
# Comprehensive security scanning script
# Runs multiple security tools and generates consolidated report

set -euo pipefail

# Configuration
REPORT_DIR="reports/security"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
CONSOLIDATED_REPORT="$REPORT_DIR/security_report_$TIMESTAMP.md"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Create reports directory
mkdir -p "$REPORT_DIR"

# Initialize consolidated report
cat > "$CONSOLIDATED_REPORT" << EOF
# Security Scan Report

**Generated:** $(date)
**Project:** tiny-llm-edge-profiler
**Scan ID:** $TIMESTAMP

## Executive Summary

This report contains the results of automated security scanning across multiple dimensions:
- Code security analysis (Bandit, Semgrep)
- Dependency vulnerability scanning (Safety, Pip-audit)
- Container security scanning (if Docker available)
- SBOM generation and analysis
- License compliance checking

---

EOF

# Track overall security status
TOTAL_ISSUES=0
CRITICAL_ISSUES=0
HIGH_ISSUES=0
MEDIUM_ISSUES=0
LOW_ISSUES=0

# Function to add section to report
add_section() {
    local title="$1"
    local content="$2"
    
    cat >> "$CONSOLIDATED_REPORT" << EOF

## $title

$content

EOF
}

# Function to run bandit security scan
run_bandit() {
    log_info "Running Bandit security scan..."
    
    local bandit_report="$REPORT_DIR/bandit_$TIMESTAMP.json"
    local bandit_summary=""
    
    if bandit -r src/ -f json -o "$bandit_report" 2>/dev/null; then
        local issues_count
        issues_count=$(jq '.results | length' "$bandit_report" 2>/dev/null || echo "0")
        bandit_summary="✅ **Bandit**: $issues_count security issues found"
        
        if [[ "$issues_count" -gt 0 ]]; then
            local high_issues
            local medium_issues
            local low_issues
            
            high_issues=$(jq '[.results[] | select(.issue_severity == "HIGH")] | length' "$bandit_report" 2>/dev/null || echo "0")
            medium_issues=$(jq '[.results[] | select(.issue_severity == "MEDIUM")] | length' "$bandit_report" 2>/dev/null || echo "0")
            low_issues=$(jq '[.results[] | select(.issue_severity == "LOW")] | length' "$bandit_report" 2>/dev/null || echo "0")
            
            HIGH_ISSUES=$((HIGH_ISSUES + high_issues))
            MEDIUM_ISSUES=$((MEDIUM_ISSUES + medium_issues))
            LOW_ISSUES=$((LOW_ISSUES + low_issues))
            
            bandit_summary="⚠️  **Bandit**: $issues_count issues (High: $high_issues, Medium: $medium_issues, Low: $low_issues)"
        fi
    else
        bandit_summary="❌ **Bandit**: Scan failed or not available"
    fi
    
    add_section "Bandit Code Security Analysis" "$bandit_summary

See detailed report: [\`bandit_$TIMESTAMP.json\`]($bandit_report)"
    
    log_success "Bandit scan completed"
}

# Function to run safety dependency scan
run_safety() {
    log_info "Running Safety dependency scan..."
    
    local safety_report="$REPORT_DIR/safety_$TIMESTAMP.json"
    local safety_summary=""
    
    if safety check --json --output "$safety_report" 2>/dev/null; then
        safety_summary="✅ **Safety**: No known vulnerabilities found in dependencies"
    else
        local vuln_count
        vuln_count=$(jq '. | length' "$safety_report" 2>/dev/null || echo "unknown")
        safety_summary="⚠️  **Safety**: $vuln_count vulnerable dependencies found"
        
        if [[ "$vuln_count" != "unknown" && "$vuln_count" -gt 0 ]]; then
            HIGH_ISSUES=$((HIGH_ISSUES + vuln_count))
        fi
    fi
    
    add_section "Safety Dependency Vulnerability Scan" "$safety_summary

See detailed report: [\`safety_$TIMESTAMP.json\`]($safety_report)"
    
    log_success "Safety scan completed"
}

# Function to run pip-audit
run_pip_audit() {
    log_info "Running pip-audit dependency scan..."
    
    local audit_report="$REPORT_DIR/pip-audit_$TIMESTAMP.json"
    local audit_summary=""
    
    if command -v pip-audit &> /dev/null; then
        if pip-audit --format=json --output="$audit_report" 2>/dev/null; then
            audit_summary="✅ **Pip-audit**: No vulnerabilities found"
        else
            local vuln_count
            vuln_count=$(jq '.vulnerabilities | length' "$audit_report" 2>/dev/null || echo "unknown")
            audit_summary="⚠️  **Pip-audit**: $vuln_count vulnerabilities found"
            
            if [[ "$vuln_count" != "unknown" && "$vuln_count" -gt 0 ]]; then
                HIGH_ISSUES=$((HIGH_ISSUES + vuln_count))
            fi
        fi
        
        add_section "Pip-audit Dependency Analysis" "$audit_summary

See detailed report: [\`pip-audit_$TIMESTAMP.json\`]($audit_report)"
    else
        log_warning "pip-audit not available, skipping"
        add_section "Pip-audit Dependency Analysis" "❌ **Pip-audit**: Not available - install with \`pip install pip-audit\`"
    fi
    
    log_success "Pip-audit scan completed"
}

# Function to run semgrep
run_semgrep() {
    log_info "Running Semgrep code analysis..."
    
    local semgrep_report="$REPORT_DIR/semgrep_$TIMESTAMP.json"
    local semgrep_summary=""
    
    if command -v semgrep &> /dev/null; then
        if semgrep --config=auto --json --output="$semgrep_report" src/ 2>/dev/null; then
            local findings_count
            findings_count=$(jq '.results | length' "$semgrep_report" 2>/dev/null || echo "0")
            
            if [[ "$findings_count" -eq 0 ]]; then
                semgrep_summary="✅ **Semgrep**: No security issues found"
            else
                local critical_count
                local high_count
                local medium_count
                local low_count
                
                critical_count=$(jq '[.results[] | select(.extra.severity == "ERROR")] | length' "$semgrep_report" 2>/dev/null || echo "0")
                high_count=$(jq '[.results[] | select(.extra.severity == "WARNING")] | length' "$semgrep_report" 2>/dev/null || echo "0")
                medium_count=$(jq '[.results[] | select(.extra.severity == "INFO")] | length' "$semgrep_report" 2>/dev/null || echo "0")
                
                CRITICAL_ISSUES=$((CRITICAL_ISSUES + critical_count))
                HIGH_ISSUES=$((HIGH_ISSUES + high_count))
                MEDIUM_ISSUES=$((MEDIUM_ISSUES + medium_count))
                
                semgrep_summary="⚠️  **Semgrep**: $findings_count findings (Critical: $critical_count, High: $high_count, Medium: $medium_count)"
            fi
        else
            semgrep_summary="❌ **Semgrep**: Scan failed"
        fi
        
        add_section "Semgrep Code Analysis" "$semgrep_summary

See detailed report: [\`semgrep_$TIMESTAMP.json\`]($semgrep_report)"
    else
        log_warning "Semgrep not available, skipping"
        add_section "Semgrep Code Analysis" "❌ **Semgrep**: Not available - install with \`pip install semgrep\`"
    fi
    
    log_success "Semgrep scan completed"
}

# Function to check container security
run_container_security() {
    log_info "Running container security checks..."
    
    local container_summary=""
    
    if command -v docker &> /dev/null && docker info &> /dev/null; then
        # Build production image for scanning
        if docker build --target production -t tiny-llm-profiler:security-scan . &> /dev/null; then
            container_summary="✅ **Container Build**: Production image builds successfully"
            
            # Check for common security issues
            local dockerfile_issues=0
            
            # Check if running as root
            if docker run --rm tiny-llm-profiler:security-scan whoami 2>/dev/null | grep -q "root"; then
                container_summary="$container_summary
⚠️  **Warning**: Container runs as root user"
                dockerfile_issues=$((dockerfile_issues + 1))
            else
                container_summary="$container_summary
✅ **User**: Container runs as non-root user"
            fi
            
            # Check image size
            local image_size
            image_size=$(docker images tiny-llm-profiler:security-scan --format "{{.Size}}")
            container_summary="$container_summary
📊 **Image Size**: $image_size"
            
            if [[ "$dockerfile_issues" -gt 0 ]]; then
                MEDIUM_ISSUES=$((MEDIUM_ISSUES + dockerfile_issues))
            fi
            
            # Cleanup
            docker rmi tiny-llm-profiler:security-scan &> /dev/null || true
        else
            container_summary="❌ **Container Build**: Failed to build production image"
            HIGH_ISSUES=$((HIGH_ISSUES + 1))
        fi
    else
        container_summary="❌ **Docker**: Not available or not running"
    fi
    
    add_section "Container Security Analysis" "$container_summary"
    
    log_success "Container security check completed"
}

# Function to generate SBOM
generate_sbom() {
    log_info "Generating Software Bill of Materials (SBOM)..."
    
    local sbom_file="$REPORT_DIR/sbom_$TIMESTAMP.json"
    local sbom_summary=""
    
    if command -v cyclonedx-py &> /dev/null; then
        if cyclonedx-py --format json --output "$sbom_file" . 2>/dev/null; then
            local components_count
            components_count=$(jq '.components | length' "$sbom_file" 2>/dev/null || echo "unknown")
            sbom_summary="✅ **SBOM Generated**: $components_count components catalogued"
        else
            sbom_summary="❌ **SBOM Generation**: Failed"
        fi
    elif python scripts/generate_sbom.py --format json --output "$sbom_file" 2>/dev/null; then
        local components_count
        components_count=$(jq '.components | length' "$sbom_file" 2>/dev/null || echo "unknown")
        sbom_summary="✅ **SBOM Generated**: $components_count components catalogued (custom generator)"
    else
        sbom_summary="❌ **SBOM**: No SBOM generator available"
        log_warning "Install cyclonedx-py for SBOM generation: pip install cyclonedx-bom"
    fi
    
    add_section "Software Bill of Materials (SBOM)" "$sbom_summary

See SBOM: [\`sbom_$TIMESTAMP.json\`]($sbom_file)"
    
    log_success "SBOM generation completed"
}

# Function to run license compliance check
check_licenses() {
    log_info "Checking license compliance..."
    
    local license_report="$REPORT_DIR/licenses_$TIMESTAMP.json"
    local license_summary=""
    
    if command -v pip-licenses &> /dev/null; then
        pip-licenses --format=json --output-file="$license_report" 2>/dev/null || true
        
        # Check for potentially problematic licenses
        local problematic_licenses=("GPL" "AGPL" "SSPL")
        local license_issues=0
        
        for license in "${problematic_licenses[@]}"; do
            if grep -i "$license" "$license_report" &> /dev/null; then
                license_issues=$((license_issues + 1))
            fi
        done
        
        if [[ "$license_issues" -eq 0 ]]; then
            license_summary="✅ **License Compliance**: No problematic licenses detected"
        else
            license_summary="⚠️  **License Compliance**: $license_issues potentially problematic licenses found"
            MEDIUM_ISSUES=$((MEDIUM_ISSUES + license_issues))
        fi
    else
        license_summary="❌ **License Check**: pip-licenses not available"
        log_warning "Install pip-licenses for license checking: pip install pip-licenses"
    fi
    
    add_section "License Compliance Check" "$license_summary

See license report: [\`licenses_$TIMESTAMP.json\`]($license_report)"
    
    log_success "License compliance check completed"
}

# Function to finalize report
finalize_report() {
    TOTAL_ISSUES=$((CRITICAL_ISSUES + HIGH_ISSUES + MEDIUM_ISSUES + LOW_ISSUES))
    
    local summary=""
    local status_icon=""
    
    if [[ "$CRITICAL_ISSUES" -gt 0 ]]; then
        status_icon="🔴"
        summary="**CRITICAL**: $CRITICAL_ISSUES critical security issues require immediate attention"
    elif [[ "$HIGH_ISSUES" -gt 0 ]]; then
        status_icon="🟠"
        summary="**HIGH**: $HIGH_ISSUES high-priority security issues found"
    elif [[ "$MEDIUM_ISSUES" -gt 0 ]]; then
        status_icon="🟡"
        summary="**MEDIUM**: $MEDIUM_ISSUES medium-priority security issues found"
    elif [[ "$LOW_ISSUES" -gt 0 ]]; then
        status_icon="🟢"
        summary="**LOW**: $LOW_ISSUES low-priority issues found"
    else
        status_icon="✅"
        summary="**CLEAN**: No security issues detected"
    fi
    
    # Update executive summary
    local temp_file="/tmp/security_report_temp"
    
    # Replace the executive summary section
    awk '
    /^## Executive Summary/ {
        print $0
        print ""
        print "'"$status_icon"' **Overall Status**: '"$summary"'"
        print ""
        print "**Issue Breakdown:**"
        print "- 🔴 Critical: '"$CRITICAL_ISSUES"'"
        print "- 🟠 High: '"$HIGH_ISSUES"'"
        print "- 🟡 Medium: '"$MEDIUM_ISSUES"'"
        print "- 🔵 Low: '"$LOW_ISSUES"'"
        print "- **Total**: '"$TOTAL_ISSUES"'"
        print ""
        print "**Recommendations:**"
        if ('$CRITICAL_ISSUES' > 0) {
            print "1. 🚨 Address critical issues immediately"
            print "2. Review and fix high-priority issues"
            print "3. Plan remediation for medium-priority issues"
        } else if ('$HIGH_ISSUES' > 0) {
            print "1. Review and fix high-priority issues"
            print "2. Plan remediation for medium-priority issues"
        } else if ('$MEDIUM_ISSUES' > 0) {
            print "1. Plan remediation for medium-priority issues"
            print "2. Monitor for new vulnerabilities"
        } else {
            print "1. Continue regular security monitoring"
            print "2. Keep dependencies updated"
        }
        print ""
        
        # Skip original executive summary content
        while (getline && !/^---/) continue
        print "---"
        next
    }
    { print }
    ' "$CONSOLIDATED_REPORT" > "$temp_file"
    
    mv "$temp_file" "$CONSOLIDATED_REPORT"
    
    # Add final sections
    add_section "Scan Metadata" "- **Scan Date**: $(date)
- **Scan Duration**: $(( $(date +%s) - $(date -d "1 minute ago" +%s) )) seconds
- **Report Format**: Markdown
- **Tools Used**: Bandit, Safety, Semgrep, pip-audit, Docker
- **Report Location**: $CONSOLIDATED_REPORT"
    
    log_info "Security scan completed"
    log_info "Consolidated report: $CONSOLIDATED_REPORT"
    
    # Print summary to console
    echo
    echo "======================================"
    echo "       SECURITY SCAN SUMMARY"
    echo "======================================"
    echo -e "$status_icon $summary"
    echo
    echo "Issue Breakdown:"
    echo "  Critical: $CRITICAL_ISSUES"
    echo "  High:     $HIGH_ISSUES"
    echo "  Medium:   $MEDIUM_ISSUES"  
    echo "  Low:      $LOW_ISSUES"
    echo "  Total:    $TOTAL_ISSUES"
    echo
    echo "Full report: $CONSOLIDATED_REPORT"
    echo "======================================"
    
    # Exit with appropriate code
    if [[ "$CRITICAL_ISSUES" -gt 0 ]]; then
        exit 2  # Critical issues
    elif [[ "$HIGH_ISSUES" -gt 0 ]]; then
        exit 1  # High priority issues
    else
        exit 0  # Success or only low/medium issues
    fi
}

# Main execution
main() {
    log_info "Starting comprehensive security scan..."
    log_info "Report directory: $REPORT_DIR"
    
    # Run all security checks
    run_bandit
    run_safety
    run_pip_audit
    run_semgrep
    run_container_security
    generate_sbom
    check_licenses
    
    # Finalize and present results
    finalize_report
}

# Execute main function
main "$@"