# Pull Request

## Summary
<!-- Provide a brief description of what this PR accomplishes -->

**Type of Change:**
- [ ] 🐛 Bug fix (non-breaking change that fixes an issue)
- [ ] ✨ New feature (non-breaking change that adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to change)
- [ ] 📚 Documentation update
- [ ] 🎨 Code style/formatting changes
- [ ] ♻️ Refactoring (no functional changes)
- [ ] ⚡ Performance improvement
- [ ] 🔒 Security enhancement
- [ ] 🧪 Test improvements
- [ ] 🔧 Build/CI improvements
- [ ] 🏗️ Hardware support addition
- [ ] 📦 Dependency updates

## Changes Made
<!-- List the specific changes made in this PR -->

### Core Changes
- 
- 
- 

### Additional Changes
- 
- 

## Related Issues
<!-- Link to issues this PR addresses -->
- Fixes #
- Closes #
- Related to #

## Hardware/Platform Impact
<!-- Which platforms are affected by this change? -->
- [ ] ESP32 / ESP32-S3
- [ ] STM32 (F4, F7, H7 series)
- [ ] RP2040 (Raspberry Pi Pico)
- [ ] Nordic nRF52 series
- [ ] RISC-V (K210, BL602)
- [ ] All microcontrollers
- [ ] Single Board Computers
- [ ] Platform-agnostic changes
- [ ] No hardware impact

## Testing

### Test Coverage
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Hardware-in-the-loop tests added/updated
- [ ] Performance tests added/updated
- [ ] Documentation tests added/updated

### Test Results
```
# Paste test output here
pytest results:
Coverage: 
Hardware tests:
Performance benchmarks:
```

### Manual Testing
<!-- Describe manual testing performed -->
- Platform tested: 
- Model tested: 
- Key scenarios validated:
  - [ ] 
  - [ ] 
  - [ ] 

### Hardware Testing
<!-- If hardware testing was performed -->
- **Devices tested:**
  - Device 1: [Platform] - Status: ✅/❌
  - Device 2: [Platform] - Status: ✅/❌

- **Performance Impact:**
  - Latency change: 
  - Memory usage change: 
  - Power consumption change: 

## Breaking Changes
<!-- List any breaking changes and migration guide -->
- [ ] No breaking changes
- [ ] Breaking changes (describe below)

**Migration Guide:**
<!-- If there are breaking changes, provide migration instructions -->
```python
# Before
old_api_usage()

# After  
new_api_usage()
```

**Deprecation Timeline:**
- [ ] Immediate breaking change
- [ ] Deprecated in this version, removed in next major
- [ ] Deprecated with migration path

## Documentation
<!-- How is this change documented? -->
- [ ] README updated
- [ ] API documentation updated
- [ ] Architecture documentation updated
- [ ] Example code updated
- [ ] CHANGELOG.md updated
- [ ] Migration guide created
- [ ] No documentation changes needed

## Code Quality

### Pre-commit Checks
- [ ] All pre-commit hooks pass
- [ ] Code formatting (black, isort) applied
- [ ] Linting (flake8, ruff) passes
- [ ] Type checking (mypy) passes
- [ ] Security scanning (bandit) passes

### Code Review Checklist
- [ ] Code follows project conventions
- [ ] Functions/classes have appropriate docstrings
- [ ] Complex logic is commented
- [ ] Error handling is appropriate
- [ ] Resource cleanup is handled properly
- [ ] Thread safety considered (if applicable)

## Performance Impact
<!-- Expected performance impact of this change -->
- [ ] Performance improvement (specify metrics)
- [ ] Performance neutral
- [ ] Potential performance regression (justified)

**Performance Metrics:**
<!-- If performance testing was done -->
- Before: 
- After: 
- Change: 
- Benchmark details:

## Security Considerations
<!-- Security impact of this change -->
- [ ] No security implications
- [ ] Security enhancement
- [ ] Potential security impact (reviewed)

**Security Review:**
- [ ] Input validation reviewed
- [ ] Authentication/authorization considered
- [ ] Cryptographic usage reviewed
- [ ] Dependency security checked

## Deployment Notes
<!-- Any special deployment considerations -->
- [ ] No special deployment requirements
- [ ] Requires configuration changes
- [ ] Requires database migration
- [ ] Requires hardware setup changes
- [ ] Backward compatibility maintained

**Environment Configuration:**
<!-- Any new environment variables or configuration -->
- New config: 
- Default values: 
- Migration required:

## Additional Context
<!-- Any additional context, screenshots, or information -->

**Screenshots/Demos:**
<!-- If applicable, add screenshots or demos -->

**References:**
<!-- Links to relevant documentation, issues, or research -->
- 
- 

**Future Work:**
<!-- Related work that could be done in follow-up PRs -->
- [ ] 
- [ ] 

---

## Reviewer Checklist
<!-- For reviewers - check off as you review -->

### Functionality Review
- [ ] Changes match the description
- [ ] Edge cases are handled appropriately
- [ ] Error conditions are handled gracefully
- [ ] API changes are backward compatible (or properly documented)

### Code Quality Review  
- [ ] Code is clear and maintainable
- [ ] Appropriate abstractions are used
- [ ] No obvious performance issues
- [ ] Security best practices followed

### Testing Review
- [ ] Test coverage is adequate
- [ ] Tests are meaningful and not trivial
- [ ] Hardware testing completed (if applicable)
- [ ] Performance impact verified

### Documentation Review
- [ ] Documentation is accurate and complete
- [ ] Examples are functional
- [ ] Breaking changes are clearly documented