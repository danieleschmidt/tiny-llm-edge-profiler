# Support

Welcome to the tiny-llm-edge-profiler community! This document provides guidance on how to get help with the project.

## 🆘 Getting Help

### Before Seeking Help
1. **Check the [README](README.md)** - Most common questions are answered there
2. **Review the [Documentation](docs/)** - Comprehensive guides and API reference
3. **Search existing [Issues](https://github.com/terragon-labs/tiny-llm-edge-profiler/issues)** - Your question might already be answered
4. **Check [Discussions](https://github.com/terragon-labs/tiny-llm-edge-profiler/discussions)** - Community Q&A

### Quick References
- **Installation**: See [README Installation](README.md#installation)
- **Hardware Setup**: See [DEVELOPMENT.md](docs/DEVELOPMENT.md)
- **API Usage**: See examples in [README](README.md#quick-start)
- **Troubleshooting**: See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)

## 📋 Types of Support

### 🐛 Bug Reports
If you've found a bug, please [create an issue](https://github.com/terragon-labs/tiny-llm-edge-profiler/issues/new?template=bug_report.md) using our bug report template.

**For Hardware-Specific Issues:**
- Use the [Hardware Support template](https://github.com/terragon-labs/tiny-llm-edge-profiler/issues/new?template=hardware_support.md)
- Include device model, firmware version, and connection details
- Attach hardware photos if relevant

### ✨ Feature Requests
Request new features using our [Feature Request template](https://github.com/terragon-labs/tiny-llm-edge-profiler/issues/new?template=feature_request.md).

### ⚡ Performance Issues  
Report performance problems using our [Performance Issue template](https://github.com/terragon-labs/tiny-llm-edge-profiler/issues/new?template=performance_issue.md).

### 💬 General Questions
For general questions, please use [GitHub Discussions](https://github.com/terragon-labs/tiny-llm-edge-profiler/discussions) rather than issues.

## 🏷️ Support Categories

### Hardware Platforms
**Officially Supported:**
- ESP32/ESP32-S3 (Primary support)
- STM32F4/F7/H7 series (Full support)
- Raspberry Pi Pico/RP2040 (Full support)
- Nordic nRF52 series (Full support)

**Community Supported:**
- RISC-V platforms (K210, BL602)
- Additional ARM Cortex-M variants
- Single Board Computers

**Support Level Legend:**
- 🟢 **Full Support**: Actively maintained, comprehensive testing
- 🟡 **Community Support**: Community-maintained, best-effort testing
- 🔴 **Experimental**: Early development, limited testing

### Model Support
**Quantization Formats:**
- 2-bit, 3-bit, 4-bit quantization (Full support)
- GGML format models (Full support)
- Custom quantization schemes (Community support)

**Model Types:**
- Language models (LLaMA, Phi, TinyLLaMA, OPT)
- Custom transformer architectures
- Encoder-only models (experimental)

## 🚀 Response Times

### Issue Response Times
- **Critical Bugs**: 24-48 hours
- **Hardware Issues**: 2-5 business days
- **Feature Requests**: 1-2 weeks for initial triage
- **General Questions**: Best effort, typically 1-7 days

### Community Support
The community often provides faster responses than maintainers. Please:
- Be patient and respectful
- Provide detailed information
- Search before asking
- Help others when you can

## 📞 Contact Information

### Primary Channels
- **GitHub Issues**: For bugs, features, and hardware support
- **GitHub Discussions**: For questions and community interaction
- **Email**: tiny-llm@terragon.dev (for security issues or urgent matters)

### Community Channels
- **Discord**: [Terragon Labs Community](https://discord.gg/terragon-labs)
- **Matrix**: [#tiny-llm-profiler:matrix.org](https://matrix.to/#/#tiny-llm-profiler:matrix.org)

### Social Media
- **Twitter**: [@terragon_labs](https://twitter.com/terragon_labs)
- **LinkedIn**: [Terragon Labs](https://linkedin.com/company/terragon-labs)

## 🛡️ Security Issues

**For security vulnerabilities**, please do NOT create public issues. Instead:
1. Email us at: security@terragon.dev
2. Review our [Security Policy](SECURITY.md)
3. Use our [Security Advisory process](https://github.com/terragon-labs/tiny-llm-edge-profiler/security/advisories)

## 🤝 Contributing Support

### Ways to Help the Community
- **Answer Questions**: Help others in discussions and issues
- **Improve Documentation**: Submit PRs for documentation improvements
- **Test Hardware**: Test with different platforms and report results
- **Share Examples**: Provide real-world usage examples
- **Write Tutorials**: Create community tutorials or blog posts

### Maintainer Support
If you're interested in becoming a maintainer or contributing regularly:
- Review [CONTRIBUTING.md](CONTRIBUTING.md)
- Join our [contributor discussions](https://github.com/terragon-labs/tiny-llm-edge-profiler/discussions/categories/contributors)
- Reach out at: contributors@terragon.dev

## 📚 Additional Resources

### Documentation
- **Architecture Guide**: [ARCHITECTURE.md](ARCHITECTURE.md)
- **Development Setup**: [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)
- **Hardware Guides**: [docs/hardware/](docs/hardware/)
- **API Reference**: [docs/api/](docs/api/)

### External Resources
- **TinyML Community**: [tinyml.org](https://tinyml.org)
- **Edge AI Resources**: [edge-ai-resources.com](https://edge-ai-resources.com)
- **Embedded ML Papers**: [papers.tinyml.org](https://papers.tinyml.org)

### Learning Resources
- **Online Courses**: Edge AI and TinyML courses
- **Workshops**: Hands-on hardware profiling workshops
- **Conferences**: TinyML Summit, Embedded World, Edge AI Summit

## ⚖️ Code of Conduct

All interactions in this community are governed by our [Code of Conduct](CODE_OF_CONDUCT.md). Please read and follow it to ensure a welcoming environment for everyone.

## 🔄 Support Process Updates

This support document is updated regularly. Last updated: January 2024

**Version History:**
- v1.2: Added Matrix channel, updated response times
- v1.1: Added Discord community link
- v1.0: Initial support document

---

## Quick Help Commands

**Common Commands:**
```bash
# Check installation
tiny-profiler --version

# List connected devices
tiny-profiler devices

# Run basic health check
tiny-profiler doctor

# Get help for specific command
tiny-profiler [command] --help
```

**Emergency Debugging:**
```bash
# Generate comprehensive system report
tiny-profiler debug-report

# Test hardware connectivity
tiny-profiler test-hardware --platform esp32

# Validate installation
python -c "import tiny_llm_profiler; print('OK')"
```

---

Thank you for being part of the tiny-llm-edge-profiler community! 🚀