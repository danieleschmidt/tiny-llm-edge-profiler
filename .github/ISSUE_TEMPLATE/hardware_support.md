---
name: Hardware Support Request
about: Request support for a new hardware platform or device
title: '[HARDWARE] Add support for [Device/Platform Name]'
labels: ['hardware', 'enhancement', 'needs-triage']
assignees: ['terragon-labs/hardware-team']
---

## Hardware Support Request

**Device/Platform Name:**
<!-- Full name and model of the hardware -->

**Manufacturer:**
<!-- Hardware manufacturer/vendor -->

**Architecture:**
- [ ] ARM Cortex-M0/M0+
- [ ] ARM Cortex-M3
- [ ] ARM Cortex-M4
- [ ] ARM Cortex-M7
- [ ] ARM Cortex-A (specify core)
- [ ] RISC-V (specify variant)
- [ ] Xtensa (ESP32 family)
- [ ] Other: ___________

## Hardware Specifications

**Memory Configuration:**
- RAM: _______ KB/MB
- Flash: _______ KB/MB
- External memory: _______

**Clock Speed:**
- Base frequency: _______ MHz
- Max frequency: _______ MHz
- Configurable frequencies: _______

**Special Features:**
- [ ] Hardware FPU
- [ ] DSP instructions
- [ ] SIMD support
- [ ] Hardware crypto
- [ ] Hardware random number generator
- [ ] DMA controllers
- [ ] External memory controller
- [ ] Other: ___________

**Power Characteristics:**
- Operating voltage: _______ V
- Current consumption (active): _______ mA
- Current consumption (sleep): _______ µA
- Power domains: _______

## Connectivity & Interfaces

**Communication Interfaces:**
- [ ] UART/USART
- [ ] USB (specify type)
- [ ] SPI
- [ ] I2C
- [ ] CAN
- [ ] Ethernet
- [ ] WiFi
- [ ] Bluetooth
- [ ] Other: ___________

**Debug/Programming Interface:**
- [ ] SWD
- [ ] JTAG  
- [ ] UART bootloader
- [ ] USB bootloader
- [ ] Custom bootloader
- [ ] Other: ___________

## Development Environment

**Official SDK/Framework:**
<!-- Name and version of official development framework -->
- Framework: 
- Version:
- Language: 
- License:

**Toolchain:**
- Compiler: 
- Debugger:
- Flash tools:
- Development board: 

**Available Libraries:**
- [ ] Standard C library
- [ ] Math libraries
- [ ] DSP libraries
- [ ] Communication stacks
- [ ] RTOS support
- [ ] Other: ___________

## Use Case & Justification

**Primary Use Case:**
<!-- Why do you need this hardware supported? -->

**Target Applications:**
- [ ] Smart home devices
- [ ] Wearables
- [ ] Industrial IoT
- [ ] Automotive
- [ ] Medical devices
- [ ] Audio processing
- [ ] Computer vision
- [ ] Other: ___________

**Expected Model Constraints:**
- Max model size: _______ MB
- Target quantization: _______ bits
- Expected latency: _______ ms
- Power budget: _______ mW

## Hardware Availability

**Development Board:**
- Name: 
- Price: 
- Availability: 
- Purchase link: 

**Documentation:**
- [ ] Datasheet available
- [ ] Reference manual available
- [ ] Application notes available
- [ ] Example code available
- [ ] Community resources available

**Hardware Access:**
- [ ] I own this hardware
- [ ] I can provide hardware for testing
- [ ] Hardware is commercially available
- [ ] Hardware is in development
- [ ] Other: ___________

## Technical Implementation

**Similar Supported Platforms:**
<!-- Which currently supported platforms are most similar? -->

**Expected Implementation Complexity:**
- [ ] Low - Similar to existing platform
- [ ] Medium - Some unique features
- [ ] High - Significant new development
- [ ] Unknown - Needs investigation

**Potential Challenges:**
<!-- Any known technical challenges or limitations -->
- [ ] Limited documentation
- [ ] Unique architecture features
- [ ] Power management complexity
- [ ] Memory layout constraints
- [ ] Toolchain limitations
- [ ] Other: ___________

## Community & Ecosystem

**Community Size:**
- [ ] Small (< 1000 developers)
- [ ] Medium (1000-10000 developers)
- [ ] Large (> 10000 developers)
- [ ] Unknown

**Commercial Adoption:**
- [ ] Widely used in industry
- [ ] Growing adoption
- [ ] Niche applications
- [ ] Experimental/prototype

**Long-term Support:**
- [ ] Manufacturer committed to long-term support
- [ ] Active development roadmap
- [ ] Stable platform
- [ ] End-of-life planned
- [ ] Unknown

## Additional Information

**References:**
<!-- Links to datasheets, development boards, examples, etc. -->
- Datasheet: 
- Development board: 
- SDK documentation:
- Community resources:

**Related Requests:**
<!-- Link to similar requests or discussions -->
- Similar to #
- Blocks #
- Related to #

**Timeline:**
<!-- When do you need this support? -->
- [ ] ASAP - Critical for current project
- [ ] Within 1 month
- [ ] Within 3 months
- [ ] Within 6 months
- [ ] No specific timeline

---

**For Maintainers:**

**Hardware Team Assessment:**
- [ ] Hardware specifications reviewed
- [ ] Development tools assessed
- [ ] Technical feasibility confirmed
- [ ] Resource requirements estimated

**Implementation Priority:**
- [ ] High - Strategic platform
- [ ] Medium - Good community fit
- [ ] Low - Niche use case
- [ ] Research needed

**Resource Requirements:**
- Development time: _______ weeks
- Hardware needed: _______
- Team members: _______