# NeuroDrive: An Open-Source RL Policy Inference ASIC for BLDC Motor Control

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Platform](https://img.shields.io/badge/Platform-Caravel%20SKY130-6E40C9.svg)](https://chipfoundry.io)
[![Category](https://img.shields.io/badge/Category-Industrial%20%2F%20Edge--IoT-green.svg)](#)

> **ChipFoundry Reference Application Design Contest — Proposal Submission**
> March 2026

---

## Table of Contents

- [1. Executive Summary](#1-executive-summary)
- [2. The Problem: Why Motor Control Needs to Evolve](#2-the-problem-why-motor-control-needs-to-evolve)
- [3. The Idea: NeuroDrive](#3-the-idea-neurodrive)
- [4. Project Scope](#4-project-scope)
- [5. Chip Architecture](#5-chip-architecture)
- [6. Caravel Integration](#6-caravel-integration)
- [7. Repository and File Architecture](#7-repository-and-file-architecture)
- [8. System Architecture: Complete BLDC Motor Controller](#8-system-architecture-complete-bldc-motor-controller)
- [9. Bill of Materials and Cost Estimate](#9-bill-of-materials-and-cost-estimate)
- [10. Implementation Phases](#10-implementation-phases)
- [11. Challenges and Mitigations](#11-challenges-and-mitigations)
- [12. Judging Criteria Alignment](#12-judging-criteria-alignment)
- [13. References](#13-references)

---

## 1. Executive Summary

**NeuroDrive** is an open-source, application-specific integrated circuit (ASIC) designed to deploy pre-trained Reinforcement Learning (RL) policies for Brushless DC (BLDC) motor control at hardware-deterministic speeds. Built on the Caravel SoC harness using the SKY130 130 nm process, the chip integrates a pipelined INT8 neural network inference engine, a 3-phase PWM generator with dead-time insertion, a classical hardware safety envelope, and Wishbone-mapped weight memory — all within the 10 mm² Caravel user area.

The design addresses a concrete gap in both the open-source silicon and motor control ecosystems: no dedicated motor control ASIC exists on SKY130, and no commercial or open-source chip provides a hardware path to deploy RL-based motor control policies at deterministic loop rates. NeuroDrive is not a replacement for Field-Oriented Control (FOC); it is a **research-to-deployment bridge** that allows the motor control research community to move RL algorithms from Python simulations to real silicon controlling real motors, while maintaining classical safety guarantees in hardware.

The reference design includes custom silicon (Caravel ASIC), a 2-layer PCBA (inverter + sensing + digital interface), firmware for the Caravel RISC-V management core (weight provisioning, logging, configuration), and a 3D-printable test fixture for a standard gimbal motor. The total system BOM (excluding fabrication) targets under $80.

### Key Specifications

| Parameter | Value |
|---|---|
| Process | SKY130 130 nm (Skywater) |
| Platform | Caravel SoC (10 mm² user area) |
| Target Clock | 40 MHz (conservative) |
| NN Topology | 8 → 64 → 64 → 6 MLP, INT8 weights/activations |
| Parameters | ~5,126 INT8 (≈5 KB) |
| MAC Units | 8 parallel INT8 MACs |
| Inference Latency | < 30 µs at 40 MHz |
| Control Loop Rate | 20 kHz (50 µs period) |
| PWM Outputs | 6 (3-phase complementary, programmable dead-time) |
| Digital Sensor Inputs | SPI (external ADC + position sensor) |
| Weight Update | Wishbone bus from RISC-V management core |
| Weight Storage | ~6 KB on-chip SRAM (3 × 2 KB OpenRAM macros) |
| Safety | Hardware overcurrent latch, watchdog, PWM-disable fault pin |
| License | Apache 2.0 |

---

## 2. The Problem: Why Motor Control Needs to Evolve

### 2.1 The Dominance and Limits of Field-Oriented Control

Field-Oriented Control (FOC) has been the gold standard for high-performance BLDC/PMSM motor control for over three decades. FOC transforms three-phase stator currents into a rotating d-q reference frame using Clarke and Park transformations, enabling independent control of torque (q-axis) and flux (d-axis) through cascaded PI controllers. The result is smooth, efficient operation across the full speed range, with control loops running at 10–40 kHz on inexpensive microcontrollers.

However, FOC has well-documented limitations that become acute in robotics and adaptive applications:

- **Parameter Sensitivity**: FOC performance depends on accurate knowledge of motor inductance, resistance, and flux linkage — parameters that drift with temperature, aging, and magnetic saturation. Periodic retuning is required and rarely performed in practice.
- **Tuning Burden**: A typical FOC system has 6–11 parameters (PI gains for speed, d-axis, and q-axis loops, plus feedforward terms) that must be manually tuned per motor variant. This process requires domain expertise and oscilloscope-level iteration.
- **Nonlinearity Handling**: Cross-coupling between d-q axes, cogging torque, and magnetic saturation introduce nonlinearities that PI controllers handle poorly. Compensation techniques exist but add complexity.
- **Static Optimization**: FOC optimizes for a single operating point or a pre-computed gain schedule. It cannot autonomously adapt to new load profiles, mechanical wear, or environmental conditions.

Every major motor controller ecosystem — STMicroelectronics MCSDK, TI MotorWare, Infineon MOTIX, Microchip motorBench — implements FOC with PI controllers as the primary algorithm. The tooling is mature, the supply chain is established, and the performance is well-understood. FOC on a $2 MCU at 20 kHz with deterministic behavior is an extremely high bar.

### 2.2 The Promise of Reinforcement Learning for Motor Control

Over the past five years (2020–2025), Reinforcement Learning has emerged as a credible alternative and complement to FOC for motor control. Key findings from the research literature include:

- **Comparable or superior dynamic performance**: RL controllers have demonstrated 50% faster settling times and elimination of overshoot versus tuned PI controllers on real hardware (Mastanaiah et al., IET Power Electronics, 2025).
- **Inherent robustness**: RL policies trained with domain randomization adapt to parameter variations without retuning — the core advantage over model-based approaches.
- **Multi-objective optimization**: RL can simultaneously minimize tracking error, torque ripple, acoustic noise, and losses — trade-offs that are difficult to encode in cascaded PI structures.
- **Cross-motor generalization**: Meta-RL techniques have demonstrated single policies that generalize across motor power classes from watts to hundreds of kilowatts (Jakobeit et al., IEEE Trans. Power Electronics, 2023).

However, RL-based motor control remains confined to simulation and laboratory bench setups using general-purpose MCUs, FPGAs, or DSPs with edge computing workstations. There is **no dedicated hardware path** for deploying RL policies at deterministic motor-control rates.

### 2.3 The Gap This Project Fills

The missing piece is not a better algorithm — it is a **hardware deployment platform**. Researchers train RL policies in Python (gym-electric-motor, MATLAB/Simulink) and validate them in simulation, but face a steep cliff when trying to deploy to real motors at real control rates:

- General-purpose MCUs running NN inference frameworks (TFLite Micro) achieve inference in 50–500 µs — too slow or too variable for the inner current loop.
- FPGAs provide the right performance but require HDL expertise orthogonal to the RL research community.
- Edge computing architectures (GPU workstation + embedded controller) add latency, complexity, and cost.

NeuroDrive fills this gap by providing a standardized, open-source ASIC that takes a trained INT8 MLP policy and executes it at deterministic 20 kHz loop rates, with all motor I/O and safety logic integrated on-chip. The chip is to RL motor control what the STM32 + MCSDK is to FOC: a reference platform that lowers the barrier from "algorithm on paper" to "motor spinning on bench."

---

## 3. The Idea: NeuroDrive

### 3.1 Design Philosophy

NeuroDrive follows three principles:

1. **Inference only, not training**: On-chip training in 130 nm is impractical within 10 mm². Training happens offline (PC/GPU), and only the trained INT8 weight vector is loaded onto the chip. This is the same paradigm used by every commercial TinyML deployment.

2. **RL decides, hardware protects**: The RL policy operates inside a classical safety envelope. Overcurrent limits, voltage bounds, temperature thresholds, and watchdog timers are enforced in dedicated digital logic that cannot be overridden by the policy. If the RL output violates any constraint, hardware clamps or disables the PWM outputs within a single clock cycle.

3. **Flexible policy, fixed interface**: The inference engine supports any MLP topology that fits in the SRAM budget (up to ~5,000 INT8 parameters). Researchers swap policies by loading new weights over the Wishbone bus — the chip's I/O pinout, PWM timing, and safety thresholds remain fixed. This makes NeuroDrive a platform, not a single-application chip.

### 3.2 What Makes This a Reference Design

The contest requires a complete, reproducible system. NeuroDrive delivers:

- **Custom silicon**: Caravel ASIC with RL inference engine, PWM generator, and safety monitor.
- **PCBA**: 2-layer KiCad board with 3-phase MOSFET inverter, current shunt amplifiers, external ADC, inductive position sensor interface, and Caravel M.2 socket.
- **Firmware**: C firmware for the Caravel RISC-V management core handling weight loading from SPI flash, runtime configuration, data logging via UART, and safe-start sequencing.
- **Mechanicals**: OpenSCAD 3D-printable test fixture mounting a gimbal motor to a baseplate with the PCBA, enabling repeatable bench testing.

---

## 4. Project Scope

### 4.1 In Scope

| Deliverable | Description |
|---|---|
| RTL design | Verilog RTL for NN inference engine, PWM generator, safety monitor, Wishbone peripherals, and top-level `user_project_wrapper` integration |
| Verification | cocotb testbenches for all modules including gate-level simulation (GLS); SDC constraints for STA |
| GDSII | Hardened layout passing `cf precheck` and tapeout checks |
| PCBA | KiCad schematic + layout for motor driver reference board |
| Firmware | RISC-V C firmware for weight provisioning, configuration, and data logging |
| Mechanicals | OpenSCAD test fixture for gimbal motor bench testing |
| Documentation | This README, architecture docs, how-to video, AI session logs |
| Training pipeline | Python scripts to train a TD3/DDPG policy in gym-electric-motor and export INT8 weights |

### 4.2 Out of Scope

- On-chip RL training (experience replay, gradient computation, optimizer state)
- On-chip analog-to-digital conversion (external ADC used; on-chip ADC deferred to Phase 2)
- Sensorless motor control (position sensor required for this revision)
- Safety certification (IEC 61508, ISO 26262) — this is a research platform
- High-voltage gate driving (external gate driver IC used)

---

## 5. Chip Architecture

### 5.1 Top-Level Block Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CARAVEL USER PROJECT AREA (10 mm²)               │
│                                                                     │
│  ┌──────────────┐    ┌────────────────────────────────────────┐     │
│  │   WISHBONE   │◄──►│         WEIGHT SRAM (6 KB)             │     │
│  │   INTERFACE   │    │   3 × 2KB OpenRAM macros               │     │
│  │  (to mgmt    │    │   Dual-port: WB write / engine read    │     │
│  │   RISC-V)    │    └────────────┬───────────────────────────┘     │
│  └──────┬───────┘                 │                                 │
│         │                         ▼                                 │
│  ┌──────┴───────┐    ┌────────────────────────────────────────┐     │
│  │   CONTROL    │    │      NN INFERENCE ENGINE                │     │
│  │   REGISTERS  │───►│  8× INT8 MAC units, pipelined          │     │
│  │  (config,    │    │  ReLU activation (comparator)          │     │
│  │   topology,  │    │  Layer sequencer FSM                   │     │
│  │   thresholds)│    │  Activation SRAM (1 KB scratch)        │     │
│  └──────────────┘    └────────────┬───────────────────────────┘     │
│                                   │ action vector                   │
│                                   ▼                                 │
│  ┌────────────────┐  ┌────────────────────────────────────────┐     │
│  │  SENSOR INPUT  │  │        SAFETY MONITOR                  │     │
│  │  INTERFACE     │  │  • Action clamping (min/max per output)│     │
│  │  • SPI master  │  │  • Overcurrent comparator (digital)    │     │
│  │    (ext ADC +  │──►│  • Watchdog timer (NN must complete    │     │
│  │    position)   │  │    within deadline or PWM disabled)    │     │
│  │  • GPIO direct │  │  • Fault latch + external fault pin    │     │
│  │    (hall/enc)  │  │  • Temperature threshold register      │     │
│  └────────────────┘  └────────────┬───────────────────────────┘     │
│                                   │ safe action vector              │
│                                   ▼                                 │
│                      ┌────────────────────────────────────────┐     │
│                      │       PWM GENERATOR                    │     │
│                      │  • 3-phase center-aligned PWM          │     │
│                      │  • Programmable dead-time (10ns res.)  │     │
│                      │  • Configurable frequency (5–50 kHz)   │     │
│                      │  • Hardware fault-disable input         │     │
│                      └────────────┬───────────────────────────┘     │
│                                   │                                 │
│  ┌────────────────┐               │                                 │
│  │ LOGIC ANALYZER │               │                                 │
│  │ PROBES (128b)  │               │                                 │
│  │ (debug access  │               │                                 │
│  │  to internal   │               │                                 │
│  │  state vectors)│               │                                 │
│  └────────────────┘               │                                 │
└───────────────────────────────────┼─────────────────────────────────┘
                                    │
                              mprj_io[37:0]
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
              6× PWM out      SPI bus         Fault/Enable
              (to gate       (to ext ADC     (to/from
               driver)       + pos sensor)    system)
```

### 5.2 Neural Network Inference Engine

The inference engine is the core IP block. It implements a configurable MLP forward pass using a layer-sequential architecture.

**Datapath**: 8 parallel INT8×INT8 multiply-accumulate (MAC) units with INT24 accumulators. Each MAC unit computes one output-neuron partial sum per cycle. A 64-neuron hidden layer with 64 inputs requires 64×64 = 4,096 MACs, executed as 64 output neurons × (64 inputs / 8 MACs) = 512 cycles per layer. At 40 MHz, this is 12.8 µs per layer.

**Activation function**: ReLU is a zero-cost comparator (MSB check). Tanh/sigmoid can be approximated with a 256-entry INT8 lookup table (256 bytes), stored in the activation scratch SRAM.

**Layer sequencer**: A finite state machine that iterates through layers, reading weights from SRAM, feeding inputs to the MAC array, writing outputs back to activation scratch memory, and advancing to the next layer. The topology (number of layers, neurons per layer) is configured via Wishbone registers, enabling different network architectures without re-synthesis.

**Timing budget** for the default 8→64→64→6 network at 40 MHz:

| Phase | Cycles | Time |
|---|---|---|
| Input normalization | 32 | 0.8 µs |
| Layer 1 (8→64): 64 × ⌈8/8⌉ = 64 cycles + pipeline | 80 | 2.0 µs |
| Layer 2 (64→64): 64 × ⌈64/8⌉ = 512 cycles + pipeline | 528 | 13.2 µs |
| Layer 3 (64→6): 6 × ⌈64/8⌉ = 48 cycles + pipeline | 64 | 1.6 µs |
| Output scaling + safety check | 16 | 0.4 µs |
| **Total inference** | **720** | **18.0 µs** |

With the remaining 32 µs in a 50 µs (20 kHz) control period allocated to SPI sensor reads (~10 µs), state vector assembly (~2 µs), and PWM update (~0.1 µs), the timing budget has **~20 µs of margin**.

### 5.3 Weight Memory Architecture

The weight SRAM uses **three 2 KB OpenRAM macros** (1024×16 configuration), providing 6 KB total. The default network requires ~5 KB:

| Data | Size |
|---|---|
| Layer 1 weights (8×64) | 512 B |
| Layer 1 biases (64) | 64 B |
| Layer 2 weights (64×64) | 4,096 B |
| Layer 2 biases (64) | 64 B |
| Layer 3 weights (64×6) | 384 B |
| Layer 3 biases (6) | 6 B |
| Normalization params (16 × 2) | 32 B |
| Activation LUT (256) | 256 B |
| **Total** | **5,414 B** |

The SRAM is dual-ported via time-division multiplexing: the Wishbone bus writes during weight-update windows (between inference cycles), and the inference engine reads during forward passes. A simple bank-swap mechanism allows atomic weight updates: new weights are written to a shadow bank while the active bank runs inference, then banks swap at the start of the next control cycle.

### 5.4 PWM Generator

The PWM generator produces 6 complementary outputs (high/low for each of 3 phases) with:

- Center-aligned PWM with configurable carrier frequency (5–50 kHz in steps of the base clock divider)
- Programmable dead-time per phase (0–2.55 µs in 10 ns steps at 100 MHz counter, or 25 ns steps at 40 MHz)
- Duty-cycle inputs from the safety monitor (INT8 scaled to timer resolution)
- Hardware fault-disable input: an external pin or internal overcurrent flag immediately forces all outputs low within 1 clock cycle
- ADC trigger output: a sync pulse at the PWM center point for center-aligned current sampling

### 5.5 Safety Monitor

The safety monitor is a **purely combinational and registered** block (no NN involvement) that enforces:

- **Action clamping**: Each of the 6 PWM duty-cycle values is clamped to configurable min/max registers before reaching the PWM generator.
- **Overcurrent detection**: A digital input pin (`fault_n`, active-low) from an external comparator circuit triggers immediate PWM shutdown and sets a latched fault flag readable via Wishbone.
- **Watchdog timer**: If the inference engine does not produce a new action vector within a configurable deadline (default: 2× the expected inference time), the PWM outputs are disabled. This catches NN hangs or clock failures.
- **Temperature threshold**: A digital temperature value read via SPI is compared against a programmable threshold register. Exceeding it triggers a soft-limit (reduced maximum duty cycle) or hard-disable.
- **Startup sequencing**: PWM outputs remain disabled after reset until the management core writes an explicit enable sequence to a control register, preventing uncontrolled motor activation on power-up.

### 5.6 Area Estimate

| Block | Estimated Area | Notes |
|---|---|---|
| Weight SRAM (3 × 2 KB OpenRAM) | 1.5–2.0 mm² | ~0.5–0.7 mm² per 2 KB macro |
| Activation scratch SRAM (1 × 1 KB) | 0.3–0.5 mm² | Single OpenRAM macro |
| NN inference engine (8 MACs + FSM) | 0.2–0.4 mm² | ~15K gates |
| PWM generator | 0.03–0.05 mm² | ~2K gates |
| Safety monitor | 0.02–0.04 mm² | ~1.5K gates |
| SPI master + GPIO logic | 0.03–0.05 mm² | ~2K gates |
| Wishbone interface + registers | 0.05–0.08 mm² | ~3K gates |
| Clock domain crossing + misc. | 0.05–0.10 mm² | Synchronizers, resets |
| Routing overhead (~40% of logic) | 1.0–2.0 mm² | Metal fill, clock tree |
| **Total estimated** | **3.2–5.3 mm²** | **Well within 10 mm²** |

The conservative total of ~5 mm² leaves **~5 mm² of margin** — enough to accommodate routing congestion, guard rings, and potential additional features (e.g., a second SPI port, encoder counter, or additional SRAM).

---

## 6. Caravel Integration

### 6.1 Integration with Caravel Harness

NeuroDrive's user logic instantiates inside `user_project_wrapper.v`, connecting to Caravel's standard interfaces:

- **Wishbone bus**: The RISC-V management core uses the 32-bit Wishbone port to write NN weights into SRAM, configure control registers (PWM frequency, dead-time, safety thresholds, NN topology), read status/fault registers, and trigger weight-bank swaps. All configuration is memory-mapped.
- **Logic analyzer probes (128 bits)**: Connected to internal state vectors for non-intrusive debugging. Probes expose: current NN input vector, NN output vector, active layer counter, safety monitor flags, PWM duty-cycle values, and fault status.
- **GPIO pins (`mprj_io[5:37]`)**: Allocated as shown in the pinout table below.

### 6.2 GPIO Pin Allocation

| GPIO Pin(s) | Direction | Function | Mode |
|---|---|---|---|
| `mprj_io[5:10]` | Output | PWM_AH, PWM_AL, PWM_BH, PWM_BL, PWM_CH, PWM_CL | `user_output` |
| `mprj_io[11]` | Output | ADC_TRIGGER (sync pulse for ext. ADC sampling) | `user_output` |
| `mprj_io[12]` | Output | SPI_SCK (to external ADC + position sensor) | `user_output` |
| `mprj_io[13]` | Output | SPI_MOSI | `user_output` |
| `mprj_io[14]` | Input | SPI_MISO | `user_input_nopull` |
| `mprj_io[15]` | Output | SPI_CS_ADC (chip select for external ADC) | `user_output` |
| `mprj_io[16]` | Output | SPI_CS_POS (chip select for position sensor) | `user_output` |
| `mprj_io[17]` | Input | FAULT_N (active-low overcurrent from comparator) | `user_input_pullup` |
| `mprj_io[18]` | Output | ENABLE (system enable to gate driver) | `user_output` |
| `mprj_io[19]` | Input | HALL_A (optional Hall sensor input) | `user_input_nopull` |
| `mprj_io[20]` | Input | HALL_B | `user_input_nopull` |
| `mprj_io[21]` | Input | HALL_C | `user_input_nopull` |
| `mprj_io[22:23]` | I/O | DEBUG_UART_TX/RX (auxiliary debug port) | `user_output` / `user_input_nopull` |
| `mprj_io[24:37]` | — | Reserved / unused | `mgmt_input_nopull` |

### 6.3 Clocking Strategy

- **External clock**: 10 MHz crystal oscillator on the Caravel evaluation board.
- **PLL**: Caravel's on-chip PLL multiplies to **40 MHz** system clock (feedback divider = 4).
- **Clock domains**: Single 40 MHz domain for all user logic. The PWM timer uses a clock divider for carrier frequency generation. SPI operates at clock/4 = 10 MHz.

### 6.4 Management Core Firmware Role

The Caravel RISC-V management core (PicoRV32) runs firmware responsible for:

1. **Boot sequence**: Initialize Caravel housekeeping, configure PLL, set GPIO modes, hold PWM disabled.
2. **Weight provisioning**: Read INT8 weight binary from external SPI flash (user area flash2 interface) into weight SRAM via Wishbone writes.
3. **Configuration**: Write NN topology registers, PWM frequency, dead-time, safety thresholds.
4. **Enable**: Arm the safety monitor and enable PWM outputs.
5. **Runtime monitoring**: Periodically read fault status, temperature, and optionally log state/action pairs via UART for offline analysis.
6. **Weight update**: On command (UART or GPIO trigger), load a new weight vector from flash and perform a bank swap.

---

## 7. Repository and File Architecture

The project follows the Caravel user project template structure exactly:

```
neurodrive/
├── .cf/
│   └── project.json                    # ChipFoundry project metadata (generated by cf init)
├── gds/
│   └── user_project_wrapper.gds        # Final GDSII (generated by cf harden)
├── lef/
│   └── user_project_wrapper.lef        # LEF for top-level macro
├── def/
│   └── user_project_wrapper.def        # DEF for top-level macro
├── verilog/
│   ├── rtl/
│   │   ├── user_project_wrapper.v      # Top-level wrapper (Caravel template)
│   │   ├── user_defines.v              # GPIO configuration defines
│   │   ├── neurodrive_top.v            # NeuroDrive top-level module
│   │   ├── nn_engine/
│   │   │   ├── nn_inference_engine.v   # Layer sequencer FSM + datapath control
│   │   │   ├── mac_array.v             # 8× INT8 MAC unit array
│   │   │   ├── mac_unit.v              # Single INT8×INT8→INT24 MAC
│   │   │   ├── activation_relu.v       # ReLU activation (combinational)
│   │   │   ├── activation_lut.v        # LUT-based tanh/sigmoid (optional)
│   │   │   ├── weight_addr_gen.v       # Weight SRAM address generator
│   │   │   └── output_scaler.v         # INT8→duty-cycle scaling
│   │   ├── motor_ctrl/
│   │   │   ├── pwm_generator.v         # 3-phase center-aligned PWM
│   │   │   ├── deadtime_insert.v       # Dead-time insertion logic
│   │   │   ├── safety_monitor.v        # Action clamping, watchdog, fault latch
│   │   │   └── adc_trigger.v           # Center-aligned ADC sync pulse
│   │   ├── peripherals/
│   │   │   ├── spi_master.v            # SPI master for external ADC + pos sensor
│   │   │   ├── sensor_interface.v      # Sensor data unpacking + normalization
│   │   │   └── hall_decoder.v          # Optional Hall sensor decoder
│   │   ├── bus/
│   │   │   ├── wb_interconnect.v       # Wishbone address decoder
│   │   │   ├── wb_sram_bridge.v        # Wishbone-to-SRAM interface
│   │   │   └── wb_registers.v          # Configuration + status registers
│   │   └── utils/
│   │       ├── sync_ff.v               # 2-FF synchronizer
│   │       └── reset_sync.v            # Reset synchronizer
│   ├── gl/
│   │   └── user_project_wrapper.v      # Gate-level netlist (generated)
│   ├── dv/
│   │   ├── cocotb/
│   │   │   ├── test_nn_engine/
│   │   │   │   ├── test_nn_engine.py       # NN inference correctness vs. Python reference
│   │   │   │   └── nn_reference_model.py   # NumPy golden reference model
│   │   │   ├── test_pwm_generator/
│   │   │   │   └── test_pwm.py             # PWM timing, dead-time, fault-disable
│   │   │   ├── test_safety_monitor/
│   │   │   │   └── test_safety.py          # Clamping, watchdog, overcurrent response
│   │   │   ├── test_spi_master/
│   │   │   │   └── test_spi.py             # SPI protocol, external ADC emulation
│   │   │   ├── test_wb_interface/
│   │   │   │   └── test_wishbone.py        # Register read/write, SRAM access
│   │   │   ├── test_integration/
│   │   │   │   └── test_full_loop.py       # Full control loop: sensor→NN→PWM
│   │   │   └── test_caravel/
│   │   │       └── test_neurodrive.py      # Caravel-level integration test
│   │   └── includes/
│   │       ├── includes.rtl.caravel_user_project
│   │       └── includes.gl.caravel_user_project
│   └── stub/
│       └── user_project_wrapper.v      # Black-box stub for Caravel integration
├── openlane/
│   ├── neurodrive_top/
│   │   └── config.json                 # OpenLane config for NeuroDrive macro
│   ├── nn_inference_engine/
│   │   └── config.json                 # OpenLane config for NN engine sub-macro
│   └── user_project_wrapper/
│       └── config.json                 # Top-level wrapper hardening config
├── firmware/
│   ├── neurodrive_fw/
│   │   ├── main.c                      # RISC-V firmware: boot, config, weight load
│   │   ├── neurodrive_hal.h            # Hardware abstraction: register map defines
│   │   ├── weight_loader.c             # Flash-to-SRAM weight provisioning
│   │   ├── uart_logger.c               # Runtime state/action logging
│   │   └── Makefile
│   └── common/
│       └── caravel_mgmt.h              # Caravel management SoC register definitions
├── pcba/
│   ├── kicad/
│   │   ├── neurodrive_board.kicad_pro  # KiCad project
│   │   ├── neurodrive_board.kicad_sch  # Schematic
│   │   ├── neurodrive_board.kicad_pcb  # PCB layout
│   │   └── libs/                       # Custom footprints/symbols
│   ├── bom/
│   │   └── bom.csv                     # Bill of materials
│   └── gerbers/                        # Manufacturing files
├── mechanical/
│   ├── test_fixture.scad               # OpenSCAD test fixture
│   └── test_fixture.stl                # Pre-rendered STL
├── training/
│   ├── train_policy.py                 # gym-electric-motor TD3 training script
│   ├── export_weights.py               # Convert trained model → INT8 binary
│   ├── quantize.py                     # Post-training INT8 quantization
│   ├── requirements.txt                # Python dependencies
│   └── policies/
│       └── default_gimbal.bin          # Pre-trained INT8 weight file for test motor
├── ai_logs/
│   ├── rtl_generation/                 # LLM session logs for RTL coding
│   ├── testbench_generation/           # LLM session logs for verification
│   └── README.md                       # Summary of AI usage
├── docs/
│   ├── architecture.md                 # Detailed architecture document
│   ├── register_map.md                 # Wishbone register map
│   ├── pinout.md                       # GPIO pin allocation
│   ├── pcba_assembly.md                # Board assembly guide
│   └── how_to_video.md                 # Video script/storyboard
├── constraints/
│   └── neurodrive.sdc                  # Timing constraints for STA
├── LICENSE                             # Apache 2.0
└── README.md                           # This file
```

---

## 8. System Architecture: Complete BLDC Motor Controller

### 8.1 System Block Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        HOST PC (Training & Monitoring)                  │
│  ┌─────────────┐  ┌──────────────────┐  ┌─────────────────────────┐   │
│  │ gym-electric │  │  Weight export   │  │  UART monitor /        │   │
│  │ -motor (GEM) │  │  (Python→INT8)   │  │  data logger           │   │
│  └─────────────┘  └──────────────────┘  └─────────────────────────┘   │
└───────────────────────────┬─────────────────────┬─────────────────────┘
                     SPI flash programming        UART (115200 baud)
                            │                          │
┌───────────────────────────┼──────────────────────────┼─────────────────┐
│                    NEURODRIVE REFERENCE PCBA                           │
│                           │                          │                 │
│  ┌────────────────────────┼──────────────────────────┼──────────┐     │
│  │              CARAVEL ASIC (M.2 daughter card)                 │     │
│  │  ┌──────────┐  ┌──────────────┐  ┌─────────────────────┐    │     │
│  │  │ RISC-V   │  │  NeuroDrive  │  │   SPI Flash         │    │     │
│  │  │ Mgmt Core│  │  User Logic  │  │  (weight storage)   │    │     │
│  │  └──────────┘  └──────┬───────┘  └─────────────────────┘    │     │
│  └───────────────────────┼──────────────────────────────────────┘     │
│                          │ 6× PWM + SPI + fault                       │
│          ┌───────────────┼───────────────┐                            │
│          ▼               ▼               ▼                            │
│  ┌──────────────┐ ┌─────────────┐ ┌────────────┐                     │
│  │ Gate Driver  │ │ External    │ │ Inductive  │                      │
│  │ (DRV8320)    │ │ ADC         │ │ Position   │                      │
│  │ 3-phase      │ │ (ADS7038)   │ │ Sensor IC  │                      │
│  │ half-bridge  │ │ 8ch 12-bit  │ │ (IPS2200)  │                      │
│  └──────┬───────┘ └──────┬──────┘ └─────┬──────┘                     │
│         │                │              │                              │
│  ┌──────┴───────┐ ┌──────┴──────┐       │                             │
│  │ 3× MOSFET    │ │ Current     │  ┌────┴──────┐                      │
│  │ Half-Bridge  │ │ Shunt       │  │ PCB coil  │                      │
│  │ (CSD18540Q5B)│ │ Amplifiers  │  │ (on motor │                      │
│  │ 60V, 100A    │ │ (INA240)    │  │  stator)  │                      │
│  └──────┬───────┘ └──────┬──────┘  └────┬──────┘                      │
│         │                │              │                              │
└─────────┼────────────────┼──────────────┼──────────────────────────────┘
          │                │              │
          ▼                ▼              ▼
    ┌──────────┐    ┌──────────┐   ┌──────────┐
    │  BLDC    │    │  Phase   │   │  Rotor   │
    │  Motor   │◄───│ Currents │   │ Position │
    │ (Gimbal) │    │ (sensed) │   │ (sensed) │
    └──────────┘    └──────────┘   └──────────┘
```

### 8.2 Signal Flow (One Control Cycle — 50 µs at 20 kHz)

1. **ADC Trigger** (t = 0 µs): PWM center-aligned trigger fires. External ADC (ADS7038) begins simultaneous sampling of 3 phase currents + bus voltage.
2. **Sensor Read** (t = 1–10 µs): SPI master reads ADC results (3 × 12-bit) and position sensor value (12-bit angle). Total: 4 × 16-bit SPI frames at 10 MHz = ~6.4 µs.
3. **State Assembly** (t = 10–12 µs): Sensor interface normalizes raw values to INT8 range [-128, +127] using pre-loaded scale/offset registers. Constructs 8-element state vector: [i_a, i_b, i_c, v_bus, ω, θ, e_speed, e_angle].
4. **NN Inference** (t = 12–30 µs): Inference engine executes 3-layer forward pass. Produces 6 INT8 action values (duty cycles for each half-bridge switch).
5. **Safety Check** (t = 30–31 µs): Safety monitor clamps actions, checks watchdog, verifies no fault conditions.
6. **PWM Update** (t = 31 µs): New duty-cycle values loaded into PWM compare registers, taking effect at next PWM cycle boundary.
7. **Idle / Logging** (t = 31–50 µs): Management core optionally reads state/action via Wishbone for UART logging.

---

## 9. Bill of Materials and Cost Estimate

### 9.1 Software / EDA Tools (All Open-Source, $0)

| Tool | Purpose | License |
|---|---|---|
| OpenLane 2 | RTL-to-GDSII flow | Apache 2.0 |
| Yosys | Logic synthesis | ISC |
| OpenROAD | Place and route | BSD |
| Magic | DRC, parasitic extraction | MIT |
| Netgen | LVS | GPL |
| OpenSTA | Static timing analysis | BSD |
| Verilator | Fast RTL simulation | LGPL |
| Icarus Verilog | Behavioral simulation | GPL |
| cocotb | Python-based verification | BSD |
| OpenRAM | SRAM compiler | BSD |
| KiCad | PCBA schematic + layout | GPL |
| OpenSCAD | Mechanical CAD | GPL |
| gym-electric-motor (GEM) | RL training environment | MIT |
| Stable-Baselines3 | RL algorithm library (TD3) | MIT |

### 9.2 Hardware BOM — Reference PCBA (per unit)

| Component | Part Number | Qty | Unit Cost | Total |
|---|---|---|---|---|
| Caravel M.2 daughter card | ChipFoundry (included) | 1 | incl. | — |
| 3-phase gate driver | DRV8320HRTAR | 1 | $2.50 | $2.50 |
| N-ch MOSFETs (60V/100A) | CSD18540Q5B | 6 | $0.85 | $5.10 |
| Current shunt amplifiers | INA240A1 | 3 | $1.20 | $3.60 |
| Shunt resistors (5 mΩ) | CSS2H-2512R-L500F | 3 | $0.30 | $0.90 |
| External ADC (8ch 12-bit SPI) | ADS7038 | 1 | $2.80 | $2.80 |
| Inductive position sensor IC | IPS2200 | 1 | $4.50 | $4.50 |
| SPI flash (16 MB) | W25Q128JVSIQ | 1 | $0.80 | $0.80 |
| 3.3V LDO regulator | AP2112K-3.3 | 1 | $0.30 | $0.30 |
| 1.8V LDO regulator | AP2112K-1.8 | 1 | $0.30 | $0.30 |
| 10 MHz crystal oscillator | LFXTAL082070 | 1 | $0.40 | $0.40 |
| Decoupling capacitors | Various (0402, 100nF/10µF) | 25 | $0.02 | $0.50 |
| Connectors (motor, power, debug) | Various | 5 | $0.50 | $2.50 |
| PCB fabrication (2-layer, 10×8 cm) | JLCPCB | 1 | $5.00 | $5.00 |
| **PCBA total (per unit)** | | | | **~$29.20** |

### 9.3 Test Setup BOM

| Item | Est. Cost |
|---|---|
| Gimbal BLDC motor (GBM2804H-100T or similar) | $15 |
| 24V power supply | $20 |
| USB-UART adapter (for Caravel debug) | $5 |
| 3D-printed test fixture (PLA filament) | $3 |
| **Test setup total** | **~$43** |

### 9.4 Project Cost Summary

| Category | Cost |
|---|---|
| ChipFoundry fabrication (if contest winner) | $0 (sponsored) |
| ChipFoundry fabrication (self-funded) | $14,950 |
| Reference PCBA (5 units prototype run) | ~$250 |
| Test setup | ~$43 |
| EDA tools | $0 (open-source) |
| **Total (contest-sponsored path)** | **~$293** |
| **Total (self-funded path)** | **~$15,243** |

---

## 10. Implementation Phases

### Target: Final design submission by April 30, 2026

The schedule assumes a single engineer working with aggressive AI-assisted RTL coding, starting from the proposal submission date.

```
March 25                              April 30
  │                                      │
  ▼                                      ▼
  ├──── Phase 1 ────┤── Phase 2 ──┤─ P3 ─┤
  │  Mar 25–Apr 6   │ Apr 7–20    │21-30 │
  │  RTL + unit TB  │ Integration │Tapeout│
  │  (12 days)      │ (14 days)   │(10d) │
```

### Phase 1: RTL Development + Unit Verification (Mar 25 – Apr 6, 12 days)

| Day(s) | Task | AI Assist | Output |
|---|---|---|---|
| 1 | Repository setup, `cf init`, `cf setup`, architecture finalization | Low | Working dev environment |
| 2–3 | MAC unit, MAC array, activation functions | High (70%) | `mac_unit.v`, `mac_array.v`, `activation_relu.v` + cocotb tests |
| 4–5 | NN inference engine (layer sequencer FSM, weight address gen) | Medium (50%) | `nn_inference_engine.v` + golden-model cocotb test |
| 6 | Weight SRAM integration (OpenRAM macros), WB-SRAM bridge | Low (30%) | `wb_sram_bridge.v`, SRAM macro instantiation |
| 7–8 | PWM generator + dead-time insertion | High (70%) | `pwm_generator.v`, `deadtime_insert.v` + timing tests |
| 9 | Safety monitor (clamping, watchdog, fault latch) | High (70%) | `safety_monitor.v` + fault-injection tests |
| 10 | SPI master + sensor interface | High (80%) | `spi_master.v`, `sensor_interface.v` + protocol tests |
| 11 | Wishbone interconnect + register file | Medium (50%) | `wb_interconnect.v`, `wb_registers.v` |
| 12 | `neurodrive_top.v` integration, first full-loop simulation | Low (20%) | Integrated RTL, passing integration test |

### Phase 2: Caravel Integration + System Verification (Apr 7 – Apr 20, 14 days)

| Day(s) | Task | Output |
|---|---|---|
| 13–14 | `user_project_wrapper.v` integration, GPIO config (`cf gpio-config`), `user_defines.v` | Caravel-integrated RTL |
| 15–16 | Caravel-level cocotb tests (Wishbone weight load → NN → PWM) | Passing Caravel integration tests |
| 17–18 | OpenLane hardening: NN engine sub-macro, then NeuroDrive top macro | Hardened GDS for sub-macros |
| 19–20 | `user_project_wrapper` hardening, timing closure, DRC/LVS fix iteration | Clean top-level GDS |
| 21 | Gate-level simulation (GLS) of Caravel integration tests | Passing GLS |
| 22 | STA (OpenSTA), SDC constraint tuning | Clean timing report |
| 23 | `cf precheck` pass | Green precheck |
| 24 | RISC-V firmware development (weight loader, config, UART logger) | Working firmware |
| 25–26 | KiCad PCBA schematic + layout | Board design files |

### Phase 3: Documentation + Submission (Apr 21 – Apr 30, 10 days)

| Day(s) | Task | Output |
|---|---|---|
| 27 | OpenSCAD test fixture design | Mechanical files |
| 28 | Python training pipeline (gym-electric-motor → INT8 export) | Training scripts + default policy |
| 29 | AI session log compilation, prompt archive | `ai_logs/` directory |
| 30–32 | Documentation: architecture, register map, pinout, assembly guide | `docs/` complete |
| 33 | 3-minute how-to video (screen recording + narration) | Video file |
| 34 | Final `cf precheck`, repo cleanup, tag release | Submission-ready repo |
| 35 | **Submit GitHub repo URL via contest form** | ✅ **Submitted** |

---

## 11. Challenges and Mitigations

### 11.1 Technical Challenges

| Challenge | Severity | Mitigation |
|---|---|---|
| **SRAM area dominance** — OpenRAM density (~5–7 KB/mm²) means 6 KB of weight storage consumes ~1.5–2 mm² | High | INT8 quantization keeps weights to 5 KB. Network sized conservatively (64 neurons). Register-based fallback for activation scratch if macro count is limited. Post-contest: commercial SRAM macros (16 KB = 0.67 mm²) dramatically improve density. |
| **Timing closure at 40 MHz** — MAC datapath (8-bit multiply + 24-bit accumulate) must close timing through OpenROAD | Medium | Pipeline the multiply and accumulate into 2 stages. Target 40 MHz (conservative for SKY130). Use `sky130_fd_sc_hd` library with 50% utilization. |
| **OpenRAM macro integration** — DRC waivers may be needed for OpenRAM macros; macro placement interacts with routing | Medium | Use pre-validated `sky130_sram_macros` from VLSIDA/Efabless. Follow OpenLane SRAM integration tutorial. Manual floorplan guidance in `config.json`. |
| **SPI timing for sensor reads** — 10 MHz SPI to external ADC must complete within control period budget | Low | 4 × 16-bit frames at 10 MHz = 6.4 µs. Budget allocates 10 µs. Comfortable margin. |
| **No on-chip ADC** — Motor current sensing requires external ADC, adding board complexity and SPI latency | Low | External ADC is the industry-standard approach for research boards. Simplifies the ASIC to purely digital, maximizing chance of first-pass success. On-chip ADC deferred to Phase 2 (OpenFrame). |

### 11.2 Schedule Challenges

| Challenge | Mitigation |
|---|---|
| **36-day design window is aggressive** | AI-assisted coding for boilerplate RTL (SPI, PWM, register files). Reuse proven patterns from existing Caravel tapeouts (Riscduino PWM, Tiny Tapeout designs). Prioritize inference engine correctness over feature completeness. |
| **Gate-level simulation may reveal issues late** | Run GLS on sub-macros early (day 21) before top-level integration. Maintain RTL-GLS parity by avoiding unsynthesizable constructs. |
| **Timing closure iteration** | Start with conservative 40 MHz target (well within SKY130 capability). No multi-clock-domain complexity. Single power domain. |

### 11.3 AI-Assisted Development Risk

The contest requires disclosure of all AI prompts and session logs. Our approach:

- **AI will be used heavily (60–80%)** for: peripheral RTL (SPI, UART, PWM), MAC unit generation, cocotb test scaffolding, SDC constraint templates, Makefile/script generation, and documentation.
- **AI will be used moderately (30–50%)** for: NN inference engine FSM, Wishbone interconnect, safety monitor logic.
- **Human expertise required (100%)** for: architecture decisions, SRAM macro integration, OpenLane configuration, timing closure, DRC/LVS debugging, Caravel integration, and overall verification strategy.
- **All sessions logged** in `ai_logs/` with full prompt history as required by contest rules.

---

## 12. Judging Criteria Alignment

| Criterion | How NeuroDrive Addresses It |
|---|---|
| **Technical Innovation** | First-ever RL inference ASIC for motor control on open-source silicon. Fills a genuine gap — no motor control chip exists on SKY130, and no commercial chip provides a hardware path for RL policy deployment at deterministic control rates. Combines two active research frontiers (RL for motor control + open-source silicon). |
| **Verification Coverage** | Module-level cocotb tests for every RTL block with golden-model comparison (Python NumPy reference for NN, analytical timing for PWM). Gate-level simulation. STA with SDC constraints. Fault-injection tests for safety monitor. |
| **Documentation Quality** | Complete architecture document, register map, GPIO pinout, PCBA assembly guide, training pipeline tutorial, and 3-minute how-to video. Repository structure follows Caravel template exactly. Third-party reproduction requires only: `cf setup` → train policy → program flash → spin motor. |
| **Feasibility & Cost** | PCBA BOM under $30/unit. Chip area estimate (~5 mm²) well within 10 mm² budget with ~50% margin. Network sized conservatively. No analog risk (external ADC). 40 MHz clock is within proven SKY130 range. Schedule has buffer days built in. |

---

## 13. References

1. Schenke, M., Kirchgässner, W., & Wallscheid, O. (2020). Controller design for electrical drives by deep reinforcement learning: A proof of concept. *IEEE Trans. Ind. Informatics*, 16(7), 4650–4658.
2. Traue, A., Book, G., Kirchgässner, W., & Wallscheid, O. (2022). Toward a reinforcement learning environment toolbox for intelligent electric motor control. *IEEE Trans. Neural Networks and Learning Systems*, 33(3), 919–928.
3. Schenke, M., Haucke-Korber, B., & Wallscheid, O. (2023). Finite-set direct torque control via edge-computing-assisted safe reinforcement learning for a PMSM. *IEEE Trans. Power Electronics*, 38(11), 13741–13756.
4. Jakobeit, D., Schenke, M., & Wallscheid, O. (2023). Meta-reinforcement learning-based current control of PMSM drives for a wide range of power classes. *IEEE Trans. Power Electronics*, 38(7), 8062–8074.
5. Mastanaiah et al. (2025). Deep reinforcement learning agent based speed controller for DTC-SVM of PMSM drive. *IET Power Electronics*.
6. ChipFoundry Platform Documentation. https://chipfoundry.io/knowledge-base/platforms
7. Caravel Harness Repository. https://github.com/efabless/caravel
8. gym-electric-motor (GEM). https://github.com/upb-lea/gym-electric-motor
9. OpenRAM SKY130 SRAM macros. https://github.com/VLSIDA/sky130_sram_macros
10. Hammond, P. (2023). Chip-Chat: First LLM-designed tapeout on SKY130. QTcore-A1 via Tiny Tapeout.

---

## License

This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

---

> **Contact**: [Juan Medrano]
> **email**: juanmedrano<dot>ec09<youknowwhat>gmail.com
---

<div align="center">

<img src="https://umsousercontent.com/lib_lnlnuhLgkYnZdkSC/hj0vk05j0kemus1i.png" alt="ChipFoundry Logo" height="140" />

[![Typing SVG](https://readme-typing-svg.demolab.com?font=Inter&size=44&duration=3000&pause=600&color=4C6EF5&center=true&vCenter=true&width=1100&lines=Caravel+User+Project+Template;OpenLane+%2B+ChipFoundry+Flow;Verification+and+Shuttle-Ready)](https://git.io/typing-svg)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![ChipFoundry Marketplace](https://img.shields.io/badge/ChipFoundry-Marketplace-6E40C9.svg)](https://platform.chipfoundry.io/marketplace)

</div>

## Table of Contents
- [Overview](#overview)
- [Documentation & Resources](#documentation--resources)
- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Starting Your Project](#starting-your-project)
- [Development Flow](#development-flow)
- [GPIO Configuration](#gpio-configuration)
- [Local Precheck](#local-precheck)
- [Checklist for Shuttle Submission](#checklist-for-shuttle-submission)

## Overview
This repository contains a user project designed for integration into the **Caravel chip user space**. Use it as a template for integrating custom RTL with Caravel's system-on-chip (SoC) utilities, including:

* **IO Pads:** Configurable general-purpose input/output.
* **Logic Analyzer Probes:** 128 signals for non-intrusive hardware debugging.
* **Wishbone Port:** A 32-bit standard bus interface for communication between the RISC-V management core and your custom hardware.

---

## Documentation & Resources
For detailed hardware specifications and register maps, refer to the following official documents:

* **[Caravel Datasheet](https://github.com/chipfoundry/caravel/blob/main/docs/caravel_datasheet_2.pdf)**: Detailed electrical and physical specifications of the Caravel harness.
* **[Caravel Technical Reference Manual (TRM)](https://github.com/chipfoundry/caravel/blob/main/docs/caravel_datasheet_2_register_TRM_r2.pdf)**: Complete register maps and programming guides for the management SoC.
* **[ChipFoundry Marketplace](https://platform.chipfoundry.io/marketplace)**: Access additional IP blocks, EDA tools, and shuttle services.

---

## Prerequisites
Ensure your environment meets the following requirements:

1. **Docker** [Linux](https://docs.docker.com/desktop/setup/install/linux/ubuntu/) | [Windows](https://docs.docker.com/desktop/setup/install/windows-install/) | [Mac](https://docs.docker.com/desktop/setup/install/mac-install/)
2. **Python 3.8+** with `pip`.
3. **Git**: For repository management.

---

## Project Structure
A successful Caravel project requires a specific directory layout for the automated tools to function:

| Directory | Description |
| :--- | :--- |
| `openlane/` | Configuration files for hardening macros and the wrapper. |
| `verilog/rtl/` | Source Verilog code for the project. |
| `verilog/gl/` | Gate-level netlists (generated after hardening). |
| `verilog/dv/` | Design Verification (cocotb and Verilog testbenches). |
| `gds/` | Final GDSII binary files for fabrication. |
| `lef/` | Library Exchange Format files for the macros. |

---

## Starting Your Project

### 1. Repository Setup
Create a new repository based on the `caravel_user_project` template and clone it to your local machine:

```bash
git clone <your-github-repo-URL>
pip install chipfoundry-cli
cd <project_name>
```

### 2. Project Initialization

> [!IMPORTANT]
> Run this first! Initialize your project configuration:

```bash
cf init
```

This creates `.cf/project.json` with project metadata. **This must be run before any other commands** (`cf setup`, `cf gpio-config`, `cf harden`, `cf precheck`, `cf verify`).

### 3. Environment Setup
Install the ChipFoundry CLI tool and set up the local environment (PDKs, OpenLane, and Caravel lite):

```bash
cf setup
```

The `cf setup` command installs:

- Caravel Lite: The Caravel SoC template.
- Management Core: RISC-V management area required for simulation.
- OpenLane: The RTL-to-GDS hardening flow.
- PDK: Skywater 130nm process design kit.
- Timing Scripts: For Static Timing Analysis (STA).

---

## Development Flow

### Hardening the Design
Hardening is the process of synthesizing your RTL and performing Place & Route (P&R) to create a GDSII layout.

#### Macro Hardening
Create a subdirectory for each custom macro under `openlane/` containing your `config.tcl`.

```bash
cf harden --list         # List detected configurations
cf harden <macro_name>   # Harden a specific macro
```

#### Integration
Instantiate your module(s) in `verilog/rtl/user_project_wrapper.v`.

Update `openlane/user_project_wrapper/config.json` environment variables (`VERILOG_FILES_BLACKBOX`, `EXTRA_LEFS`, `EXTRA_GDS_FILES`) to point to your new macros.

#### Wrapper Hardening
Finalize the top-level user project:

```bash
cf harden user_project_wrapper
```

### Verification

#### 1. Simulation
We use cocotb for functional verification. Ensure your file lists are updated in `verilog/includes/`.

**Configure GPIO settings first (required before verification):**

```bash
cf gpio-config
```

This interactive command will:
- Configure all GPIO pins interactively
- Automatically update `verilog/rtl/user_defines.v`
- Automatically run `gen_gpio_defaults.py` to generate GPIO defaults for simulation

GPIO configuration is required before running any verification tests.

Run RTL Simulation:

```bash
cf verify <test_name>
```

Run Gate-Level (GL) Simulation:

```bash
cf verify <test_name> --sim gl
```

Run all tests:

```bash
cf verify --all
```

#### 2. Static Timing Analysis (STA)
Verify that your design meets timing constraints using OpenSTA:

```bash
make extract-parasitics
make create-spef-mapping
make caravel-sta
```

> [!NOTE]
> Run `make setup-timing-scripts` if you need to update the STA environment.

---

## GPIO Configuration
Configure the power-on default configuration for each GPIO using the interactive CLI tool.

**Use the GPIO configuration command:**
```bash
cf gpio-config
```

This command will:
- Present an interactive form for configuring GPIO pins 5-37 (GPIO 0-4 are fixed system pins)
- Show available GPIO modes with descriptions
- Allow selection by number, partial key, or full mode name
- Save configuration to `.cf/project.json` (as hex values)
- Automatically update `verilog/rtl/user_defines.v` with the new configuration
- Automatically run `gen_gpio_defaults.py` to generate GPIO defaults for simulation (if Caravel is installed)

**GPIO Pin Information:**
- GPIO[0] to GPIO[4]: Preset system pins (do not change).
- GPIO[5] to GPIO[37]: User-configurable pins.

**Available GPIO Modes:**
- Management modes: `mgmt_input_nopull`, `mgmt_input_pulldown`, `mgmt_input_pullup`, `mgmt_output`, `mgmt_bidirectional`, `mgmt_analog`
- User modes: `user_input_nopull`, `user_input_pulldown`, `user_input_pullup`, `user_output`, `user_bidirectional`, `user_output_monitored`, `user_analog`

> [!NOTE]
> GPIO configuration is required before running `cf precheck` or `cf verify`. Invalid modes cannot be saved - all GPIOs must have valid configurations.

---

## Local Precheck
Before submitting your design for fabrication, run the local precheck to ensure it complies with all shuttle requirements:

> [!IMPORTANT]
> GPIO configuration is required before running precheck. Make sure you've run `cf gpio-config` first.

```bash
cf precheck
```

You can also run specific checks or disable LVS:

```bash
cf precheck --disable-lvs                    # Skip LVS check
cf precheck --checks license --checks makefile  # Run specific checks only
```
---

## Checklist for Shuttle Submission
- [ ] Top-level macro is named user_project_wrapper.
- [ ] Full Chip Simulation passes for both RTL and GL.
- [ ] Hardened Macros are LVS and DRC clean.
- [ ] user_project_wrapper matches the required pin order/template.
- [ ] Design passes the local cf precheck.
- [ ] Documentation (this README) is updated with project-specific details.
