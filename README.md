# NeuroDrive: A Tapeout-Realistic RL Policy Inference ASIC for BLDC Motor Control

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Platform](https://img.shields.io/badge/Platform-Caravel%20SKY130-6E40C9.svg)](https://chipfoundry.io)
[![Category](https://img.shields.io/badge/Category-Industrial%20%2F%20Edge--IoT-green.svg)](#)

> ChipFoundry Reference Application Design Contest
> Proposal revised after architecture, Caravel, marketplace, and Tiny Tapeout feasibility review
> March 26, 2026

---

## Table of Contents

- [1. Executive Summary](#1-executive-summary)
- [2. Review Outcome: What Changed and Why](#2-review-outcome-what-changed-and-why)
- [3. Rev A Project Scope](#3-rev-a-project-scope)
- [4. Updated System Architecture](#4-updated-system-architecture)
- [5. Digital Architecture](#5-digital-architecture)
- [6. Memory and Area Plan](#6-memory-and-area-plan)
- [7. Caravel Integration Plan](#7-caravel-integration-plan)
- [8. Reference Board Plan](#8-reference-board-plan)
- [9. Verification Strategy](#9-verification-strategy)
- [10. Implementation Schedule](#10-implementation-schedule)
- [11. Marketplace IP Plan](#11-marketplace-ip-plan)
- [12. Cost Estimate](#12-cost-estimate)
- [13. Why This Scope Is Credible](#13-why-this-scope-is-credible)
- [14. Risks and Mitigations](#14-risks-and-mitigations)
- [15. References](#15-references)

---

## 1. Executive Summary

**NeuroDrive** is a digital policy-inference ASIC for low-voltage BLDC motor control research on the Caravel SKY130 platform. The chip does **not** attempt to replace a complete industrial motor-control SoC in one shuttle cycle. Rev A instead targets a realistic and reproducible reference design:

- a fixed-topology INT8 MLP accelerator for a pre-trained RL policy,
- deterministic sensor acquisition from external ADC and rotor-angle devices,
- a hard real-time 3-phase PWM modulator with complementary outputs,
- a hardware safety envelope that can override the policy immediately,
- and enough on-chip memory and trace capture to use most of the Caravel user area in a way that is technically justified.

The design is intentionally **memory-dominated**. The Caravel user project wrapper exposes about **2.920 mm x 3.520 mm = 10.28 mm2** of area, and this proposal now targets roughly **8.3 to 9.2 mm2** of that space by using commercial SRAM macros from the ChipFoundry marketplace for weight bandwidth, trace capture, and bring-up buffers.

### Rev A Specifications

| Parameter | Rev A Target |
|---|---|
| Process | SKY130 |
| Platform | Caravel user area, 2.920 mm x 3.520 mm |
| User logic style | Digital only |
| Policy topology | Fixed MLP, `12 -> 128 -> 128 -> 3`, INT8 |
| MAC architecture | 16-lane output-parallel INT8 MAC array |
| Weight bandwidth | 128 bits/cycle via 4 interleaved 32-bit SRAM macros |
| Nominal clock target | 25 MHz |
| Stretch clock target | 40 MHz if timing allows |
| PWM carrier | 20 kHz |
| Policy update rate | 10 kHz baseline, 20 kHz stretch |
| Outputs | 6 complementary PWM outputs generated from 3 commanded duties |
| Sensor interface | Shared SPI to external ADC and angle sensor, optional Hall fallback |
| Safety | Async fault kill, watchdog, stale-sensor detect, duty clamp, slew limiter |
| Bring-up modes | Direct duty mode and open-loop electrical-angle sweep mode |
| Marketplace IP | Commercial SRAM macros, `CF_UART` |
| Marketplace IP cost | $2,500 per project for all SRAM instances |

### Rev A Positioning

This project is a **reference application**, not a claim that RL will immediately outperform tuned FOC on every motor. The value is a complete open-source path from:

`offline RL training -> fixed-point export -> deterministic silicon inference -> safe low-voltage BLDC bench setup`

That fits the contest goals better than a broader but less believable plan.

---

## 2. Review Outcome: What Changed and Why

The original draft had the right direction, but several parts were not tapeout-realistic for the ChipFoundry schedule ending on **April 30, 2026**.

### 2.1 Major Deletions from the Original Draft

| Original idea | Why it was not believable enough | Rev A correction |
|---|---|---|
| Run-time configurable NN topology via registers | Verification burden is too high for a 35-day tapeout window | Freeze one topology at synthesis time |
| RL directly emits 6 half-bridge outputs | Unsafe abstraction; high-side and low-side commands should not be independently learned | RL emits 3 phase-duty commands; hardware generates the 6 complementary outputs |
| Weight hot-swap during active control | Single-port SRAM and control-loop safety make this risky | Weight loading only while the motor is disabled |
| 2-layer board with discrete 60 V / 100 A stage | Overkill for a Caravel reference design and noisy for bring-up | 4-layer low-voltage board for a gimbal or small outrunner motor |
| Dependence on a PLL from day one | Adds unnecessary risk | Baseline closes at 25 MHz without requiring a PLL |
| Promise of a complete physical motor demo by April 30 | Silicon returns in October / November 2026 | Final contest deliverable is tapeout-ready silicon + board design + firmware + docs; live-silicon motor demo is post-return |

### 2.2 Major Additions

| Added item | Why it is needed |
|---|---|
| Fixed bring-up modes independent of the NN | Needed to debug the PWM, power stage, and sensors before trusting the policy |
| Commercial SRAM architecture sized for bandwidth | Four 32-bit banks are justified by the 16-lane MAC datapath, not just by capacity |
| Trace SRAM and post-fault capture | Essential for repeatable debug and third-party replication |
| CRC-checked policy image format | Prevents invalid weight loads and improves reproducibility |
| Duty slew limiter and stale-sensor faulting | Practical safety features missing from the first draft |
| Explicit use of `CF_UART` marketplace IP | Reduces schedule risk for diagnostics |

### 2.3 The Correct Rev A Objective

Rev A is now defined as:

> A fixed-topology INT8 RL policy coprocessor with safety-wrapped 3-phase PWM generation for a **low-voltage, sensored BLDC research platform**, tapeout-ready by April 30, 2026.

That is both innovative and believable.

---

## 3. Rev A Project Scope

### 3.1 In Scope

- Caravel-integrated user project with fixed-topology policy inference
- Deterministic sensor acquisition from external digital parts
- 3-phase complementary PWM generation with dead-time and synchronous duty updates
- Hardware safety interlocks and fault reporting
- Firmware for configuration, policy loading, trace readout, and safe start/stop
- 4-layer reference board design for a low-voltage BLDC test setup
- Mechanical fixture and assembly notes for post-silicon validation
- Full RTL, GLS, STA constraints, OpenLane hardening, and precheck package
- Open-source training/export scripts for the frozen Rev A policy format

### 3.2 Explicitly Out of Scope for Rev A

- On-chip RL training
- Sensorless startup
- Full FOC implementation as a second controller on silicon
- Runtime weight updates while the motor is spinning
- High-voltage or high-current industrial inverter hardware
- Safety certification
- Claiming a pre-silicon hardware motor demo before chips return

---

## 4. Updated System Architecture

### 4.1 System-Level View

```
Host PC
  |
  |  train / quantize / export policy
  v
SPI flash or management-core firmware image
  |
  v
Caravel management core -> Wishbone config + weight loading + trace readout
  |
  v
NeuroDrive user logic
  |- sensor SPI front-end
  |- policy inference core
  |- safety envelope
  |- 3-phase PWM modulator
  |- trace capture
  `- user UART diagnostics
  |
  v
Low-voltage BLDC reference board
  |- external ADC for phase currents and bus voltage
  |- magnetic angle sensor or Hall inputs
  |- comparator-based fault path
  `- 3-phase power stage
```

### 4.2 Real-Time Control Partition

The management core is **not** in the control loop. It only:

- loads policy weights,
- programs limits and timing registers,
- enables or disables control,
- services faults,
- and reads trace memory after a run.

The user logic performs:

- SPI sensor transactions,
- state-vector assembly,
- policy inference,
- PWM update timing,
- and fault handling.

This is the only partitioning that is realistic for deterministic motor control on Caravel.

### 4.3 Control Abstraction

The policy does **not** command transistor gates directly. Rev A uses this signal chain:

`policy output[2:0] -> clamp + slew limit -> 3 signed phase duty commands -> complementary PWM generator -> 6 gate-driver inputs`

This removes an unnecessary safety hazard from the original proposal.

### 4.4 Fixed State and Action Format

Rev A freezes a single policy interface:

- **12-element state vector**:
  - `i_a`, `i_b`, `i_c`
  - `v_bus`
  - `sin(theta_e)`, `cos(theta_e)`
  - `omega_est`
  - `reference`
  - `duty_a_prev`, `duty_b_prev`, `duty_c_prev`
  - `derate_or_temperature_margin`

- **3-element action vector**:
  - `duty_a_cmd`
  - `duty_b_cmd`
  - `duty_c_cmd`

The action outputs are signed fixed-point quantities interpreted by the hardware modulator.

---

## 5. Digital Architecture

### 5.1 Top-Level Blocks

```
Wishbone slave + control/status registers
        |
        +-- policy loader / SRAM arbiter
        +-- trace reader
        +-- diagnostics UART
        |
        +-- real-time subsystem
             |- SPI sensor engine
             |- state normalizer
             |- fixed-topology MLP engine
             |- safety envelope
             |- PWM generator
             `- bring-up mode generator
```

### 5.2 Policy Inference Engine

Rev A uses a fixed MLP:

- `12 -> 128 -> 128 -> 3`
- INT8 weights and activations
- ReLU in hidden layers
- linear output layer

The accelerator uses a **16-lane output-parallel** datapath:

- each cycle, one input activation is broadcast,
- 16 signed weights are fetched in parallel,
- 16 accumulators update 16 output neurons at once.

This organization is deliberate. It keeps activation-memory bandwidth modest while making weight bandwidth the dominant requirement. That directly motivates the four-bank SRAM plan in Section 6.

### 5.3 Inference Timing

For the fixed `12 -> 128 -> 128 -> 3` network:

| Layer | Cycles |
|---|---|
| `12 -> 128` | `12 x ceil(128/16) = 96` |
| `128 -> 128` | `128 x ceil(128/16) = 1024` |
| `128 -> 3` | `128 x ceil(3/16) = 128` |
| Control overhead | ~80 |
| **Total** | **~1328 cycles** |

At **25 MHz**, inference takes about **53 us**.

That supports a credible **10 kHz** policy rate with margin for sensor transactions and PWM update bookkeeping. If post-layout timing supports **40 MHz**, the same architecture can be evaluated at roughly **33 us** inference time, enabling an exploratory **20 kHz** policy rate.

### 5.4 Bring-Up and Safe Modes

Rev A adds two non-NN modes:

1. **Direct Duty Mode**
   - software writes the three phase duties directly,
   - used for board bring-up and power-stage validation.

2. **Open-Loop Angle Sweep Mode**
   - hardware emits a slow rotating electrical angle and modulation ramp,
   - used to verify encoder polarity, phase ordering, and inverter wiring.

These two modes make post-silicon bring-up materially more realistic.

### 5.5 Safety Envelope

The safety subsystem is entirely outside the neural network.

It enforces:

- asynchronous `FAULT_N` kill path from the external comparator,
- synchronized fault logging into status registers,
- watchdog timeout on missing policy completions,
- stale-sensor detection,
- duty clamp and minimum-off enforcement,
- duty slew-rate limiting,
- startup interlock,
- and forced neutral output on fault or invalid-policy state.

The external fault signal is intended to disable both the **ASIC PWM path** and the **power-stage enable** path. The chip is therefore not the single point of safety.

### 5.6 Policy Image Format

The weight image loaded by firmware includes:

- magic/version field,
- topology ID,
- quantization scale metadata,
- per-policy clamps,
- CRC32 of the payload,
- and the interleaved weight/bias payload.

Weights are loaded only while PWM is disabled. The hardware refuses to arm motor control until the image passes header and CRC checks.

---

## 6. Memory and Area Plan

### 6.1 Why Memory Dominates the Area

The revised proposal intentionally uses most of the Caravel project area for a reason:

- the NN engine needs **bandwidth**, not just capacity,
- the contest asks for a complete reference design, so traceability and diagnostics matter,
- and a large mostly-empty wrapper would look less credible than a well-justified, memory-centered architecture.

### 6.2 Selected Memory Architecture

| Macro Use | Instance Count | Size Each | Total Capacity | Area Each | Total Area |
|---|---|---|---|---|---|
| Interleaved weight banks | 4 | 32 KB | 128 KB | 1.34 mm2 | 5.36 mm2 |
| Trace ring buffer | 1 | 16 KB | 16 KB | 0.67 mm2 | 0.67 mm2 |
| Staging / metadata / debug capture | 1 | 16 KB | 16 KB | 0.67 mm2 | 0.67 mm2 |
| Activation scratch A | 1 | 4 KB | 4 KB | 0.165 mm2 | 0.165 mm2 |
| Activation scratch B | 1 | 4 KB | 4 KB | 0.165 mm2 | 0.165 mm2 |
| **Total memory** |  |  | **168 KB** |  | **7.03 mm2** |

### 6.3 Why Four 32-Bit Weight Banks Are the Right Choice

Each cycle, the 16-lane MAC array needs **16 signed 8-bit weights**. Four 32-bit SRAM banks provide:

- 4 words/cycle,
- 16 bytes/cycle,
- 128 bits/cycle total.

That is the minimum clean architecture for the chosen datapath.

This is a much stronger story than the original draft, which had enough capacity but no convincing bandwidth plan.

### 6.4 Estimated Total Area

| Block | Estimated Area |
|---|---|
| SRAM macros | 7.03 mm2 |
| MLP engine + normalization + safety + PWM + SPI + Wishbone + UART | 0.9 to 1.3 mm2 |
| Routing, CTS, control logic margin | 0.4 to 0.9 mm2 |
| **Total** | **8.33 to 9.23 mm2** |

That uses roughly **81% to 90%** of the available Caravel user area while still leaving routing margin.

### 6.5 Floorplan Intent

- weight SRAM banks placed close to the MLP core to minimize wide-bus routing,
- activation scratch adjacent to the core for ping-pong layer buffering,
- trace and staging SRAM placed closer to the Wishbone and UART side,
- PWM and safety placed near the GPIO edge,
- SPI and sensor front-end placed near the sensor I/O cluster.

This is a realistic macro-first hardening plan for OpenLane.

---

## 7. Caravel Integration Plan

### 7.1 Interfaces Used

- **Wishbone slave**
  - configuration,
  - policy loading,
  - trace readout,
  - status and fault reporting.

- **Logic analyzer**
  - bring-up visibility,
  - test overrides,
  - internal state observation during DV.

- **GPIO**
  - PWM outputs,
  - shared SPI,
  - diagnostics UART,
  - fault and enable pins,
  - optional Hall inputs.

- **User IRQ**
  - `irq[0]`: latched fault,
  - `irq[1]`: trace buffer full,
  - `irq[2]`: policy loaded / heartbeat.

### 7.2 GPIO Allocation

| GPIO | Direction | Function |
|---|---|---|
| `mprj_io[5:10]` | Output | `PWM_AH`, `PWM_AL`, `PWM_BH`, `PWM_BL`, `PWM_CH`, `PWM_CL` |
| `mprj_io[11]` | Output | `DRV_ENABLE` |
| `mprj_io[12]` | Input | `FAULT_N` |
| `mprj_io[13]` | Output | `ADC_SYNC` |
| `mprj_io[14]` | Output | `SPI_SCK` |
| `mprj_io[15]` | Output | `SPI_MOSI` |
| `mprj_io[16]` | Input | `SPI_MISO` |
| `mprj_io[17]` | Output | `SPI_CS_ADC` |
| `mprj_io[18]` | Output | `SPI_CS_POS` |
| `mprj_io[19]` | Output | `UART_TX` |
| `mprj_io[20]` | Input | `UART_RX` |
| `mprj_io[21:23]` | Input | Optional `HALL_A/B/C` or spare digital inputs |
| `mprj_io[24:37]` | Mixed | Spare debug and expansion pins |

This allocation stays within the normal Caravel model where GPIO `5:37` are the user-configurable pins.

### 7.3 Clocking

Rev A baseline:

- one synchronous user-logic domain,
- target timing closure at **25 MHz**,
- no mandatory PLL dependency.

This is safer than the original plan and still compatible with a later 40 MHz experiment if timing and bring-up allow it.

---

## 8. Reference Board Plan

### 8.1 Board Objective

The board is not a production inverter. It is a **low-voltage post-silicon validation platform** for:

- current sensing,
- rotor position sensing,
- PWM generation,
- fault behavior,
- and closed-loop policy evaluation on a small BLDC motor.

### 8.2 Board Changes from the Original Draft

| Original draft | Rev A correction |
|---|---|
| 2-layer board | 4-layer board |
| 60 V / 100 A discrete power stage | low-voltage, low-current power stage sized for a gimbal or small outrunner |
| industrial-style BOM | bench-friendly lab board |
| one-shot analog assumptions | explicit comparator fault path and cleaner grounding plan |

### 8.3 Planned Board Contents

- Caravel module connector
- low-voltage 3-phase inverter stage
- current shunt and amplifier chain
- external SPI ADC
- SPI magnetic angle sensor or Hall connector
- hardware comparator for fault shutdown
- regulators, oscillator, debug headers, motor and power connectors

### 8.4 Board-Level Safety

The board includes a comparator-driven shutdown path that:

- feeds `FAULT_N` into the ASIC for logging,
- and independently disables the motor-driver enable path.

This avoids treating the digital controller as the only safety barrier.

---

## 9. Verification Strategy

The contest explicitly requires RTL tests, GLS, STA constraints, and passing precheck. Rev A therefore prioritizes verification over optional features.

### 9.1 Block-Level Verification

- MLP engine against a Python fixed-point golden model
- weight-bank interleaving and address generation
- activation scratch ping-pong operation
- PWM timing, dead-time, and synchronized duty updates
- safety envelope fault injection
- SPI master transactions with ADC and angle-sensor models
- policy image header and CRC checking

### 9.2 Integration Verification

- sensor transaction -> state build -> inference -> PWM update
- fault during inference
- stale-sensor timeout
- policy load / arm / disarm sequences
- bring-up modes without the NN enabled

### 9.3 Gate-Level and Signoff

- one short GLS smoke test for each major mode
- SDF-backannotated motor-control integration smoke test
- STA with the contest-required SDC
- `cf precheck` and platform tapeout checks

### 9.4 What Will Not Be Pretended

The verification plan does **not** claim full analog motor-plant proof. Instead it provides:

- digital correctness,
- timing signoff,
- safety-sequence coverage,
- and a reproducible software golden model.

That is the right level of rigor for this contest timeline.

---

## 10. Implementation Schedule

The challenge sets:

- proposal deadline: **March 25, 2026**
- final tapeout-ready submission: **April 30, 2026**
- shuttle tapeout: **May 13, 2026**
- silicon return and assembly: **October / November 2026**

That means the tapeout-ready phase is about **35 days** long. The plan must be narrow.

### Phase 0: Scope Freeze and IP Setup

**March 26 - March 30**

- freeze topology, pinout, and policy image format
- install marketplace IP and create wrappers
- finalize floorplan assumptions and memory banking
- write golden-model format and test vectors

### Phase 1: Core RTL and Unit Tests

**March 31 - April 8**

- MLP engine
- SRAM bank wrappers and arbiter
- activation scratch and state normalizer
- PWM and safety blocks
- SPI front-end
- unit cocotb regressions

### Phase 2: Integration and Firmware Skeleton

**April 9 - April 18**

- Caravel wrapper integration
- Wishbone register map
- policy loader firmware
- trace capture path
- diagnostics UART integration
- integration regression

### Phase 3: Hardening and Signoff Closure

**April 19 - April 25**

- macro placement
- OpenLane hardening
- timing closure
- DRC and LVS fix iteration
- GLS smoke tests

### Phase 4: Submission Package

**April 26 - April 30**

- `cf precheck`
- board schematic and layout package
- docs and assembly notes
- final BOM
- AI session logs and release cleanup

### What Is Deliberately Deferred Until Chips Return

- live motor-spin demonstration on fabricated silicon
- board assembly results
- post-silicon controller tuning across multiple motors

This change makes the schedule consistent with the contest timeline instead of pretending that silicon will be available before April 30, 2026.

---

## 11. Marketplace IP Plan

### 11.1 Selected IP

| IP | Use | Cost | Selection Rationale |
|---|---|---|---|
| Commercial SRAM macros (`1024x32`, `4096x32`, `8192x32`) | weight banks, activation scratch, trace buffers | $2,500 per project | Required for realistic density and bandwidth |
| `CF_UART` | diagnostics and trace dump interface | $0 | Reduces schedule risk versus writing another UART block |

### 11.2 Evaluated but Not Selected for the Main Path

| IP | Reason not selected as primary solution |
|---|---|
| `CF_TMR32` | good standalone timer/PWM IP, but synchronized 3-phase complementary motor PWM is cleaner to verify as one custom block |
| `OL-DFFRAM` for bulk weights | free but too area-inefficient for the target memory footprint |

### 11.3 Why Marketplace SRAM Is Justified

Without the commercial SRAM option, Rev A would have to shrink to a much smaller network and leave a large fraction of the Caravel area unused. Since this proposal assumes the organizers will cover the marketplace IP cost in the sponsored path, the SRAM option is the correct choice for:

- a more credible memory system,
- higher confidence physical design,
- and a better use of the available silicon area.

---

## 12. Cost Estimate

### 12.1 EDA and Core Tooling

| Item | Cost |
|---|---|
| Open-source RTL, PnR, STA, verification, and board tools | $0 |

### 12.2 Marketplace IP

| Item | Cost | Notes |
|---|---|---|
| Commercial SRAM macros | $2,500 | one project fee, unlimited instances per ChipFoundry page |
| `CF_UART` | $0 | free catalog IP |

### 12.3 Reference Board and Bench Hardware

These are intentionally stated as realistic low-volume estimates, not inflated industrial BOM claims.

| Item | Estimated Cost |
|---|---|
| Low-voltage 4-layer BLDC reference board, assembled | $45 to $70 |
| Small BLDC motor | $15 to $25 |
| Lab supply / adapter / debug accessories | $25 to $40 |
| Mechanical fixture and printed parts | $5 to $10 |
| **Post-silicon bench total** | **$90 to $145** |

### 12.4 Contest-Sponsored Path

| Item | Cost to Project |
|---|---|
| Fabrication and packaging | covered by contest |
| Initial prototype PCBA support | covered by contest |
| Mechanical support | covered by contest |
| Commercial SRAM IP | covered by organizers in the sponsored path |
| Miscellaneous bench hardware | ~$90 to $145 |

This is a more realistic cost story than the original draft because it separates:

- silicon and sponsored contest costs,
- marketplace IP cost,
- and the actual post-silicon bench bring-up cost.

---

## 13. Why This Scope Is Credible

This revised proposal is based on practical lessons from previous Tiny Tapeout projects and from the Caravel contest rules.

### 13.1 Lessons Taken from Tiny Tapeout

1. **Scope discipline matters**

   QTCore-A1 and the locked QTCore-A1 variants are a useful reminder that even small designs often cut features late to fit and verify cleanly. Rev A therefore removes dynamic-topology support and other nonessential flexibility.

2. **AI or accelerator projects succeed when the interface is fixed**

   The Tiny Neural Network Accelerator project by Greg Chadwick uses a fixed external protocol and keeps the larger software and documentation stack outside the tiny taped-out core. Rev A follows the same principle: fixed topology on silicon, training and model tooling off-chip.

3. **Serialized compute is a realistic hardware strategy**

   The `Neural Network dinamic` project shows that reusing a small neuron datapath over time is a practical way to tape out neural-network logic. Rev A uses a serialized 16-lane output-parallel datapath instead of a fully parallel array.

4. **PWM and register blocks tape out well when kept deterministic**

   Projects such as `spi_pwm` show that simple, register-driven PWM hardware is credible in open-source flows. Rev A keeps PWM, fault handling, and bring-up modes deterministic and outside the policy core.

### 13.2 Lessons Taken from the ChipFoundry Contest Rules

The challenge judges:

- technical innovation,
- verification coverage,
- documentation quality,
- and feasibility/cost.

That means the winning plan is not the broadest plan. It is the plan with the strongest ratio of:

`useful novelty / verification risk`

The revised NeuroDrive proposal is much stronger by that metric than the original draft.

---

## 14. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Large macro count makes top-level floorplanning harder | High | freeze macro plan early and harden around a memory-first floorplan |
| 40 MHz may not close after layout | Medium | baseline the project at 25 MHz and 10 kHz policy rate |
| Motor bring-up can fail for board-level reasons unrelated to the NN | High | add direct-duty mode and open-loop sweep mode |
| Weight image corruption causes unsafe startup | Medium | CRC-protected policy format and arm-after-validate sequence |
| Sensor interface bugs appear late | Medium | emulate ADC and encoder in cocotb from the start |
| Full physical board assembly may slip beyond tapeout deadline | Low | board design files are in scope; physical validation is post-silicon by contest timeline |

---

## 15. References

### Contest and Platform

1. ChipFoundry Reference Application Design Contest  
   https://chipfoundry.io/challenges/application

2. ChipFoundry Commercial SRAM page  
   https://chipfoundry.io/commercial-sram

3. ChipFoundry IP Catalog  
   https://platform.chipfoundry.io/ip-catalog

4. Local Caravel wrapper area and integration template  
   `openlane/user_project_wrapper/config.json`  
   `verilog/rtl/user_project_wrapper.v`

### Tiny Tapeout precedents reviewed for this revision

5. Tiny Tapeout chips index  
   https://tinytapeout.com/chips/

6. QTCore-A1  
   https://tinytapeout.com/chips/tt03/kiwih_tt_top

7. RTL Locked QTCore-A1  
   https://tinytapeout.com/runs/tt03/072

8. Tiny Neural Network Accelerator  
   https://tinytapeout.com/chips/ttihp25a/tt_um_gregac_tiny_nn

9. Neural Network dinamic  
   https://tinytapeout.com/runs/tt07/tt_um_neural_network

10. spi_pwm  
    https://tinytapeout.com/chips/ttihp25a/tt_um_spi_pwm_djuara

### License

This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
