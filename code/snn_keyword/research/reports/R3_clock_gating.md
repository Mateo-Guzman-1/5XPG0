# R3: clock gating and BRAM power for the B1/B2 cascades

*Dates:* 2026-10-20
*Branch:* Pedro (local tag `r3-base` = the commit before R3)
*Plan:* [PLAN_R3_GATING.md](../PLAN_R3_GATING.md); pre-registered predictions: `results/R3/phase1/hypothesis.json`,
`results/R3/phase2/hypothesis_rw1.json` (READ_WAIT 1, written at G1), `results/R3/phase2/hypothesis_decomp.json`
(address-decoded BRAM, written before its build)
*Data:* `results/R3/`
*LaTeX:* plan report `reports/latex/clock_gating_plan/main.pdf`; results report: Phase 3

## Design codes
| Code | Design |
|---|---|
| B1 | window SNN + fine-tuned GRU verifier, highest-recall rule (R2) |
| B2 | the same, at most 2,150 verifier calls per hour (R2) |
| busy | the S6 `ksdot` fabric as built: the core busy-polls, both BRAM ports read every cycle (0.367 W) |
| G8 | clock gating (CPU sleep + gated CPU clock), READ_WAIT 8, BRAM as built (`r3_gated_rw8.bit`) |
| G1 | the same with READ_WAIT 1 (`r3_gated_rw1.bit`): the main design chosen at G1 |
| G1-D | G1 with the address-decoded BRAM (`R3_RAM_DECOMP`, `ram_decomp = "power"`, option B), Vivado implementation strategy Performance_NetDelay_high (`r3_gated_rw1_decomp.bit`) |

The streaming design's branches `opt/cpu-sleep` and `opt/bram-power` were ported by hand (no merge).

## Summary
| | |
|---|---|
| **Question** | B1 and B2 run on our fabric as one cascade firmware. How much average PL power does an idle mode save: the CPU sleeps between requests (stalled store, sticky doorbell), its clock stops (BUFGCE), and the BRAM stops reading when nothing needs it? |
| **Method** | Cascade firmware (`cascade_main.c`, verifier unchanged from the streaming design) verified bit-exact in a new full-SoC xsim testbench against the integer oracles; RTL behind parameters that are off by default; Vivado builds; per-state `report_power` with the simulated CPU-clock duty and the BRAM enable probability of the built netlist; energy per hour from the simulated cycles. |
| **Speed** | Verifier call: 21.17 M cycles (211.7 ms) at READ_WAIT 8, 10.69 M (106.9 ms) at READ_WAIT 1. Worst request 212.2 ms (fails the 200 ms requirement) and 107.2 ms (passes). G1 meets all seven requirements. |
| **Energy, as built (G1)** | B1 0.352 W, B2 0.351 W: **-4.0 % / -4.3 %** against busy-polling. Only the clock gate saves power: with byte writes Vivado ties the enables of the width-sliced BRAM to 1, so the RTL read enables are not in the netlist (below). |
| **Energy, G1-D** | B1 0.106 W, B2 0.105 W: **-71.1 % / -71.5 %** (381 / 376 J/h against 1,321 J/h busy-polling); BRAM 0.254 -> 0.004 W; idle 0.102 W, awake 0.143 W. Cost +1,645 LUT, +183 FF against G1. With Vivado's default strategies it missed timing (WNS -0.288 ns on BRAM -> 64-way selector -> kdot DSP input); the plan's fallback, a timing-driven implementation strategy with no design change, closes it: WNS +0.366 ns (Performance_NetDelay_high; ExplorePostRoutePhysOpt +0.106 ns, ExtraTimingOpt +0.364 ns). |
| **Cost of G1** | +60 LUT, +84 FF, one BUFGCE, 64 BRAM, 8 DSP; WNS +0.479 ns. |

## Hypotheses and outcomes
| ID | Prediction | Outcome |
|---|---|---|
| H1 | T_V 19-25 M cycles at READ_WAIT 8; worst compute > 200 ms, < 250 ms | **Confirmed:** 21.17 M; 212.2 ms. |
| H1 (RW1) | T_V 10.3-11.2 M; worst compute <= 200 ms | **Confirmed:** 10.69 M; 107.2 ms. |
| H2 | 100 % bit-exact; F2 anchor = board cycles; compute cycles equal with POWER_OPT 0 and 1 | **Confirmed** at READ_WAIT 8 and 1 (all sets; release firmware gives the board's 58,299 / 58,670.8 / 59,040 cycles). |
| S1 (RW1) | stage 1 11-18 k cycles at READ_WAIT 1 | **Confirmed:** 17,695 (mean over 55 windows; 34,311 at READ_WAIT 8). |
| H3 | P_idle 0.085-0.105 W | **Refuted as built** (G8/G1 0.350-0.351 W); **met by G1-D** (0.102 W). |
| H4 | B1 <= 0.150 W and B2 <= 0.130 W; refuted if a saving < 50 % | **Refuted as built** (-4 %); **confirmed for G1-D** (0.106 / 0.105 W, -71 %). |
| H5 | gate alone saves <= 0.03 W; the read enables give >= 85 % of the saving | First part **confirmed** (0.017 W); second **not met**: the read enables are not in the built netlist. The BRAM saving comes from the address-decoded layout instead. |
| H6 | awake 40-80 k cycles per request (RW8), asleep >= 79 % (B1), >= 86 % (B2) | **Confirmed:** 47.6 k; 81.9 % / 88.6 %. |
| H6 (RW1) | awake 15-40 k; asleep >= 90 % / >= 93 % | **Confirmed:** 23.8 k; 90.9 % / 94.2 %. |
| H7 | <= +150 LUT, <= +120 FF, BUFGCTRL 2, 64 BRAM, 8 DSP, WNS >= 0 | **Confirmed** for G8 and G1. |
| H8 | POWER_OPT 0 equivalent; release bit-exact with identical cycles; `--compare deploy` identical; utilization = S6 | Equivalence, release and compare **confirmed**; utilization **+11 LUT** (registers, BRAM, DSP equal; WNS +0.645 ns). Rebuilding r3-base with the S6 flow gives S6 exactly, so the 11 LUT come from the reworded RTL. Accepted. |
| H9 | address-decoded BRAM: verifier-state BRAM power -80 % or more; about +1,000 LUT | Power **confirmed** (-98 %, idle 0.102 W, B1/B2 in the predicted 0.10-0.15 W; the layout alone, without sleeping, 0.143 W < 0.17 W). Cost **refuted**: +1,645 LUT (in the predicted 1,000-2,200), +183 FF (> 150), and timing met only with a non-default implementation strategy. |
| H10 | READ_WAIT 1 lowers the gated power by only 1-5 mW | **Confirmed** (as built): 3.0 mW (B1), 2.2 mW (B2). |
| Requirements (RW1) | all seven met by B1 and B2 | **Confirmed** for G1 and G1-D (detection unchanged from R2; 107.2 ms; 170 / 178 ms latency; 64.5 KiB; 0.35 W as built, 0.106 W G1-D). |

## Checks (`results/R3/phase2/verification.json`, all pass)
- **G-O:** the firmware oracle reproduces R2's representative configuration: live 0.8726 (B1), 0.8585 (B2); 2,752.5 and 2,128.8 verifier calls per hour (`phase2/gate_GO.json`).
- **Vector sets** (13 verifier calls; `results/R3/sim/vectors.json`): V1 40 stage-1 windows, V2 3 verifier windows, V3a a live keyword (hold-off and k-1 rule, partial last request), V3b 40 stream requests with one proposal, V4 a margin between the B1 and B2 thresholds (on both images), V5 the stage-1-only start of V3b. All bit-exact on the r3-base RTL, then with POWER_OPT 1 at READ_WAIT 8 and 1.
- **Equivalence:** POWER_OPT 0 against the r3-base RTL every clock (CPU bus, PCPI, port-A data, LEDs, trap, timer, AXI): 0 mismatches over 3.5 M cycles (release firmware and a cascade sequence).
- **Sleep:** wake only on a PS write of MB[0]; race sweep (MB[0] written 58 clocks before to 4,050 after the SLEEP accept, incl. -3..+3): no request lost; reset while asleep clears sleeping and the doorbell; mask 0 does not stall; POWER counters equal the testbench's counts; no gated clock edge and no port-B read while asleep.
- **Regressions:** `run_tb_kdot.sh` 48/48; release firmware `--compare deploy` identical; `deploy/` unchanged since r3-base; protected files unchanged.
- **Vivado:** exactly one BUFGCE under the CPU gate (G8, G1, G1-D); CE path slack +2.7 ns; DRC as S6 (two DSP pipelining warnings).

## The BRAM finding (`results/R3/vivado/bram_enable/`)
- The 64K x 32 memory is built as 64 width-sliced RAMB36 (32K x 1, cascade pairs). With byte writes, **synthesis ties every enable to 1**, for both coding styles tried (Synth 8-6841 on test RAMs); opt_design then gates only the lower block of each pair by the address MSB. The RTL read enables are not in the netlist; behaviour is unchanged ("always read" is the r3-base behaviour).
- A first set of per-state reports gave the BRAM enables the simulated read duties (idle 0.222 W, "-38 %"). They did not describe the built netlist and were withdrawn before any commit; the method now takes the enable probability from the built layout (1 as built, 1/64 for G1-D).
- With `ram_decomp = "power"` the memory is 64 address-decoded 1K x 36 blocks: one block per port is enabled per cycle (also asleep, at the frozen address), and the read hold is built in fabric. A further step, B+ (64 explicit banks whose enables also need a read or write, zero blocks while asleep), is deferred.

## Deviations from the plan (all reported to Pedro)
- G1 decision: READ_WAIT 1 is the main design; READ_WAIT 8 shows gating alone. Predictions for READ_WAIT 1 were added before any such run.
- Option B (`R3_RAM_DECOMP`) was declined at G1 and then chosen after the BRAM finding, with its own predictions. It is a second RTL commit (619cacd, an attribute behind a define, default off).
- The plan's sleep check "while asleep pb_re = pa_re = 0": port A legitimately reads while the CPU sleeps (the host reads the reply), so the check is "no gated clock edge and no port-B read while asleep".
- The equivalence testbench is a mode of `tb_r3_soc.sv` (`R3_EQUIV`) rather than a separate `tb_r3_equiv.sv`; a small sleep self-test firmware (`r3_sleep_test.c`) covers the mask-0 check.
- Command 7 does not compute the logits hash (MB[30] = 0) so that T_V is the cascade's; the hash is checked with command 6.
- The race sweep covers 0-120 clocks after the reply instead of only -3..+3 around the accept.
- The RTL is compiled as plain Verilog in xsim (`ps_if.v` uses `rstate` before its declaration); no RTL change.

## Threats to validity
- **Estimates, not measurements.** Vivado `report_power` (vectorless defaults, 25 C, "Medium" confidence) with simulated clock duty and the layout's enable probability; no board measurement yet. ENERGY.md section 3 (meter within 7 % for a loaded design) is the only validation, and it did not test idle states.
- **Toggle-rate sensitivity is small:** 6.25 % and 25 % change B1/B2 by <= 2 mW; the day reports agree with the model within 2 mW.
- **Verifier calls** come from R2 (19 h of dense speech); B1's sd (984 calls/h) moves G1-D's B1 by +-1.2 mW.
- **G1-D's timing margin** (+0.366 ns) needs a non-default implementation strategy; a later RTL change could lose it. The B+ variant or a register stage before the kdot multiplier would give more margin.
- **The enable probability of G1-D** (one block per port in every cycle) is the netlist's decode logic; the power of the 64-way selector and the read hold in fabric is in Vivado's vectorless estimate (signals and logic rise from 0.016 to 0.038 W while awake).
- **The board run is still to do** (bit-exactness on hardware, BUFGCE glitch-free wake, meter: busy vs gated).

## Board kit (written, not run)
`research/tools/r3_board_cascade.py` runs on the PYNQ: loads a bitstream and the cascade firmware, replays an R3 request file, writes REQ lines that `r3_sim.py check <log> <set>.expect.json` compares with the oracle and the simulated cycles, and with `--idle-seconds` reports the asleep share for the USB-meter comparison. How to run later:
1. Copy `results/R3/r3_gated_rw1_decomp.bit` (G1-D; or `r3_gated_rw1.bit` for G1), `build/r3_fw/b1/cascade.bin` and `build/r3_sim/phase2/v3a.req` to the board.
2. `sudo env XILINX_XRT=/usr BOARD=Pynq-Z2 python3 r3_board_cascade.py --bit r3_gated_rw1.bit --fw cascade.bin --req v3a.req --out v3a.board.txt --idle-seconds 60`
3. On the PC: `r3_sim.py check v3a.board.txt build/r3_sim/phase2/v3a.expect.json`.
4. Meter: board input power with the busy demo (`run_board.ps1`) and with the gated bitstream idle; then restore the demo with `run_board.ps1`.

## Reproduce
From the repo root (`.venv`; xsim and Vivado 2026.1):
```bash
python code/snn_keyword/research/tools/r3_hypothesis.py                 # phase1/hypothesis.json (and --addendum-rw1)
python code/snn_keyword/research/tools/r3_build_cascade.py              # build/r3_fw/b1, b2 (+ --sleep-test)
python code/snn_keyword/research/tools/r3_oracle.py gate                # G-O
python code/snn_keyword/research/tools/r3_sim.py vectors v1|v2|v3a|v3b|v4|rel|sleep --out-dir code/snn_keyword/build/r3_sim/phase2
python code/snn_keyword/research/tools/r3_sim.py run <name>=<fw.bin>,<set>.req ... --out-dir ... --generic POWER_OPT=1 --generic READ_WAIT=1 [--equiv]
python code/snn_keyword/research/tools/r3_verify.py collect             # verification.json
python code/snn_keyword/research/tools/r3_verify.py duties              # activity.json, duties_<variant>.tcl
# Vivado, from a git archive copy of code/pynqz2_riscv_flow (vivado/ folder, r3_vivado.tcl copied next to build.tcl):
vivado -mode batch -nojournal -source r3_vivado.tcl -tclargs <proj> <bit> <reports> <POWER_OPT> <READ_WAIT> [ram_decomp [<impl strategy> [<synth strategy>]]]
vivado -mode batch -nojournal -source r3_power_states.tcl -tclargs <reports>/routed.dcp <outdir> duties_<variant>.tcl
python code/snn_keyword/research/tools/power/parse_power.py results/R3/vivado/asbuilt_rw8 .../asbuilt_rw1 .../decomp_rw1 --out results/R3/power.json
python code/snn_keyword/research/tools/r3_verify.py fabric              # fabric.json
python code/snn_keyword/research/tools/r3_energy.py                     # energy.json, energy.csv
```
