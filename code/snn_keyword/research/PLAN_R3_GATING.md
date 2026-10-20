# Plan R3: clock gating for the B1 and B2 cascades

Written 2026-10-20, before any R3 result, for Pedro and for a fresh Claude Code session. Follows R2 ([PLAN_R2_GRU.md](PLAN_R2_GRU.md), [reports/R2_cascade_gru.md](reports/R2_cascade_gru.md)).

## 0. How to use this plan (new session: read this first)
**Prompt to start:**

> "Read 5XPG0/CLAUDE.md, then 5XPG0/code/snn_keyword/research/PLAN_R3_GATING.md, and carry out R3 from Phase 1. Stop at every gate and report to me."

**Read, in this order:**
1. this file;
2. `research/STATUS.md` (sections R2, S0–S1, S6);
3. `research/ENERGY.md` (power terms and the three scenarios);
4. `research/reports/R2_cascade_gru.md` (design codes, results).

Read other files only when a step needs them.

**Rules** (from `5XPG0/CLAUDE.md`):
- Work only on branch `Pedro`. Read other branches only with `git show <ref>:<path>` or `git ls-tree`; never check them out.
- Commit locally at the end of each phase. **Never push without asking.**
- Never modify `code/snn_keyword/deploy/`.
- No board session in R3. The board test comes later, after Phase 3.
- Every RTL change sits behind a parameter that is **off by default** (section 8).

**Pedro:**
- He is a beginner: use plain language and say what a long command does before running it.
- Report real numbers only.
- **Reports name no people.** Use the design codes (W0, B1, B2, ...) and "the streaming design" / "the parallel-processing design" (memory note `report-naming`).

**Gates:** stop at **G1** (end of Phase 1), **G2** (end of Phase 2) and **G3** (end of Phase 3), and on any failure listed as G2a. Also stop before installing system software.

## 1. Context

### 1.1 Why R3
- R2 found two cascades that meet all seven group requirements (test set, thresholds fixed on validation):

  | Code | Design | Live | FA/h | Verifier calls/h |
  |---|---|---|---|---|
  | **B1** | window SNN + fine-tuned GRU verifier, highest-recall rule | 87.8 ± 1.0 % | 1.48 | 3,044 ± 984 |
  | **B2** | the same, ≤ 2,150 verifier calls/h | 85.8 ± 0.8 % | 1.21 | 1,908 ± 200 |

- On today's hardware the core **busy-polls** the mailbox. Every design therefore draws the same 0.367 W of PL power (1,321 J/h), whatever work it does.
- Fewer verifier calls only save energy if the chip can idle. R3 builds that idle mode on our fabric and measures its effect on B1 and B2:
  1. **CPU sleep with a gated clock**: a BUFGCE stops the core's clock between requests, and the ARM's mailbox write wakes it.
  2. **BRAM read enables.** Both BRAM ports read on every clock cycle today (`spike_soc.v` l.93 and l.104), and the BRAM is 0.250 of the 0.282 W PL dynamic power. Gating the CPU clock alone would save only about 10 mW (u_cpu), so the read enables carry most of the saving.
- **Pedro's scope choice:** B1 and B2 must actually run on our fabric. That needs:
  - cascade firmware: window SNN stage 1 + fine-tuned GRU verifier stage 2;
  - the PC sends the 25 newest 10 ms frames each 250 ms, which the core keeps in a 1.5 s buffer for the verifier;
  - gating and read enables;
  - bit-exact xsim verification and a bitstream.

  The board run is deferred.

### 1.2 Design codes (from R2; use them in every report)
| Code | Meaning | Result-file name |
|---|---|---|
| W0 | window SNN alone (pruned H48 K64, our network) | `stage1_R1` |
| S1 | streaming SNN + original verifier (the streaming design) | `mateo_cascade` |
| B1 | window SNN + fine-tuned verifier, highest-recall rule | `cascade_B` |
| B2 | window SNN + fine-tuned verifier, ≤ 2,150 calls/h | `cascade_B_calls2150` |

**Representative configuration** (at the design means, used in the R2 figures): stage-1 seed 2 + fine-tuned verifier seed 1.

| | t1 | t2 | Calls/h | Source |
|---|---|---|---|---|
| B1 | −914 | 1402 | 2,752.5 | `results/R2/B/v1/h48_k64_seed2.json` |
| B2 | −596 | 1402 | 2,128.8 | `results/R2/B/v1/budget2150/h48_k64_seed2.json` |

- Read the thresholds programmatically (keys `cascade.t1`, `cascade.t2`); never type them by hand.
- Models:
  - stage 1: `results/S3/models/h48_k64_seed2.npz`;
  - verifier: `results/R2/verifier/sheila_head_r2_s1.pt` (h1 64, h2 64, stack 2, 40 classes, head).

### 1.3 Hardware facts (branch Pedro, HEAD 8b09978)
- **`code/pynqz2_riscv_flow/rtl/`:**
  - `spike_top.v`: single clock FCLK0 at 100 MHz, already behind the PS7's BUFG.
  - `spike_soc.v`:
    - PicoRV32 with `ENABLE_IRQ(0)` and `ENABLE_PCPI(1)`: no `wfi`/`waitirq`.
    - `READ_WAIT = 4'd8` (l.59).
    - BRAM `(* ram_style="block" *) reg [31:0] mem[0:65535]`, synthesized as 64 width-sliced RAMB36, so all 64 are active per access.
    - Port A (PS) and port B (CPU/kdot) read every cycle (l.93, l.104).
    - `timer_lo`, LED countdown and `u_poisson` run on `clk`; poisson steps its LFSRs every cycle, 5 mW.
    - VERSION/ABI `0x0002_0003` (l.187).
  - `kdot_pcpi.v`: custom-0, funct7 0. funct3 0 kdot, 1 klen, 2 kxload, 3 ksdot.
  - `ps_if.v`: AXI4-Lite slave. CTRL bit0 = core reset; ABI at l.196; `bram_addr` held, so port A re-reads the last address.
- **Build:**
  - `vivado/build.tcl` adds `[glob ../rtl/*.v]` (l.25), so **do not add new `.v` files to `rtl/`**; put any new module inside `spike_soc.v`.
  - `set_property top spike_top` is at l.110.
  - `spike_top.xdc` has no multicycle constraints.
  - Power flow: `research/tools/power/vivado_power.tcl` on a `git archive` copy under `code/snn_keyword/build/vivado_power/`. The exact S6 command is in `build/vivado_power/vivado_ksdot.log`. A full run takes about 4.5 min.
  - S6 ksdot fabric: 3,243 LUT, 2,831 FF, 64 RAMB36, 8 DSP, BUFGCTRL 1, WNS +0.545 ns (`results/S6/fabric.json`).
- **Power terms** (`research/tools/energy_model.py`, `load_power()['ksdot']`):

  | Term | W |
  |---|---|
  | P_PL,dyn | 0.282 |
  | P_PL,stat | 0.085 |
  | P_core (u_soc) | 0.273 |
  | P_infra | 0.009 |
  | Block RAM | 0.250 |
  | u_cpu | 0.010 |
  | u_kdot | 0.006 |
  | u_poisson | 0.005 |

- **Simulation:**
  - The only xsim test is `code/pynqz2_riscv_flow/sim/tb_kdot.sv` with `run_tb_kdot.sh` (kdot unit).
  - The full-SoC harness `code/snn_keyword/sim/soc_main.cpp` is Verilator/WSL and **unusable on this PC**.
  - `code/snn_keyword/sim/tb_axi.sv` has AXI-Lite master tasks to reuse.

### 1.4 Firmware and host facts
- **`code/snn_keyword/firmware/main.c`:**
  - Busy-polls `if (seq == MB[1]) continue;`.
  - Mailbox MB at 0x10400: [0] seq_in (the ARM writes it **last**), [1] seq_out, [2] opcode, [3] length, [6] status, [7..8] scores, [9] cycles, [10] spikes, [11] detected, [12..15] magic/thresholds.
  - Input buffer at 0x10800. TIMER at 0x10001000 (free-running). LED at 0x10002000.
- **`firmware/linker.ld`:** code 0x0–0xFFFF; `.model` 0x18000–0x3BFFF (144 KiB); stack top 0x40000.
- **`firmware/inference.c`:** USE_KSDOT / USE_LIF_TABLE / USE_KDOT paths. USE_KSDOT is tested before USE_KDOT.
- **`research/tools/build_firmware.py`:** `EXTRA_IMAGES` includes `keyword_ksdot_lut`; `--compare deploy` checks byte identity. **Do not modify it**; import from it.
- **`board_server.py`:**
  - writes the payload, MB[2], MB[3], then MB[0] last; polls MB[1] every 1 ms;
  - checks ABI bits 0 and 1 and the 100 MHz clock;
  - **do not modify** (a fabric with an extra ABI bit stays compatible).
- **S6 board cycle anchor** (`results/S6/board/h48_k64_seed2.json`): image `keyword_ksdot_lut.bin`, sha256 `d06c685c…`; 40 vectors × 3 took 33,785 / 34,165.6 / 34,617 cycles (min / mean / max). Board cycles equal RTL cycles (JOURNAL).

### 1.5 Teammate work to port (read only, with `git show`)
The streaming-design teammate already built the parts R3 needs. They target **another SoC** (READ_WAIT 1, neuron engine, different ABI bits; merge-base with Pedro is 05ad009), were verified only with Verilator in WSL, and have **never run on a board**. So port by hand; do not merge.

**`origin/opt/cpu-sleep` (f03b86f):**
- `code/pynqz2_riscv_flow/rtl/spike_soc.v`: sleep FSM about l.174–260, `clk_gate` about l.288–365.
  - POWER block at 0x1000_5000: a store of the wake mask to +0x00 stalls the CPU (no `mem_ready`, `sleeping = 1`).
  - Sticky doorbell set by the PS port-A write to word `16'h4100` (MB[0]).
  - Counters at +0x04 and +0x08.
  - BUFGCE under `` `ifdef SYNTHESIS ``, a latch-AND model in simulation.
- `code/snn_keyword/sim/tb_sleep.sv`; `docs/opt/cpu-sleep.md` (why a stalled store rather than IRQ + waitirq).
- `docs/opt/cpu-sleep_power.tcl`: report_power ignores the BUFGCE CE (Power 33-324), so it uses `create_clock` on the BUFGCE output with the duty.
- `code/snn_keyword/measure_sleep.py`.

**`origin/opt/bram-power` (d1938e3):**
- `pb_re = kd_busy || (acc_rd && sel_bram)`; port A `bram_re` from the `ps_if` read state; `if (pa_re)` / `if (pb_re)` around the reads.
- Optional `(* ram_decomp = "power" *)`: address-decoded blocks, +994 LUT, WNS +0.285 ns.
- `docs/opt/bram-power.md`, `docs/opt/bram_power.tcl`, `docs/opt/bram_power_energy.json`.

**`origin/integrate/power` (5274a41):** both merged; `docs/power/MERGED_POWER.md`. It puts sleep on ABI bit 4, so **use bit 2 here**.

**Verifier firmware at a856a03:** `code/snn_keyword/firmware/verifier.c`, `verifier.h`.
- Uses only plain kdot (funct3 0) and klen (funct3 1), which are identical on our `kdot_pcpi`.
- `stream_main.c` shows the cascade logic (`-DCASCADE`, vhist double buffer 2 × 150 × 24 B, opcodes 4/5/6, MB[26..33] verifier scores).
- **Do not port the KX instructions** (dual-core-alif): their opcodes clash with our kxload/ksdot.

**Already vendored Python** (`research/tools/mateo_eval/`):
- `verifier_export.py`: `load_quantized`, `export(q, out, frames, warmup, boundary, thresholds, cascade_t1)` writing `verifier_config.h` and `verifier_data.h` into section `.model`; `oracle()` = bit-exact integer verifier.
- `verifier_model.py`: `integer_forward`, `head_score`.
- `verifier_cascade.py`: `cascade_requests()` = the per-request firmware oracle of the streaming design.

**Our stage-1 oracle:** `model.integer_forward` / `research/tools/req_eval.py` `fast_margin`. Window j sits at frame 98 + 25j, inside request j + 3 (`research/tools/r2_cascade.py`).

**Data caches** (git-ignored, present):
- `code/snn_keyword/data_verifier/cache_test_sheila_r2_2_m.npz`: frames + window margins (`live`, `stream`, `neg`).
- `data_verifier/vscores_r2_s1_test.npz`: verifier heads per request.
- `code/snn_keyword/build/r1_cache/{live_test,stream_test}.npz`: window features.

### 1.6 Energy model inputs (`results/R2/designs.csv`)
| | B1 | B2 |
|---|---|---|
| Calls/h | 3,044.2 | 1,908.2 |
| Busy share at T_V 105.83 ms | 8.95 % | 5.61 % |
| Gated (R2 calculation) | 0.1188 W (−67.6 %) | 0.1097 W (−70.1 %) |

- Busy-polling: 0.367 W.
- Gated floor: P_stat + P_infra = 0.094 W.
- **T_V is unknown on our fabric.** R2 assumed 105.83 ms mean / 107.3 ms worst, measured on the teammate SoC with READ_WAIT = 1. With READ_WAIT = 8 the estimate is 10.73 M cycles + 1.6 M CPU reads × 7 extra cycles ≈ **21.9 M cycles ≈ 219 ms**. That would fail the ≤ 200 ms compute requirement while staying under the 250 ms hop.

## 2. Pre-registered hypotheses (commit before any R3 measurement)
| ID | Prediction | Test |
|---|---|---|
| H1 | T_V ≈ 21.9 M cycles (19–25 M) = 190–250 ms at READ_WAIT = 8 (10.3–11.2 M at READ_WAIT = 1). Worst compute ≈ 219 ms > 200 ms (**predicted fail**), < 250 ms | MB[21] in xsim |
| H2 | 100 % bit-exact on both fabrics. The F2 anchor equals the board cycles. Compute cycles are identical with POWER_OPT 0 and 1 | xsim |
| H3 | P_idle (gated + read enables) ≈ 0.094 W (accept 0.085–0.105) | Vivado `idle` state |
| H4 | Average PL power, three predictions: **(a)** R2 model B1 0.119 W (−67.6 %, 428 J/h), B2 0.110 W (−70.1 %, 395 J/h); **(b)** R3 central prior (T_V 219 ms, verifier increment ≈ 0.079 W) B1 0.109 W (−70 %), B2 0.103 W (−72 %); **(c)** pessimistic (whole core 0.273 W while awake) B1 0.145 W, B2 0.126 W. **Confirmed** if B1 ≤ 0.150 W and B2 ≤ 0.130 W; **refuted** if either saving < 50 % | `r3_energy.py` |
| H5 | Gating the CPU clock alone saves ≤ 0.03 W (≤ 8 %); the read enables give ≥ 85 % of the saving | `idle_gate_only` vs `idle` |
| H6 | Awake cycles per request without a verifier call 40–80 k; CPU asleep ≥ 79 % of the time (B1), ≥ 86 % (B2) | xsim counters |
| H7 | Cost ≤ +150 LUT, ≤ +120 FF, BUFGCTRL 1 → 2, 64 BRAM, 8 DSP, WNS ≥ 0 | Vivado reports |
| H8 | POWER_OPT = 0 is equivalent to r3-base; the release firmware runs bit-exact with identical cycles on the R3 fabric; `--compare deploy` identical; POWER_OPT = 0 utilization equal to S6 | section 6 |
| H9 | Only if the `ram_decomp` variant is built: verifier-state BRAM power −80 % or more; +about 1,000 LUT | Vivado |

Also predicted: latency (word end to decision) ≈ 275–290 ms, which passes 500 ms.

## 3. Phase 1: feasibility, pre-registration, plan mini report
1. **Tag and pre-register.**
   - `git tag r3-base` (local only).
   - Write `research/tools/r3_hypothesis.py`. It reads `results/R2/designs.csv`, the B JSONs, `results/R1/models.csv` and `energy_model.load_power()['ksdot']`, and writes `research/results/R3/phase1/hypothesis.json` with the section-2 numbers and their derivation.
   - Add an R3 "in progress" row to STATUS.md.
   - Commit ("R3: pre-registered hypotheses").
2. **Firmware v0.**
   - `git show a856a03:code/snn_keyword/firmware/verifier.c > code/snn_keyword/firmware/verifier.c` (and `verifier.h`), unchanged.
   - New `firmware/cascade_main.c` with opcodes 1 (window, as `main.c`), 2 (status) and 6 (verifier only: 150 frames in INPUT with length 3600, or the buffer with length 0; writes head, scores and a logits hash).
   - New `research/tools/r3_build_cascade.py`:
     - imports FLAGS, CROSS, run, section_sizes and the model export from `build_firmware.py`;
     - calls the vendored `verifier_export.export(q, out, frames=150, warmup=5, boundary=10, thresholds=(t2, NEG), cascade_t1=t1)` with `KWS_KEYWORD=sheila`;
     - compiles `../../pynqz2_riscv_flow/firmware/start.S cascade_main.c inference.c verifier.c runtime.c` with `-DUSE_KSDOT -DUSE_LIF_TABLE -DUSE_KDOT`, run from `code/snn_keyword/firmware` (as `build_firmware.SOURCES`);
     - compiler: `build_firmware.CROSS` (the `riscv64-unknown-elf-gcc.exe` reports itself as `riscv32-xilinx-elf-gcc` 13.4.0, the compiler of the S6 images);
     - writes `code/snn_keyword/build/r3_fw/<b1|b2>/cascade.{bin,elf,map}` + JSON (sha256, sizes).
   - **F1 (memory):** `.model` ≈ 12.7 + 53.3 KB ≤ 144 KiB; `.bss` end < 0x10000 (vhist 7.2 KB + logits).
3. **Full-SoC xsim testbench** `code/pynqz2_riscv_flow/sim/tb_r3_soc.sv`:
   - `ps_if` + `spike_soc` wired as in `spike_top`, without the block design;
   - AXI tasks from `tb_axi.sv`;
   - firmware by backdoor `$readmemh(+fw, u_soc.mem)` from a full 65,536-word hex file; zero `cpuregs` in the TB if X values appear;
   - release CTRL bit0 over AXI;
   - replay a request file (`+req`: opcode, length, gap, flags, payload words) with MB[0] written last; poll MB[1] every 1,000 cycles; a 40 M-cycle watchdog;
   - print one line per request: MB[6..35], cycles, gated-clock-on cycles, sleeping cycles, pb/pa enable and write cycles. On unchanged RTL, compute the "would-be" enables from `kd_busy || (acc_rd && sel_bram)` through hierarchical references.
4. **Runner** `research/tools/r3_sim.py` (subcommands `image`, `vectors`, `run`, `check`):
   - copies the sources to a space-free temp dir per run (`tempfile.mkdtemp(prefix='5xpg0_r3_')`, as `run_tb_kdot.sh` does);
   - runs `xvlog.bat --sv`, `xelab.bat tb_r3_soc -s snap --generic_top "POWER_OPT=1"`, `xsim.bat snap --runall --testplusarg ...`;
   - runs independent jobs in parallel (`concurrent.futures`), always as background commands.
5. **Feasibility runs.**
   - **F2, cycle anchor.**
     - `build_firmware.py research/results/S3/models/h48_k64_seed2.npz --out code/snn_keyword/build/r3_s6_anchor --images keyword_ksdot_lut`; its sha256 must start `d06c685c`.
     - In xsim, the 40 vectors of `results/sheila_hardneg/verification_vectors.npz` must give scores and spikes equal to `model.integer_forward`, and cycles exactly 33,785 / 34,165.6 / 34,617.
     - Record the simulator speed (cycles/s).
   - **F3, verifier.** Opcode 6 on 2–3 windows (a live keyword, a negative, one zero-padded at the stream start) must equal `verifier_export.oracle()` and the cached head. **T_V = MB[21]**.
   - Write `research/results/R3/phase1/feasibility.json`.
6. **Plan mini report.**
   - New `research/tools/make_r3_latex.py` (style of `make_r2_latex.py`). `make_r3_latex.py plan` writes `research/reports/latex/clock_gating_plan/`:
     - `main.tex`: IEEEtran with the same preamble as `reports/latex/cascade_gru/main.tex`; no fillbetween; copy `figures/r1_style.tex`;
     - `figures/fig_r3_idea.tex`;
     - `.gitignore` as in `cascade_gru`.
   - Content: **one concise paragraph** (about 150–200 words) — what clock gating is, why the BRAM read enables matter, the pre-registered savings for B1 and B2, the F2/F3 results.
   - **One figure:** (a) TikZ sketch: PS write of MB[0] → doorbell → sleep FSM → BUFGCE → CPU + kdot clock, and read enables → BRAM; (b) pgfplots P(t) over 1 s (4 requests): busy-polling flat at 0.367 W against gated at about 0.094 W with short stage-1 spikes and one verifier burst of the measured T_V. Shade with `\fill ... rectangle` or `\closedcycle`.
   - Build with `latexmk -pdf`; check every page at high resolution for overlaps (render with PyMuPDF installed into the scratchpad); commit only `main.pdf` and `.gitignore` (`*.tex` is git-ignored).
7. **Commit** ("R3 Phase 1: feasibility + plan report"), then **Gate G1**. Report T_V, simulator speed, the anchor result, memory fit and the plan PDF. Pedro decides:
   - **If T_V > 200 ms:** report the requirement failure as it is, or also build and verify a `READ_WAIT = 1` variant. Recommend the variant if T_V > 245 ms, where no real-time margin is left.
   - Whether to add the `ram_decomp = "power"` BRAM variant (H9).
   - The vector budget, if xsim is slow.

## 4. Phase 2: implement, verify, build, measure
### 2A. Firmware v1 and oracle (on the unchanged RTL)
- **`cascade_main.c` v1**, adding opcodes 5 and 7.
  - Opcode 5 = reset: buffer zeroed, prev_reach = 0, frames = 0.
  - Opcode 7 = cascade request. Payload: optional 768 B window, then k × 24 B frames (1 ≤ k ≤ 25); length = 24k (no stage 1, reach = 0) or 768 + 24k. Steps:
    1. push the frames into the double ring;
    2. stage 1: margin = s1 − s0; reach = margin ≥ T1;
    3. if reach or prev_reach: run the verifier on the last 150 frames;
    4. detection = head ≥ T2 and frames − last_event ≥ 100;
    5. LED for 1 s on a detection.
  - Sleep store: `if (seq == MB[1]) { POWER_SLEEP = 1; continue; }` (POWER_SLEEP = 0x10005000). On an old fabric the store completes at once.
  - MB layout:

    | Word | Content |
    |---|---|
    | MB[11] | bit0 detection, bit1 reach, bit2 verifier ran |
    | MB[12] | magic "KWC1" 0x4b574331 |
    | MB[13] | T1 |
    | MB[18] | frames since reset |
    | MB[20] | stage-1 cycles |
    | MB[21] | verifier cycles |
    | MB[22] | frame-push cycles |
    | MB[25] | T2 |
    | MB[26..29] | path scores |
    | MB[30] | logits hash |
    | MB[31] | 1 (cascade build) |
    | MB[32..33] | head, head step |
    | MB[34..35] | sleep counters, only if ABI bit 2 is set |

  - All timing via TIMER.
- **`research/tools/r3_oracle.py`:**
  - stage 1 = `model.integer_forward`;
  - verifier = vendored `oracle()`;
  - `cascade_requests_window(...)` = the logic of `verifier_cascade.cascade_requests` with the window margin at request j + 3.
- **Gate G-O (Python only, minutes):** from the cached margins and heads, the oracle reproduces the R2 representative-config test numbers: live 0.8726 (B1), 0.8585 (B2); 2,752.5 and 2,128.8 calls/h on `neg`. Any difference: stop (G2a).
- **Vectors** (`r3_sim.py vectors`; deterministic; item ids saved in `results/R3/sim/vectors.json`). Assert `fast_margin(r1_cache x) == r2 cache raw` at the window frames, and oracle head == cached head.

  | Set | Content |
  |---|---|
  | V1 | Opcode 1 × 40 (stage-1 bit-exactness with the cascade image) |
  | V2 | Opcode 6 × 3 (verifier, logits hash, T_V) |
  | V3-B1 | A live sequence with reach in ≥ 2 consecutive requests and ≥ 2 runs with head ≥ t2 within 100 frames (hold-off and the k−1 rule), incl. a partial last request; plus a stream stretch of about 40 requests far from keywords with exactly one proposal |
  | V4-B2 | A stretch where one margin is in [−914, −596) (B1 calls the verifier, B2 does not) + one detection |
  | V5 | POWER_OPT 0 vs 1 cycle identity on V1 + one V2 window + a stage-1-only part of V3 |

  Budget: about 13 verifier calls ≈ 290 M cycles. All sets must be 100 % bit-exact on the **unchanged** RTL before any RTL edit. Record the stage-1 cycles C_S1, T_V and the would-be enable duties.

### 2B. RTL (all default off)
- **`spike_soc.v`:** `parameter integer POWER_OPT = 0`, `parameter integer READ_WAIT = 8`.
  - With POWER_OPT = 1:
    - POWER block, sleep FSM, sticky doorbell, counters;
    - `clk_gate` (BUFGCE / latch model) for `u_cpu`, `u_kdot`, `u_poisson`;
    - `acc = acc_req && !kd_busy && !sleeping`.
  - Read enables:
    ```verilog
    wire pb_rd_en = (POWER_OPT != 0) ? (kd_busy || (acc_rd && sel_bram)) : 1'b1;
    wire pa_rd_en = (POWER_OPT != 0) ? pa_re : 1'b1;
    if (pa_rd_en) pa_rdata_i <= mem[pa_addr];      // l.93
    if (pb_rd_en) pb_rdata   <= mem[pb_word_addr]; // l.104
    ```
  - `initial sleeping = 0`.
- **`ps_if.v`:** `parameter [31:0] ABI = 32'h0002_0003`; output `bram_re = (rstate == 2'd1)`.
- **`spike_top.v`:** pass the parameters; `ps_if #(.ABI(POWER_OPT ? 32'h0002_0007 : 32'h0002_0003))`.
- Optional `` `ifdef R3_RAM_DECOMP `` adds `ram_decomp = "power"` to `mem` (only if chosen at G1).

### 2C. Verification (all must pass before the RTL commit)
1. **Equivalence:** `code/pynqz2_riscv_flow/sim/tb_r3_equiv.sv`.
   - Reference = `git show r3-base:` copies of `spike_soc.v`/`ps_if.v`, modules renamed `*_ref` by Python. New instance with POWER_OPT = 0, same stimulus.
   - Compare every cycle: CPU bus, PCPI, `pa_rdata`, `led_o`, trap, `timer_lo`, AXI responses.
   - Stimulus: release image `deploy/keyword_kdot.bin` (40 vectors) + a short stage-1-only cascade sequence.
2. **POWER_OPT = 1** with all vector sets and the B1/B2 images: bit-exact; per-request compute cycles (MB[9], MB[20], MB[21]) identical to 2A.
3. **Sleep checks** (directed requests):
   - wakes on an MB[0] write; does not wake on other words;
   - race sweep: PS write −3 to +3 cycles around the sleep accept, no lost request;
   - mask 0 does not stall;
   - while asleep the gated clock is stopped and `pb_re` = `pa_re` = 0 (assertion);
   - CTRL reset while asleep clears `sleeping` and the doorbell;
   - counters correct.
4. **Regressions:**
   - `bash code/pynqz2_riscv_flow/sim/run_tb_kdot.sh` passes;
   - release image on POWER_OPT = 1: bit-exact, cycles equal POWER_OPT = 0;
   - `build_firmware.py research/results/S2/models/h48_b8_seed1.npz --out code/snn_keyword/build/r3_release_check --images keyword_lut keyword_kdot_lut --compare code/snn_keyword/deploy` identical;
   - `git diff r3-base -- code/snn_keyword/deploy` empty.
- Commit ("R3: POWER_OPT RTL (default off) + cascade firmware, verified in xsim"). **Keep the RTL diff in one commit.**

### 2D. Vivado build
- New `research/tools/power/r3_vivado.tcl` (logic of `vivado_power.tcl`, with the same board-path workaround):
  - after the `set_property top` line, inject `set_property generic {POWER_OPT=1 READ_WAIT=8} [current_fileset]` (plus `verilog_define R3_RAM_DECOMP` if chosen);
  - `eval` the build;
  - **assert exactly one `BUFG*` cell under the CPU clock gate** (otherwise a latch was synthesized: stop);
  - write timing (incl. the BUFGCE CE path), utilization, DRC, `routed.dcp`, and the 12.5 / 6.25 / 25 % power reports.
- Run from a `git archive HEAD code/pynqz2_riscv_flow` copy at `code/snn_keyword/build/vivado_power/src_r3`. Use the S6 log's command, with `proj_r3`, `r3_gated.bit` and `reports_r3`.
- Outputs: `research/results/R3/r3_gated.bit`, `research/results/R3/vivado/*.rpt`, `research/results/R3/fabric.json` (S6 format: commit, RTL hashes, ABI 0x00020007, WNS, utilization, bit sha256).
- **Regression build POWER_OPT = 0** from the same commit: utilization must equal S6.
- WNS < 0, missing BUFGCE or DRC errors: stop (G2a), then section 9.

### 2E. Power and energy
- New `research/tools/power/r3_power_states.tcl <routed.dcp> <outdir> <duties.tcl>`, porting `git show origin/opt/cpu-sleep:docs/opt/cpu-sleep_power.tcl` and `origin/opt/bram-power:docs/opt/bram_power.tcl`.
  - Common settings, as in S0: `set_switching_activity -default_toggle_rate 12.5 -default_static_probability 0.5`.
  - Gated clock: `create_clock -period [expr {10.0/$d}]` on the BUFGCE output (works around Power 33-324).
  - BRAM EN/WE static probability from the simulated duties (per block for the width-sliced layout), toggle = min(99, 199·p).
  - States: `idle`, `idle_gate_only` (no read enables), `stage1`, `verifier`, `day_B1`, `day_B2`; each also at 6.25 / 25 %.
  - `duties.tcl` is written by `r3_sim.py`; parse the reports with `research/tools/power/parse_power.py` into `results/R3/power.json`.
- New `research/tools/r3_energy.py` → `results/R3/energy.{json,csv}` + hypothesis outcomes:
  ```
  E_h = 3600·P_idle + 14,400·(P_S1 − P_idle)·C_S1/f + N_V·(P_V − P_idle)·C_V/f,   f = 100 MHz
  ```
  - N_V = `designs.csv` means (B1 3,044.2, B2 1,908.2) ± sd, and the representative config (2,752.5 / 2,128.8).
  - P_avg = E_h / 3600; saving = 1 − P_avg / 0.367.
  - Awake share < 1 (check). Compare with the `day_*` reports.
  - Derived: t_max = max C_S1 + max C_V; latency = R2 latency − 107.3 + T_V,max (ms).
  - Label every number "Vivado estimate + RTL-simulated cycles" (not a board measurement).

### 2F. Board kit (written, not run)
- New `research/tools/r3_board_cascade.py` (runs on the PYNQ): mmap like `board_server.py`; `Bitstream(...).download()`; replays the same request files; writes JSON that `r3_sim.py check` can read.
- A "how to run later" section in the record.
- `board_server.py`, `start_board.sh`, `run_board.ps1` and `deploy/` stay unchanged.

### 2G. Record and gate
- `research/reports/R3_clock_gating.md` (style of R2_cascade_gru.md: design codes, summary, hypotheses and outcomes, checks, threats to validity, reproduce).
- An ENERGY.md section "R3: clock gating, measured duty, per-state Vivado".
- STATUS.md.
- Commit, then **Gate G2**: report bit-exactness, cost and timing, per-state power, the B1/B2 energy table and the hypothesis outcomes. Wait before Phase 3.

## 5. Phase 3: results mini report
- `make_r3_latex.py results` writes `research/reports/latex/clock_gating_results/` (a new, separate report).
- Very concise: **What**, **How**, **Why** as short paragraphs, then results.
- **Fig. 1:** PL power per state (busy, idle, idle with gate only, stage 1, verifier), stacked by component (static, BRAM, CPU+kdot+poisson, clocks/signals, interface).
- **Fig. 2:** average PL power against verifier calls/h. Busy flat at 0.367 W; measured gated line; R2 model dashed; B1 and B2 marked with calls-sd error bars (hollow = pre-registered, filled = measured).
- **Optional Fig. 3:** energy per hour for B1 and B2, busy vs gated, broken down into static, idle, stage 1 and verifier.
- **Tables:**
  - hypotheses vs outcomes;
  - cost: LUT, FF, BUFG, WNS, T_V, worst compute, latency;
  - requirements still met?
- **Pros and cons:** a short bullet list Pedro can expand later.
- Design codes only; no personal names; check all figures at high resolution for overlaps.
- Commit `main.pdf`, update STATUS.md, then **Gate G3**: ask Pedro about pushing, and plan the board test (bitstream, `r3_board_cascade.py`, USB meter busy vs gated).

## 6. Verification summary
- **Oracles:**
  - stage 1 = `model.integer_forward` (board-exact);
  - verifier = `verifier_export.oracle()`;
  - cascade = `r3_oracle.cascade_requests_window`.
- **Cross-checks:**
  - fast_margin == r2 cache `raw`;
  - oracle head == cached `vscores` head;
  - G-O reproduces the R2 numbers;
  - F2 reproduces the board cycles.
- **Order:** firmware on the unchanged RTL → equivalence (POWER_OPT = 0) → POWER_OPT = 1 bit-exact + identical cycles → sleep checks → regressions → Vivado (BUFGCE assertion, WNS ≥ 0, utilization) → power states.
- **xsim speed:** unknown until F2; expect 20–200 k cycles/s while awake.
  - Backdoor loading, no waveforms or `--debug`, one print line per request, 2–5 k cycle gaps between requests, parallel background runs.
  - If F2 predicts more than about 3 h in total, cut to V1 + 2×V2 + one live + one negative sequence (about 8 calls).

## 7. Power method notes
- The S0 method (vectorless 12.5 %, routed design, 25 °C) stays the reference. The busy reference is the unchanged S6 ksdot report: 0.367 W.
- Vivado ignores the BUFGCE CE. So each state is a separate `report_power` with `create_clock` on the gated clock (duty d) and BRAM enable activity from the simulated duties.
- **Optional cross-check:** SAIF from xsim (`open_saif` / `log_saif` / `close_saif` around about 1 M cycles of a verifier call), then `read_saif` in Vivado. Report "Design Nets Matched"; if matching is low, keep it as a note only.
- These remain estimates. The USB-meter validation of ENERGY.md section 3 (Vivado within 7 %) is the precedent; the board measurement is deferred.

## 8. Reversibility
- Local tag `r3-base` (HEAD before R3).
- Parameters default off (`POWER_OPT = 0`, `READ_WAIT = 8`, no `R3_RAM_DECOMP`); `build.tcl` unedited; no new `rtl/*.v`. Default builds are therefore unchanged, proved by the equivalence testbench, the POWER_OPT = 0 utilization and the release regression.
- Never modified: `code/snn_keyword/deploy/`, `firmware/main.c`, `firmware/inference.c`, `board_server.py`, `build_firmware.py`, `start_board.sh`, `research/results/S6/keyword_ksdot.bit`.
- New outputs only under `research/results/R3/`; builds under `code/snn_keyword/build/r3_*` and `build/vivado_power/*_r3*` (git-ignored).
- The RTL is committed as one commit, only after 2C passes. **Rollback:** `git revert <RTL commit>` or `git checkout r3-base -- code/pynqz2_riscv_flow/rtl`. If Phase 2 fails before the commit: `git checkout -- code/pynqz2_riscv_flow/rtl`.

## 9. Risks and fallbacks
| Risk | Fallback |
|---|---|
| T_V too long at READ_WAIT = 8 (likely 200–250 ms) | The `READ_WAIT` parameter is timing-neutral (no multicycle constraints): a READ_WAIT = 1 variant, decided at G1. Otherwise report the failure. KX acceleration stays out of scope (opcode clash) |
| xsim too slow | Smaller vector set, parallel runs, overnight background runs; the F2 anchor keeps T_V credible with few calls |
| X values in 4-state sim | Full-image hex; TB-only `cpuregs` zeroing; `initial` values on `sleeping` and the doorbell |
| BUFGCE timing (BUFG→BUFGCE skew on the CPU→64-BRAM path) | (a) Performance strategy for the R3 run only; (b) PS7 FCLK buffer off, with parallel BUFG and BUFGCE in `spike_top`; (c) no clock gate: sleep stall + read enables only, which keeps about 90 % of the saving (H5) |
| `SYNTHESIS` macro not defined, so a latch ends up on the clock | The build assertion stops the run; switch to `` `ifndef XILINX_SIMULATOR `` |
| BRAM enables folded or tied by Vivado | Check the layout report: `idle` must show BRAM power falling |
| Wake race | Sticky doorbell + race sweep; board-level glitch check deferred to the board run |
| Call-rate uncertainty (B1 sd 984/h) | Error bars; also report the representative config |
| Spaces in Windows paths | Space-free temp work dirs for xsim; the board-path workaround of `vivado_power.tcl` |

## 10. Practical notes (learned in R1/R2)
- **Bash heredocs mangle backslashes** (and `sed` patterns with `\` often fail). Write Python, LaTeX, Tcl and SV files with the Write tool, and use the Edit tool for edits.
- **Run long jobs** (xsim, Vivado) with `run_in_background` and wait for the notification; foreground `sleep` is blocked.
- **The repo path has spaces.** Quote every path; xsim and Vivado work dirs must be space-free (temp dirs).
- **`*.tex` is git-ignored at the repo root.** The LaTeX generator writes the files; only `main.pdf` (and the folder `.gitignore`) are committed.
- **Checking PDFs:** `.venv/Scripts/python.exe -m pip install --target <scratchpad>/pylib pymupdf`, then render pages and each figure at about 250 dpi and look for overlapping text, markers and labels (R2 needed two rounds of fixes).
- **The group report preamble** (FINAL REPORT/main.tex) has `pgfplots` and the TikZ libraries `arrows.meta,fit` only. Compile every figure and table inside a copy of that preamble (plus `\input{figures/r1_style}`).
- **Python:** `.venv/Scripts/python.exe` (torch CPU, numpy). `.venv-rocm` is the GPU environment from R2 (not needed in R3).
- **RISC-V gcc:** `C:/AMDDesignTools/2026.1/gnu/riscv/nt/bin/riscv64-unknown-elf-*`. Vivado/xsim: `C:/AMDDesignTools/2026.1/Vivado/bin/` (`xvlog.bat`, `xelab.bat`, `xsim.bat`, `vivado.bat`).

## 11. Rough schedule
| Phase | Estimate |
|---|---|
| 1 | Half a day to a day (the testbench is the main work; F3 needs one or two verifier calls in xsim) |
| 2 | 1–2 sessions (firmware + oracle, RTL + verification, 2 Vivado builds of about 5 min each, power states) |
| 3 | About half a day |
