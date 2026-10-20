# Why the BRAM read enables did not reach the hardware (R3, G2 finding, 2026-10-20)

**Question.** With POWER_OPT = 1, `spike_soc.v` reads port B only when kdot owns it or a CPU BRAM read is accepted,
and port A only in ps_if's read state. Do these read enables reach the BRAM enable pins of the built design?

**Answer: no, not with the width-sliced 64K x 32 memory.**

| Evidence | File | What it shows |
|---|---|---|
| Routed main design (POWER_OPT 1, READ_WAIT 1) | `en_logic_rw1.txt` | Upper block of each cascade pair: ENARDEN = ENBWREN = VCC. Lower block: CPU-port EN = NOT(port-B word-address bit 15) (LUT3 INIT 0x47 of kd_busy, kd_addr[15], cpu_mem_addr[17]); PS-port EN = 1 after power-on. No read-enable term. |
| Synthesized vs opt netlist | `en_synth_rw1.txt` | After synthesis every EN is VCC; opt_design's BRAM power optimization ("cooolgate") then gates only the lower blocks by the address MSB. |
| Idle probes on the routed design | `probe1_lines.txt`, `probe2_lines.txt`, `probe1_*.txt` | Forcing EN = 0 leaves 0.125 W: the 32 upper blocks, whose EN pins are constant. Declaring the BRAM clock pins slow does not change it. |
| `spike_soc.v` in Vivado's standard "port with enable" template | `spike_soc_template_en_report.txt` | Still EN = VCC after synthesis. |
| Test RAMs (`research/tools/power/bram_enable/`) | `test_ram_report1..5.txt`, `test_ram_synth_warning.txt` | 64K x 32 without byte writes: EN = write OR read (mapped). With byte writes (both coding styles): EN = VCC, read hold in fabric (Synth 8-6841: byte-wide write enable cannot be used). With `ram_decomp = "power"`: 64 blocks of 1K x 36, EN = 6-bit address decode (one block per port per cycle), read hold in fabric (about +1,950 LUT). |

**Consequences.**
- The built POWER_OPT 1 designs read the BRAM every cycle, also while the CPU sleeps. Only the clock gate saves
  power: B1 0.352 W (-4.0 %), B2 0.351 W (-4.3 %) at READ_WAIT 1.
- The first per-state reports, which gave the BRAM enables the simulated read duties (idle 0.222 W, "-38 %"), did
  not describe the built netlist; they were withdrawn and never committed.
- Option B (`R3_RAM_DECOMP`, `ram_decomp = "power"`): one block per port enabled per cycle. Option B+ (64 explicit
  banks whose enables include the read/write activity, zero blocks while asleep) is deferred.
- Function is unaffected: "always read" is the r3-base behaviour; all xsim results stand.

Scripts: `research/tools/power/bram_enable/` (`r3_en_logic.tcl`, `r3_en_synth.tcl`, `r3_bram_probe*.tcl` on the
build checkpoints; `t*.tcl` with `ram*.v` for the test RAMs; `spike_soc_template_synth.tcl` for the template test).
