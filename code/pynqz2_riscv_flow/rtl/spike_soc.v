// spike_soc.v — minimal RISC-V SoC: PicoRV32 + 256 KB BRAM + peripherals.
//
// This is a stripped-down version of the AURA controller. All the
// neuromorphic-chip GPIO and bias-generator hardware is gone; what is left
// is the minimum needed to run C code on a soft core and collect spikes:
//
//   0x0000_0000  BRAM (256 KB)  program + data + console + spike log
//   0x1000_0000  SYSCTRL        0x00 MAGIC  0x04 VERSION
//                               0x08 SCRATCH 0x0C TRAP
//   0x1000_1000  TIMER          0x00 TIME   (32-bit free-running counter)
//   0x1000_2000  LED            0x00 OUT    ([9:0] PYNQ-Z2 board LEDs)
//   0x1000_3000  POISSON        hardware spike generator (see rtl/poisson.v)
//   0x1000_5000  POWER          (POWER_OPT = 1 only) 0x00 SLEEP (store: wake mask, bit0 = mailbox)
//                               0x04 clocks asleep  0x08 sleeps (both free-running)
//   PCPI custom-0               kdot/klen dense and kxload/ksdot sparse dot products (rtl/kdot_pcpi.v)
//
// The BRAM has two ports: port A belongs to the PS (via ps_if, used to load
// the program and read the console / spike log), port B belongs to the CPU.
// The CPU fetches and loads/stores from the SAME BRAM (execute-in-place).
//
// POWER_OPT = 1 (research R3, code/snn_keyword/research/PLAN_R3_GATING.md; default 0 = the design as
// before, bit for bit): the CPU sleeps between requests and its clock stops, and each BRAM port reads only
// in the cycles whose data is used.
//   - SLEEP: a CPU store to POWER 0x00 with wake-mask bit0 set stalls (no mem_ready, sleeping = 1) until
//     the doorbell rings. The doorbell is sticky: a PS port-A write of MB[0] (word 0x4100, the request's
//     sequence word, which the host writes last) sets it, a completed SLEEP clears it, and a PS write in
//     the same clock wins. A doorbell that rang before the SLEEP makes it return at once, so no request
//     is lost; the firmware re-reads MB[0] after every SLEEP. Mask 0 never stalls.
//   - Clock gate: the CPU, kdot and Poisson clocks stop while the CPU sleeps (one BUFGCE). A sleeping CPU
//     is stalled on a bus store and only waits for mem_ready, so stopping its clock changes nothing it
//     computes (the cycle CSR aside; the firmware times with TIMER). TIMER, the LED countdown and the
//     bus logic keep the free-running clock.
//   - BRAM read enables: port B reads when kdot owns it or on the accept cycle of a CPU BRAM read (the
//     output register then holds the word until it is used); port A reads in ps_if's read state.
//   - VERSION (and ps_if's ABI word) 0x0002_0007: bit2 = CPU sleep + gated clock + read enables.
// READ_WAIT: CPU read latency in clocks (the BRAM returns data after one); default 8 as before.
//
// Peripheral bit layouts (LED / POISSON) MUST match firmware/board.h.

module spike_soc #(
    parameter integer MEM_WORDS = 65536,  // 256 KB at 32 bits/word
    parameter integer POWER_OPT = 0,      // 1 = CPU sleep, gated CPU clock, BRAM read enables (R3)
    parameter integer READ_WAIT = 8       // clocks from a CPU read accept to mem_ready (1..15)
)(
    input  wire        clk,
    input  wire        core_rst_n,        // 0 = CPU held in reset

    // BRAM port A (PS side). pa_addr is a WORD address.
    input  wire             pa_we,
    input  wire             pa_re,        // ps_if reads pa_addr this cycle (used only with POWER_OPT = 1)
    input  wire [15:0]      pa_addr,
    input  wire [31:0]      pa_wdata,
    input  wire [3:0]       pa_be,
    output wire [31:0]      pa_rdata,

    // status / housekeeping
    output wire         core_running,
    output wire         core_trap,
    output reg  [31:0]  timer_lo,          // free-running (snapped to the PS)
    output reg  [9:0]   led_o              // board LEDs, written by the CPU
);

    // ------------------------------------------------------------------
    // CPU memory interface (PicoRV32 holds req signals until mem_ready)
    // ------------------------------------------------------------------
    wire        cpu_mem_valid;
    wire        cpu_mem_instr;
    reg         cpu_mem_ready;
    wire [31:0] cpu_mem_addr;
    wire [31:0] cpu_mem_wdata;
    wire [3:0]  cpu_mem_wstrb;
    reg  [31:0] cpu_mem_rdata;

    wire sel_bram = (cpu_mem_addr[31:28] == 4'h0);   // 0x0000_0000 region
    wire sel_peri = (cpu_mem_addr[31:28] == 4'h1);   // 0x1000_0000 region
    wire req_write = (|cpu_mem_wstrb);

    // Read data phase: BRAM is sync-read, so reads complete a few cycles
    // after accept. Writes complete on the accept cycle itself.
    reg  [3:0] xfer;                 // 0 idle, counting up to READ_WAIT
    localparam [3:0] RW = READ_WAIT;
    reg        xfer_bram;            // 1 = data comes from BRAM
    reg  [31:0] peri_rdata_r;        // peripheral value captured at accept

    // CPU sleep (POWER_OPT = 1): sleep_q = 1 while a SLEEP store is stalled; 0 otherwise.
    reg  sleeping = 1'b0;
    wire sleep_q = (POWER_OPT != 0) && sleeping;

    // The kdot coprocessor borrows port B while the CPU is stalled on it.
    wire        kd_busy;
    wire [15:0] kd_addr;
    wire acc_req = cpu_mem_valid && !cpu_mem_ready && (xfer == 4'd0);
    wire acc     = acc_req && !kd_busy && !sleep_q;   // a sleeping CPU's held store is not accepted again
    wire kd_mem_idle = (xfer == 4'd0) && !acc_req;
    wire acc_wr  = acc && req_write;
    wire acc_rd  = acc && !req_write;
    wire bram_wr_now = acc_wr && sel_bram;

    // ------------------------------------------------------------------
    // Main memory: 256 KB true dual-port block RAM (one always block per
    // port, byte-enable writes, 1-cycle registered read).
    // ------------------------------------------------------------------
    // `define R3_RAM_DECOMP (research R3 option B, set only by research/tools/power/r3_vivado.tcl): Vivado builds
    // the array from 64 address-decoded 1K x 36 blocks, so one block per port is enabled per cycle instead of the
    // width-sliced layout's 64 (with byte writes Vivado cannot put the read enables below on the width-sliced
    // blocks' enable pins; it ties them to 1). Behaviour is unchanged; without the define the array is as before.
`ifdef R3_RAM_DECOMP
    (* ram_style = "block", ram_decomp = "power" *) reg [31:0] mem [0:MEM_WORDS-1];
`else
    (* ram_style = "block" *) reg [31:0] mem [0:MEM_WORDS-1];
`endif

    reg  [31:0] pb_rdata;            // port B read data (CPU)
    reg  [31:0] pa_rdata_i;          // port A read data (PS)
    assign pa_rdata = pa_rdata_i;

    wire [15:0] pb_word_addr = kd_busy ? kd_addr : cpu_mem_addr[17:2];
    // Read enables (POWER_OPT = 1): kdot reads every cycle it owns port B, the CPU only on the accept cycle
    // of a BRAM read; the PS in ps_if's read state. Otherwise both ports read every cycle, as before.
    wire pb_rd_en = (POWER_OPT != 0) ? (kd_busy || (acc_rd && sel_bram)) : 1'b1;
    wire pa_rd_en = (POWER_OPT != 0) ? pa_re : 1'b1;

    // Port A (PS)
    always @(posedge clk) begin
        if (pa_we) begin
            if (pa_be[0]) mem[pa_addr][7:0]   <= pa_wdata[7:0];
            if (pa_be[1]) mem[pa_addr][15:8]  <= pa_wdata[15:8];
            if (pa_be[2]) mem[pa_addr][23:16] <= pa_wdata[23:16];
            if (pa_be[3]) mem[pa_addr][31:24] <= pa_wdata[31:24];
        end
        if (pa_rd_en) pa_rdata_i <= mem[pa_addr];
    end

    // Port B (CPU)
    always @(posedge clk) begin
        if (bram_wr_now) begin
            if (cpu_mem_wstrb[0]) mem[pb_word_addr][7:0]   <= cpu_mem_wdata[7:0];
            if (cpu_mem_wstrb[1]) mem[pb_word_addr][15:8]  <= cpu_mem_wdata[15:8];
            if (cpu_mem_wstrb[2]) mem[pb_word_addr][23:16] <= cpu_mem_wdata[23:16];
            if (cpu_mem_wstrb[3]) mem[pb_word_addr][31:24] <= cpu_mem_wdata[31:24];
        end
        if (pb_rd_en) pb_rdata <= mem[pb_word_addr];
    end

    // ------------------------------------------------------------------
    // CPU clock gate (POWER_OPT = 1): CPU, kdot and Poisson stop while the CPU sleeps; the clock
    // runs in reset. The wake raises mem_ready and the clock enable in the same cycle.
    // ------------------------------------------------------------------
    wire cpu_clk_en = !core_rst_n || !sleep_q;
    wire cpu_clk;
    clk_gate #(.GATE(POWER_OPT != 0)) u_cpu_gate (.clk(clk), .en(cpu_clk_en), .gclk(cpu_clk));

    // ------------------------------------------------------------------
    // Peripherals
    // ------------------------------------------------------------------
    reg [31:0] sys_scratch;
    // LED pulse duration in fabric clocks; independent of firmware workload.
    reg [31:0] led_remaining;
    // FPGA register INIT also gives a defined timer in four-state simulation.
    initial timer_lo = 32'd0;

    // Hardware Poisson spike generator.
    wire        po_wr  = acc_wr && sel_peri && (cpu_mem_addr[15:12] == 4'h3);
    wire [7:0]  po_addr = cpu_mem_addr[7:0];
    wire [31:0] po_rdata;
    wire [7:0]  po_pending;      // (unused by the SoC, available for expansion)
    wire [31:0] po_total;

    poisson #(
        .CLK_HZ (32'd100_000_000)
    ) u_poisson (
        .clk     (cpu_clk),
        .rst_n   (core_rst_n),
        .wr_en   (po_wr),
        .wr_addr (po_addr),
        .wr_data (cpu_mem_wdata),
        .rd_addr (po_addr),
        .rd_data (po_rdata),
        .pending (po_pending),
        .total   (po_total)
    );

    // POWER: CPU sleep (POWER_OPT = 1).
    localparam [15:0] MB_SEQ_WORD = 16'h4100;        // MB[0] = 0x10400 / 4
    reg        doorbell = 1'b0;
    reg [31:0] sleep_cycles = 32'd0, sleep_count = 32'd0;
    wire pwr_sleep = (POWER_OPT != 0) && acc_wr && sel_peri && (cpu_mem_addr[15:12] == 4'h5) &&
                     (cpu_mem_addr[3:2] == 2'd0);
    wire wake_now  = !cpu_mem_wdata[0] || doorbell;   // mask 0, or the doorbell already rang
    wire wake_slp  = sleep_q && doorbell;
    wire db_clear  = (pwr_sleep && cpu_mem_wdata[0] && doorbell) || wake_slp;
    always @(posedge clk) begin
        if (!core_rst_n) doorbell <= 1'b0;
        else if (pa_we && pa_addr == MB_SEQ_WORD) doorbell <= 1'b1;   // a PS write in the same clock wins
        else if (db_clear) doorbell <= 1'b0;
    end
    always @(posedge clk) begin
        if (sleep_q) sleep_cycles <= sleep_cycles + 32'd1;
        if (pwr_sleep && !wake_now) sleep_count <= sleep_count + 32'd1;
    end

    // Free-running timer (not reset by core_rst_n, so the PS can tell the
    // fabric is alive even while the CPU is held in reset).
    always @(posedge clk)
        timer_lo <= timer_lo + 32'h1;

    // ------------------------------------------------------------------
    // CPU bus FSM + peripheral write/read decode
    // ------------------------------------------------------------------
    always @(posedge clk) begin
        if (!core_rst_n) begin
            cpu_mem_ready <= 1'b0;
            xfer          <= 4'd0;
            sys_scratch   <= 32'h0;
            led_o         <= 10'h0;
            led_remaining <= 32'd0;
            sleeping      <= 1'b0;
        end else begin
            cpu_mem_ready <= 1'b0;
            if (wake_slp) begin                  // the stalled SLEEP store completes
                sleeping      <= 1'b0;
                cpu_mem_ready <= 1'b1;
            end
            if (led_remaining != 0) begin
                led_remaining <= led_remaining - 1'b1;
                if (led_remaining == 1) led_o <= 10'd0;
            end

            if (xfer == 4'd0) begin
                if (acc_wr) begin
                    cpu_mem_ready <= 1'b1;
                    if (sel_peri) begin
                        case (cpu_mem_addr[15:12])
                        4'h0: if (cpu_mem_addr[3:2] == 2'd2)        // SYSCTRL SCRATCH
                                  sys_scratch <= cpu_mem_wdata;
                        4'h2: begin
                            if (cpu_mem_addr[3:2] == 2'd0) begin
                                led_o <= cpu_mem_wdata[9:0];
                                led_remaining <= 0;
                            end
                            // Writing duration starts/retriggers LED0 atomically.
                            if (cpu_mem_addr[3:2] == 2'd1) begin
                                led_remaining <= cpu_mem_wdata;
                                led_o <= cpu_mem_wdata == 0 ? 10'd0 : 10'd1;
                            end
                        end
                        4'h5: if (pwr_sleep && !wake_now) begin   // POWER SLEEP: stall
                            cpu_mem_ready <= 1'b0;
                            sleeping      <= 1'b1;
                        end
                        default: ;   // TIMER/POISSON have no CPU writes here
                        endcase
                    end
                    // (BRAM writes happen in the port-B block above)
                end else if (acc_rd) begin
                    xfer      <= 4'd1;
                    xfer_bram <= sel_bram;
                    case (cpu_mem_addr[15:12])
                    4'h0: case (cpu_mem_addr[3:2])                  // SYSCTRL
                          2'd0: peri_rdata_r <= 32'h534B_454C;     // "SKEL"
                          // v2.0 + bit0: kdot, bit1: kxload/ksdot, bit2: CPU sleep + gated clock + read enables
                          2'd1: peri_rdata_r <= (POWER_OPT != 0) ? 32'h0002_0007 : 32'h0002_0003;
                          2'd2: peri_rdata_r <= sys_scratch;
                          2'd3: peri_rdata_r <= {31'b0, core_trap};
                          endcase
                    4'h1: peri_rdata_r <= timer_lo;                 // TIMER
                    4'h2: peri_rdata_r <= cpu_mem_addr[3:2] == 2'd1 ? led_remaining : {22'b0, led_o};
                    4'h3: peri_rdata_r <= po_rdata;                 // POISSON
                    4'h5: peri_rdata_r <= (POWER_OPT == 0) ? 32'hDEAD_BEEF :     // POWER
                                          cpu_mem_addr[3:2] == 2'd1 ? sleep_cycles :
                                          cpu_mem_addr[3:2] == 2'd2 ? sleep_count : 32'd0;
                    default: peri_rdata_r <= 32'hDEAD_BEEF;
                    endcase
                end
            end else if (xfer != RW) begin
                xfer <= xfer + 4'd1;
            end else begin
                // read completion
                xfer          <= 4'd0;
                cpu_mem_ready <= 1'b1;
                cpu_mem_rdata <= xfer_bram ? pb_rdata : peri_rdata_r;
            end
        end
    end

    assign core_running = core_rst_n;

    // ------------------------------------------------------------------
    // CPU
    // ------------------------------------------------------------------
    wire        pcpi_valid, pcpi_wr, pcpi_wait, pcpi_ready;
    wire [31:0] pcpi_insn, pcpi_rs1, pcpi_rs2, pcpi_rd;

    kdot_pcpi u_kdot (
        .clk        (cpu_clk),
        .resetn     (core_rst_n),
        .pcpi_valid (pcpi_valid),
        .pcpi_insn  (pcpi_insn),
        .pcpi_rs1   (pcpi_rs1),
        .pcpi_rs2   (pcpi_rs2),
        .pcpi_wr    (pcpi_wr),
        .pcpi_rd    (pcpi_rd),
        .pcpi_wait  (pcpi_wait),
        .pcpi_ready (pcpi_ready),
        .mem_idle   (kd_mem_idle),
        .mem_busy   (kd_busy),
        .mem_addr   (kd_addr),
        .mem_rdata  (pb_rdata)
    );

    picorv32 #(
        .ENABLE_PCPI       (1),
        .ENABLE_COUNTERS   (1),
        .ENABLE_COUNTERS64 (0),
        .ENABLE_MUL        (0),
        .ENABLE_FAST_MUL   (1),
        .ENABLE_DIV        (1),
        .ENABLE_IRQ        (0),
        .ENABLE_TRACE      (0),
        .PROGADDR_RESET    (32'h0000_0000),
        .STACKADDR         (32'h0000_0000)
    ) u_cpu (
        .clk         (cpu_clk),
        .resetn      (core_rst_n),
        .trap        (core_trap),
        .mem_valid   (cpu_mem_valid),
        .mem_instr   (cpu_mem_instr),
        .mem_ready   (cpu_mem_ready),
        .mem_addr    (cpu_mem_addr),
        .mem_wdata   (cpu_mem_wdata),
        .mem_wstrb   (cpu_mem_wstrb),
        .mem_rdata   (cpu_mem_rdata),
        .pcpi_valid  (pcpi_valid), .pcpi_insn (pcpi_insn), .pcpi_rs1 (pcpi_rs1), .pcpi_rs2 (pcpi_rs2),
        .pcpi_wr     (pcpi_wr), .pcpi_rd (pcpi_rd), .pcpi_wait (pcpi_wait), .pcpi_ready (pcpi_ready),
        .irq         (32'h0),
        .eoi         (),
        .trace_valid (), .trace_data ()
    );

endmodule

// Glitch-free clock gate (ported from the streaming design's opt/cpu-sleep branch): gclk follows clk while
// en, sampled before the rising edge, is high. FPGA: a BUFGCE global buffer. Simulation: the equivalent latch
// (open while clk is low) ANDed with the clock, so registers on gclk update at an edge exactly when en was
// high just before it, as with a clock enable. GATE = 0: no gate (gclk = clk).
module clk_gate #(
    parameter integer GATE = 1
)(
    input  wire clk,
    input  wire en,
    output wire gclk
);
    generate if (GATE == 0) begin : g_pass
        assign gclk = clk;
    end else begin : g_gate
`ifdef SYNTHESIS
        BUFGCE u_bufgce (.I(clk), .CE(en), .O(gclk));
`else
        reg en_q = 1'b1;
        always @(clk or en) if (!clk) en_q = en;      // latch, open while clk is low
        assign gclk = clk & en_q;
`endif
    end endgenerate
endmodule
