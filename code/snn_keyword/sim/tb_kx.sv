// tb_kx.sv -- instruction-level test of rtl/kdot_pcpi.v (kdot/klen and the KX extension).
// Drives the PCPI interface like PicoRV32 and serves the unit's BRAM reads like spike_soc.v
// (registered address in the unit, registered BRAM read: data two cycles after issue).
// Vectors and expected results come from sim/kx_vectors.py (golden semantics in Python).
// mem_idle drops at random while the unit waits for the bus and while it runs; between
// instructions the core idles for the gap given by the vector (0 = back to back).
// Prints per-instruction cycle totals for legacy kdot/klen so two RTL versions can be
// compared for "no slower". Run with sim/kx_unit.sh.
`include "tb.vh"
`timescale 1ns/1ps
module tb_kx;
    reg clk = 0;
    always #5 clk = ~clk;
    reg resetn = 0;
    reg         pcpi_valid = 0;
    reg  [31:0] pcpi_insn = 0, pcpi_rs1 = 0, pcpi_rs2 = 0;
    wire        pcpi_wr, pcpi_wait, pcpi_ready;
    wire [31:0] pcpi_rd;
    reg         mem_idle = 1;
    wire        mem_busy;
    wire [15:0] mem_addr;
    reg  [31:0] mem_rdata;

`ifdef NO_KX_PARAM
    kdot_pcpi dut (                          // RTL without the KX parameter (legacy comparison)
`else
    kdot_pcpi #(.KX(`KX)) dut (
`endif
        .clk(clk), .resetn(resetn), .pcpi_valid(pcpi_valid), .pcpi_insn(pcpi_insn),
        .pcpi_rs1(pcpi_rs1), .pcpi_rs2(pcpi_rs2), .pcpi_wr(pcpi_wr), .pcpi_rd(pcpi_rd),
        .pcpi_wait(pcpi_wait), .pcpi_ready(pcpi_ready),
        .mem_idle(mem_idle), .mem_busy(mem_busy), .mem_addr(mem_addr), .mem_rdata(mem_rdata));

    reg [31:0] mem [0:65535];
    reg [31:0] prog [0:5*`NPROG-1];
    always @(posedge clk) mem_rdata <= mem[mem_addr];      // spike_soc port B: registered read

    integer seed = `SEED;
    // While the unit owns the port the CPU is stalled, so mem_idle only matters before it
    // takes the port; drop it at random anyway (the unit must ignore it once busy).
    always @(posedge clk) mem_idle <= ($urandom(seed) % 4) != 0;

    integer i, errors = 0, results = 0, claimed_ok = 0, cyc, wait_cyc;
    longint legacy_cycles = 0, kx_cycles = 0, total_cycles = 0;
    reg [31:0] w, r1, r2, flags, expv, got;
    reg        got_wr;
    initial begin
        $readmemh("mem.hex", mem);
        $readmemh("prog.hex", prog);
        repeat (3) @(posedge clk);
        resetn = 1;
        @(posedge clk);
        for (i = 0; i < `NPROG; i = i + 1) begin
            w = prog[5*i]; r1 = prog[5*i+1]; r2 = prog[5*i+2]; flags = prog[5*i+3]; expv = prog[5*i+4];
            repeat (flags[15:8]) @(posedge clk);
            pcpi_insn = w; pcpi_rs1 = r1; pcpi_rs2 = r2; pcpi_valid = 1;
            #1;
            if (pcpi_wait !== flags[1]) begin
                errors = errors + 1;
                if (errors < 10) $display("BAD pcpi_wait=%b for insn %08x (expected %b) [%0d]", pcpi_wait, w, flags[1], i);
            end
            if (!flags[1]) begin                                  // not ours: core would time out
                repeat (20) begin
                    @(posedge clk); #1;
                    if (pcpi_ready || pcpi_wait) begin
                        errors = errors + 1;
                        if (errors < 10) $display("BAD unclaimed insn %08x answered [%0d]", w, i);
                    end
                end
                pcpi_valid = 0;
                continue;
            end
            cyc = 0; got_wr = 0;
            while (1) begin
                @(posedge clk); #1; cyc = cyc + 1;
                if (pcpi_ready) begin got_wr = pcpi_wr; got = pcpi_rd; break; end
                if (cyc > 100000) begin $display("BAD timeout on insn %08x [%0d]", w, i); $finish; end
            end
            pcpi_valid = 0;                                        // core drops valid after ready
            total_cycles = total_cycles + cyc;
            if (flags[2]) legacy_cycles = legacy_cycles + cyc; else kx_cycles = kx_cycles + cyc;
            claimed_ok = claimed_ok + 1;
            if (got_wr !== flags[0]) begin
                errors = errors + 1;
                if (errors < 10) $display("BAD pcpi_wr=%b for insn %08x [%0d]", got_wr, w, i);
            end else if (flags[0]) begin
                results = results + 1;
                if (got !== expv) begin
                    errors = errors + 1;
                    if (errors < 10) $display("BAD insn %08x rs1=%08x rs2=%08x: rd=%08x expected %08x [%0d]", w, r1, r2, got, expv, i);
                end
            end
        end
        $display("instructions %0d, results checked %0d, legacy cycles %0d, KX cycles %0d", `NPROG, results, legacy_cycles, kx_cycles);
        if (errors == 0) $display("RESULT: PASS (%0d instructions, %0d results)", `NPROG, results);
        else             $display("RESULT: FAIL (%0d errors)", errors);
        $finish;
    end
endmodule
