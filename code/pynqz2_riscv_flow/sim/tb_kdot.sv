// tb_kdot.sv — unit test of rtl/kdot_pcpi.v (kdot, klen, kxload, ksdot) in Vivado xsim.
//
// The BRAM model has the SoC's latency: the unit's registered word address is sampled
// by a registered read, so data arrives two cycles after the address is issued.
// mem_idle stays low for a random 0-12 cycles after each instruction starts, like an
// instruction prefetch still in flight. A PicoRV32-style driver holds pcpi_valid
// until pcpi_ready and drops it the next cycle. Every result is compared with an
// int32 (wrap-around) reference computed here. Run: run_tb_kdot.sh

`timescale 1ns/1ps
module tb_kdot;
    reg clk = 0, resetn = 0;
    always #5 clk = ~clk;

    reg         pcpi_valid = 0;
    reg  [31:0] pcpi_insn = 0, pcpi_rs1 = 0, pcpi_rs2 = 0;
    wire        pcpi_wr, pcpi_wait, pcpi_ready;
    wire [31:0] pcpi_rd;
    wire        mem_busy;
    wire [15:0] mem_addr;
    reg  [31:0] mem_rdata;
    reg  [31:0] mem [0:65535];
    integer     idle_left = 0;
    wire        mem_idle = (idle_left == 0);

    kdot_pcpi dut (
        .clk(clk), .resetn(resetn),
        .pcpi_valid(pcpi_valid), .pcpi_insn(pcpi_insn), .pcpi_rs1(pcpi_rs1), .pcpi_rs2(pcpi_rs2),
        .pcpi_wr(pcpi_wr), .pcpi_rd(pcpi_rd), .pcpi_wait(pcpi_wait), .pcpi_ready(pcpi_ready),
        .mem_idle(mem_idle), .mem_busy(mem_busy), .mem_addr(mem_addr), .mem_rdata(mem_rdata));

    // Port B as in spike_soc.v: pb_rdata <= mem[kd_busy ? kd_addr : cpu address].
    always @(posedge clk) begin
        mem_rdata <= mem[mem_busy ? mem_addr : 16'd0];
        if (idle_left > 0) idle_left <= idle_left - 1;
    end

    function automatic [31:0] insn(input [2:0] funct3, input [4:0] rd);
        insn = {7'd0, 5'd2, 5'd1, funct3, rd, 7'b0001011};
    endfunction

    integer errors = 0, checks = 0;
    integer last_cycles;

    // One PCPI instruction; returns rd (or 0) and checks the wait/ready protocol.
    task automatic issue(input [2:0] funct3, input [31:0] rs1, input [31:0] rs2,
                         input bit expect_wr, output [31:0] rd);
        integer n;
        begin
            idle_left = $urandom_range(0, 12);
            @(negedge clk);
            pcpi_insn = insn(funct3, expect_wr ? 5'd10 : 5'd0); pcpi_rs1 = rs1; pcpi_rs2 = rs2;
            pcpi_valid = 1;
            #1 if (!pcpi_wait) begin $display("ERROR funct3=%0d: pcpi_wait not asserted", funct3); errors++; end
            n = 0;
            while (!pcpi_ready) begin @(posedge clk); #1 n++; if (n > 200000) begin $display("ERROR: no ready"); $finish; end end
            last_cycles = n;
            if (pcpi_wr !== expect_wr) begin $display("ERROR funct3=%0d: pcpi_wr=%b", funct3, pcpi_wr); errors++; end
            rd = pcpi_rd;
            @(negedge clk); pcpi_valid = 0;
        end
    endtask

    // Reference sums (int is 32-bit signed and wraps like the C accumulation).
    function automatic int ref_dense(input int w_word, input int x_word, input int len);
        int s = 0;
        for (int i = 0; i < len; i++) begin
            int w = $signed(mem[w_word + i / 2][16 * (i % 2) +: 16]);
            int x = mem[x_word + i / 4][8 * (i % 4) +: 8];
            s += w * x;
        end
        return s;
    endfunction
    function automatic int ref_sparse(input int e_word, input int count, input int x_word);
        int s = 0;
        for (int j = 0; j < count; j++) begin
            int w = $signed(mem[e_word + j][31:16]);
            int idx = mem[e_word + j][9:0];
            int x = mem[x_word + idx / 4][8 * (idx % 4) +: 8];
            s += w * x;
        end
        return s;
    endfunction

    task automatic check(input string what, input [31:0] got, input int expected);
        checks++;
        if ($signed(got) !== expected) begin
            errors++;
            $display("ERROR %s: got %0d, expected %0d", what, $signed(got), expected);
        end
    endtask

    localparam int X = 16'h0200;      // input region (word address), 768+ bytes
    localparam int W = 16'h1000;      // dense weights
    localparam int E = 16'h4000;      // sparse synapse words
    reg [31:0] rd;
    int len, count, ksdot_cycles_512;
    int len_list[] = '{4, 8, 768, 516, 12};

    initial begin
        for (int a = 0; a < 65536; a++) mem[a] = $urandom;
        // Extremes: weights -32768 / 32767, inputs 0 / 255.
        mem[W] = {16'h7fff, 16'h8000}; mem[W + 1] = {16'h8000, 16'h7fff};
        mem[X] = 32'hff00_ff00;
        repeat (5) @(posedge clk);
        resetn = 1;

        // 1. Dense kdot (regression of the release instruction).
        foreach (len_list[k]) begin
            len = len_list[k];
            issue(3'd1, len, 0, 0, rd);                                   // klen
            issue(3'd0, W << 2, X << 2, 1, rd);                           // kdot
            check($sformatf("kdot len=%0d", len), rd, ref_dense(W, X, len));
        end

        // 2. kxload the 768-byte input, then ksdot over random synapse lists.
        issue(3'd2, X << 2, 768, 0, rd);
        for (int t = 0; t < 40; t++) begin
            count = (t == 0) ? 0 : (t == 1) ? 1 : (t == 2) ? 512 : $urandom_range(1, 600);
            for (int j = 0; j < count; j++) begin
                automatic int idx = (j == 0) ? 0 : (j == 1) ? 767 : $urandom_range(0, 767);
                automatic bit [15:0] w = (j == 2) ? 16'h8000 : (j == 3) ? 16'h7fff : $urandom;
                mem[E + j] = {w, 6'd0, idx[9:0]};
            end
            issue(3'd3, E << 2, count, 1, rd);
            if (count == 512) ksdot_cycles_512 = last_cycles;
            check($sformatf("ksdot count=%0d", count), rd, ref_sparse(E, count, X));
        end

        // 3. Back-to-back: kdot after ksdot, ksdot again without reloading, full 1024-byte load.
        issue(3'd1, 768, 0, 0, rd);
        issue(3'd0, W << 2, X << 2, 1, rd);
        check("kdot after ksdot", rd, ref_dense(W, X, 768));
        issue(3'd3, E << 2, 100, 1, rd);
        check("ksdot without reload", rd, ref_sparse(E, 100, X));
        issue(3'd2, X << 2, 1024, 0, rd);
        for (int j = 0; j < 64; j++) mem[E + j] = {16'($urandom), 6'd0, 10'(1023 - j)};
        issue(3'd3, E << 2, 64, 1, rd);
        check("ksdot indices 960-1023 after a 1024-byte load", rd, ref_sparse(E, 64, X));

        // 4. Other funct3 values are not answered (the core would trap after 16 cycles).
        for (int f = 4; f < 8; f++) begin
            @(negedge clk);
            pcpi_insn = insn(f[2:0], 5'd10); pcpi_valid = 1;
            repeat (20) begin
                @(posedge clk); #1
                if (pcpi_wait || pcpi_ready) begin $display("ERROR: funct3=%0d answered", f); errors++; end
            end
            @(negedge clk); pcpi_valid = 0;
        end

        $display("ksdot with 512 synapses took %0d cycles from issue to ready", ksdot_cycles_512);
        $display("%s: %0d checks, %0d errors", errors == 0 ? "PASS" : "FAIL", checks, errors);
        $finish;
    end

endmodule
