// tb_spike_top_dual.v -- system simulation of rtl/spike_top.v with two cores,
// running the unmodified firmware/spike.bin on both.
//
// ps_bd_wrapper (the Vivado block design: PS7 + interconnect) is replaced by a
// stub that makes the 100 MHz clock and acts as the AXI4-Lite master, so the
// real spike_top -> ps_if -> 2 x spike_soc path is exercised. The test follows
// host/smoke_test_dual.sh:
//   A. load both cores, start only core 0: core 1 must stay in reset and its
//      console must stay empty while core 0 prints its banner and answers echo;
//   B. reload both, start core 0 then core 1: both TIMEs advance, both print
//      the banner, each answers echo with its own arguments, and (with the
//      Poisson rates raised through the mailbox so it happens within a
//      simulation) each core logs output spikes in its own spike log.
//
// Run: sim/run_spike_top_dual.sh   (needs firmware/spike.bin). Ends with RESULT: PASS / FAIL.
`timescale 1ns/1ps

// ---------------------------------------------------------------------------
// stand-in for the Vivado block design
// ---------------------------------------------------------------------------
module ps_bd_wrapper (
    inout wire [14:0] DDR_addr, inout wire [2:0] DDR_ba,
    inout wire DDR_cas_n, DDR_ck_n, DDR_ck_p, DDR_cke, DDR_cs_n,
    inout wire [3:0] DDR_dm, inout wire [31:0] DDR_dq,
    inout wire [3:0] DDR_dqs_n, DDR_dqs_p,
    inout wire DDR_odt, DDR_ras_n, DDR_reset_n, DDR_we_n,
    inout wire FIXED_IO_ddr_vrn, FIXED_IO_ddr_vrp,
    inout wire [53:0] FIXED_IO_mio,
    inout wire FIXED_IO_ps_clk, FIXED_IO_ps_porb, FIXED_IO_ps_srstb,
    output reg         FCLK_CLK0,
    input  wire        aresetn,
    output reg  [31:0] M00_AXI_awaddr, output wire [2:0] M00_AXI_awprot,
    output reg         M00_AXI_awvalid, input wire M00_AXI_awready,
    output reg  [31:0] M00_AXI_wdata, output reg [3:0] M00_AXI_wstrb,
    output reg         M00_AXI_wvalid, input wire M00_AXI_wready,
    input  wire [1:0]  M00_AXI_bresp, input wire M00_AXI_bvalid, output reg M00_AXI_bready,
    output reg  [31:0] M00_AXI_araddr, output wire [2:0] M00_AXI_arprot,
    output reg         M00_AXI_arvalid, input wire M00_AXI_arready,
    input  wire [31:0] M00_AXI_rdata, input wire [1:0] M00_AXI_rresp,
    input  wire        M00_AXI_rvalid, output reg M00_AXI_rready
);
    assign M00_AXI_awprot = 3'b000;
    assign M00_AXI_arprot = 3'b000;
    initial begin
        FCLK_CLK0 = 1'b0;
        M00_AXI_awvalid = 0; M00_AXI_wvalid = 0; M00_AXI_bready = 0;
        M00_AXI_arvalid = 0; M00_AXI_rready = 0;
        M00_AXI_awaddr = 0; M00_AXI_wdata = 0; M00_AXI_wstrb = 0; M00_AXI_araddr = 0;
    end
    always #5 FCLK_CLK0 = ~FCLK_CLK0;

    task wr(input [31:0] a, input [31:0] d);
        begin
            @(posedge FCLK_CLK0);
            M00_AXI_awaddr <= a; M00_AXI_awvalid <= 1'b1;
            M00_AXI_wdata <= d; M00_AXI_wstrb <= 4'hF; M00_AXI_wvalid <= 1'b1;
            M00_AXI_bready <= 1'b1;
            @(posedge FCLK_CLK0);
            while (M00_AXI_awvalid || M00_AXI_wvalid) begin
                if (M00_AXI_awready) M00_AXI_awvalid <= 1'b0;
                if (M00_AXI_wready)  M00_AXI_wvalid  <= 1'b0;
                @(posedge FCLK_CLK0);
            end
            while (!M00_AXI_bvalid) @(posedge FCLK_CLK0);
            if (M00_AXI_bresp != 2'b00) $display("AXI write %h: bresp %b", a, M00_AXI_bresp);
            M00_AXI_bready <= 1'b0;
        end
    endtask
    task rd(input [31:0] a, output [31:0] d);
        begin
            @(posedge FCLK_CLK0);
            M00_AXI_araddr <= a; M00_AXI_arvalid <= 1'b1; M00_AXI_rready <= 1'b1;
            @(posedge FCLK_CLK0);
            while (!M00_AXI_arready) @(posedge FCLK_CLK0);
            M00_AXI_arvalid <= 1'b0;
            while (!M00_AXI_rvalid) @(posedge FCLK_CLK0);
            d = M00_AXI_rdata;
            if (M00_AXI_rresp != 2'b00) $display("AXI read %h: rresp %b", a, M00_AXI_rresp);
            M00_AXI_rready <= 1'b0;
        end
    endtask
endmodule

// ---------------------------------------------------------------------------
module tb_spike_top_dual;
    localparam [31:0] BASE = 32'h4000_0000, STRIDE = 32'h0010_0000, SYS = 32'h0004_0000;
    localparam [31:0] CONSOLE = 32'h1_0000, MAILBOX = 32'h1_0400, SPLOG = 32'h1_1000;
    localparam [31:0] SPLOG_DATA = SPLOG + 16;

    wire [9:0] led;
    spike_top dut (.led(led));
    wire clk = dut.u_bd.FCLK_CLK0;

    // timer_lo has no reset (spike_soc.v: it must run while the CPU is in reset);
    // on the FPGA it powers up as 0 (register INIT), in simulation it would stay x.
    initial begin dut.u_soc0.timer_lo = 32'd0; dut.u_soc1.timer_lo = 32'd0; end

    integer errors = 0, checks = 0;
    task check(input cond, input [8*96-1:0] what);
        begin
            checks = checks + 1;
            if (cond !== 1'b1) begin errors = errors + 1; $display("BAD  t=%0t  %0s", $time, what); end
            else $display("ok   %0s", what);
        end
    endtask

    function [31:0] bram(input integer core, input [31:0] off); bram = BASE + core * STRIDE + off; endfunction
    function [31:0] sys(input integer core, input [31:0] off);  sys  = BASE + core * STRIDE + SYS + off; endfunction

    // firmware image (32-bit little-endian words, made by run_spike_top_dual.sh)
    reg [31:0] img [0:16383];
    integer img_words;

    task load(input integer core);
        integer w;
        begin
            dut.u_bd.wr(sys(core, 0), 32'h1);                        // hold in reset
            // as spike_pynq.py load-elf: image, zeros up to the console, then a clean
            // console + mailbox + spike-log header
            for (w = 0; w < CONSOLE / 4; w = w + 1) dut.u_bd.wr(bram(core, 4 * w), w < img_words ? img[w] : 32'h0);
            for (w = CONSOLE; w < SPLOG_DATA + 4 * 64; w = w + 4) dut.u_bd.wr(bram(core, w), 32'h0);
        end
    endtask

    // console ring -> string (as spike_pynq.py console)
    reg [8*600-1:0] con;
    integer con_len;
    task console(input integer core);
        reg [31:0] head, tail, word;
        integer k;
        begin
            dut.u_bd.rd(bram(core, CONSOLE), head);
            dut.u_bd.rd(bram(core, CONSOLE + 4), tail);
            con = 0; con_len = 0;
            for (k = 0; k < head - tail && k < 512; k = k + 1) begin
                dut.u_bd.rd(bram(core, CONSOLE + 8 + (((tail + k) & 32'h1FF) & ~32'h3)), word);
                con = {con[8*599-1:0], word[8*((tail + k) % 4) +: 8]};
                con_len = con_len + 1;
            end
            dut.u_bd.wr(bram(core, CONSOLE + 4), head);
        end
    endtask
    function has(input [8*600-1:0] s, input integer len, input [8*24-1:0] pat, input integer plen);
        integer p, q, ok;
        begin
            has = 0;
            for (p = 0; p + plen <= len; p = p + 1) begin
                ok = 1;
                for (q = 0; q < plen; q = q + 1)
                    if (s[8*(len - 1 - p - q) +: 8] != pat[8*(plen - 1 - q) +: 8]) ok = 0;
                if (ok) has = 1;
            end
        end
    endfunction

    // mailbox command (as spike_pynq.py cmd); r[0..3] = reply
    reg [31:0] r [0:3];
    reg        mb_ok;
    task mb(input integer core, input [31:0] op, input [31:0] a0, input [31:0] a1, input [31:0] a2);
        reg [31:0] seq, ack;
        integer k, tries;
        begin
            dut.u_bd.rd(bram(core, MAILBOX), seq);
            dut.u_bd.wr(bram(core, MAILBOX + 8), op);
            dut.u_bd.wr(bram(core, MAILBOX + 12), a0);
            dut.u_bd.wr(bram(core, MAILBOX + 16), a1);
            dut.u_bd.wr(bram(core, MAILBOX + 20), a2);
            dut.u_bd.wr(bram(core, MAILBOX), seq + 1);
            mb_ok = 0;
            for (tries = 0; tries < 2000 && !mb_ok; tries = tries + 1) begin
                repeat (50) @(posedge clk);
                dut.u_bd.rd(bram(core, MAILBOX + 4), ack);
                if (ack == seq + 1) mb_ok = 1;
            end
            for (k = 0; k < 4; k = k + 1) dut.u_bd.rd(bram(core, MAILBOX + 24 + 4 * k), r[k]);
        end
    endtask

    task wait_banner(input integer core, output found);
        reg [31:0] head;
        integer tries;
        begin
            found = 0;
            for (tries = 0; tries < 400 && !found; tries = tries + 1) begin
                repeat (500) @(posedge clk);
                dut.u_bd.rd(bram(core, CONSOLE), head);
                if (head >= 80) found = 1;       // both banner lines are > 80 characters
            end
        end
    endtask

    reg [31:0] d, ta0, ta1, tb0, tb1, h0, h1, s0, s1;
    reg found;
    integer k, fd, n;

    initial begin
        // spike.bin as hex words
        for (k = 0; k < 16384; k = k + 1) img[k] = 32'h0;
        $readmemh("spike_bin.hex", img, 0, 16383);
        img_words = 0;
        for (k = 0; k < 16384; k = k + 1) if (img[k] !== 32'h0) img_words = k + 1;
        $display("spike.bin: %0d words", img_words);

        wait (dut.aresetn === 1'b1);
        repeat (5) @(posedge clk);

        // ================= A. independence =================
        $display("== A. independence: only core 0 started");
        dut.u_bd.rd(sys(0, 32'h14), d); check(d == 32'h534B_454C, "core 0 MAGIC");
        dut.u_bd.rd(sys(1, 32'h14), d); check(d == 32'h534B_454C, "core 1 MAGIC");
        load(0);
        load(1);
        check(dut.u_soc0.core_rst_n === 1'b0 && dut.u_soc1.core_rst_n === 1'b0, "both loaded and held in reset");
        dut.u_bd.wr(sys(0, 0), 32'h0);                   // start core 0 only
        for (k = 0; k < 5; k = k + 1) begin
            repeat (3000) @(posedge clk);
            dut.u_bd.rd(sys(1, 32'h04), d);
            check(d[0] == 1'b0, "A.3 core 1 STATUS running bit stays 0 while only core 0 runs");
        end
        wait_banner(0, found);
        check(found, "A.4 core 0 printed its banner");
        console(0);
        check(has(con, con_len, "single LIF neuron", 17), "A.4 core 0 console contains 'single LIF neuron'");
        mb(0, 1, 32'hDEADBEEF, 7, 8);
        check(mb_ok && r[0] == 32'hDEADBEEF && r[1] == 7 && r[2] == 8, "A.4 core 0 mailbox echo deadbeef 7 8");
        dut.u_bd.rd(bram(1, CONSOLE), h1);
        check(h1 == 0, "A.4 core 1 console still empty (core 1 never ran)");
        check(dut.u_soc1.core_rst_n === 1'b0, "A.4 core 1 still held in reset");
        dut.u_bd.rd(sys(1, 32'h04), d);
        check(d[0] == 1'b0, "A.4 core 1 STATUS still not running");
        dut.u_bd.wr(sys(0, 0), 32'h1);                   // A.5 stop core 0, both in reset
        dut.u_bd.wr(sys(1, 0), 32'h1);

        // ================= B. concurrency =================
        $display("== B. concurrency: core 0 and core 1 started back to back");
        load(0);
        load(1);
        dut.u_bd.wr(sys(0, 0), 32'h0);
        dut.u_bd.wr(sys(1, 0), 32'h0);
        dut.u_bd.rd(sys(0, 32'h08), ta0); dut.u_bd.rd(sys(1, 32'h08), ta1);
        repeat (1000) @(posedge clk);
        dut.u_bd.rd(sys(0, 32'h08), tb0); dut.u_bd.rd(sys(1, 32'h08), tb1);
        check(ta0 != 0 && ta1 != 0 && tb0 > ta0 && tb1 > ta1, "B.2 both TIME registers non-zero and advancing");
        dut.u_bd.rd(sys(0, 32'h04), s0); dut.u_bd.rd(sys(1, 32'h04), s1);
        check(s0 == 32'h1 && s1 == 32'h1, "B.3 both cores report running");
        wait_banner(0, found); check(found, "B.3 core 0 printed its banner");
        wait_banner(1, found); check(found, "B.3 core 1 printed its banner");
        console(0); check(has(con, con_len, "single LIF neuron", 17), "B.3 core 0 console contains 'single LIF neuron'");
        console(1); check(has(con, con_len, "single LIF neuron", 17), "B.3 core 1 console contains 'single LIF neuron'");
        mb(0, 1, 32'hDEADBEEF, 32'h10, 32'h11);
        check(mb_ok && r[0] == 32'hDEADBEEF && r[1] == 32'h10 && r[2] == 32'h11, "B.3 core 0 echo returns its own arguments (deadbeef 10 11)");
        mb(1, 1, 32'hDEADBEEF, 32'h20, 32'h21);
        check(mb_ok && r[0] == 32'hDEADBEEF && r[1] == 32'h20 && r[2] == 32'h21, "B.3 core 1 echo returns its own arguments (deadbeef 20 21)");
        // spikes: raise channel 0's rate so an output spike happens within the simulation
        mb(0, 3, 0, 32'd10_000_000, 0); check(mb_ok, "B.3 core 0 set_rate ch0 10 MHz (simulation only)");
        mb(1, 3, 0, 32'd10_000_000, 0); check(mb_ok, "B.3 core 1 set_rate ch0 10 MHz (simulation only)");
        h0 = 0; h1 = 0;
        for (k = 0; k < 400 && (h0 == 0 || h1 == 0); k = k + 1) begin
            repeat (1000) @(posedge clk);
            dut.u_bd.rd(bram(0, SPLOG), h0);
            dut.u_bd.rd(bram(1, SPLOG), h1);
        end
        check(h0 > 0, "B.3 core 0 logged output spikes");
        check(h1 > 0, "B.3 core 1 logged output spikes");
        dut.u_bd.rd(bram(0, SPLOG + 12), s0); dut.u_bd.rd(bram(1, SPLOG + 12), s1);
        check(s0 == 32'h5A11_0001 && s1 == 32'h5A11_0001, "B.3 both spike logs initialised (version word)");
        dut.u_bd.rd(bram(0, SPLOG_DATA), s0); dut.u_bd.rd(bram(1, SPLOG_DATA), s1);
        $display("     first spike word: core 0 %h, core 1 %h", s0, s1);
        check(s0 != s1, "B.4 first spike words differ between the cores (two separate BRAMs)");
        check(dut.u_soc0.mem[SPLOG_DATA >> 2] === s0 && dut.u_soc1.mem[SPLOG_DATA >> 2] === s1,
              "B.4 each AXI read came from its own core's BRAM (checked against the RTL arrays)");

        $display("");
        if (errors == 0) $display("RESULT: PASS (%0d checks, 0 bad)", checks);
        else             $display("RESULT: FAIL (%0d checks, %0d bad)", checks, errors);
        $finish;
    end

    initial begin
        #200_000_000;
        $display("RESULT: FAIL (timeout)");
        $finish;
    end
endmodule
