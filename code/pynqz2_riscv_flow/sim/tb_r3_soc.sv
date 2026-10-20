// tb_r3_soc.sv — full-SoC xsim testbench for R3 (code/snn_keyword/research/PLAN_R3_GATING.md).
//
// ps_if + spike_soc wired as in spike_top.v, without the PS block design: the testbench is
// the AXI4-Lite master (the ARM). The firmware is loaded by backdoor ($readmemh of a full
// 65,536-word hex file into u_soc.mem), then CTRL bit0 is released over AXI and requests
// are replayed from a text file like board_server.py does: payload words to INPUT (0x10800),
// MB[2] opcode, MB[3] length, MB[0] seq last; MB[1] is polled every POLL cycles.
//
// Plusargs: +fw=<hex> (required), +req=<file> (required), +poll=<cycles> (default 1000),
//           +watchdog=<cycles per request> (default 40,000,000).
// Request file, repeated: "<opcode> <length> <gap> <flags> <nwords>" (decimal), then nwords
// payload words in hex (little-endian bytes of the payload). gap = idle cycles after the
// request completes; flags are printed only (reserved).
//
// Output, one line per request:
//   REQ n= op= len= flags= seq= svc= pb_re= pb_we= pa_re= pa_we= clk_on= sleep= mb=<MB[6..35] hex>
// svc counts the cycles from the PS write of MB[0] to the CPU write of MB[1]; the other
// counters count cycles inside that window. pb_re / pa_re are the cycles a BRAM port must
// read: kd_busy || (acc_rd && sel_bram) for port B and ps_if rstate == 1 for port A. On the
// unchanged RTL (both ports read every cycle) these are the "would-be" read enables.
// At the end: SUM (the same counters from the CPU release to the end) and DONE.
`timescale 1ns/1ps
module tb_r3_soc;
    parameter integer POWER_OPT = 0;          // passed to spike_soc once the RTL has it (Phase 2B)

    localparam [31:0] AXI_BRAM = 32'h4000_0000, CTRL = 32'h4004_0000;
    localparam [31:0] MB = AXI_BRAM + 32'h0001_0400, INPUT = AXI_BRAM + 32'h0001_0800;
    localparam [15:0] MB0_WORD = 16'h4100, MB1_WORD = 16'h4101;

    reg clk = 0;
    always #5 clk = ~clk;                     // 100 MHz
    reg aresetn = 0;

    // AXI4-Lite master
    reg  [31:0] awaddr = 0, wdata = 0, araddr = 0;
    reg  [3:0]  wstrb = 0;
    reg         awvalid = 0, wvalid = 0, bready = 0, arvalid = 0, rready = 0;
    wire        awready, wready, bvalid, arready, rvalid;
    wire [1:0]  bresp, rresp;
    wire [31:0] rdata;

    // ps_if <-> spike_soc, as in spike_top.v
    wire        core_rst, bram_we;
    wire [15:0] bram_addr;
    wire [31:0] bram_wdata, bram_rdata, scratch, timer_lo;
    wire [3:0]  bram_be;
    wire        core_running, core_trap;
    wire [9:0]  led_o;

    ps_if #(.CLK_HZ(32'd100_000_000)) u_ps_if (
        .aclk(clk), .aresetn(aresetn),
        .s_axil_awaddr(awaddr), .s_axil_awvalid(awvalid), .s_axil_awready(awready),
        .s_axil_wdata(wdata), .s_axil_wstrb(wstrb), .s_axil_wvalid(wvalid), .s_axil_wready(wready),
        .s_axil_bresp(bresp), .s_axil_bvalid(bvalid), .s_axil_bready(bready),
        .s_axil_araddr(araddr), .s_axil_arvalid(arvalid), .s_axil_arready(arready),
        .s_axil_rdata(rdata), .s_axil_rresp(rresp), .s_axil_rvalid(rvalid), .s_axil_rready(rready),
        .bram_we(bram_we), .bram_addr(bram_addr), .bram_wdata(bram_wdata), .bram_be(bram_be),
        .bram_rdata(bram_rdata), .core_rst(core_rst), .core_running(core_running),
        .core_trap(core_trap), .timer_now(timer_lo), .scratch(scratch), .led(led_o));

    spike_soc #(.MEM_WORDS(65536)) u_soc (
        .clk(clk), .core_rst_n(~core_rst),
        .pa_we(bram_we), .pa_addr(bram_addr), .pa_wdata(bram_wdata), .pa_be(bram_be), .pa_rdata(bram_rdata),
        .core_running(core_running), .core_trap(core_trap), .timer_lo(timer_lo), .led_o(led_o));

    // ------------------------------------------------------------------
    // Activity counters (hierarchical references; no RTL change)
    // ------------------------------------------------------------------
    wire pb_re = u_soc.kd_busy || (u_soc.acc_rd && u_soc.sel_bram);
    wire pb_we = u_soc.bram_wr_now;
    wire pa_re = (u_ps_if.rstate == 2'd1);
    wire pa_we = bram_we;
`ifdef R3_POWER_RTL
    wire clk_on = u_soc.cpu_clk_en;           // gated CPU clock enabled (Phase 2B names)
    wire asleep = u_soc.sleeping;
`else
    wire clk_on = 1'b1;
    wire asleep = 1'b0;
`endif
    wire mb0_wr = bram_we && bram_addr == MB0_WORD;
    wire mb1_wr = u_soc.bram_wr_now && u_soc.pb_word_addr == MB1_WORD;

    reg        counting = 0, active = 0;
    reg [63:0] a_cyc = 0, a_pb_re = 0, a_pb_we = 0, a_pa_re = 0, a_pa_we = 0, a_clk = 0, a_sleep = 0;
    reg [63:0] r_cyc = 0, r_pb_re = 0, r_pb_we = 0, r_pa_re = 0, r_pa_we = 0, r_clk = 0, r_sleep = 0;
    always @(posedge clk) begin
        if (counting) begin
            a_cyc <= a_cyc + 1; a_pb_re <= a_pb_re + pb_re; a_pb_we <= a_pb_we + pb_we;
            a_pa_re <= a_pa_re + pa_re; a_pa_we <= a_pa_we + pa_we; a_clk <= a_clk + clk_on;
            a_sleep <= a_sleep + asleep;
        end
        if (mb0_wr) begin
            active <= 1;
            r_cyc <= 0; r_pb_re <= 0; r_pb_we <= 0; r_pa_re <= 0; r_pa_we <= 0; r_clk <= 0; r_sleep <= 0;
        end else if (active) begin
            r_cyc <= r_cyc + 1; r_pb_re <= r_pb_re + pb_re; r_pb_we <= r_pb_we + pb_we;
            r_pa_re <= r_pa_re + pa_re; r_pa_we <= r_pa_we + pa_we; r_clk <= r_clk + clk_on;
            r_sleep <= r_sleep + asleep;
            if (mb1_wr) active <= 0;
        end
        if (core_trap === 1'b1 && !core_rst) begin
            $display("TRAP at %0t", $time);
            $finish;
        end
    end

    // ------------------------------------------------------------------
    // AXI4-Lite master tasks (signals change on the falling edge, sampled on the rising edge)
    // ------------------------------------------------------------------
    task automatic axi_write(input [31:0] a, input [31:0] d);
        bit aw_done, w_done;
        aw_done = 0; w_done = 0;
        @(negedge clk);
        awaddr = a; awvalid = 1; wdata = d; wstrb = 4'hf; wvalid = 1;
        do begin
            @(posedge clk);
            if (awready) aw_done = 1;
            if (wready) w_done = 1;
            @(negedge clk);
            if (aw_done) awvalid = 0;
            if (w_done) wvalid = 0;
        end while (!(aw_done && w_done));
        bready = 1;
        do @(posedge clk); while (!bvalid);
        if (bresp !== 2'b00) begin $display("ERROR write %h: bresp %b", a, bresp); $finish; end
        @(negedge clk); bready = 0;
    endtask

    task automatic axi_read(input [31:0] a, output [31:0] d);
        @(negedge clk); araddr = a; arvalid = 1;
        do @(posedge clk); while (!arready);
        @(negedge clk); arvalid = 0; rready = 1;
        do @(posedge clk); while (!rvalid);
        d = rdata;
        if (rresp !== 2'b00) begin $display("ERROR read %h: rresp %b", a, rresp); $finish; end
        @(negedge clk); rready = 0;
    endtask

    // ------------------------------------------------------------------
    // Request replay
    // ------------------------------------------------------------------
    string fw, req, mbs;
    integer fd, opcode, length, gap, flags, nwords, n, i, poll, waited;
    longint watchdog;
    reg [31:0] word, seq, v;

    initial begin
        if (!$value$plusargs("fw=%s", fw)) begin $display("ERROR: +fw=<hex> missing"); $finish; end
        if (!$value$plusargs("req=%s", req)) begin $display("ERROR: +req=<file> missing"); $finish; end
        if (!$value$plusargs("poll=%d", poll)) poll = 1000;
        if (!$value$plusargs("watchdog=%d", watchdog)) watchdog = 40_000_000;
        $readmemh(fw, u_soc.mem);
        for (i = 0; i < 32; i = i + 1) u_soc.u_cpu.cpuregs[i] = 32'h0;   // no X in the register file
        repeat (20) @(negedge clk);
        aresetn = 1;
        repeat (20) @(negedge clk);
        counting = 1;
        axi_write(CTRL, 32'h0);                       // release the CPU
        waited = 0;
        do begin                                      // firmware ready: magic in MB[12]
            repeat (poll) @(posedge clk);
            waited = waited + poll;
            axi_read(MB + 48, v);
            if (waited > 1_000_000) begin $display("ERROR: no magic in MB[12]"); $finish; end
        end while (v == 0);
        $display("READY magic=%h after %0d cycles", v, waited);
        fd = $fopen(req, "r");
        if (fd == 0) begin $display("ERROR: cannot open %s", req); $finish; end
        seq = 0; n = 0;
        while ($fscanf(fd, "%d %d %d %d %d", opcode, length, gap, flags, nwords) == 5) begin
            for (i = 0; i < nwords; i = i + 1) begin
                if ($fscanf(fd, "%h", word) != 1) begin $display("ERROR: short payload"); $finish; end
                axi_write(INPUT + 4 * i, word);
            end
            axi_write(MB + 8, opcode);
            axi_write(MB + 12, length);
            seq = seq + 1;
            axi_write(MB + 0, seq);                   // last, as board_server.py
            waited = 0;
            do begin
                repeat (poll) @(posedge clk);
                waited = waited + poll;
                if (waited > watchdog) begin $display("ERROR: request %0d watchdog", n); $finish; end
                axi_read(MB + 4, v);
            end while (v != seq);
            mbs = "";
            for (i = 6; i <= 35; i = i + 1) begin
                axi_read(MB + 4 * i, v);
                if ($isunknown(v)) $display("WARNING: MB[%0d] has X", i);
                mbs = {mbs, (i == 6) ? "" : ",", $sformatf("%h", v)};
            end
            $display("REQ n=%0d op=%0d len=%0d flags=%0d seq=%0d svc=%0d pb_re=%0d pb_we=%0d pa_re=%0d pa_we=%0d clk_on=%0d sleep=%0d mb=%s",
                     n, opcode, length, flags, seq, r_cyc, r_pb_re, r_pb_we, r_pa_re, r_pa_we, r_clk, r_sleep, mbs);
            n = n + 1;
            repeat (gap) @(posedge clk);
        end
        $fclose(fd);
        $display("SUM requests=%0d cyc=%0d pb_re=%0d pb_we=%0d pa_re=%0d pa_we=%0d clk_on=%0d sleep=%0d",
                 n, a_cyc, a_pb_re, a_pb_we, a_pa_re, a_pa_we, a_clk, a_sleep);
        $display("DONE");
        $finish;
    end
endmodule
