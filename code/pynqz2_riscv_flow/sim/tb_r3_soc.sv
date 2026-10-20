// tb_r3_soc.sv — full-SoC xsim testbench for R3 (code/snn_keyword/research/PLAN_R3_GATING.md).
//
// ps_if + spike_soc wired as in spike_top.v, without the PS block design: the testbench is
// the AXI4-Lite master (the ARM). The firmware is loaded by backdoor ($readmemh of a full
// 65,536-word hex file into u_soc.mem), then CTRL bit0 is released over AXI and requests
// are replayed from a text file like board_server.py does: payload words to INPUT (0x10800),
// MB[2] opcode, MB[3] length, MB[0] seq last; MB[1] is polled every POLL cycles.
//
// Parameters (xelab --generic_top): POWER_OPT (0/1), READ_WAIT (1..15), passed to spike_soc and ps_if.
// Plusargs: +fw=<hex> (required), +req=<file> (required), +poll=<cycles> (default 1000),
//           +watchdog=<cycles per request> (default 40,000,000), +dump=<first>:<last> (print MB words
//           after the firmware is ready, e.g. the sleep self-test's MB[40..42]).
// Request file, repeated: "<opcode> <length> <gap> <flags> <nwords>" (decimal), then nwords
// payload words in hex (little-endian bytes of the payload). gap = idle cycles after the
// request completes. flags:
//   bit0  wait for the CPU's MB[1] write itself (exact cycle) instead of polling MB[1];
//   bit1  race request: write only MB[0], `gap` cycles after the previous request's MB[1] write
//         (the request words are unchanged from the previous request); no gap afterwards;
//   bit2  before this request, while the CPU sleeps: CTRL reset, check that sleeping and the
//         doorbell clear, then release and wait for the firmware's magic again.
//
// Output, one line per request:
//   REQ n= op= len= flags= seq= svc= pb_re= pb_we= pa_re= pa_we= clk_on= sleep= tb_asleep= tb_sleeps= awake= race= mb=<MB[6..35]>
// svc counts the cycles from the PS write of MB[0] to the CPU write of MB[1]; the other counters
// count inside that window: pb_re / pa_re = cycles a BRAM port must read (kd_busy || (acc_rd && sel_bram)
// for port B, ps_if rstate == 1 for port A; with POWER_OPT = 0 the ports read every cycle and these are
// the "would-be" enables), clk_on = edges of the gated CPU clock, sleep = cycles with the CPU asleep.
// tb_asleep / tb_sleeps = the testbench's own count of asleep cycles / sleeps since start, taken at the
// MB[1] write (compare with the POWER counters in MB[34..35]). awake = gated clock edges from the MB[0]
// write to the sleep that follows the reply (0 with POWER_OPT = 0, which never sleeps). race = cycle of the MB[0] write minus
// the cycle of the CPU's SLEEP store accept after the previous request (none if it did not sleep).
// Checks on every cycle (ERROR lines): asleep => no gated clock edge and no port-B read; every wake
// follows a PS write of MB[0] in the sleep's first cycle or during the sleep.
// At the end: SUM (the same counters from the CPU release to the end), ERRORS=<n> and DONE.
//
// `define R3_EQUIV (r3_sim.py run --equiv): a reference SoC (spike_soc_ref / ps_if_ref, the r3-base
// RTL renamed) runs from the same AXI stimulus and firmware, and every clock its CPU bus, PCPI,
// port-A data, LEDs, trap, timer and AXI responses are compared with the instance under test
// (POWER_OPT = 0): EQUIV mismatches=<n>.
`timescale 1ns/1ps
module tb_r3_soc;
    parameter integer POWER_OPT = 0;
    parameter integer READ_WAIT = 8;

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
    wire        core_rst, bram_we, bram_re;
    wire [15:0] bram_addr;
    wire [31:0] bram_wdata, bram_rdata, scratch, timer_lo;
    wire [3:0]  bram_be;
    wire        core_running, core_trap;
    wire [9:0]  led_o;

    ps_if #(.CLK_HZ(32'd100_000_000), .ABI(POWER_OPT != 0 ? 32'h0002_0007 : 32'h0002_0003)) u_ps_if (
        .aclk(clk), .aresetn(aresetn),
        .s_axil_awaddr(awaddr), .s_axil_awvalid(awvalid), .s_axil_awready(awready),
        .s_axil_wdata(wdata), .s_axil_wstrb(wstrb), .s_axil_wvalid(wvalid), .s_axil_wready(wready),
        .s_axil_bresp(bresp), .s_axil_bvalid(bvalid), .s_axil_bready(bready),
        .s_axil_araddr(araddr), .s_axil_arvalid(arvalid), .s_axil_arready(arready),
        .s_axil_rdata(rdata), .s_axil_rresp(rresp), .s_axil_rvalid(rvalid), .s_axil_rready(rready),
        .bram_we(bram_we), .bram_re(bram_re), .bram_addr(bram_addr), .bram_wdata(bram_wdata), .bram_be(bram_be),
        .bram_rdata(bram_rdata), .core_rst(core_rst), .core_running(core_running),
        .core_trap(core_trap), .timer_now(timer_lo), .scratch(scratch), .led(led_o));

    spike_soc #(.MEM_WORDS(65536), .POWER_OPT(POWER_OPT), .READ_WAIT(READ_WAIT)) u_soc (
        .clk(clk), .core_rst_n(~core_rst),
        .pa_we(bram_we), .pa_re(bram_re), .pa_addr(bram_addr), .pa_wdata(bram_wdata), .pa_be(bram_be),
        .pa_rdata(bram_rdata), .core_running(core_running), .core_trap(core_trap), .timer_lo(timer_lo),
        .led_o(led_o));

    integer errors = 0;

`ifdef R3_EQUIV
    // ------------------------------------------------------------------
    // Reference SoC (r3-base RTL), same stimulus; compared every clock
    // ------------------------------------------------------------------
    wire        r_awready, r_wready, r_bvalid, r_arready, r_rvalid, r_core_rst, r_bram_we;
    wire [1:0]  r_bresp, r_rresp;
    wire [31:0] r_rdata, r_bram_wdata, r_bram_rdata, r_scratch, r_timer_lo;
    wire [15:0] r_bram_addr;
    wire [3:0]  r_bram_be;
    wire        r_core_running, r_core_trap;
    wire [9:0]  r_led_o;
    ps_if_ref #(.CLK_HZ(32'd100_000_000)) u_ps_if_ref (
        .aclk(clk), .aresetn(aresetn),
        .s_axil_awaddr(awaddr), .s_axil_awvalid(awvalid), .s_axil_awready(r_awready),
        .s_axil_wdata(wdata), .s_axil_wstrb(wstrb), .s_axil_wvalid(wvalid), .s_axil_wready(r_wready),
        .s_axil_bresp(r_bresp), .s_axil_bvalid(r_bvalid), .s_axil_bready(bready),
        .s_axil_araddr(araddr), .s_axil_arvalid(arvalid), .s_axil_arready(r_arready),
        .s_axil_rdata(r_rdata), .s_axil_rresp(r_rresp), .s_axil_rvalid(r_rvalid), .s_axil_rready(rready),
        .bram_we(r_bram_we), .bram_addr(r_bram_addr), .bram_wdata(r_bram_wdata), .bram_be(r_bram_be),
        .bram_rdata(r_bram_rdata), .core_rst(r_core_rst), .core_running(r_core_running),
        .core_trap(r_core_trap), .timer_now(r_timer_lo), .scratch(r_scratch), .led(r_led_o));
    spike_soc_ref #(.MEM_WORDS(65536)) u_ref (
        .clk(clk), .core_rst_n(~r_core_rst),
        .pa_we(r_bram_we), .pa_addr(r_bram_addr), .pa_wdata(r_bram_wdata), .pa_be(r_bram_be),
        .pa_rdata(r_bram_rdata), .core_running(r_core_running), .core_trap(r_core_trap),
        .timer_lo(r_timer_lo), .led_o(r_led_o));

    wire [511:0] dut_sig = {awready, wready, bvalid, bresp, arready, rvalid, rresp, rvalid ? rdata : 32'h0,
                            bram_rdata, led_o, core_trap, timer_lo, core_rst,
                            u_soc.cpu_mem_valid, u_soc.cpu_mem_instr, u_soc.cpu_mem_ready,
                            u_soc.cpu_mem_valid ? {u_soc.cpu_mem_addr, u_soc.cpu_mem_wdata, u_soc.cpu_mem_wstrb} : 68'h0,
                            u_soc.cpu_mem_ready ? u_soc.cpu_mem_rdata : 32'h0,
                            u_soc.pcpi_valid, u_soc.pcpi_valid ? {u_soc.pcpi_insn, u_soc.pcpi_rs1, u_soc.pcpi_rs2} : 96'h0,
                            u_soc.pcpi_wr, u_soc.pcpi_wait, u_soc.pcpi_ready, u_soc.pcpi_ready ? u_soc.pcpi_rd : 32'h0};
    wire [511:0] ref_sig = {r_awready, r_wready, r_bvalid, r_bresp, r_arready, r_rvalid, r_rresp, r_rvalid ? r_rdata : 32'h0,
                            r_bram_rdata, r_led_o, r_core_trap, r_timer_lo, r_core_rst,
                            u_ref.cpu_mem_valid, u_ref.cpu_mem_instr, u_ref.cpu_mem_ready,
                            u_ref.cpu_mem_valid ? {u_ref.cpu_mem_addr, u_ref.cpu_mem_wdata, u_ref.cpu_mem_wstrb} : 68'h0,
                            u_ref.cpu_mem_ready ? u_ref.cpu_mem_rdata : 32'h0,
                            u_ref.pcpi_valid, u_ref.pcpi_valid ? {u_ref.pcpi_insn, u_ref.pcpi_rs1, u_ref.pcpi_rs2} : 96'h0,
                            u_ref.pcpi_wr, u_ref.pcpi_wait, u_ref.pcpi_ready, u_ref.pcpi_ready ? u_ref.pcpi_rd : 32'h0};
    longint eq_cycles = 0, eq_bad = 0;
    always @(posedge clk) if (aresetn) begin
        eq_cycles <= eq_cycles + 1;
        if (dut_sig !== ref_sig) begin
            if (eq_bad < 10) $display("EQUIV DIFF at %0t: dut %h ref %h", $time, dut_sig, ref_sig);
            eq_bad <= eq_bad + 1;
        end
    end
`endif

    // ------------------------------------------------------------------
    // Activity counters and checks (hierarchical references)
    // ------------------------------------------------------------------
    wire pb_re = u_soc.kd_busy || (u_soc.acc_rd && u_soc.sel_bram);
    wire pb_we = u_soc.bram_wr_now;
    wire pa_re = (u_ps_if.rstate == 2'd1);
    wire pa_we = bram_we;
    wire asleep = u_soc.sleep_q;
    wire mb0_wr = bram_we && bram_addr == MB0_WORD;
    wire mb1_wr = u_soc.bram_wr_now && u_soc.pb_word_addr == MB1_WORD;
    wire sleep_start = u_soc.pwr_sleep && !u_soc.wake_now;   // a SLEEP store that stalls (from the next clock)

    longint cyc = 0, gclk_edges = 0;
    reg     counting = 0;
    always @(posedge u_soc.cpu_clk) begin
        if (counting) gclk_edges <= gclk_edges + 1;
        if (asleep && !core_rst) begin              // in reset the clock runs on purpose (sleeping clears)
            if (errors < 20) $display("ERROR at %0t: gated CPU clock edge while asleep", $time);
            errors = errors + 1;
        end
    end

    reg        active = 0, armed = 0;
    reg [63:0] a_cyc = 0, a_pb_re = 0, a_pb_we = 0, a_pa_re = 0, a_pa_we = 0, a_sleep = 0, a_sleeps = 0;
    reg [63:0] r_cyc = 0, r_pb_re = 0, r_pb_we = 0, r_pa_re = 0, r_pa_we = 0, r_sleep = 0;
    longint    r_g0 = 0, r_clk = 0, snap_asleep = 0, snap_sleeps = 0, last_mb1_cyc = 0, mb0_cyc = 0;
    longint    sleep_acc_cyc = -1;                    // first SLEEP store accept after the last MB[1] write
    longint    race_acc = -1;                         // the same, latched at this request's MB[1] write
    longint    awake = 0;                             // gated clock edges from MB[0] write to the next sleep start
    always @(posedge clk) begin
        cyc <= cyc + 1;
        if (counting) begin
            a_cyc <= a_cyc + 1; a_pb_re <= a_pb_re + pb_re; a_pb_we <= a_pb_we + pb_we;
            a_pa_re <= a_pa_re + pa_re; a_pa_we <= a_pa_we + pa_we; a_sleep <= a_sleep + asleep;
            a_sleeps <= a_sleeps + sleep_start;
        end
        if (asleep && pb_re) begin
            if (errors < 20) $display("ERROR at %0t: port-B read while asleep", $time);
            errors = errors + 1;
        end
        // every wake follows a PS write of MB[0] (in the clock the sleep began, or during it)
        if (sleep_start) armed <= mb0_wr;
        else if (asleep && mb0_wr) armed <= 1;
        if (u_soc.wake_slp && !(armed || mb0_wr)) begin
            if (errors < 20) $display("ERROR at %0t: woke without a PS write of MB[0]", $time);
            errors = errors + 1;
        end
        if (u_soc.pwr_sleep && sleep_acc_cyc < 0) sleep_acc_cyc <= cyc;
        if (sleep_start && !active) awake <= gclk_edges - r_g0;
        if (mb0_wr) begin
            active <= 1; mb0_cyc <= cyc; r_g0 <= gclk_edges;
            r_cyc <= 0; r_pb_re <= 0; r_pb_we <= 0; r_pa_re <= 0; r_pa_we <= 0; r_sleep <= 0;
        end else if (active) begin
            r_cyc <= r_cyc + 1; r_pb_re <= r_pb_re + pb_re; r_pb_we <= r_pb_we + pb_we;
            r_pa_re <= r_pa_re + pa_re; r_pa_we <= r_pa_we + pa_we; r_sleep <= r_sleep + asleep;
            if (mb1_wr) begin
                active <= 0; r_clk <= gclk_edges - r_g0; last_mb1_cyc <= cyc; sleep_acc_cyc <= -1;
                race_acc <= sleep_acc_cyc;
                snap_asleep <= a_sleep; snap_sleeps <= a_sleeps;
            end
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

    task automatic wait_magic(input integer poll);
        reg [31:0] v;
        integer waited;
        waited = 0;
        do begin                                      // firmware ready: magic in MB[12]
            repeat (poll) @(posedge clk);
            waited = waited + poll;
            axi_read(MB + 48, v);
            if (waited > 2_000_000) begin $display("ERROR: no magic in MB[12]"); $finish; end
        end while (v == 0);
        $display("READY magic=%h after %0d cycles", v, waited);
    endtask

    // ------------------------------------------------------------------
    // Request replay
    // ------------------------------------------------------------------
    string fw, req, mbs, dump, race;
    integer fd, opcode, length, gap, flags, nwords, n, i, poll, waited, d0, d1;
    longint watchdog;
    reg [31:0] word, seq, v;

    initial begin
        if (!$value$plusargs("fw=%s", fw)) begin $display("ERROR: +fw=<hex> missing"); $finish; end
        if (!$value$plusargs("req=%s", req)) begin $display("ERROR: +req=<file> missing"); $finish; end
        if (!$value$plusargs("poll=%d", poll)) poll = 1000;
        if (!$value$plusargs("watchdog=%d", watchdog)) watchdog = 40_000_000;
        $readmemh(fw, u_soc.mem);
        for (i = 0; i < 32; i = i + 1) u_soc.u_cpu.cpuregs[i] = 32'h0;   // no X in the register file
`ifdef R3_EQUIV
        $readmemh(fw, u_ref.mem);
        for (i = 0; i < 32; i = i + 1) u_ref.u_cpu.cpuregs[i] = 32'h0;
`endif
        $display("CONFIG POWER_OPT=%0d READ_WAIT=%0d", POWER_OPT, READ_WAIT);
        repeat (20) @(negedge clk);
        aresetn = 1;
        repeat (20) @(negedge clk);
        counting = 1;
        axi_write(CTRL, 32'h0);                       // release the CPU
        wait_magic(poll);
        if ($value$plusargs("dump=%d:%d", d0, d1))
            for (i = d0; i <= d1; i = i + 1) begin axi_read(MB + 4 * i, v); $display("DUMP MB[%0d]=%h", i, v); end
        fd = $fopen(req, "r");
        if (fd == 0) begin $display("ERROR: cannot open %s", req); $finish; end
        seq = 0; n = 0;
        while ($fscanf(fd, "%d %d %d %d %d", opcode, length, gap, flags, nwords) == 5) begin
            if (flags & 4) begin                      // CTRL reset while the CPU sleeps
                wait (asleep === 1'b1);
                axi_write(CTRL, 32'h1);
                repeat (5) @(posedge clk);
                $display("RESET_ASLEEP sleeping=%0d doorbell=%0d", u_soc.sleeping, u_soc.doorbell);
                if (u_soc.sleeping !== 1'b0 || u_soc.doorbell !== 1'b0) errors = errors + 1;
                axi_write(MB + 48, 32'h0);
                axi_write(CTRL, 32'h0);
                wait_magic(poll);
            end
            if (flags & 2) begin                      // race: only MB[0], gap cycles after the last MB[1] write
                while (cyc < last_mb1_cyc + gap) @(posedge clk);
                for (i = 0; i < nwords; i = i + 1) void'($fscanf(fd, "%h", word));
            end else begin
                for (i = 0; i < nwords; i = i + 1) begin
                    if ($fscanf(fd, "%h", word) != 1) begin $display("ERROR: short payload"); $finish; end
                    axi_write(INPUT + 4 * i, word);
                end
                axi_write(MB + 8, opcode);
                axi_write(MB + 12, length);
            end
            seq = seq + 1;
            axi_write(MB + 0, seq);                   // last, as board_server.py
            waited = 0;
            if (flags & 1) begin
                while (!(active === 1'b1 && mb1_wr === 1'b1)) begin
                    @(posedge clk);
                    waited = waited + 1;
                    if (waited > watchdog) begin $display("ERROR: request %0d watchdog", n); $finish; end
                end
                @(posedge clk);
            end else begin
                do begin
                    repeat (poll) @(posedge clk);
                    waited = waited + poll;
                    if (waited > watchdog) begin $display("ERROR: request %0d watchdog", n); $finish; end
                    axi_read(MB + 4, v);
                end while (v != seq);
            end
            race = (race_acc < 0) ? "none" : $sformatf("%0d", mb0_cyc - race_acc);
            mbs = "";
            for (i = 6; i <= 35; i = i + 1) begin
                // exact-timing requests read by backdoor (no AXI traffic before the next, timed request)
                if (flags & 1) v = u_soc.mem[MB0_WORD + i];
                else axi_read(MB + 4 * i, v);
                if ($isunknown(v)) $display("WARNING: MB[%0d] has X", i);
                mbs = {mbs, (i == 6) ? "" : ",", $sformatf("%h", v)};
            end
            $display("REQ n=%0d op=%0d len=%0d flags=%0d seq=%0d svc=%0d pb_re=%0d pb_we=%0d pa_re=%0d pa_we=%0d clk_on=%0d sleep=%0d tb_asleep=%0d tb_sleeps=%0d awake=%0d race=%s mb=%s",
                     n, opcode, length, flags, seq, r_cyc, r_pb_re, r_pb_we, r_pa_re, r_pa_we, r_clk, r_sleep,
                     snap_asleep, snap_sleeps, awake, race, mbs);
            n = n + 1;
            if (!(flags & 2)) repeat (gap) @(posedge clk);
        end
        $fclose(fd);
        $display("SUM requests=%0d cyc=%0d pb_re=%0d pb_we=%0d pa_re=%0d pa_we=%0d clk_on=%0d sleep=%0d sleeps=%0d",
                 n, a_cyc, a_pb_re, a_pb_we, a_pa_re, a_pa_we, gclk_edges, a_sleep, a_sleeps);
`ifdef R3_EQUIV
        $display("EQUIV mismatches=%0d cycles=%0d", eq_bad, eq_cycles);
`endif
        $display("ERRORS=%0d", errors);
        $display("DONE");
        $finish;
    end
endmodule
