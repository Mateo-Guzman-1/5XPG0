// tb_ps_if_dual.v -- self-checking testbench for the NUM_CORES = 2 decode of
// rtl/ps_if.v, driven directly on its AXI4-Lite slave port.
//
//   1. directed: core 0 / core 1 BRAM and syscon reach their own port group
//      (no crosstalk: SCRATCH, CTRL/reset, STATUS, TIME, LED, BRAM contents,
//      bram_we pulses counted per core), unimplemented cores 2..7 and unused
//      in-core offsets answer SLVERR on both B and R and change nothing.
//   2. lockstep regression: random core-0 traffic (BRAM, syscon, unused
//      offsets; random strobes and handshake delays) goes to this ps_if AND to
//      the original single-core ps_if (ps_if_ref, made by run_ps_if_dual.sh
//      from git); every output is compared every clock.
//   3. random two-core scoreboard: random reads/writes to both cores' BRAM and
//      SCRATCH, sometimes a write to one core while reading the other,
//      checked against a per-core model.
//
// Run: sim/run_ps_if_dual.sh   (iverilog + vvp). Ends with RESULT: PASS / FAIL.
`timescale 1ns/1ps

module tb_ps_if_dual;
    localparam integer N = 2;
    localparam [31:0] BASE = 32'h4000_0000;
    localparam [31:0] CORE_STRIDE = 32'h0010_0000;
    localparam [31:0] SYSCON = 32'h0004_0000;
    localparam [1:0]  OKAY = 2'b00, SLVERR = 2'b10;

    reg clk = 1'b0;
    always #5 clk = ~clk;
    reg aresetn = 1'b0;

    integer errors = 0;
    integer checks = 0;
    task automatic check(input cond, input [8*96-1:0] what);
        begin
            checks = checks + 1;
            if (cond !== 1'b1) begin
                errors = errors + 1;
                $display("BAD  t=%0t  %0s", $time, what);
            end
        end
    endtask

    // ------------------------------------------------------------------
    // AXI4-Lite master signals (shared by the DUT and, in lockstep, the reference)
    // ------------------------------------------------------------------
    reg  [31:0] awaddr = 0;  reg awvalid = 0;
    reg  [31:0] wdata = 0;   reg [3:0] wstrb = 0; reg wvalid = 0;
    reg         bready = 0;
    reg  [31:0] araddr = 0;  reg arvalid = 0;
    reg         rready = 0;

    wire        awready, wready, bvalid, arready, rvalid;
    wire [1:0]  bresp, rresp;
    wire [31:0] rdata;

    // ---- DUT port groups ----
    wire [N-1:0]    bram_we;
    wire [16*N-1:0] bram_addr;
    wire [32*N-1:0] bram_wdata;
    wire [4*N-1:0]  bram_be;
    wire [32*N-1:0] bram_rdata;
    wire [N-1:0]    core_rst;
    wire [32*N-1:0] scratch;
    reg  [32*N-1:0] timer_now;
    wire [N-1:0]    core_running = ~core_rst;          // like spike_soc
    wire [N-1:0]    core_trap    = 2'b10;              // core 1 only: routing check
    wire [10*N-1:0] led          = {10'h2AA, 10'h155}; // core 1, core 0

    ps_if #(.CLK_HZ(32'd100_000_000), .NUM_CORES(N)) dut (
        .aclk(clk), .aresetn(aresetn),
        .s_axil_awaddr(awaddr), .s_axil_awvalid(awvalid), .s_axil_awready(awready),
        .s_axil_wdata(wdata), .s_axil_wstrb(wstrb), .s_axil_wvalid(wvalid), .s_axil_wready(wready),
        .s_axil_bresp(bresp), .s_axil_bvalid(bvalid), .s_axil_bready(bready),
        .s_axil_araddr(araddr), .s_axil_arvalid(arvalid), .s_axil_arready(arready),
        .s_axil_rdata(rdata), .s_axil_rresp(rresp), .s_axil_rvalid(rvalid), .s_axil_rready(rready),
        .bram_we(bram_we), .bram_addr(bram_addr), .bram_wdata(bram_wdata), .bram_be(bram_be),
        .bram_rdata(bram_rdata),
        .core_rst(core_rst), .core_running(core_running), .core_trap(core_trap),
        .timer_now(timer_now), .scratch(scratch), .led(led)
    );

    // ---- reference: the original single-core ps_if ----
    wire        r_awready, r_wready, r_bvalid, r_arready, r_rvalid;
    wire [1:0]  r_bresp, r_rresp;
    wire [31:0] r_rdata;
    wire        r_bram_we;
    wire [15:0] r_bram_addr;
    wire [31:0] r_bram_wdata, r_bram_rdata, r_scratch;
    wire [3:0]  r_bram_be;
    wire        r_core_rst;

    ps_if_ref #(.CLK_HZ(32'd100_000_000)) ref0 (
        .aclk(clk), .aresetn(aresetn),
        .s_axil_awaddr(awaddr), .s_axil_awvalid(awvalid), .s_axil_awready(r_awready),
        .s_axil_wdata(wdata), .s_axil_wstrb(wstrb), .s_axil_wvalid(wvalid), .s_axil_wready(r_wready),
        .s_axil_bresp(r_bresp), .s_axil_bvalid(r_bvalid), .s_axil_bready(bready),
        .s_axil_araddr(araddr), .s_axil_arvalid(arvalid), .s_axil_arready(r_arready),
        .s_axil_rdata(r_rdata), .s_axil_rresp(r_rresp), .s_axil_rvalid(r_rvalid), .s_axil_rready(rready),
        .bram_we(r_bram_we), .bram_addr(r_bram_addr), .bram_wdata(r_bram_wdata), .bram_be(r_bram_be),
        .bram_rdata(r_bram_rdata),
        .core_rst(r_core_rst), .core_running(~r_core_rst), .core_trap(core_trap[0]),
        .timer_now(timer_now[31:0]), .scratch(r_scratch), .led(led[9:0])
    );

    // ------------------------------------------------------------------
    // BRAM port-A models (same behaviour as spike_soc: byte enables,
    // registered read), one per DUT core plus one for the reference.
    // ------------------------------------------------------------------
    reg [31:0] mem0 [0:65535];
    reg [31:0] mem1 [0:65535];
    reg [31:0] memr [0:65535];
    reg [31:0] rd0, rd1, rdr;
    assign bram_rdata   = {rd1, rd0};
    assign r_bram_rdata = rdr;
    integer i;
    initial for (i = 0; i < 65536; i = i + 1) begin mem0[i] = 0; mem1[i] = 0; memr[i] = 0; end

    task automatic bwrite(inout [31:0] word, input [31:0] d, input [3:0] be);
        begin
            if (be[0]) word[7:0]   = d[7:0];
            if (be[1]) word[15:8]  = d[15:8];
            if (be[2]) word[23:16] = d[23:16];
            if (be[3]) word[31:24] = d[31:24];
        end
    endtask
    reg [31:0] tmpw;
    always @(posedge clk) begin
        if (bram_we[0]) begin tmpw = mem0[bram_addr[15:0]];  bwrite(tmpw, bram_wdata[31:0],  bram_be[3:0]); mem0[bram_addr[15:0]]  <= tmpw; end
        if (bram_we[1]) begin tmpw = mem1[bram_addr[31:16]]; bwrite(tmpw, bram_wdata[63:32], bram_be[7:4]); mem1[bram_addr[31:16]] <= tmpw; end
        if (r_bram_we)  begin tmpw = memr[r_bram_addr];      bwrite(tmpw, r_bram_wdata,       r_bram_be);   memr[r_bram_addr]      <= tmpw; end
        rd0 <= mem0[bram_addr[15:0]];
        rd1 <= mem1[bram_addr[31:16]];
        rdr <= memr[r_bram_addr];
    end

    // distinct free-running timers per core
    always @(posedge clk)
        if (!aresetn) timer_now <= {32'h8000_0000, 32'h0000_0000};
        else          timer_now <= {timer_now[63:32] + 32'd1, timer_now[31:0] + 32'd1};

    // write-enable pulses per core
    integer we_cnt0 = 0, we_cnt1 = 0;
    always @(posedge clk) begin
        if (bram_we[0]) we_cnt0 = we_cnt0 + 1;
        if (bram_we[1]) we_cnt1 = we_cnt1 + 1;
    end

    // ------------------------------------------------------------------
    // AXI master tasks (random handshake delays when 'jitter' is set)
    // ------------------------------------------------------------------
    reg jitter = 0;
    integer seed = 32'h5EED;
    function integer rdly(input integer maxd);
        rdly = jitter ? ({$random(seed)} % (maxd + 1)) : 0;
    endfunction

    task automatic send_aw(input [31:0] a);
        begin
            repeat (rdly(3)) @(posedge clk);
            awaddr <= a; awvalid <= 1'b1;
            @(posedge clk); while (!awready) @(posedge clk);
            awvalid <= 1'b0;
        end
    endtask
    task automatic send_w(input [31:0] d, input [3:0] s);
        begin
            repeat (rdly(3)) @(posedge clk);
            wdata <= d; wstrb <= s; wvalid <= 1'b1;
            @(posedge clk); while (!wready) @(posedge clk);
            wvalid <= 1'b0;
        end
    endtask
    task automatic axi_write(input [31:0] a, input [31:0] d, input [3:0] s, output [1:0] resp);
        begin
            fork
                send_aw(a);
                send_w(d, s);
            join
            repeat (rdly(3)) @(posedge clk);
            bready <= 1'b1;
            @(posedge clk); while (!bvalid) @(posedge clk);
            resp = bresp;
            bready <= 1'b0;
        end
    endtask
    task automatic axi_read(input [31:0] a, output [31:0] d, output [1:0] resp);
        begin
            repeat (rdly(3)) @(posedge clk);
            araddr <= a; arvalid <= 1'b1;
            @(posedge clk); while (!arready) @(posedge clk);
            arvalid <= 1'b0;
            repeat (rdly(3)) @(posedge clk);
            rready <= 1'b1;
            @(posedge clk); while (!rvalid) @(posedge clk);
            d = rdata; resp = rresp;
            rready <= 1'b0;
        end
    endtask

    function [31:0] bram_a(input integer core, input [31:0] off);
        bram_a = BASE + core * CORE_STRIDE + off;
    endfunction
    function [31:0] sys_a(input integer core, input [31:0] reg_off);
        sys_a = BASE + core * CORE_STRIDE + SYSCON + reg_off;
    endfunction

    // ------------------------------------------------------------------
    // Lockstep comparison against the single-core reference (phase 2)
    // ------------------------------------------------------------------
    reg lockstep = 0;
    reg [31:0] scratch1_frozen;
    always @(negedge clk) if (lockstep) begin
        check(awready == r_awready, "lockstep: awready differs from single-core ps_if");
        check(wready  == r_wready,  "lockstep: wready differs");
        check(bvalid  == r_bvalid,  "lockstep: bvalid differs");
        check(arready == r_arready, "lockstep: arready differs");
        check(rvalid  == r_rvalid,  "lockstep: rvalid differs");
        if (bvalid) check(bresp == r_bresp, "lockstep: bresp differs");
        if (rvalid) check(rresp == r_rresp, "lockstep: rresp differs");
        if (rvalid && rresp == OKAY) check(rdata == r_rdata, "lockstep: rdata differs");
        check(core_rst[0] == r_core_rst, "lockstep: core 0 reset differs");
        check(scratch[31:0] == r_scratch, "lockstep: core 0 SCRATCH differs");
        check(bram_we[0] == r_bram_we, "lockstep: core 0 bram_we differs");
        if (bram_we[0]) begin
            check(bram_addr[15:0] == r_bram_addr, "lockstep: core 0 write address differs");
            check(bram_wdata[31:0] == r_bram_wdata, "lockstep: core 0 write data differs");
            check(bram_be[3:0] == r_bram_be, "lockstep: core 0 byte enables differ");
        end
        // core-0-only traffic must leave core 1 alone
        check(bram_we[1] == 1'b0, "lockstep: core 1 bram_we pulsed by core-0 traffic");
        check(core_rst[1] == 1'b1, "lockstep: core 1 reset changed by core-0 traffic");
        check(scratch[63:32] == scratch1_frozen, "lockstep: core 1 SCRATCH changed by core-0 traffic");
    end

    // ------------------------------------------------------------------
    // Scoreboard model (phase 3)
    // ------------------------------------------------------------------
    reg [31:0] exp0 [0:65535];
    reg [31:0] exp1 [0:65535];
    reg [31:0] exp_scr [0:1];

    // ------------------------------------------------------------------
    // Test sequence
    // ------------------------------------------------------------------
    reg [31:0] tmpe;
    reg [31:0] d, d2, t0a, t0b, t1a, t1b;
    reg [1:0]  r, r2;
    integer k, core, n, we0_before, we1_before, op, widx;
    reg [31:0] a, rnd;
    reg [3:0]  s;

    initial begin
        $display("tb_ps_if_dual: NUM_CORES=%0d", N);
        repeat (5) @(posedge clk);
        aresetn <= 1'b1;
        repeat (3) @(posedge clk);

        // ---------------- 1. directed ----------------
        $display("== 1. directed checks");
        check(core_rst == 2'b11, "after reset both cores must be held in reset");
        for (core = 0; core < N; core = core + 1) begin
            axi_read(sys_a(core, 32'h14), d, r);
            check(r == OKAY && d == 32'h534B_454C, "MAGIC of a core");
            axi_read(sys_a(core, 32'h10), d, r);
            check(r == OKAY && d == 32'd100_000_000, "CLK_HZ of a core");
            axi_read(sys_a(core, 32'h00), d, r);
            check(r == OKAY && d == 32'h1, "CTRL reset value 1 of a core");
            axi_read(sys_a(core, 32'h18), d, r);
            check(r == OKAY && d == (core == 0 ? 32'h155 : 32'h2AA), "LED register routed to its own core");
        end

        // BRAM: same offset in both cores, different data
        we0_before = we_cnt0; we1_before = we_cnt1;
        axi_write(bram_a(0, 32'h100), 32'h1111_2222, 4'hF, r);
        check(r == OKAY, "core 0 BRAM write OKAY");
        check(we_cnt0 == we0_before + 1 && we_cnt1 == we1_before, "core 0 BRAM write pulses only core 0 bram_we");
        check(bram_addr[15:0] == 16'h0040, "core 0 BRAM write word address 0x40");
        axi_write(bram_a(1, 32'h100), 32'h3333_4444, 4'hF, r);
        check(r == OKAY, "core 1 BRAM write OKAY");
        check(we_cnt1 == we1_before + 1 && we_cnt0 == we0_before + 1, "core 1 BRAM write pulses only core 1 bram_we");
        check(bram_addr[31:16] == 16'h0040, "core 1 BRAM write word address 0x40");
        @(posedge clk);
        check(mem0[16'h40] == 32'h1111_2222, "core 0 BRAM holds core 0's word");
        check(mem1[16'h40] == 32'h3333_4444, "core 1 BRAM holds core 1's word");
        axi_read(bram_a(0, 32'h100), d, r);
        check(r == OKAY && d == 32'h1111_2222, "core 0 BRAM read-back");
        axi_read(bram_a(1, 32'h100), d, r);
        check(r == OKAY && d == 32'h3333_4444, "core 1 BRAM read-back");
        // last word of each 256 KB BRAM, and byte strobes on core 1
        axi_write(bram_a(1, 32'h3_FFFC), 32'hA1B2_C3D4, 4'b0101, r);
        axi_read(bram_a(1, 32'h3_FFFC), d, r);
        check(r == OKAY && d == 32'h00B2_00D4, "core 1 byte-strobe write to the last BRAM word");
        axi_read(bram_a(0, 32'h3_FFFC), d, r);
        check(r == OKAY && d == 32'h0, "core 0 last BRAM word untouched by core 1's write");

        // SCRATCH, both directions
        axi_write(sys_a(1, 32'h0C), 32'hCAFE_F00D, 4'hF, r);
        check(r == OKAY, "core 1 SCRATCH write OKAY");
        check(scratch[63:32] == 32'hCAFE_F00D && scratch[31:0] == 32'h0, "core 1 SCRATCH write leaves core 0's");
        axi_read(sys_a(0, 32'h0C), d, r);
        check(r == OKAY && d == 32'h0, "core 0 SCRATCH reads 0 after core 1's write");
        axi_write(sys_a(0, 32'h0C), 32'h1234_5678, 4'hF, r);
        check(scratch[31:0] == 32'h1234_5678 && scratch[63:32] == 32'hCAFE_F00D, "core 0 SCRATCH write leaves core 1's");
        axi_read(sys_a(1, 32'h0C), d, r);
        check(r == OKAY && d == 32'hCAFE_F00D, "core 1 SCRATCH read-back");

        // CTRL / reset independence and STATUS routing
        axi_write(sys_a(0, 32'h00), 32'h0, 4'hF, r);
        check(core_rst == 2'b10, "starting core 0 leaves core 1 in reset");
        axi_read(sys_a(0, 32'h04), d, r);
        check(r == OKAY && d == 32'h1, "core 0 STATUS: running, no trap");
        axi_read(sys_a(1, 32'h04), d, r);
        check(r == OKAY && d == 32'h2, "core 1 STATUS: not running, its own trap bit");
        axi_write(sys_a(1, 32'h00), 32'h0, 4'hF, r);
        check(core_rst == 2'b00, "starting core 1 too");
        axi_write(sys_a(0, 32'h00), 32'h1, 4'hF, r);
        check(core_rst == 2'b01, "stopping core 0 leaves core 1 running");
        axi_read(sys_a(1, 32'h00), d, r);
        check(r == OKAY && d == 32'h0, "core 1 CTRL reads 0 (running)");
        axi_write(sys_a(1, 32'h00), 32'h1, 4'hF, r);
        check(core_rst == 2'b11, "both back in reset");

        // TIME: each core's own counter, advancing
        axi_read(sys_a(0, 32'h08), t0a, r);
        axi_read(sys_a(1, 32'h08), t1a, r);
        axi_read(sys_a(0, 32'h08), t0b, r);
        axi_read(sys_a(1, 32'h08), t1b, r);
        check(t0a < 32'h8000_0000 && t1a >= 32'h8000_0000, "TIME routed to its own core's timer");
        check(t0b > t0a && t1b > t1a, "both TIME values advance");

        // unimplemented cores 2..7: SLVERR on B and R, nothing changes
        we0_before = we_cnt0; we1_before = we_cnt1;
        for (core = N; core < 8; core = core + 1) begin
            axi_write(bram_a(core, 32'h100), 32'hBAD0_0000 | core, 4'hF, r);
            check(r == SLVERR, "write to an unimplemented core's BRAM range answers SLVERR");
            axi_write(sys_a(core, 32'h00), 32'h0, 4'hF, r);
            check(r == SLVERR, "write to an unimplemented core's CTRL answers SLVERR");
            axi_write(sys_a(core, 32'h0C), 32'hBAD0_0000, 4'hF, r);
            check(r == SLVERR, "write to an unimplemented core's SCRATCH answers SLVERR");
            axi_read(bram_a(core, 32'h100), d, r);
            check(r == SLVERR, "read of an unimplemented core's BRAM range answers SLVERR");
            axi_read(sys_a(core, 32'h14), d, r);
            check(r == SLVERR, "read of an unimplemented core's MAGIC answers SLVERR");
        end
        axi_read(BASE + 32'h007F_FFFC, d, r);
        check(r == SLVERR, "top of the 8 MB window answers SLVERR");
        check(we_cnt0 == we0_before && we_cnt1 == we1_before, "unimplemented-core writes pulse no bram_we");
        check(core_rst == 2'b11, "unimplemented-core CTRL writes start no core");
        check(scratch[31:0] == 32'h1234_5678 && scratch[63:32] == 32'hCAFE_F00D,
              "unimplemented-core SCRATCH writes change no SCRATCH");
        // unused offsets inside an implemented core (as before: SLVERR)
        for (core = 0; core < N; core = core + 1) begin
            axi_write(BASE + core * CORE_STRIDE + 32'h5_0000, 32'h0, 4'hF, r);
            check(r == SLVERR, "write to an unused in-core offset answers SLVERR");
            axi_read(BASE + core * CORE_STRIDE + 32'h8_0000, d, r);
            check(r == SLVERR, "read of an unused in-core offset answers SLVERR");
        end
        check(we_cnt0 == we0_before && we_cnt1 == we1_before, "unused-offset writes pulse no bram_we");
        $display("   directed: %0d checks, %0d bad", checks, errors);

        // ---------------- 2. lockstep regression against the single-core ps_if ----------------
        $display("== 2. lockstep: random core-0 traffic vs. the single-core ps_if");
        // bring the reference to the same state as the DUT's core 0
        aresetn <= 1'b0; repeat (3) @(posedge clk); aresetn <= 1'b1; repeat (2) @(posedge clk);
        for (i = 0; i < 65536; i = i + 1) begin memr[i] = mem0[i]; end
        axi_write(sys_a(1, 32'h0C), 32'h5A5A_A5A5, 4'hF, r);   // core 1 state to protect
        scratch1_frozen = scratch[63:32];
        jitter = 1;
        lockstep = 1;
        n = checks;
        for (k = 0; k < 3000; k = k + 1) begin
            rnd = $random(seed);
            op  = {$random(seed)} % 10;
            case (op)
            0, 1, 2, 3: a = BASE + {rnd[17:2], 2'b00} & 32'h4003_FFFC;          // BRAM, anywhere
            4, 5:       a = BASE + {rnd[9:2], 2'b00};                              // BRAM, low words
            6, 7:       a = BASE + SYSCON + ({$random(seed)} % 9) * 4;            // syscon 0x00..0x20
            8:          a = BASE + 32'h0005_0000 + ({rnd[19:2], 2'b00} % 32'hB_0000); // unused in core 0
            default:    a = BASE + SYSCON + {rnd[15:2], 2'b00};                   // syscon aliases
            endcase
            s = $random(seed);
            if ($random(seed) & 1) axi_write(a, $random(seed), s, r);
            else                   axi_read(a, d, r);
        end
        lockstep = 0;
        jitter = 0;
        $display("   lockstep: %0d cycle checks, %0d bad so far", checks - n, errors);

        // ---------------- 3. random two-core scoreboard ----------------
        $display("== 3. random two-core traffic against a per-core model");
        for (i = 0; i < 65536; i = i + 1) begin exp0[i] = mem0[i]; exp1[i] = mem1[i]; end
        exp_scr[0] = scratch[31:0]; exp_scr[1] = scratch[63:32];
        jitter = 1;
        n = checks;
        for (k = 0; k < 3000; k = k + 1) begin
            core = {$random(seed)} % 2;
            widx = {$random(seed)} % 64;                  // a small set: lots of reuse
            if (widx == 63) widx = 65535;
            op = {$random(seed)} % 6;
            d2 = $random(seed);
            s  = $random(seed);
            case (op)
            0: begin                                      // BRAM write
                axi_write(bram_a(core, widx * 4), d2, s, r);
                check(r == OKAY, "scoreboard: BRAM write OKAY");
                if (core == 0) begin tmpe = exp0[widx]; bwrite(tmpe, d2, s); exp0[widx] = tmpe; end
                else           begin tmpe = exp1[widx]; bwrite(tmpe, d2, s); exp1[widx] = tmpe; end
            end
            1, 2: begin                                   // BRAM read
                axi_read(bram_a(core, widx * 4), d, r);
                check(r == OKAY && d == (core == 0 ? exp0[widx] : exp1[widx]), "scoreboard: BRAM read");
            end
            3: begin                                      // SCRATCH write + read of both
                axi_write(sys_a(core, 32'h0C), d2, s, r);
                tmpe = exp_scr[core]; bwrite(tmpe, d2, s); exp_scr[core] = tmpe;
                axi_read(sys_a(0, 32'h0C), d, r);
                check(r == OKAY && d == exp_scr[0], "scoreboard: core 0 SCRATCH");
                axi_read(sys_a(1, 32'h0C), d, r);
                check(r == OKAY && d == exp_scr[1], "scoreboard: core 1 SCRATCH");
            end
            default: begin                                // write one core while reading the other
                fork
                    axi_write(bram_a(core, widx * 4), d2, 4'hF, r);
                    axi_read(bram_a(1 - core, widx * 4), d, r2);
                join
                check(r == OKAY && r2 == OKAY, "scoreboard: concurrent write/read OKAY");
                check(d == (core == 0 ? exp1[widx] : exp0[widx]),
                      "scoreboard: read of the other core during a write sees the other core's data");
                if (core == 0) exp0[widx] = d2; else exp1[widx] = d2;
            end
            endcase
        end
        jitter = 0;
        repeat (3) @(posedge clk);                       // last write lands one clock after B
        for (i = 0; i < 64; i = i + 1) begin
            if (mem0[i] !== exp0[i] || mem1[i] !== exp1[i])
                $display("     word %0d: core0 %h (model %h)  core1 %h (model %h)", i, mem0[i], exp0[i], mem1[i], exp1[i]);
            check(mem0[i] == exp0[i] && mem1[i] == exp1[i], "scoreboard: final BRAM contents");
        end
        check(mem0[65535] == exp0[65535] && mem1[65535] == exp1[65535], "scoreboard: final last words");
        $display("   scoreboard: %0d checks, %0d bad so far", checks - n, errors);

        $display("");
        if (errors == 0) $display("RESULT: PASS (%0d checks, 0 bad)", checks);
        else             $display("RESULT: FAIL (%0d checks, %0d bad)", checks, errors);
        $finish;
    end

    initial begin
        #50_000_000;
        $display("RESULT: FAIL (timeout - a handshake hung)");
        $finish;
    end
endmodule
