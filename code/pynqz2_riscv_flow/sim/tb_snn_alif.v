// tb_snn_alif.v -- self-checking testbench for the ALIF rtl/snn_layer.v.
//
// Drives the DUT through exactly the buses the CPU uses (register strobes and the
// weight window), loads the weights and per-neuron parameters of a vector set made
// by golden_alif.py, replays its command stream and compares against the golden
// model:
//   * after EVERY tick: the fire vector (SPK) and u and a of every neuron (U, A)
//   * at the end: every spike counter and the tick counter, evt_ovf = 0
// plus directed tests: ID / CONFIG / parameter reset values, parameter read-back
// (BQ sign extension), CLR keeps parameters and zeroes state, one event written
// while a TICK is busy is kept, a second one sets evt_ovf and is dropped, CLR in
// the middle of a TICK aborts it.
//
// In GAP mode (TB_GAP >= 2) events are pushed every GAP clocks without polling,
// and the first event of each step is written one clock after its TICK, while
// the layer is still busy with the tick.
//
// Run from a vector directory (run_snn_alif.sh does this) so the $readmemh file
// names resolve: weights.mem params.mem cmds.mem expect.mem and tb_params.vh.
`include "tb_params.vh"

module tb_snn_alif;
    localparam integer P      = `TB_P;
    localparam integer N_IN   = `TB_NIN;
    localparam integer XBITS  = `TB_XBITS;
    localparam integer WORDS  = `TB_WORDS;
    localparam integer NCMD   = `TB_NCMD;
    localparam integer NTICK  = `TB_NTICK;
    localparam integer GAP    = `TB_GAP;
    localparam integer SPKW   = (P + 31) / 32;
    localparam integer NEXP   = NTICK * (SPKW + 2 * P) + P + 1;

    reg clk = 1'b0;
    always #5 clk = ~clk;

    reg          rst_n = 1'b0;
    reg          reg_wr = 1'b0;
    reg  [11:0]  reg_waddr = 0;
    reg  [31:0]  reg_wdata = 0;
    reg  [11:0]  reg_raddr = 0;
    wire [31:0]  reg_rdata;
    reg          wgt_wr = 1'b0;
    reg  [13:0]  wgt_waddr = 0;
    reg  [31:0]  wgt_wdata = 0;
    wire [P-1:0] out_spk;
    wire         out_valid;

    snn_layer #(.P(P), .N_IN(N_IN), .WBITS(8), .XBITS(XBITS)) dut (
        .clk(clk), .rst_n(rst_n),
        .reg_wr(reg_wr), .reg_waddr(reg_waddr), .reg_wdata(reg_wdata),
        .reg_raddr(reg_raddr), .reg_rdata(reg_rdata),
        .wgt_wr(wgt_wr), .wgt_waddr(wgt_waddr), .wgt_wdata(wgt_wdata),
        .out_spk(out_spk), .out_valid(out_valid));

    reg [31:0] weights [0:WORDS-1];
    reg [31:0] params  [0:3*P-1];
    reg [31:0] cmds    [0:NCMD-1];
    reg [31:0] exp_mem [0:NEXP-1];

    integer errors = 0, checks = 0;
    task check(input [31:0] got, input [31:0] want, input [8*24-1:0] what, input integer a, input integer b);
        begin
            checks = checks + 1;
            if (got !== want) begin
                errors = errors + 1;
                if (errors <= 10)
                    $display("MISMATCH %0s [%0d/%0d]: got %h want %h", what, a, b, got, want);
            end
        end
    endtask

    // one-cycle CPU write strobes, combinational reads (as spike_soc drives them)
    task wr(input [11:0] a, input [31:0] d);
        begin
            @(posedge clk) #1;
            reg_waddr = a; reg_wdata = d; reg_wr = 1'b1;
            @(posedge clk) #1;
            reg_wr = 1'b0;
        end
    endtask
    task rd(input [11:0] a, output [31:0] d);
        begin
            reg_raddr = a; #1; d = reg_rdata;
        end
    endtask
    task wgt(input [13:0] a, input [31:0] d);
        begin
            @(posedge clk) #1;
            wgt_waddr = a; wgt_wdata = d; wgt_wr = 1'b1;
            @(posedge clk) #1;
            wgt_wr = 1'b0;
        end
    endtask
    task wait_idle;
        reg [31:0] s;
        integer n;
        begin
            n = 0;
            rd(12'h004, s);
            while (s[0]) begin
                @(posedge clk) #1;
                rd(12'h004, s);
                n = n + 1;
                if (n > 1000) begin
                    $display("MISMATCH busy stuck");
                    errors = errors + 1;
                    s = 0;
                end
            end
        end
    endtask

    task check_tick(input integer t);
        integer n, base;
        reg [31:0] v;
        begin
            base = t * (SPKW + 2 * P);
            for (n = 0; n < SPKW; n = n + 1) begin
                rd(12'h010 + 4 * n, v); check(v, exp_mem[base + n], "fire", t, n);
            end
            for (n = 0; n < P; n = n + 1) begin
                rd(12'h800 + 4 * n, v); check(v, exp_mem[base + SPKW + n], "u", t, n);
                rd(12'h900 + 4 * n, v); check(v, exp_mem[base + SPKW + P + n], "a", t, n);
            end
        end
    endtask

    integer i, n, t, op, ci;
    reg [31:0] v, c;

    initial begin
        $readmemh("weights.mem", weights);
        $readmemh("params.mem", params);
        $readmemh("cmds.mem", cmds);
        $readmemh("expect.mem", exp_mem);
        $display("TB: P=%0d N_IN=%0d XBITS=%0d ticks=%0d cmds=%0d gap=%0d", P, N_IN, XBITS, NTICK, NCMD, GAP);

        repeat (3) @(posedge clk);
        #1 rst_n = 1'b1;
        repeat (2) @(posedge clk);

        // ---- reset state ----
        rd(12'h034, v); check(v, 32'h414C4946, "ID", 0, 0);
        rd(12'h00C, v); check(v, (N_IN << 16) | (XBITS << 8) | P, "CONFIG", 0, 0);
        rd(12'h004, v); check(v, 0, "STATUS@rst", 0, 0);
        rd(12'h030, v); check(v, 0, "TICKS@rst", 0, 0);
        for (n = 0; n < P; n = n + 1) begin
            rd(12'h500 + 4 * n, v); check(v, 32'h0033_0400, "NP@rst", n, 0);
            rd(12'h600 + 4 * n, v); check(v, 0, "BQ@rst", n, 0);
            rd(12'h700 + 4 * n, v); check(v, 0, "BIAS@rst", n, 0);
        end

        // ---- weights and parameters, parameter read-back ----
        for (i = 0; i < WORDS; i = i + 1) wgt(4 * i, weights[i]);
        for (n = 0; n < P; n = n + 1) begin
            wr(12'h500 + 4 * n, params[3 * n]);
            wr(12'h600 + 4 * n, params[3 * n + 1]);
            wr(12'h700 + 4 * n, params[3 * n + 2]);
        end
        for (n = 0; n < P; n = n + 1) begin
            rd(12'h500 + 4 * n, v); check(v, params[3 * n], "NP", n, 0);
            rd(12'h600 + 4 * n, v); check(v, params[3 * n + 1], "BQ", n, 0);
            rd(12'h700 + 4 * n, v); check(v, params[3 * n + 2], "BIAS", n, 0);
        end
        wr(12'h000, 32'h1);                                  // CLR

        // ---- replay the command stream ----
        t = 0;
        ci = 0;
        while (ci < NCMD) begin
            c = cmds[ci];
            op = c[31:30];
            ci = ci + 1;
            if (op == 0) begin                               // EVT
                wr(12'h008, c & 32'h00FF_FFFF);
                if (GAP == 0) wait_idle;
                else repeat (GAP - 2) @(posedge clk);        // wr itself takes 2 clocks
            end else if (op == 1) begin                      // TICK
                wr(12'h000, 32'h2);
                if (GAP != 0 && cmds[ci][31:30] == 2'd0) begin   // next step's first event, during the tick
                    rd(12'h004, v); check(v[0], 1, "busy during tick", t, 0);
                    wr(12'h008, cmds[ci] & 32'h00FF_FFFF);
                    ci = ci + 1;
                end
                wait_idle;
                check_tick(t);
                t = t + 1;
            end else begin
                ci = NCMD;                                   // END
            end
        end
        check(t, NTICK, "ticks run", 0, 0);
        for (n = 0; n < P; n = n + 1) begin
            rd(12'h400 + 4 * n, v); check(v, exp_mem[NTICK * (SPKW + 2 * P) + n], "cnt", n, 0);
        end
        rd(12'h030, v); check(v, exp_mem[NTICK * (SPKW + 2 * P) + P], "TICKS", 0, 0);
        rd(12'h004, v); check(v, 0, "STATUS (evt_ovf)", 0, 0);

        // ---- CLR keeps parameters, zeroes state ----
        wr(12'h000, 32'h1);
        @(posedge clk) #1;
        rd(12'h030, v); check(v, 0, "TICKS after CLR", 0, 0);
        for (n = 0; n < P; n = n + 1) begin
            rd(12'h800 + 4 * n, v); check(v, 0, "U after CLR", n, 0);
            rd(12'h900 + 4 * n, v); check(v, 0, "A after CLR", n, 0);
            rd(12'h400 + 4 * n, v); check(v, 0, "CNT after CLR", n, 0);
            rd(12'h500 + 4 * n, v); check(v, params[3 * n], "NP kept by CLR", n, 0);
            rd(12'h600 + 4 * n, v); check(v, params[3 * n + 1], "BQ kept by CLR", n, 0);
        end

        // ---- event register while a TICK is busy: 1st kept, 2nd dropped + evt_ovf ----
        wr(12'h000, 32'h2);                                  // tick (busy TICK_CYCLES clocks)
        wr(12'h008, 32'h0001_0000);                          // kept
        wr(12'h008, 32'h0001_0001);                          // dropped
        rd(12'h004, v); check(v[1], 1, "evt_ovf set", 0, 0);
        wait_idle;
        rd(12'h030, v); check(v, 1, "tick ran", 0, 0);
        wr(12'h000, 32'h1);
        @(posedge clk) #1;
        rd(12'h004, v); check(v, 0, "CLR clears evt_ovf", 0, 0);

        // ---- CLR in the middle of a TICK aborts it ----
        wr(12'h000, 32'h2);
        wr(12'h000, 32'h1);                                  // 2 clocks into the tick
        repeat (8) @(posedge clk);
        #1;
        rd(12'h004, v); check(v, 0, "idle after CLR mid-tick", 0, 0);
        rd(12'h030, v); check(v, 0, "no tick counted after CLR mid-tick", 0, 0);
        for (n = 0; n < P; n = n + 1) begin
            rd(12'h800 + 4 * n, v); check(v, 0, "U after CLR mid-tick", n, 0);
        end

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
