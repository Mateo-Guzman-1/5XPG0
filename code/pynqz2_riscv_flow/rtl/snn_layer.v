// snn_layer.v -- SKETCH. NOT compiled, NOT simulated, NOT synthesized.
// Treat it as a starting point / discussion aid, not a verified block.
//
// P parallel LIF neurons fed by ONE row-wide weight block RAM.
//
//   input spike i  -->  read row i of the weight matrix (= the P weights from
//   input i to the P neurons)  -->  add all P weights in the SAME clock cycle.
//
// So the parallelism (P) is set by the BRAM row width (P*WBITS bits), not by the
// RISC-V core.  Same idea as the "weight memory mapping" + "neuron cluster" of
// Gyro (Corradi 2021) and Sankaran et al. (ICONS 2022).
//
// Neuron update, matched to snnTorch snn.Leaky (defaults: reset_mechanism =
// "subtract", threshold "strict >", reset_delay = True), integer arithmetic:
//
//   during a time step :  v <- v + w * x           (one add per input event)
//   at TICK (end of step): spike = (v > thr)
//                          v    <- v - (v >>> K) - (spike ? thr : 0)
//
//   i.e. beta = 1 - 2^-K (a shift, no multiplier) and the subtract-reset is
//   applied to the LEAKED value, exactly like snnTorch with reset_delay=True.
//   !! Verify against the snnTorch version you train with, and against a
//   !! bit-exact integer golden model, before trusting any accuracy number.
//
// Register map (CPU view).  Two regions, decoded in spike_soc.v:
//
//  WEIGHT window   0x1000_4000 .. 0x1000_7FFF   write-only, 32-bit words
//     word index w = row*GROUPS + g      (GROUPS = P / WPW, WPW = 32/WBITS)
//     word bits [8j+7 : 8j] = weight for neuron (g*WPW + j) from input 'row'
//     (assumes WBITS = 8, P a multiple of 4, N_IN*GROUPS <= 4096 words)
//
//  REGS            0x1000_8000 .. 0x1000_8FFF
//     0x000 CTRL    W   bit0 CLR  : zero v, counters, status
//                       bit1 TICK : evaluate one time step (after pending events)
//     0x004 STATUS  R   bit0 busy, bit1 evt_ovf (sticky, cleared by CLR)
//     0x008 EVT     W   [AW-1:0] input index (row);  [16+XBITS-1:16] input value x
//                       (spike mode, XBITS=1: bit16 = 1)
//     0x00C THR     RW  signed threshold, same fixed-point scale as the weights
//     0x010 SPK     R   fire vector of the last TICK (word n = neurons 32n..32n+31)
//     0x030 TICKS   R   ticks executed since CLR
//     0x400+4c CNT  R   output-spike count of neuron c since CLR
//     0x800+4c VMEM R   membrane potential of neuron c (debug, sign-extended)
//
// NOT modelled here (see the discussion): bias row, chaining layer1 -> layer2
// (serialise out_spk into events of a 2nd snn_layer), refractory period,
// hoisting the constant-current MAC out of the time loop, FIFO on the event port.

// ---------------------------------------------------------------------------
module lif_neuron #(
    parameter integer WBITS      = 8,     // weight width (signed)
    parameter integer XBITS      = 1,     // input width: 1 = binary spike, 8 = direct value
    parameter integer VBITS      = 24,    // membrane width (signed, saturating); must be >= WBITS+XBITS+1
    parameter integer LEAK_SHIFT = 3      // beta = 1 - 2^-LEAK_SHIFT
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     clr,      // synchronous clear (start of a sample)
    input  wire                     syn_en,   // an input event arrived: v += w*x
    input  wire signed [WBITS-1:0]  w,
    input  wire        [XBITS-1:0]  x,
    input  wire                     tick,     // end of time step: leak + fire + reset
    input  wire signed [VBITS-1:0]  thr,
    output wire                     fire,     // valid in the tick cycle (uses pre-update v)
    output reg  signed [VBITS-1:0]  v
);
    localparam integer PW = WBITS + XBITS + 1;                 // product width

    // synaptic add (XBITS=1 -> product is just w or 0: an AND gate, no DSP)
    wire signed [XBITS:0]   xs   = $signed({1'b0, x});
    wire signed [PW-1:0]    prod = w * xs;
    wire signed [VBITS:0]   sum  = v + prod;                   // one guard bit
    wire                    ovf  = (sum[VBITS] != sum[VBITS-1]);
    localparam signed [VBITS-1:0] VMAX = {1'b0, {(VBITS-1){1'b1}}};
    localparam signed [VBITS-1:0] VMIN = {1'b1, {(VBITS-1){1'b0}}};
    wire signed [VBITS-1:0] sum_sat = ovf ? (sum[VBITS] ? VMIN : VMAX) : sum[VBITS-1:0];

    // leak by arithmetic shift, fire on strict '>', subtract-reset
    wire signed [VBITS-1:0] v_shift = v >>> LEAK_SHIFT;
    wire signed [VBITS-1:0] v_leak  = v - v_shift;
    assign fire = (v > thr);
    wire signed [VBITS-1:0] v_next  = fire ? (v_leak - thr) : v_leak;

    // tick and syn_en are mutually exclusive (the layer FSM guarantees it)
    always @(posedge clk) begin
        if (!rst_n || clr) v <= {VBITS{1'b0}};
        else if (tick)     v <= v_next;
        else if (syn_en)   v <= sum_sat;
    end
endmodule

// ---------------------------------------------------------------------------
module snn_layer #(
    parameter integer P          = 64,    // physical neurons running in parallel
    parameter integer N_IN       = 256,   // inputs = weight-RAM rows
    parameter integer WBITS      = 8,     // this sketch assumes 8
    parameter integer XBITS      = 1,     // 1 = binary spikes, 8 = direct 8-bit input
    parameter integer VBITS      = 24,
    parameter integer LEAK_SHIFT = 3,
    parameter integer CBITS      = 8,     // per-neuron output spike counter width
    parameter         INIT_FILE  = ""     // optional $readmemh file, one row (P*WBITS bits) per line
)(
    input  wire         clk,
    input  wire         rst_n,

    // CPU side, same style as poisson.v: 1-cycle write strobes, combinational reads
    input  wire         reg_wr,
    input  wire [11:0]  reg_waddr,        // byte offset in the 4 KB register window
    input  wire [31:0]  reg_wdata,
    input  wire [11:0]  reg_raddr,
    output reg  [31:0]  reg_rdata,
    input  wire         wgt_wr,
    input  wire [13:0]  wgt_waddr,        // byte offset in the 16 KB weight window
    input  wire [31:0]  wgt_wdata,

    // for chaining to a next layer (valid for exactly one cycle)
    output wire [P-1:0] out_spk,
    output wire         out_valid
);
    localparam integer AW     = $clog2(N_IN);
    localparam integer WPW    = 32 / WBITS;        // weights per 32-bit bus word
    localparam integer GROUPS = P / WPW;           // bus words per weight row
    localparam integer ROWW   = P * WBITS;         // weight-RAM row width in bits
    localparam integer SPKW   = (P + 31) / 32;     // 32-bit words needed for the fire vector
    localparam signed [VBITS-1:0] THR_INIT = 64;   // = 1.0 if the scale is 2^6

    // ---------------- state (declared first) ----------------
    localparam [1:0] S_IDLE = 2'd0, S_ACC = 2'd1, S_TICK = 2'd2;
    reg  [1:0]              st;
    reg                     evt_v, evt_ovf, tick_req, clr_r;
    reg  [AW-1:0]           evt_row;
    reg  [XBITS-1:0]        evt_x, x_q;
    reg  signed [VBITS-1:0] thr;
    reg  [31:0]             ticks;
    reg  [P-1:0]            spk_last;
    reg  [P*CBITS-1:0]      cnt;
    wire [P-1:0]            fire;
    wire [P*VBITS-1:0]      vmem;
    wire idle   = (st == S_IDLE);
    wire syn_en = (st == S_ACC);
    wire tick   = (st == S_TICK);
    wire busy   = evt_v | tick_req | ~idle;

    // ---------------- weight RAM: 1 row = P weights ----------------
    // Column-based write (one column per weight) follows the UG901 byte-write-
    // enable template; columns must be 8/9/16/18 bits wide to map onto BRAM
    // byte enables.  Synchronous read of a whole row -> 1 cycle latency.
    (* ram_style = "block" *) reg [ROWW-1:0] wmem [0:N_IN-1];

    generate
        if (INIT_FILE != "") begin : g_init_file
            initial $readmemh(INIT_FILE, wmem);
        end else begin : g_init_zero
            integer r;
            initial for (r = 0; r < N_IN; r = r + 1) wmem[r] = {ROWW{1'b0}};
        end
    endgenerate

    wire [11:0]     wgt_widx = wgt_waddr[13:2];
    wire [AW-1:0]   wgt_row  = wgt_widx / GROUPS;          // GROUPS is a power of two
    wire [31:0]     wgt_grp  = wgt_widx % GROUPS;
    wire [ROWW-1:0] wgt_di;
    reg  [P-1:0]    wgt_we;

    genvar c;
    generate
        for (c = 0; c < P; c = c + 1) begin : g_wdi
            assign wgt_di[c*WBITS +: WBITS] = wgt_wdata[(c % WPW)*WBITS +: WBITS];
        end
    endgenerate

    integer k;
    always @(*) begin
        for (k = 0; k < P; k = k + 1)
            wgt_we[k] = wgt_wr && ((k / WPW) == wgt_grp);
    end

    integer i;
    always @(posedge clk) begin
        for (i = 0; i < P; i = i + 1)
            if (wgt_we[i])
                wmem[wgt_row][i*WBITS +: WBITS] <= wgt_di[i*WBITS +: WBITS];
    end

    reg [ROWW-1:0] wrow;
    always @(posedge clk) wrow <= wmem[evt_row];            // valid one cycle after evt_row

    // ---------------- P neurons, all updated in the same cycle ----------------
    genvar n;
    generate
        for (n = 0; n < P; n = n + 1) begin : g_neuron
            lif_neuron #(.WBITS(WBITS), .XBITS(XBITS), .VBITS(VBITS),
                         .LEAK_SHIFT(LEAK_SHIFT)) u_n (
                .clk(clk), .rst_n(rst_n), .clr(clr_r),
                .syn_en(syn_en), .w(wrow[n*WBITS +: WBITS]), .x(x_q),
                .tick(tick), .thr(thr),
                .fire(fire[n]), .v(vmem[n*VBITS +: VBITS]));
        end
    endgenerate

    assign out_spk   = fire;
    assign out_valid = tick;

    // ---------------- control FSM + register writes ----------------
    //  IDLE --evt_v--> ACC   (row read issued in IDLE, weights valid in ACC)
    //  IDLE --tick_req--> TICK (only when no event is pending)
    // 2 cycles per event; can be pipelined to 1 event/cycle later.
    integer j;
    always @(posedge clk) begin
        if (!rst_n) begin
            st <= S_IDLE;  evt_v <= 1'b0;  evt_ovf <= 1'b0;  tick_req <= 1'b0;  clr_r <= 1'b0;
            evt_row <= {AW{1'b0}};  evt_x <= {XBITS{1'b0}};  x_q <= {XBITS{1'b0}};
            thr <= THR_INIT;  ticks <= 32'd0;
            spk_last <= {P{1'b0}};  cnt <= {(P*CBITS){1'b0}};
        end else begin
            clr_r <= 1'b0;

            case (st)
            S_IDLE: if (evt_v) begin
                        st <= S_ACC;  x_q <= evt_x;  evt_v <= 1'b0;
                    end else if (tick_req) begin
                        st <= S_TICK; tick_req <= 1'b0;
                    end
            S_ACC:  st <= S_IDLE;
            S_TICK: begin
                        st       <= S_IDLE;
                        spk_last <= fire;
                        ticks    <= ticks + 32'd1;
                        for (j = 0; j < P; j = j + 1)
                            if (fire[j])
                                cnt[j*CBITS +: CBITS] <= cnt[j*CBITS +: CBITS] + 1'b1;
                    end
            default: st <= S_IDLE;
            endcase

            // register writes: last assignment wins, so a CPU push in the same
            // cycle the FSM pops the previous event is safe (row already captured).
            if (reg_wr) begin
                case (reg_waddr)
                12'h000: begin
                    if (reg_wdata[0]) begin
                        clr_r <= 1'b1;  evt_v <= 1'b0;  tick_req <= 1'b0;  evt_ovf <= 1'b0;
                        ticks <= 32'd0; spk_last <= {P{1'b0}};  cnt <= {(P*CBITS){1'b0}};
                    end
                    if (reg_wdata[1]) tick_req <= 1'b1;
                end
                12'h008: begin
                    if (evt_v && !idle) evt_ovf <= 1'b1;        // previous event not consumed: drop
                    else begin
                        evt_v   <= 1'b1;
                        evt_row <= reg_wdata[AW-1:0];
                        evt_x   <= reg_wdata[16 +: XBITS];
                    end
                end
                12'h00C: thr <= reg_wdata[VBITS-1:0];
                default: ;
                endcase
            end
        end
    end

    // ---------------- register reads (combinational, like poisson.v) ----------------
    wire [32*SPKW-1:0] spk_pad = spk_last;                   // zero-extended by assignment
    always @(*) begin
        reg_rdata = 32'h0;
        if      (reg_raddr == 12'h000) reg_rdata = {31'b0, busy};
        else if (reg_raddr == 12'h004) reg_rdata = {30'b0, evt_ovf, busy};
        else if (reg_raddr == 12'h00C) reg_rdata = thr;                          // signed -> sign-extends
        else if (reg_raddr >= 12'h010 && reg_raddr < 12'h010 + 4*SPKW)
            reg_rdata = spk_pad[32*((reg_raddr - 12'h010) >> 2) +: 32];
        else if (reg_raddr == 12'h030) reg_rdata = ticks;
        else if (reg_raddr >= 12'h400 && reg_raddr < 12'h400 + 4*P)
            reg_rdata = {{(32-CBITS){1'b0}}, cnt[((reg_raddr - 12'h400) >> 2)*CBITS +: CBITS]};
        else if (reg_raddr >= 12'h800 && reg_raddr < 12'h800 + 4*P)
            reg_rdata = $signed(vmem[((reg_raddr - 12'h800) >> 2)*VBITS +: VBITS]);
    end
endmodule
