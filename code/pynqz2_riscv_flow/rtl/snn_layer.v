// snn_layer.v -- P parallel ALIF (adaptive-threshold LIF) neurons fed by ONE
// row-wide weight block RAM. Bit-exact with sim/golden_alif.py, whose neuron is
// _alif() / layer 1 of model.integer_forward_stream() on branch ALIF-Layer1
// (snn_keyword/model.py, and rtl/neuron_engine.v there).
//
//   input event (row i, value x)  -->  read row i of the weight matrix (= the P
//   weights from input i to the P neurons)  -->  acc[n] += w[i][n] * x for all P
//   neurons at the same time (multiply in one clock, add in the next).
//
// So the parallelism (P) is set by the BRAM row width (P*WBITS bits), not by the
// RISC-V core.  Same idea as the "weight memory mapping" + "neuron cluster" of
// Gyro (Corradi 2021) and Sankaran et al. (ICONS 2022).
//
// Neuron (integer, shifts are arithmetic / floor), per time step:
//
//   during the step :  acc  <- acc + w * x                  (one add per event)
//   at TICK         :  cur  =  (acc >>> r) + b
//                      a    <- a - (a >> ka) + (s << 8)     adaptation, Q8 (256 = 1 spike)
//                      thr  =  theta + ((bq * a) >>> 8)     adaptive threshold
//                      u    <- sat16(u - (u >>> km) + cur)  leak, then integrate
//                      s    <- (u >= thr)
//                      u    <- sat16(u - s * thr)           subtract reset
//                      acc  <- 0
//
// s is the spike of the previous tick when a is updated (as in _alif). With
// bq = 0 the threshold is theta: a plain LIF with leak 2^-km.
// Per-neuron parameters are registers (theta, r, km, ka, bq, b); the update of
// all P neurons runs in parallel, pipelined over TICK_CYCLES = 4 clocks.
//
// Widths: u int16 (saturating), a 24-bit unsigned (exact for every ka <= 15),
// acc 32-bit (exact up to 2^31 / (127 * 255) = 66k events per step), bq 18-bit
// signed (one DSP48 per neuron; the trained models use 36..448), b int32.
// thr is clamped to 18 bits, which never changes a result: u is int16.
//
// Register map (CPU view).  Two regions, decoded in spike_soc.v:
//
//  WEIGHT window   0x1000_4000 .. 0x1000_7FFF   write-only, 32-bit words
//     word index w = row*GROUPS + g      (GROUPS = P / WPW, WPW = 32/WBITS)
//     word bits [8j+7 : 8j] = weight for neuron (g*WPW + j) from input 'row'
//     (assumes WBITS = 8, P a multiple of 4, N_IN*GROUPS <= 4096 words)
//
//  REGS            0x1000_8000 .. 0x1000_8FFF
//     0x000 CTRL    W   bit0 CLR  : zero acc, u, a, s, counters, ticks, status
//                                   (parameters and weights are kept)
//                       bit1 TICK : evaluate one time step (after pending events)
//     0x004 STATUS  R   bit0 busy, bit1 evt_ovf (sticky, cleared by CLR)
//     0x008 EVT     W   [AW-1:0] input index (row);  [16+XBITS-1:16] input value x
//                       (unsigned; spike input: x = 1)
//     0x00C CONFIG  R   [7:0] P, [15:8] XBITS, [31:16] N_IN
//     0x010 SPK     R   fire vector of the last TICK (word n = neurons 32n..32n+31)
//     0x030 TICKS   R   ticks executed since CLR
//     0x034 ID      R   0x414C_4946 "ALIF"
//     0x400+4n CNT  R   output-spike count of neuron n since CLR
//     0x500+4n NP   RW  theta[15:0] | r[19:16] | km[23:20] | ka[27:24]
//                       (the NP1 table format of neuron_engine.v), reset 0x0033_0400
//     0x600+4n BQ   RW  adaptation gain, signed, bits [17:0] (reads sign-extended)
//     0x700+4n BIAS RW  b, int32
//     0x800+4n U    R   membrane u (sign-extended)
//     0x900+4n A    R   adaptation trace a
//  (P <= 64, so each per-neuron region fits its 256 bytes.)
//
// Write parameters only while idle (not between a TICK and the end of busy).
// After a TICK the layer is busy for TICK_CYCLES clocks; one EVT may be written
// during that time (it waits in the event register), a second one sets evt_ovf.
//
// NOT modelled here: chaining layer1 -> layer2 (serialise out_spk into events of
// a 2nd snn_layer), delays, recurrence, FIFO on the event port.

// ---------------------------------------------------------------------------
module alif_neuron #(
    parameter integer WBITS = 8,          // weight width (signed)
    parameter integer XBITS = 8           // input width (unsigned)
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     clr,      // synchronous clear (start of a sample)
    input  wire                     syn_en,   // an input event: acc += w*x
    input  wire signed [WBITS-1:0]  w,
    input  wire        [XBITS-1:0]  x,
    input  wire                     t0, t1, t2, t3,   // TICK pipeline stages (one-hot)
    input  wire        [31:0]       np,       // theta[15:0] r[19:16] km[23:20] ka[27:24]
    input  wire signed [17:0]       bq,
    input  wire signed [31:0]       b,
    output wire                     fire,     // valid in t3
    output reg  signed [15:0]       u,
    output reg         [23:0]       a,
    output reg                      s
);
    wire [15:0] theta = np[15:0];
    wire [3:0]  r     = np[19:16];
    wire [3:0]  km    = np[23:20];
    wire [3:0]  ka    = np[27:24];

    function signed [15:0] sat16(input signed [35:0] v);
        sat16 = (v > 36'sd32767) ? 16'sh7FFF : (v < -36'sd32768) ? 16'sh8000 : v[15:0];
    endfunction

    // events: acc += w * x (x unsigned). The product is registered in the event's
    // ACC cycle and added in the next one (always IDLE), so the BRAM -> multiply ->
    // 32-bit add path is split over two clocks; a TICK reads acc after that.
    reg  signed [31:0] acc;
    wire signed [WBITS+XBITS:0] prod = w * $signed({1'b0, x});
    reg  signed [WBITS+XBITS:0] prod_q;
    reg                         add_v;

    // t0: adaptation trace and input current
    wire [23:0]        a_new = a - (a >> ka) + {15'd0, s, 8'd0};
    wire signed [32:0] cur   = (acc >>> r) + b;
    reg  [23:0]        a1;
    reg  signed [32:0] cur1;
    // t1: membrane (leak + current, saturated) and bq * a (one DSP48)
    reg  signed [15:0] u2;
    reg  signed [42:0] pq;
    // t2: threshold, clamped to 18 bits (exact: u is int16)
    wire signed [35:0] thr_w = $signed({20'd0, theta}) + (pq >>> 8);
    reg  signed [17:0] thr3;
    // t3: fire and subtract reset
    assign fire = (u2 >= thr3);
    wire signed [18:0] u_rst = u2 - thr3;

    always @(posedge clk) begin
        if (!rst_n || clr) begin
            acc <= 32'sd0; u <= 16'sd0; a <= 24'd0; s <= 1'b0; add_v <= 1'b0;
        end else begin
            add_v <= syn_en;
            if (syn_en) prod_q <= prod;
            if (add_v)  acc <= acc + prod_q;
            if (t0) begin
                a1   <= a_new;
                cur1 <= cur;
                acc  <= 32'sd0;
            end
            if (t1) begin
                u2 <= sat16($signed({{20{u[15]}}, u}) - $signed({{20{u[15]}}, u >>> km})
                            + $signed({{3{cur1[32]}}, cur1}));
                pq <= bq * $signed({1'b0, a1});
            end
            if (t2)
                thr3 <= (thr_w > 36'sd131071) ? 18'sd131071 :
                        (thr_w < -36'sd131072) ? -18'sd131072 : thr_w[17:0];
            if (t3) begin
                u <= fire ? sat16($signed({{17{u_rst[18]}}, u_rst})) : u2;
                a <= a1;
                s <= fire;
            end
        end
    end
endmodule

// ---------------------------------------------------------------------------
module snn_layer #(
    parameter integer P          = 16,    // physical neurons running in parallel (<= 64)
    parameter integer N_IN       = 256,   // inputs = weight-RAM rows
    parameter integer WBITS      = 8,     // this design assumes 8
    parameter integer XBITS      = 8,     // input value width: 1 = binary spikes, 8 = bytes
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
    localparam [31:0]  NP_INIT = 32'h0033_0400;    // theta 1024, r 0, km 3, ka 3
    localparam [31:0]  CONFIG  = (N_IN << 16) | (XBITS << 8) | P;

    // ---------------- state (declared first) ----------------
    localparam [2:0] S_IDLE = 3'd0, S_ACC = 3'd1, S_T0 = 3'd2, S_T1 = 3'd3, S_T2 = 3'd4, S_T3 = 3'd5;
    reg  [2:0]              st;
    reg                     evt_v, evt_ovf, tick_req, clr_r;
    reg  [AW-1:0]           evt_row;
    reg  [XBITS-1:0]        evt_x, x_q;
    reg  [31:0]             ticks;
    reg  [P-1:0]            spk_last;
    reg  [P*CBITS-1:0]      cnt;
    reg  [32*P-1:0]         np_r;              // per-neuron parameters
    reg  [18*P-1:0]         bq_r;
    reg  [32*P-1:0]         b_r;
    wire [P-1:0]            fire;
    wire [16*P-1:0]         umem;
    wire [24*P-1:0]         amem;
    wire idle   = (st == S_IDLE);
    wire syn_en = (st == S_ACC);
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

    // ---------------- P neurons, all updated in the same cycles ----------------
    genvar n;
    generate
        for (n = 0; n < P; n = n + 1) begin : g_neuron
            alif_neuron #(.WBITS(WBITS), .XBITS(XBITS)) u_n (
                .clk(clk), .rst_n(rst_n), .clr(clr_r),
                .syn_en(syn_en), .w(wrow[n*WBITS +: WBITS]), .x(x_q),
                .t0(st == S_T0), .t1(st == S_T1), .t2(st == S_T2), .t3(st == S_T3),
                .np(np_r[32*n +: 32]), .bq(bq_r[18*n +: 18]), .b(b_r[32*n +: 32]),
                .fire(fire[n]), .u(umem[16*n +: 16]), .a(amem[24*n +: 24]), .s());
        end
    endgenerate

    assign out_spk   = fire;
    assign out_valid = (st == S_T3);

    // ---------------- control FSM + register writes ----------------
    //  IDLE --evt_v--> ACC   (row read issued in IDLE, weights valid in ACC)
    //  IDLE --tick_req--> T0 -> T1 -> T2 -> T3 -> IDLE  (only when no event is pending)
    // 2 cycles per event, TICK_CYCLES = 4 per tick.
    wire [11:0] widx = (reg_waddr[7:0] >> 2);              // neuron of a per-neuron write
    integer j;
    always @(posedge clk) begin
        if (!rst_n) begin
            st <= S_IDLE;  evt_v <= 1'b0;  evt_ovf <= 1'b0;  tick_req <= 1'b0;  clr_r <= 1'b0;
            evt_row <= {AW{1'b0}};  evt_x <= {XBITS{1'b0}};  x_q <= {XBITS{1'b0}};
            ticks <= 32'd0;
            spk_last <= {P{1'b0}};  cnt <= {(P*CBITS){1'b0}};
            for (j = 0; j < P; j = j + 1) begin
                np_r[32*j +: 32] <= NP_INIT;
                bq_r[18*j +: 18] <= 18'sd0;
                b_r[32*j +: 32]  <= 32'sd0;
            end
        end else begin
            clr_r <= 1'b0;

            case (st)
            S_IDLE: if (evt_v) begin
                        st <= S_ACC;  x_q <= evt_x;  evt_v <= 1'b0;
                    end else if (tick_req) begin
                        st <= S_T0; tick_req <= 1'b0;
                    end
            S_ACC:  st <= S_IDLE;
            S_T0:   st <= S_T1;
            S_T1:   st <= S_T2;
            S_T2:   st <= S_T3;
            S_T3: begin
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
                if (reg_waddr == 12'h000) begin
                    if (reg_wdata[0]) begin                     // CLR also aborts a tick
                        clr_r <= 1'b1;  evt_v <= 1'b0;  tick_req <= 1'b0;  evt_ovf <= 1'b0;
                        st <= S_IDLE;
                        ticks <= 32'd0; spk_last <= {P{1'b0}};  cnt <= {(P*CBITS){1'b0}};
                    end
                    if (reg_wdata[1]) tick_req <= 1'b1;
                end else if (reg_waddr == 12'h008) begin
                    if (evt_v && !idle) evt_ovf <= 1'b1;        // previous event not consumed: drop
                    else begin
                        evt_v   <= 1'b1;
                        evt_row <= reg_wdata[AW-1:0];
                        evt_x   <= reg_wdata[16 +: XBITS];
                    end
                end else if (widx < P) begin
                    for (j = 0; j < P; j = j + 1) if (widx == j) begin
                        if (reg_waddr[11:8] == 4'h5) np_r[32*j +: 32] <= reg_wdata;
                        if (reg_waddr[11:8] == 4'h6) bq_r[18*j +: 18] <= reg_wdata[17:0];
                        if (reg_waddr[11:8] == 4'h7) b_r[32*j +: 32]  <= reg_wdata;
                    end
                end
            end
        end
    end

    // ---------------- register reads (combinational, like poisson.v) ----------------
    wire [32*SPKW-1:0] spk_pad = spk_last;                   // zero-extended by assignment
    wire [11:0]        ridx    = (reg_raddr[7:0] >> 2);
    always @(*) begin
        reg_rdata = 32'h0;
        if      (reg_raddr == 12'h000) reg_rdata = {31'b0, busy};
        else if (reg_raddr == 12'h004) reg_rdata = {30'b0, evt_ovf, busy};
        else if (reg_raddr == 12'h00C) reg_rdata = CONFIG;
        else if (reg_raddr >= 12'h010 && reg_raddr < 12'h010 + 4*SPKW)
            reg_rdata = spk_pad[32*((reg_raddr - 12'h010) >> 2) +: 32];
        else if (reg_raddr == 12'h030) reg_rdata = ticks;
        else if (reg_raddr == 12'h034) reg_rdata = 32'h414C_4946;      // "ALIF"
        else if (reg_raddr[11:8] >= 4'h4 && reg_raddr[11:8] <= 4'h9 && ridx < P) begin
            case (reg_raddr[11:8])
            4'h4: reg_rdata = {{(32-CBITS){1'b0}}, cnt[ridx*CBITS +: CBITS]};
            4'h5: reg_rdata = np_r[32*ridx +: 32];
            4'h6: reg_rdata = {{14{bq_r[18*ridx + 17]}}, bq_r[18*ridx +: 18]};
            4'h7: reg_rdata = b_r[32*ridx +: 32];
            4'h8: reg_rdata = {{16{umem[16*ridx + 15]}}, umem[16*ridx +: 16]};
            default: reg_rdata = {8'd0, amem[24*ridx +: 24]};
            endcase
        end
    end
endmodule
