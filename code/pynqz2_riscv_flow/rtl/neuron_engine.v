// neuron_engine.v - event-driven layer 2 and readout of the streaming keyword SNN
// (snn_keyword/IMPLEMENTATION_PLAN.md, Phase 6). Bit-exact with
// model.integer_forward_stream() and firmware/stream_infer.c.
//
// The PicoRV32 computes layer 1 (kdot) and writes the indices of the layer-1
// neurons that spiked this frame to SPIKE, then writes CTRL.start. Per frame:
//   delayed   for tau = 0..31, for every layer-1 neuron that spiked tau frames
//             ago, stream its synapses of delay tau from a CSR table (one
//             synaptic event per clock) into the layer-2 accumulators
//   recurrent for every layer-2 neuron that spiked last frame, add its
//             weight column 16 lanes at a time (one 128-bit BRAM row per clock)
//   update    ALIF update of each layer-2 neuron, pipelined, one per clock:
//             a -= a>>ka; a += s<<8; thr = theta + (bq*a)>>8;
//             u = sat16(u - (u>>km) + (acc<<p) + b); spike if u >= thr; u = sat16(u - thr)
//             and the readout sums of the neurons that spike
//   readout   o = sat32(o - (o>>ko) + (acc_o<<PO) + bo); score = sat32(o[yes] - max o[other])
// Membrane updates are one per clock, not 16 lanes: they are 128 per frame
// against about 1.5k synaptic events, which set the cost. Tables and neuron
// state are synchronous-read RAMs (BRAM / LUTRAM); only the 128 accumulators,
// which need random and 16-wide access, are registers.
//
// MMIO (byte offsets from 0x1000_4000; reads combinational):
//   0x00 W CTRL     bit0 reset network state, bit1 start frame
//   0x04 R STATUS   bit0 busy
//   0x08 W SPIKE    index of a layer-1 neuron that spiked this frame
//   0x0C R SCORE    decision score of the last frame (int32)
//   0x10 R SPIKES2  layer-2 spikes of the last frame
//   0x14 R EVENTS   synaptic additions of the last frame (as stream_infer.c counts them)
//   0x18 R CYCLES   engine clocks of the last frame
//   0x1C R ID       0x4E454E47 "NENG"
//   0x20 W TADDR    table[31:28], address[15:0]: start of a table write burst
//   0x24 W TDATA    one table word; the address then increments
//   0x28 W CONFIG   yes[1:0], classes[6:4], PO[11:8]
// Tables: 0 OFF (16 bit, N1*32+1), 1 SYN ({w[15:8], post[7:0]}), 2 REC (int8
// rec[j][k] at byte k*N2 + j, four per word), 3 WOT (per j: int8 per class),
// 4 NP0 (per j: theta[15:0] | p[19:16] | km[23:20] | ka[27:24]), 5 BQ, 6 B2,
// 7 KO (per class), 8 BO (per class).
module neuron_engine #(
    parameter N1      = 128,
    parameter N2      = 128,
    parameter MAX_SYN = 16384
) (
    input  wire        clk,
    input  wire        resetn,
    input  wire        wr,
    input  wire [7:0]  addr,
    input  wire [31:0] wdata,
    input  wire [7:0]  raddr,
    output reg  [31:0] rdata
);
    localparam ROWS = N2 / 16;

    // ------------------------------------------------------------------
    // Table interface
    // ------------------------------------------------------------------
    reg [3:0]  tsel;
    reg [15:0] taddr;
    wire       twr = wr && addr == 8'h24;
    always @(posedge clk) begin
        if (wr && addr == 8'h20) begin
            tsel <= wdata[31:28]; taddr <= wdata[15:0];
        end else if (twr) begin
            taddr <= taddr + 16'd1;
        end
    end
    reg [1:0]  yes;
    reg [2:0]  classes;
    reg [3:0]  po;
    reg [3:0]  ko [0:3];
    reg signed [31:0] bo [0:3];
    always @(posedge clk) begin
        if (wr && addr == 8'h28) begin yes <= wdata[1:0]; classes <= wdata[6:4]; po <= wdata[11:8]; end
        if (twr && tsel == 4'd7) ko[taddr[1:0]] <= wdata[3:0];
        if (twr && tsel == 4'd8) bo[taddr[1:0]] <= wdata;
    end

    // ------------------------------------------------------------------
    // Sequencer registers
    // ------------------------------------------------------------------
    localparam S_IDLE = 4'd0, S_DSLOT = 4'd1, S_DIDX = 4'd2, S_DOFF = 4'd3, S_DSYN = 4'd4,
               S_REC = 4'd5, S_UPD = 4'd6, S_OUT1 = 4'd7, S_OUT2 = 4'd8, S_SC1 = 4'd9,
               S_SC2 = 4'd10, S_SC3 = 4'd11, S_RESET = 4'd12, S_SC1B = 4'd13, S_SC1C = 4'd14;
    reg [3:0]  state;
    reg        busy;
    reg [4:0]  pos;
    reg [5:0]  tau;
    reg [7:0]  n;
    reg [15:0] e, e_end;
    reg        v;
    reg [10:0] row;
    reg [2:0]  row_blk;
    reg [7:0]  j;                     // update issue index / reset counter
    reg [7:0]  new_cnt, s2_cnt;
    reg [7:0]  ring_cnt [0:31];
    reg [6:0]  s2_list [0:N2-1];
    wire [4:0] slot = pos - tau[4:0];

    // ------------------------------------------------------------------
    // Synchronous-read memories (address from this cycle, data next cycle)
    // ------------------------------------------------------------------
    reg [15:0]  off_mem [0:N1*32];
    reg [15:0]  syn_mem [0:MAX_SYN-1];
    reg [127:0] rec_mem [0:N2*ROWS-1];
    reg [6:0]   ring_mem [0:32*N1-1];
    reg [6:0]   ring_q;
    reg [15:0]  offa_q, offb_q, syn_q;
    reg [127:0] rec_q;
    wire [12:0] off_k  = {ring_q, tau[4:0]};
    wire [15:0] syn_rd = (state == S_DOFF) ? offa_q : e;
    always @(posedge clk) begin
        if (twr && tsel == 4'd0) off_mem[taddr[12:0]] <= wdata[15:0];
        offa_q <= off_mem[off_k];
    end
    always @(posedge clk) offb_q <= off_mem[off_k + 13'd1];
    always @(posedge clk) begin
        if (twr && tsel == 4'd1) syn_mem[taddr[13:0]] <= wdata[15:0];
        syn_q <= syn_mem[syn_rd[13:0]];
    end
    // 128-bit recurrent rows written 32 bits at a time: four 32-bit-wide RAMs.
    genvar g;
    generate for (g = 0; g < 4; g = g + 1) begin : rec_bank
        reg [31:0] mem [0:N2*ROWS-1];
        always @(posedge clk) begin
            if (twr && tsel == 4'd2 && taddr[1:0] == g) mem[taddr[12:2]] <= wdata;
            rec_q[32*g +: 32] <= mem[{s2_list[row[10:3]], row[2:0]}];
        end
    end endgenerate
    always @(posedge clk) begin
        if (wr && addr == 8'h08 && !busy) ring_mem[{pos, ring_cnt[pos][6:0]}] <= wdata[6:0];
        ring_q <= ring_mem[{slot, n[6:0]}];
    end

    // Per-neuron parameters: written by the table port, read at the issue index j.
    reg [31:0] np0_mem [0:N2-1], bq_mem [0:N2-1], b2_mem [0:N2-1], wot_mem [0:N2-1];
    reg [31:0] np0_q, bq_q, b2_q;
    wire [6:0] j_rd = j[6:0];
    always @(posedge clk) begin
        if (twr && tsel == 4'd4) np0_mem[taddr[6:0]] <= wdata;
        np0_q <= np0_mem[j_rd];
    end
    always @(posedge clk) begin
        if (twr && tsel == 4'd5) bq_mem[taddr[6:0]] <= wdata;
        bq_q <= bq_mem[j_rd];
    end
    always @(posedge clk) begin
        if (twr && tsel == 4'd6) b2_mem[taddr[6:0]] <= wdata;
        b2_q <= b2_mem[j_rd];
    end
    // Readout weights: read at stage 3's neuron, used the next clock if it fired.
    reg [31:0] wot_q;
    reg [6:0]  p3_j;
    always @(posedge clk) begin
        if (twr && tsel == 4'd3) wot_mem[taddr[6:0]] <= wdata;
        wot_q <= wot_mem[p3_j];
    end

    // Neuron state: read at issue, written at write-back or by the reset sweep.
    reg signed [15:0] u_mem [0:N2-1];
    reg [31:0] a_mem [0:N2-1];
    reg        s_mem [0:N2-1];
    reg        st_we;
    reg [6:0]  st_wa;
    reg signed [15:0] st_u;
    reg [31:0] st_a;
    reg        st_s;
    reg signed [15:0] u_q;
    reg [31:0] a_q;
    reg        s_q;
    always @(posedge clk) begin
        if (st_we) begin u_mem[st_wa] <= st_u; a_mem[st_wa] <= st_a; s_mem[st_wa] <= st_s; end
        u_q <= u_mem[j_rd]; a_q <= a_mem[j_rd]; s_q <= s_mem[j_rd];
    end

    // ------------------------------------------------------------------
    // Accumulators: random (delayed synapses), 16-wide (recurrence), read+clear (update)
    // ------------------------------------------------------------------
    reg signed [17:0] acc [0:N2-1];
    reg signed [17:0] acc_q;
    // Synaptic events, three stages after the BRAM: ev registers the synapse word;
    // stage A reads acc[post], or the sum that stage B writes to the same neuron in
    // this clock (forwarding, so back-to-back events to one neuron stay exact);
    // stage B adds the weight and writes. The read multiplexer and the adder with
    // the write decode are in different clocks (one clock was 11 logic levels,
    // JOURNAL entry 19).
    reg        ev_v, evb_v;
    reg [15:0] ev_syn;
    reg [6:0]  evb_j;
    reg signed [17:0] evb_acc;
    reg signed [7:0]  evb_w;
    wire signed [17:0] evb_sum = evb_acc + evb_w;

    // ------------------------------------------------------------------
    // Update pipeline
    //   issue (S_UPD, j):   RAM addresses; acc_q <= acc[j]; acc[j] <= 0
    //   stage 1 (p1):       a_new, current
    //   stage 2 (p2):       bq * a_new (DSP), leaked membrane
    //   stage 3 (p3):       threshold, saturated membrane
    //   stage 4 (p4):       fire, reset, write back, readout sums
    // ------------------------------------------------------------------
    reg        i_v; reg [6:0] i_j;               // issued last cycle: RAM data valid now
    reg        p1_v; reg [6:0] p1_j; reg [31:0] p1_a; reg signed [39:0] p1_cur;
    reg signed [15:0] p1_u; reg [15:0] p1_theta; reg [3:0] p1_km; reg signed [31:0] p1_bq;
    reg        p2_v; reg [6:0] p2_j; reg [31:0] p2_a; reg signed [47:0] p2_prod;
    reg signed [39:0] p2_u; reg [15:0] p2_theta;
    reg        p3_v; reg [31:0] p3_a; reg signed [39:0] p3_thr, p3_u;
    wire [3:0] i_ka = np0_q[27:24];
    wire [31:0] i_anew = a_q - (a_q >> i_ka) + {23'd0, s_q, 8'd0};
    wire signed [39:0] i_cur = ($signed({{22{acc_q[17]}}, acc_q}) <<< np0_q[19:16])
                             + $signed({{8{b2_q[31]}}, b2_q});
    wire        p3_fire = p3_u >= p3_thr;
    wire signed [39:0] p3_rst = p3_u - p3_thr;
    wire signed [15:0] p3_fin = !p3_fire ? p3_u[15:0] :
                                (p3_rst > 40'sd32767) ? 16'sd32767 :
                                (p3_rst < -40'sd32768) ? -16'sd32768 : p3_rst[15:0];

    // ------------------------------------------------------------------
    // Readout
    // ------------------------------------------------------------------
    reg signed [31:0] o [0:3];
    reg signed [31:0] acc_o [0:3];
    reg signed [47:0] ro_a [0:3], ro_b [0:3];
    reg signed [31:0] best_other, o_yes, m01, m23;
    reg signed [31:0] vo [0:3];
    reg signed [47:0] diff;
    reg signed [31:0] score;
    reg [31:0] spikes2, events, cycles;
    reg        wb_fire;                          // stage 4 fired last cycle: add wot_q now
    function signed [31:0] sat32(input signed [47:0] x);
        sat32 = (x > 48'sd2147483647) ? 32'sh7fffffff : (x < -48'sd2147483648) ? 32'sh80000000 : x[31:0];
    endfunction

    always @* begin
        case (raddr)
            8'h04: rdata = {31'd0, busy};
            8'h0c: rdata = score;
            8'h10: rdata = spikes2;
            8'h14: rdata = events;
            8'h18: rdata = cycles;
            8'h1c: rdata = 32'h4E454E47;
            default: rdata = 32'd0;
        endcase
    end

    integer l, c;
    always @(posedge clk) begin
        st_we <= 1'b0;
        if (!resetn) begin
            state <= S_RESET; busy <= 1'b1; j <= 8'd0;
            i_v <= 1'b0; p1_v <= 1'b0; p2_v <= 1'b0; p3_v <= 1'b0; wb_fire <= 1'b0; ev_v <= 1'b0; evb_v <= 1'b0;
        end else begin
            if (busy) cycles <= cycles + 32'd1;

            // ---- update pipeline (runs whenever stages hold valid data) ----
            p1_v <= i_v; p1_j <= i_j; p1_a <= i_anew; p1_cur <= i_cur; p1_u <= u_q;
            p1_theta <= np0_q[15:0]; p1_km <= np0_q[23:20]; p1_bq <= bq_q;
            p2_v <= p1_v; p2_j <= p1_j; p2_a <= p1_a; p2_theta <= p1_theta;
            p2_prod <= $signed({{16{p1_bq[31]}}, p1_bq}) * $signed({16'd0, p1_a});
            p2_u <= $signed({{24{p1_u[15]}}, p1_u}) - $signed({{24{p1_u[15]}}, p1_u >>> p1_km}) + p1_cur;
            p3_v <= p2_v; p3_j <= p2_j; p3_a <= p2_a;
            p3_thr <= $signed({24'd0, p2_theta}) + (p2_prod >>> 8);
            p3_u <= (p2_u > 40'sd32767) ? 40'sd32767 : (p2_u < -40'sd32768) ? -40'sd32768 : p2_u;
            wb_fire <= 1'b0;
            if (p3_v) begin
                st_we <= 1'b1; st_wa <= p3_j; st_u <= p3_fin; st_a <= p3_a; st_s <= p3_fire;
                if (p3_fire) begin
                    s2_list[new_cnt[6:0]] <= p3_j;
                    new_cnt <= new_cnt + 8'd1;
                    wb_fire <= 1'b1;
                    events <= events + classes;
                end
            end
            if (wb_fire)
                for (c = 0; c < 4; c = c + 1) acc_o[c] <= acc_o[c] + $signed(wot_q[8*c +: 8]);
            i_v <= 1'b0;
            ev_v <= (state == S_DSYN) && v; ev_syn <= syn_q;
            evb_v <= ev_v;
            if (ev_v) begin
                evb_j <= ev_syn[6:0]; evb_w <= ev_syn[15:8];
                evb_acc <= (evb_v && evb_j == ev_syn[6:0]) ? evb_sum : acc[ev_syn[6:0]];
                events <= events + 32'd1;
            end
            if (evb_v) acc[evb_j] <= evb_sum;

            case (state)
            S_RESET: begin
                // One neuron and one ring slot per clock.
                st_we <= 1'b1; st_wa <= j[6:0]; st_u <= 16'sd0; st_a <= 32'd0; st_s <= 1'b0;
                acc[j[6:0]] <= 18'sd0;
                if (j < 8'd32) ring_cnt[j[4:0]] <= 8'd0;
                if (j < 8'd4) begin o[j[1:0]] <= 32'sd0; acc_o[j[1:0]] <= 32'sd0; end
                if (j == N2 - 1) begin
                    state <= S_IDLE; busy <= 1'b0; pos <= 5'd0; s2_cnt <= 8'd0;
                end
                j <= j + 8'd1;
            end
            S_IDLE: begin
                if (wr && addr == 8'h08) ring_cnt[pos] <= ring_cnt[pos] + 8'd1;
                if (wr && addr == 8'h00 && wdata[0]) begin
                    state <= S_RESET; busy <= 1'b1; j <= 8'd0;
                end else if (wr && addr == 8'h00 && wdata[1]) begin
                    state <= S_DSLOT; busy <= 1'b1; cycles <= 32'd0;
                    events <= 32'd0; tau <= 6'd0; n <= 8'd0;
                end
            end
            // ---- delayed synapses, one event per clock ----
            S_DSLOT: begin
                if (tau == 6'd32) begin
                    if (!ev_v && !evb_v) begin state <= S_REC; row <= 11'd0; v <= 1'b0; end
                end else if (n >= ring_cnt[slot]) begin
                    tau <= tau + 6'd1; n <= 8'd0;
                end else begin
                    state <= S_DIDX;                 // ring_q <= ring_mem[slot, n]
                end
            end
            S_DIDX: state <= S_DOFF;                  // offa_q, offb_q <= off_mem[ring_q, tau (+1)]
            S_DOFF: begin
                if (offa_q == offb_q) begin
                    n <= n + 8'd1; state <= S_DSLOT;
                end else begin                        // syn_q <= syn_mem[offa_q]
                    e <= offa_q + 16'd1; e_end <= offb_q; v <= 1'b1; state <= S_DSYN;
                end
            end
            S_DSYN: begin
                // syn_q (valid when v) goes to the event stage above.
                if (e < e_end) begin
                    e <= e + 16'd1; v <= 1'b1;
                end else begin
                    v <= 1'b0; n <= n + 8'd1; state <= S_DSLOT;
                end
            end
            // ---- recurrence: last frame's layer-2 spikes, 16 lanes per clock ----
            S_REC: begin
                if (v) begin
                    for (l = 0; l < 16; l = l + 1)
                        acc[{row_blk, 4'd0} + l] <= acc[{row_blk, 4'd0} + l] + $signed(rec_q[8*l +: 8]);
                    if (row_blk == 3'd7) events <= events + N2;
                end
                if (row < {s2_cnt, 3'd0}) begin    // rec_q <= rec row (s2_list[row/8], row%8)
                    row_blk <= row[2:0]; row <= row + 11'd1; v <= 1'b1;
                end else begin
                    v <= 1'b0; state <= S_UPD; j <= 8'd0; new_cnt <= 8'd0;
                end
            end
            // ---- membrane update issue, one neuron per clock ----
            S_UPD: begin
                if (j < N2) begin
                    i_v <= 1'b1; i_j <= j[6:0];
                    acc_q <= acc[j[6:0]];
                    acc[j[6:0]] <= 18'sd0;
                    j <= j + 8'd1;
                end else if (!i_v && !p1_v && !p2_v && !p3_v && !wb_fire) begin
                    state <= S_OUT1;
                end
            end
            // ---- readout and score, split over clocks ----
            S_OUT1: begin
                for (c = 0; c < 4; c = c + 1) begin
                    ro_a[c] <= $signed({{16{o[c][31]}}, o[c]}) - $signed({{16{o[c][31]}}, o[c] >>> ko[c]});
                    ro_b[c] <= ($signed({{16{acc_o[c][31]}}, acc_o[c]}) <<< po) + $signed({{16{bo[c][31]}}, bo[c]});
                end
                state <= S_OUT2;
            end
            S_OUT2: begin
                for (c = 0; c < 4; c = c + 1)
                    if (c < classes) o[c] <= sat32(ro_a[c] + ro_b[c]);
                state <= S_SC1;
            end
            S_SC1: begin
                for (c = 0; c < 4; c = c + 1)
                    vo[c] <= (c < classes && c != yes) ? o[c] : 32'sh80000000;
                o_yes <= o[yes];
                state <= S_SC1B;
            end
            S_SC1B: begin
                m01 <= (vo[0] > vo[1]) ? vo[0] : vo[1];
                m23 <= (vo[2] > vo[3]) ? vo[2] : vo[3];
                state <= S_SC1C;
            end
            S_SC1C: begin
                best_other <= (m01 > m23) ? m01 : m23;
                state <= S_SC2;
            end
            S_SC2: begin
                diff <= $signed({{16{o_yes[31]}}, o_yes}) - $signed({{16{best_other[31]}}, best_other});
                state <= S_SC3;
            end
            S_SC3: begin
                score <= sat32(diff);
                spikes2 <= {24'd0, new_cnt}; s2_cnt <= new_cnt;
                for (c = 0; c < 4; c = c + 1) acc_o[c] <= 32'sd0;
                ring_cnt[pos + 5'd1] <= 8'd0;       // that slot expires; the next frame writes it
                pos <= pos + 5'd1;
                state <= S_IDLE; busy <= 1'b0;
            end
            default: state <= S_IDLE;
            endcase
        end
    end
endmodule
