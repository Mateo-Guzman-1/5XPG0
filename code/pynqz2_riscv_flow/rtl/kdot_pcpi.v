// kdot_pcpi.v — PicoRV32 PCPI coprocessor: streaming int16 x uint8 dot product.
//
// Custom-0 opcode (0001011), R-type:
//   funct3=1  klen x0, rs1, x0   length register <= rs1 (elements, multiple of 4)
//   funct3=0  kdot rd, rs1, rs2  rd <= sum_{i<len} int16 w[i] * uint8 x[i]
//                                w at byte address rs1, x at byte address rs2,
//                                both word-aligned in BRAM; 32-bit wrap-around,
//                                identical to the C int32 accumulation.
// KX extension (parameter KX = 1; KX = 0 removes it and these encodings are
// not claimed, so they trap as illegal instructions):
//   funct3=2  kdotc rd, rs1, rs2 rd <= sum_{i<len} int16 w[i] * (uint8 x[i] - 128)
//                                = kdot - 128 * sum(w), the corrected dot product
//                                of firmware/verifier.c; same operands and timing
//                                as kdot (the multiplier takes x ^ 0x80 as int8).
//   funct3=3  kload x0, rs1, x0  activation buffer[0 .. len) <= int16 w[0 .. len) at
//                                byte address rs1 (one BRAM word per cycle)
//   funct3=4  kdotb rd, rs1, x0  rd <= sum_{i<len} buffer[4*off + i] * (uint8 x[i] - 128),
//                                x at byte address rs1: only the x words are read,
//                                4 MACs per cycle (1 group of 4 elements per cycle)
//   funct3=5  kset x0, rs1, x0   setup register funct7 <= rs1:
//                                funct7=0 off   (buffer offset in groups of 4)
//                                funct7=1 ebase (byte address of the uint8 row exponents e[])
//                                funct7=2 bbase (byte address of the int32 row biases b[])
//                                funct7=3 sbase (byte address of the 512 x int16 sigmoid table)
//                                funct7=4 tbase (byte address of the 512 x int16 tanh table)
//                                funct7=5 rptr  (byte address of the next weight row)
//                                funct7=6 ridx  ({bi[15:0], ei[15:0]} of the next row)
//                                funct7=7 rstep (row stride in bytes)
//   funct3=6  kpre rd, rs1, rs2  funct7=0: rd <= P = (kdotb(rs1) >>> e[ei]) + b[bi]
//             ksig rd, rs1, rs2  funct7=1: rd <= sig(P)
//                                rs2 = {bi[15:0], ei[15:0]}; the unit reads e[ei] and b[bi]
//                                itself (2 extra BRAM reads, overlapped with the row)
//             kpren rd           funct7=2: kpre with rs1 = rptr, rs2 = ridx, then rptr += rstep,
//             ksign rd           funct7=3: ksig likewise        ei += 1, bi += 1 (rows contiguous)
//   funct3=7  ktanh rd, rs1, x0  funct7=0: rd <= tnh(rs1)
//   sig(x) = table_s[(clamp(x, -8192, 8191) + 8192) >> 5], tnh(x) = table_t[(clamp(x, -4096, 4095)
//   + 4096) >> 4], the int16 entry sign-extended (firmware/verifier.c sigmoid_q / tanh_q);
//   >>> is arithmetic, by e[4:0] (as RV32 sra). 32-bit wrap-around everywhere.
// len is the klen register in every case; the buffer holds BUF_GROUPS groups of 4.
// Latency (cycles from issue, mem_idle permitting): kdot/kdotc 3 per group + 9; kload 2 per
// group + 5; kdotb len/4 + 8; kpre(n) len/4 + 12; ksig(n) len/4 + 15; ktanh 6; klen/kset 2.
//
// While kdot runs the CPU is stalled waiting on PCPI, so the unit borrows the
// CPU's BRAM port (mem_busy). It waits for mem_idle (no CPU transfer in
// flight, e.g. an instruction prefetch) before taking the port. Per 4
// elements it reads one input word and two weight words: 3 cycles per 4 MACs.
// mem_addr is registered and the BRAM read is registered, so data for an
// address issued in cycle c is on mem_rdata in cycle c+2 (tag_i -> tag_d).

module kdot_pcpi #(
    parameter KX         = 1,           // 1: KX instructions (funct3 >= 2), 0: kdot/klen only
    parameter BUF_GROUPS = 32           // activation buffer, groups of 4 int16 (128 = the longest
                                        // verifier row, VERIFIER_H1 + VERIFIER_H2)
) (
    input  wire        clk,
    input  wire        resetn,

    input  wire        pcpi_valid,
    input  wire [31:0] pcpi_insn,
    input  wire [31:0] pcpi_rs1,
    input  wire [31:0] pcpi_rs2,
    output reg         pcpi_wr,
    output reg  [31:0] pcpi_rd,
    output wire        pcpi_wait,
    output reg         pcpi_ready,

    input  wire        mem_idle,        // CPU bus has no transfer in flight
    output reg         mem_busy,        // 1 = unit owns the BRAM port
    output reg  [15:0] mem_addr,        // word address
    input  wire [31:0] mem_rdata        // BRAM data for the previous mem_addr
);
    wire is_custom0 = pcpi_insn[6:0] == 7'b0001011 && pcpi_insn[31:25] == 7'd0;
    wire is_kdot    = is_custom0 && pcpi_insn[14:12] == 3'd0;
    wire is_klen    = is_custom0 && pcpi_insn[14:12] == 3'd1;
    wire is_kdotc   = KX && is_custom0 && pcpi_insn[14:12] == 3'd2;
    wire is_kload   = KX && is_custom0 && pcpi_insn[14:12] == 3'd3;
    wire is_kdotb   = KX && is_custom0 && pcpi_insn[14:12] == 3'd4;
    wire is_c0      = pcpi_insn[6:0] == 7'b0001011;
    wire [6:0] f7   = pcpi_insn[31:25];
    wire is_kset    = KX && is_c0 && pcpi_insn[14:12] == 3'd5 && f7 <= 7'd7;
    wire is_kpost   = KX && is_c0 && pcpi_insn[14:12] == 3'd6 && f7 <= 7'd3;
    wire is_ktanh   = KX && is_custom0 && pcpi_insn[14:12] == 3'd7;
    // Assert wait immediately so the core's 16-cycle PCPI timeout never fires.
    assign pcpi_wait = pcpi_valid && (is_kdot || is_klen || is_kdotc || is_kload || is_kdotb || is_kset ||
                                      is_kpost || is_ktanh);

    localparam [3:0] S_IDLE = 4'd0, S_BUS = 4'd1, S_RUN = 4'd2, S_DRAIN = 4'd3, S_DONE = 4'd4,
                     S_LOAD = 4'd5, S_RUNB = 4'd6, S_PREF = 4'd7, S_POST1 = 4'd8, S_POST2 = 4'd9,
                     S_TISS = 4'd10, S_TWAIT = 4'd11;
    localparam [2:0] OP_DOT = 3'd0, OP_LOAD = 3'd1, OP_DOTB = 3'd2, OP_POST = 3'd3, OP_TANH = 3'd4;
    localparam BW = $clog2(BUF_GROUPS);
    reg [3:0]  state;
    reg [2:0]  op;                     // S_BUS target: OP_*
    reg [BW-1:0] off;                  // kset 0: buffer offset (groups)
    reg [17:0] ebase, bbase, sbase, tbase;    // kset 1..4: byte addresses
    reg [17:0] rptr, rstep;                   // kset 5, 7: next row, row stride (bytes)
    reg [15:0] r_ei, r_bi;                    // kset 6: indices of the next row
    wire       pnext = f7[1];                 // kpren / ksign: operands from rptr / ridx
    wire [17:0] p_row = pnext ? rptr : pcpi_rs1[17:0];
    wire [15:0] p_ei  = pnext ? r_ei : pcpi_rs2[15:0];
    wire [15:0] p_bi  = pnext ? r_bi : pcpi_rs2[31:16];
    reg        sig;                    // kpost: 1 = ksig
    reg [17:0] e_ba;                   // kpost: byte address of e[ei]
    reg [17:0] b_ba;                   // kpost: byte address of b[bi]
    reg [4:0]  e_q;                    // e[ei][4:0]
    reg [31:0] b_q;
    reg [31:0] pq;                     // P, or the ktanh operand
    reg        ttanh;                  // table step: 1 = tanh, 0 = sigmoid
    reg [17:0] ta;                     // table byte address
    reg [1:0]  twait;
    reg [BW:0]   bidx;                 // kload: next buffer word (2 per group); kdotb: next group
    reg [13:0] len_groups;             // length / 4
    reg [13:0] groups_left;
    reg [15:0] w_ptr, x_ptr;
    reg [1:0]  phase;                  // address issue: 0 = x word, 1 = w lo, 2 = w hi
    reg [2:0]  drain;

    // Pipeline: tag of the word arriving this cycle, then multiply, then accumulate.
    reg [2:0]  tag_i, tag_d;           // 0 none, 1 x, 2 w lo, 3 w hi, 4 kload word, 5 kdotb x word
    reg [BW:0] bi_i, bi_d;             // buffer index travelling with tags 4 and 5
    reg [31:0] x_q;
    reg [31:0] w_m;
    reg [15:0] x_m;
    reg        m_valid;
    reg signed [25:0] p_q;
    reg        p_valid;
    reg [31:0] acc;
    reg        corr;                   // kdotc: x operand is x - 128 (int8), else uint8

    // 9-bit signed x operand: uint8 x, or x - 128 = (x ^ 0x80) as int8, sign-extended.
    wire signed [8:0] xs0 = (KX && corr) ? {~x_m[7],  ~x_m[7],  x_m[6:0]}  : {1'b0, x_m[7:0]};
    wire signed [8:0] xs1 = (KX && corr) ? {~x_m[15], ~x_m[15], x_m[14:8]} : {1'b0, x_m[15:8]};
    wire signed [24:0] prod0 = $signed(w_m[15:0])  * xs0;
    wire signed [24:0] prod1 = $signed(w_m[31:16]) * xs1;

    // Activation buffer (LUTRAM): lo = elements 0,1 of a group, hi = elements 2,3.
    (* ram_style = "distributed" *) reg [31:0] buf_lo [0:BUF_GROUPS-1];
    (* ram_style = "distributed" *) reg [31:0] buf_hi [0:BUF_GROUPS-1];
    // kdotb pipeline: capture (x word + buffer group), 4 products, sum of 4, accumulate.
    reg        b_cap;
    reg [31:0] bx_q;
    reg [63:0] ba_q;
    reg        b_mul;
    reg signed [24:0] bp0, bp1, bp2, bp3;
    reg        b_sum;
    reg signed [26:0] bs_q;
    function signed [8:0] xm128(input [7:0] x);       // uint8 x - 128 as int9
        xm128 = {~x[7], ~x[7], x[6:0]};
    endfunction
    // Table index: sigmoid (clamp(x, -8192, 8191) + 8192) >> 5, tanh (clamp(x, -4096, 4095) + 4096) >> 4.
    wire signed [31:0] pqs = pq;
    wire [13:0] sig_u  = (pqs < -32'sd8192) ? 14'd0 : (pqs > 32'sd8191) ? 14'd16383 : pq[13:0] + 14'd8192;
    wire [12:0] tanh_u = (pqs < -32'sd4096) ? 13'd0 : (pqs > 32'sd4095) ? 13'd8191 : pq[12:0] + 13'd4096;
    wire [8:0]  tidx   = ttanh ? tanh_u[12:4] : sig_u[13:5];
    wire [15:0] thalf  = ta[1] ? mem_rdata[31:16] : mem_rdata[15:0];

    always @(posedge clk) begin
        pcpi_ready <= 1'b0;
        pcpi_wr    <= 1'b0;
        // Data / arithmetic pipeline (runs every cycle).
        m_valid <= 1'b0;
        b_cap <= 1'b0;
        case (tag_d)
        3'd1: x_q <= mem_rdata;
        3'd2: begin w_m <= mem_rdata; x_m <= x_q[15:0];  m_valid <= 1'b1; end
        3'd3: begin w_m <= mem_rdata; x_m <= x_q[31:16]; m_valid <= 1'b1; end
        3'd4: if (KX) begin
                  if (bi_d[0]) buf_hi[bi_d[BW:1]] <= mem_rdata;
                  else         buf_lo[bi_d[BW:1]] <= mem_rdata;
              end
        3'd5: if (KX) begin
                  bx_q  <= mem_rdata;
                  ba_q  <= {buf_hi[bi_d[BW-1:0]], buf_lo[bi_d[BW-1:0]]};
                  b_cap <= 1'b1;
              end
        3'd6: if (KX) e_q <= mem_rdata[8 * e_ba[1:0] +: 5];       // e[ei], byte lane of its address
        3'd7: if (KX) b_q <= mem_rdata;                            // b[bi]
        default: ;
        endcase
        p_valid <= m_valid;
        p_q     <= prod0 + prod1;
        if (KX) begin                  // KX = 0: no kdotb pipeline (and no DSPs for it)
            b_mul <= b_cap;
            bp0   <= $signed(ba_q[15:0])  * xm128(bx_q[7:0]);
            bp1   <= $signed(ba_q[31:16]) * xm128(bx_q[15:8]);
            bp2   <= $signed(ba_q[47:32]) * xm128(bx_q[23:16]);
            bp3   <= $signed(ba_q[63:48]) * xm128(bx_q[31:24]);
            b_sum <= b_mul;
            bs_q  <= (bp0 + bp1) + (bp2 + bp3);
        end
        if (p_valid)          acc <= acc + {{6{p_q[25]}}, p_q};
        else if (KX && b_sum) acc <= acc + {{5{bs_q[26]}}, bs_q};
        tag_d <= tag_i;
        tag_i <= 3'd0;
        bi_d  <= bi_i;

        if (!resetn) begin
            state      <= S_IDLE;
            mem_busy   <= 1'b0;
            len_groups <= 14'd0;
            tag_i      <= 3'd0;
            tag_d      <= 3'd0;
            m_valid    <= 1'b0;
            p_valid    <= 1'b0;
            b_cap      <= 1'b0;
            b_mul      <= 1'b0;
            b_sum      <= 1'b0;
            off        <= {BW{1'b0}};
            ebase <= 18'd0; bbase <= 18'd0; sbase <= 18'd0; tbase <= 18'd0;
            rptr <= 18'd0; rstep <= 18'd0; r_ei <= 16'd0; r_bi <= 16'd0;
        end else case (state)
        S_IDLE:
            if (pcpi_valid && is_klen) begin
                len_groups <= pcpi_rs1[15:2];
                pcpi_ready <= 1'b1;
                state      <= S_DONE;
            end else if (pcpi_valid && (is_kdot || is_kdotc)) begin
                corr        <= is_kdotc;
                op          <= OP_DOT;
                w_ptr       <= pcpi_rs1[17:2];
                x_ptr       <= pcpi_rs2[17:2];
                groups_left <= len_groups;
                phase       <= 2'd0;
                acc         <= 32'd0;
                state       <= S_BUS;
            end else if (pcpi_valid && is_kset) begin
                case (f7[2:0])
                3'd0: off   <= pcpi_rs1[BW-1:0];
                3'd1: ebase <= pcpi_rs1[17:0];
                3'd2: bbase <= pcpi_rs1[17:0];
                3'd3: sbase <= pcpi_rs1[17:0];
                3'd4: tbase <= pcpi_rs1[17:0];
                3'd5: rptr  <= pcpi_rs1[17:0];
                3'd6: begin r_ei <= pcpi_rs1[15:0]; r_bi <= pcpi_rs1[31:16]; end
                default: rstep <= pcpi_rs1[17:0];
                endcase
                pcpi_ready <= 1'b1;
                state      <= S_DONE;
            end else if (pcpi_valid && is_kpost) begin
                op          <= OP_POST;
                sig         <= f7[0];
                x_ptr       <= p_row[17:2];
                e_ba        <= ebase + {2'd0, p_ei};
                b_ba        <= bbase + {p_bi, 2'd0};
                if (pnext) begin
                    rptr <= rptr + rstep;
                    r_ei <= r_ei + 16'd1;
                    r_bi <= r_bi + 16'd1;
                end
                groups_left <= len_groups;
                bidx        <= {1'b0, off};
                phase       <= 2'd0;
                acc         <= 32'd0;
                state       <= S_BUS;
            end else if (pcpi_valid && is_ktanh) begin
                op          <= OP_TANH;
                pq          <= pcpi_rs1;
                ttanh       <= 1'b1;
                state       <= S_BUS;
            end else if (pcpi_valid && is_kload) begin
                op          <= OP_LOAD;
                w_ptr       <= pcpi_rs1[17:2];
                groups_left <= len_groups;          // 2 words per group
                bidx        <= {(BW+1){1'b0}};
                phase       <= 2'd0;
                state       <= S_BUS;
            end else if (pcpi_valid && is_kdotb) begin
                op          <= OP_DOTB;
                x_ptr       <= pcpi_rs1[17:2];
                groups_left <= len_groups;
                bidx        <= {1'b0, off};
                acc         <= 32'd0;
                state       <= S_BUS;
            end
        S_BUS:
            if (op == OP_TANH || op == OP_POST) begin
                if (mem_idle) begin
                    mem_busy <= 1'b1;
                    state    <= op == OP_TANH ? S_POST2 : S_PREF;
                end
            end else if (groups_left == 0) begin
                state <= S_DRAIN; drain <= 3'd0;
            end else if (mem_idle) begin
                mem_busy <= 1'b1;
                state    <= op == OP_LOAD ? S_LOAD : op == OP_DOTB ? S_RUNB : S_RUN;
            end
        S_PREF: if (KX) begin
            // kpost: read e[ei] (tag 6), then b[bi] (tag 7), then the row as kdotb.
            if (phase == 2'd0) begin
                mem_addr <= e_ba[17:2]; tag_i <= 3'd6; phase <= 2'd1;
            end else begin
                mem_addr <= b_ba[17:2]; tag_i <= 3'd7; phase <= 2'd0;
                if (groups_left == 0) begin state <= S_DRAIN; drain <= 3'd5; end
                else state <= S_RUNB;
            end
        end
        S_POST1: if (KX) begin
            pq    <= $signed($signed(acc) >>> e_q) + $signed(b_q);   // both signed: >>> stays arithmetic
            ttanh <= 1'b0;
            state <= S_POST2;
        end
        S_POST2: if (KX) begin
            if (op == OP_POST && !sig) begin
                mem_busy   <= 1'b0;
                pcpi_rd    <= pq;
                pcpi_wr    <= 1'b1;
                pcpi_ready <= 1'b1;
                state      <= S_DONE;
            end else begin
                ta    <= (ttanh ? tbase : sbase) + {8'd0, tidx, 1'b0};
                state <= S_TISS;
            end
        end
        S_TISS: if (KX) begin
            mem_addr <= ta[17:2]; twait <= 2'd2; state <= S_TWAIT;
        end
        S_TWAIT: if (KX) begin
            // The table word is on mem_rdata two cycles after its address.
            if (twait != 2'd1) twait <= twait - 2'd1;
            else begin
                mem_busy   <= 1'b0;
                pcpi_rd    <= {{16{thalf[15]}}, thalf};
                pcpi_wr    <= 1'b1;
                pcpi_ready <= 1'b1;
                state      <= S_DONE;
            end
        end
        S_LOAD: if (KX) begin
            // Two activation words per group into the buffer, written two cycles later (tag 4).
            mem_addr <= w_ptr; w_ptr <= w_ptr + 16'd1; tag_i <= 3'd4; bi_i <= bidx;
            bidx <= bidx + 1'd1;
            phase <= {1'b0, ~phase[0]};
            if (phase[0]) begin
                groups_left <= groups_left - 14'd1;
                if (groups_left == 14'd1) begin state <= S_DRAIN; drain <= 3'd2; end
            end
        end
        S_RUNB: if (KX) begin
            // One x word (4 weights) per cycle, against buffer group bidx (tag 5).
            mem_addr <= x_ptr; x_ptr <= x_ptr + 16'd1; tag_i <= 3'd5; bi_i <= bidx;
            bidx <= bidx + 1'd1;
            groups_left <= groups_left - 14'd1;
            if (groups_left == 14'd1) begin state <= S_DRAIN; drain <= 3'd5; end
        end
        S_RUN: begin
            case (phase)
            2'd0: begin mem_addr <= x_ptr; x_ptr <= x_ptr + 16'd1; tag_i <= 3'd1; phase <= 2'd1; end
            2'd1: begin mem_addr <= w_ptr; w_ptr <= w_ptr + 16'd1; tag_i <= 3'd2; phase <= 2'd2; end
            default: begin
                mem_addr <= w_ptr; w_ptr <= w_ptr + 16'd1; tag_i <= 3'd3; phase <= 2'd0;
                groups_left <= groups_left - 14'd1;
                if (groups_left == 14'd1) begin state <= S_DRAIN; drain <= 3'd6; end
            end
            endcase
        end
        S_DRAIN:
            // Last issue -> mem_addr -> BRAM -> tag stage -> multiply -> accumulate.
            if (drain != 0) drain <= drain - 3'd1;
            else if (op == OP_POST) begin
                state      <= S_POST1;                  // keeps the port for the table read
            end else begin
                mem_busy   <= 1'b0;
                pcpi_rd    <= acc;
                pcpi_wr    <= op != OP_LOAD;            // kload has no result
                pcpi_ready <= 1'b1;
                state      <= S_DONE;
            end
        default:
            // One idle cycle while the core drops pcpi_valid for this instruction.
            state <= S_IDLE;
        endcase
    end
endmodule
