// kdot_pcpi.v — PicoRV32 PCPI coprocessor: streaming int16 x uint8 dot product.
//
// Custom-0 opcode (0001011), R-type:
//   funct3=1  klen x0, rs1, x0   length register <= rs1 (elements, multiple of 4)
//   funct3=0  kdot rd, rs1, rs2  rd <= sum_{i<len} int16 w[i] * uint8 x[i]
//                                w at byte address rs1, x at byte address rs2,
//                                both word-aligned in BRAM; 32-bit wrap-around,
//                                identical to the C int32 accumulation.
//
// While kdot runs the CPU is stalled waiting on PCPI, so the unit borrows the
// CPU's BRAM port (mem_busy). It waits for mem_idle (no CPU transfer in
// flight, e.g. an instruction prefetch) before taking the port. Per 4
// elements it reads one input word and two weight words: 3 cycles per 4 MACs.
// mem_addr is registered and the BRAM read is registered, so data for an
// address issued in cycle c is on mem_rdata in cycle c+2 (tag_i -> tag_d).

module kdot_pcpi (
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
    // Assert wait immediately so the core's 16-cycle PCPI timeout never fires.
    assign pcpi_wait = pcpi_valid && (is_kdot || is_klen);

    localparam [2:0] S_IDLE = 3'd0, S_BUS = 3'd1, S_RUN = 3'd2, S_DRAIN = 3'd3, S_DONE = 3'd4;
    reg [2:0]  state;
    reg [13:0] len_groups;             // length / 4
    reg [13:0] groups_left;
    reg [15:0] w_ptr, x_ptr;
    reg [1:0]  phase;                  // address issue: 0 = x word, 1 = w lo, 2 = w hi
    reg [2:0]  drain;

    // Pipeline: tag of the word arriving this cycle, then multiply, then accumulate.
    reg [1:0]  tag_i, tag_d;           // 0 none, 1 x, 2 w lo, 3 w hi
    reg [31:0] x_q;
    reg [31:0] w_m;
    reg [15:0] x_m;
    reg        m_valid;
    reg signed [25:0] p_q;
    reg        p_valid;
    reg [31:0] acc;

    wire signed [24:0] prod0 = $signed(w_m[15:0])  * $signed({1'b0, x_m[7:0]});
    wire signed [24:0] prod1 = $signed(w_m[31:16]) * $signed({1'b0, x_m[15:8]});

    always @(posedge clk) begin
        pcpi_ready <= 1'b0;
        pcpi_wr    <= 1'b0;
        // Data / arithmetic pipeline (runs every cycle).
        m_valid <= 1'b0;
        case (tag_d)
        2'd1: x_q <= mem_rdata;
        2'd2: begin w_m <= mem_rdata; x_m <= x_q[15:0];  m_valid <= 1'b1; end
        2'd3: begin w_m <= mem_rdata; x_m <= x_q[31:16]; m_valid <= 1'b1; end
        default: ;
        endcase
        p_valid <= m_valid;
        p_q     <= prod0 + prod1;
        if (p_valid) acc <= acc + {{6{p_q[25]}}, p_q};
        tag_d <= tag_i;
        tag_i <= 2'd0;

        if (!resetn) begin
            state      <= S_IDLE;
            mem_busy   <= 1'b0;
            len_groups <= 14'd0;
            tag_i      <= 2'd0;
            tag_d      <= 2'd0;
            m_valid    <= 1'b0;
            p_valid    <= 1'b0;
        end else case (state)
        S_IDLE:
            if (pcpi_valid && is_klen) begin
                len_groups <= pcpi_rs1[15:2];
                pcpi_ready <= 1'b1;
                state      <= S_DONE;
            end else if (pcpi_valid && is_kdot) begin
                w_ptr       <= pcpi_rs1[17:2];
                x_ptr       <= pcpi_rs2[17:2];
                groups_left <= len_groups;
                phase       <= 2'd0;
                acc         <= 32'd0;
                state       <= S_BUS;
            end
        S_BUS:
            if (groups_left == 0) begin
                state <= S_DRAIN; drain <= 3'd0;
            end else if (mem_idle) begin
                mem_busy <= 1'b1;
                state    <= S_RUN;
            end
        S_RUN: begin
            case (phase)
            2'd0: begin mem_addr <= x_ptr; x_ptr <= x_ptr + 16'd1; tag_i <= 2'd1; phase <= 2'd1; end
            2'd1: begin mem_addr <= w_ptr; w_ptr <= w_ptr + 16'd1; tag_i <= 2'd2; phase <= 2'd2; end
            default: begin
                mem_addr <= w_ptr; w_ptr <= w_ptr + 16'd1; tag_i <= 2'd3; phase <= 2'd0;
                groups_left <= groups_left - 14'd1;
                if (groups_left == 14'd1) begin state <= S_DRAIN; drain <= 3'd6; end
            end
            endcase
        end
        S_DRAIN:
            // Last issue -> mem_addr -> BRAM -> tag stage -> multiply -> accumulate.
            if (drain != 0) drain <= drain - 3'd1;
            else begin
                mem_busy   <= 1'b0;
                pcpi_rd    <= acc;
                pcpi_wr    <= 1'b1;
                pcpi_ready <= 1'b1;
                state      <= S_DONE;
            end
        default:
            // One idle cycle while the core drops pcpi_valid for this instruction.
            state <= S_IDLE;
        endcase
    end
endmodule
