// kw_mac_pcpi.v — multiply-accumulate (MAC) co-processor for PicoRV32.
//
// Connected to the CPU's co-processor port (PCPI). The CPU hands every
// instruction it does not know to this unit; the unit answers only for
// opcode custom-0 (0001011) and selects the operation with funct7:
//
//   funct7  name        rs1                    rs2            rd
//   0       KW_LDX      start position (0-124) 4 input bytes  -
//   1       KW_SETACC   bias                   -              -
//   2       KW_MAC4     4 int8 weights         -              new sum
//   3       KW_MAC8     8 int4 weights         -              new sum
//
// The unit keeps a 128-byte copy of the input frame (loaded once per frame
// with 32 x KW_LDX), a running sum and a position. Each MAC multiplies its
// weights (signed) with the next 4 or 8 inputs (unsigned bytes), adds them to
// the sum, moves the position on and returns the new sum. Weight lane j sits
// in the low bits first (lane 0 = bits [3:0] or [7:0]), like the firmware.
//
// Timing: KW_LDX / KW_SETACC are answered one cycle after they arrive. A MAC
// is a 3-stage pipeline (pick inputs, multiply, add) and is answered three
// cycles after it arrives; meanwhile pcpi_wait holds off the CPU's timeout.
// (Done in one cycle, the path buffer -> multiply -> add is ~14 ns, too slow
// for the 10 ns clock.)

module kw_mac_pcpi (
    input  wire        clk,
    input  wire        rst_n,

    // from the CPU
    input  wire        pcpi_valid,
    input  wire [31:0] pcpi_insn,
    input  wire [31:0] pcpi_rs1,
    input  wire [31:0] pcpi_rs2,

    // back to the CPU
    output reg         pcpi_wr,
    output reg  [31:0] pcpi_rd,
    output wire        pcpi_wait,
    output reg         pcpi_ready
);

    // ------------------------------------------------------------------
    // Instruction decode
    // ------------------------------------------------------------------
    wire [6:0] opcode = pcpi_insn[6:0];
    wire [6:0] funct7 = pcpi_insn[31:25];

    localparam [6:0] OPCODE_CUSTOM_0 = 7'b0001011;
    localparam [6:0] F_LDX    = 7'd0;   // store 4 input bytes into the buffer
    localparam [6:0] F_SETACC = 7'd1;   // sum = bias, position = 0
    localparam [6:0] F_MAC4   = 7'd2;   // int8: 4 weights x 4 inputs
    localparam [6:0] F_MAC8   = 7'd3;   // int4: 8 weights x 8 inputs

    wire is_custom0 = (opcode == OPCODE_CUSTOM_0) && pcpi_valid;

    wire is_ldx    = is_custom0 && (funct7 == F_LDX);
    wire is_setacc = is_custom0 && (funct7 == F_SETACC);
    wire is_mac4   = is_custom0 && (funct7 == F_MAC4);
    wire is_mac8   = is_custom0 && (funct7 == F_MAC8);

    wire is_mine = is_ldx || is_setacc || is_mac4 || is_mac8;

    // ------------------------------------------------------------------
    // State
    // ------------------------------------------------------------------
    reg        [7:0]  input_buffer [0:127];  // 128 bytes, one input per entry
    reg signed [31:0] sum;                   // running sum of one neuron
    reg        [6:0]  position;              // first input of the next MAC

    // MAC pipeline registers
    reg               busy;                  // a MAC is in stage 1 or 2
    reg               stage2;                // 0 = multiply next, 1 = add next
    reg               op_mac8;               // 1 = int4 lanes, 0 = int8 lanes
    reg        [31:0] w_r;                   // the packed weights (rs1)
    reg        [7:0]  x_r [0:7];             // the 8 inputs at position..+7
    reg signed [16:0] p_r [0:7];             // the 8 products

    // Busy with a MAC: keep the CPU waiting instead of letting it time out.
    assign pcpi_wait = busy;

    // ------------------------------------------------------------------
    // Arithmetic
    // ------------------------------------------------------------------
    // KW_LDX: rs1 = start position in the buffer, rs2 = 4 input bytes
    wire [6:0] ldx_pos = pcpi_rs1[6:0];

    // Signed weight of lane j: int4 nibble j (MAC8), or int8 byte j for
    // lanes 0-3 (MAC4; lanes 4-7 are unused there and count as 0).
    function signed [7:0] lane_weight(input [31:0] w, input mac8, input [2:0] j);
        begin
            if (mac8)
                lane_weight = {{4{w[4*j + 3]}}, w[4*j +: 4]};
            else if (!j[2])
                lane_weight = w[8*j[1:0] +: 8];
            else
                lane_weight = 8'sd0;
        end
    endfunction

    // Stage 3: the new sum (products are signed weights x unsigned inputs).
    wire signed [31:0] mac_sum = sum + p_r[0] + p_r[1] + p_r[2] + p_r[3]
                                     + p_r[4] + p_r[5] + p_r[6] + p_r[7];

    // ------------------------------------------------------------------
    // Handshake + state update
    // ------------------------------------------------------------------
    integer k;

    always @(posedge clk) begin
        if (!rst_n) begin
            pcpi_ready <= 1'b0;
            pcpi_wr    <= 1'b0;
            pcpi_rd    <= 32'd0;
            sum        <= 32'sd0;
            position   <= 7'd0;
            busy       <= 1'b0;
            stage2     <= 1'b0;
        end else begin
            // default: no answer this cycle
            pcpi_ready <= 1'b0;
            pcpi_wr    <= 1'b0;

            if (busy) begin
                if (!stage2) begin
                    // stage 2: multiply the 8 lanes
                    for (k = 0; k < 8; k = k + 1)
                        p_r[k] <= lane_weight(w_r, op_mac8, k)
                                  * $signed({1'b0, x_r[k]});
                    stage2 <= 1'b1;
                end else begin
                    // stage 3: add, answer the CPU with the new sum
                    sum        <= mac_sum;
                    pcpi_rd    <= mac_sum;
                    pcpi_wr    <= 1'b1;
                    pcpi_ready <= 1'b1;
                    busy       <= 1'b0;
                    stage2     <= 1'b0;
                end

            // !pcpi_ready: the CPU still shows the instruction in the cycle it
            // sees our answer; without this it would execute twice.
            end else if (is_mine && !pcpi_ready) begin
                if (is_ldx) begin
                    input_buffer[ldx_pos]        <= pcpi_rs2[ 7: 0];
                    input_buffer[ldx_pos + 7'd1] <= pcpi_rs2[15: 8];
                    input_buffer[ldx_pos + 7'd2] <= pcpi_rs2[23:16];
                    input_buffer[ldx_pos + 7'd3] <= pcpi_rs2[31:24];
                    pcpi_ready <= 1'b1;
                end else if (is_setacc) begin
                    sum        <= $signed(pcpi_rs1);  // bias
                    position   <= 7'd0;
                    pcpi_ready <= 1'b1;
                end else begin
                    // stage 1 of a MAC: latch the weights and pick the inputs
                    w_r     <= pcpi_rs1;
                    op_mac8 <= is_mac8;
                    // 7-bit sums: 124+4..7 wraps to 0..3 (MAC4 leaves those
                    // lanes unused), so the index never leaves the buffer.
                    x_r[0] <= input_buffer[position];
                    x_r[1] <= input_buffer[position + 7'd1];
                    x_r[2] <= input_buffer[position + 7'd2];
                    x_r[3] <= input_buffer[position + 7'd3];
                    x_r[4] <= input_buffer[position + 7'd4];
                    x_r[5] <= input_buffer[position + 7'd5];
                    x_r[6] <= input_buffer[position + 7'd6];
                    x_r[7] <= input_buffer[position + 7'd7];
                    position <= position + (is_mac8 ? 7'd8 : 7'd4);
                    busy     <= 1'b1;
                end
            end
        end
    end

endmodule
