// poisson.v — hardware Poisson spike generator (the "input encoding" block).
//
// Eight independent channels. Each channel c emits a spike with probability
// rate_hz[c] / CLK_HZ on EVERY clock edge, which is exactly a Poisson
// process with mean rate rate_hz[c] spikes per second. On a spike the
// channel's PENDING bit latches; the firmware reads PENDING, handles the
// spike and clears it by writing 1 to ACK (write-1-to-clear).
//
// How the probability is computed (no hardware divider!):
//     inc = rate_hz * 2^32 / CLK_HZ          (Q32 "probability per clock")
// We get this with one multiply by a constant:
//     K48   = 2^48 / CLK_HZ                  (elaboration-time constant)
//     inc   = (rate_hz * K48) >> 16
// and clamp inc to 0xFFFFFFFF (rate >= CLK_HZ -> a spike every clock).
//
// Each channel has its own 32-bit xorshift LFSR so the channels are
// independent. This is the block students are most likely to replace:
// try different rates / channel counts / encoding schemes here.
//
// Register map (byte offsets inside the peripheral, CPU view):
//   0x00  CTRL      RW  bit0 = global enable
//   0x04  CH_EN     RW  per-channel enable mask [NCH-1:0]
//   0x08  PENDING   RO  per-channel pending spike [NCH-1:0]
//   0x0C  ACK       WO  write 1 to clear the corresponding PENDING bit
//   0x10  RATE_HZ[c] RW  one 32-bit register per channel: 0x10 + 4*c
//   0x30  TOTAL     RO  number of spikes generated since reset

module poisson #(
    parameter integer NCH    = 8,
    parameter [31:0]  CLK_HZ = 32'd100_000_000
)(
    input  wire        clk,
    input  wire        rst_n,

    // CPU register interface
    input  wire        wr_en,      // 1-cycle write strobe
    input  wire [7:0]  wr_addr,    // byte offset inside this peripheral
    input  wire [31:0] wr_data,
    input  wire [7:0]  rd_addr,    // byte offset for the current read
    output reg  [31:0] rd_data,

    output wire [NCH-1:0] pending, // to the firmware (RO copy)
    output reg  [31:0]    total    // spikes generated since reset (RO)
);

    // ------------------------------------------------------------------
    // Elaboration-time constant: K48 = 2^48 / CLK_HZ (rounded).
    // For CLK_HZ = 100 MHz this is 2814750.
    // ------------------------------------------------------------------
    localparam [63:0] TWO48 = 64'h0001_0000_0000_0000;   // 2^48
    localparam [63:0] K48   = (TWO48 + (CLK_HZ >> 1)) / CLK_HZ;

    reg               en;
    reg  [NCH-1:0]    ch_en;
    reg  [NCH-1:0]    pend;
    reg  [31:0]       rate_hz  [0:NCH-1];
    reg  [31:0]       inc      [0:NCH-1];   // Q32 probability per clock
    reg  [31:0]       lfsr     [0:NCH-1];   // per-channel random state

    assign pending = pend;

    // xorshift32: cheap pseudo-random number with good-enough statistics.
    function [31:0] xs32;
        input [31:0] x;
        reg   [31:0] y;
        begin
            y = x ^ (x << 13);
            y = y ^ (y >> 17);
            y = y ^ (y << 5);
            xs32 = y;
        end
    endfunction

    // Q32 increment from a rate in Hz, computed combinationally on write.
    function [31:0] calc_inc;
        input [31:0] hz;
        reg   [63:0] q;
        begin
            q = ({32'b0, hz} * K48) >> 16;          // hz * 2^32 / CLK_HZ
            if (q >= 64'h1_0000_0000)
                calc_inc = 32'hFFFFFFFF;            // clamp: always fire
            else
                calc_inc = q[31:0];
        end
    endfunction

    integer i;

    // ------------------------------------------------------------------
    // Registers + spike generation (ONE clocked block: single driver for
    // every register, so there is no multi-driver ambiguity).
    //
    // Priority inside the cycle: reset > write (ACK) > new spike, so a
    // spike that arrives in the same cycle as an ACK is not lost.
    // ------------------------------------------------------------------
    always @(posedge clk) begin
        if (!rst_n) begin
            en      <= 1'b0;
            ch_en   <= {NCH{1'b0}};
            pend    <= {NCH{1'b0}};
            total   <= 32'h0;
            for (i = 0; i < NCH; i = i + 1) begin
                rate_hz[i] <= 32'h0;
                inc[i]     <= 32'h0;
                lfsr[i]    <= 32'h1234_5678 ^ (32'h9E37_79B9 * (i + 1));
            end
        end else begin
            // --- register writes ---
            if (wr_en) begin
                case (wr_addr)
                8'h00: en    <= wr_data[0];
                8'h04: ch_en <= wr_data[NCH-1:0];
                8'h0C: pend  <= pend & ~wr_data[NCH-1:0];   // W1C
                default: begin
                    // RATE_HZ[c] lives at 0x10 + 4*c
                    if (wr_addr >= 8'h10 && wr_addr < (8'h10 + 4*NCH)) begin
                        rate_hz[(wr_addr - 8'h10) >> 2] <= wr_data;
                        inc    [(wr_addr - 8'h10) >> 2] <= calc_inc(wr_data);
                    end
                end
                endcase
            end

            // --- spike generation ---
            // Each channel each clock: draw a random 32-bit number and fire
            // with probability inc[c]/2^32 = rate_hz[c]/CLK_HZ. This is the
            // standard discrete-time Bernoulli approximation of a Poisson
            // process. A channel that already has an unacknowledged spike is
            // not overwritten (that event would be undeliverable anyway).
            for (i = 0; i < NCH; i = i + 1) begin
                lfsr[i] <= xs32(lfsr[i]);
                if (en && ch_en[i] && !pend[i] && (lfsr[i] < inc[i])) begin
                    pend[i] <= 1'b1;
                    total   <= total + 32'h1;
                end
            end
        end
    end

    // ------------------------------------------------------------------
    // Register reads
    // ------------------------------------------------------------------
    always @(*) begin
        case (rd_addr)
        8'h00:   rd_data = {31'b0, en};
        8'h04:   rd_data = {{(32-NCH){1'b0}}, ch_en};
        8'h08:   rd_data = {{(32-NCH){1'b0}}, pend};
        8'h30:   rd_data = total;
        default: begin
            if (rd_addr >= 8'h10 && rd_addr < (8'h10 + 4*NCH))
                rd_data = rate_hz[(rd_addr - 8'h10) >> 2];
            else
                rd_data = 32'h0;
        end
        endcase
    end

endmodule
