// ps_if.v — the PS-side interface: one AXI4-Lite slave.
//
// The Zynq processing system reaches the PL through M_AXI_GP0 -> interconnect
// -> protocol converter at base 0x4000_0000 (8 MB window). This slave decodes:
//
//   offset 0x000_0000 - 0x003_FFFF   BRAM port A (256 KB program memory)
//                                     (offset[22:18] == 0)
//   offset 0x040_0000 - 0x040_00FF   syscon registers
//                                     (offset[22:16] == 4)
//
// syscon registers (word offsets from 0x4004_0000):
//   0x00 CTRL    RW  bit0 = core reset (1 = hold CPU in reset). Reset value 1.
//   0x04 STATUS  RO  bit0 = core running, bit1 = core trap
//   0x08 TIME    RO  free-running timer snapshot
//   0x0C SCRATCH RW  test register
//   0x10 CLK_HZ  RO  build constant, core clock in Hz
//   0x14 MAGIC   RO  0x534B_454C ("SKEL")
//   0x18 LED     RO  current board-LED value written by the CPU
//
// Note: unlike the AURA version there are no chip GPIO pins here, and the
// LED register is read-only from the PS (the RISC-V firmware owns the LEDs).

module ps_if #(
    parameter [31:0] CLK_HZ = 32'd100_000_000
) (
    input  wire        aclk,
    input  wire        aresetn,

    // AXI4-Lite slave
    input  wire [31:0] s_axil_awaddr,
    input  wire        s_axil_awvalid,
    output wire        s_axil_awready,
    input  wire [31:0] s_axil_wdata,
    input  wire [3:0]  s_axil_wstrb,
    input  wire        s_axil_wvalid,
    output wire        s_axil_wready,
    output reg  [1:0]  s_axil_bresp,
    output reg         s_axil_bvalid,
    input  wire        s_axil_bready,
    input  wire [31:0] s_axil_araddr,
    input  wire        s_axil_arvalid,
    output wire        s_axil_arready,
    output reg  [31:0] s_axil_rdata,
    output reg  [1:0]  s_axil_rresp,
    output reg         s_axil_rvalid,
    input  wire        s_axil_rready,

    // BRAM port A (word address) — driven only by this module
    output reg         bram_we,
    output reg  [15:0] bram_addr,
    output reg  [31:0] bram_wdata,
    output reg  [3:0]  bram_be,
    input      [31:0]  bram_rdata,

    // syscon
    output reg         core_rst,     // 1 = hold CPU in reset
    input  wire        core_running,
    input  wire        core_trap,
    input  wire [31:0] timer_now,
    output reg  [31:0] scratch,
    input  wire [9:0]  led           // LED value owned by the CPU
);

    // ------------------------------------------------------------------
    // Address decode (full AXI address; slave base 0x4000_0000)
    // ------------------------------------------------------------------
    function sel_bram_addr;
        input [31:0] a;
        sel_bram_addr = (a[22:18] == 5'b00000);
    endfunction

    function sel_syscon_addr;
        input [31:0] a;
        sel_syscon_addr = (a[22:16] == 7'b0000100);
    endfunction

    wire [4:0] syscon_ridx = s_axil_araddr[6:2];

    // ------------------------------------------------------------------
    // Write channel: latch AW and W independently, fire when both present.
    // ------------------------------------------------------------------
    reg        aw_got, w_got, b_wait;
    reg [31:0] wr_addr_q, wr_data_q;
    reg [3:0]  wr_strb_q;

    wire [4:0] syscon_widx = wr_addr_q[6:2];

    assign s_axil_awready = !aw_got && !b_wait;
    assign s_axil_wready  = !w_got  && !b_wait;
    // The shared BRAM address must remain stable until the read is captured.
    // AW/W may queue independently, but cannot execute during that read.
    wire wr_fire = aw_got && w_got && !b_wait && (rstate == 2'd0) && !s_axil_rvalid;

    // Read channel accept (blocked for one cycle while a write fires, so the
    // write and the read never fight over bram_addr in the same cycle)
    reg [1:0] rstate;         // 0 idle, 1 bram reading, 2 capture, 3 present
    assign s_axil_arready = (rstate == 2'd0) && !s_axil_rvalid && !wr_fire;
    wire ar_accept = s_axil_arvalid && s_axil_arready;

    reg        rd_is_sys;
    reg [31:0] rd_sys_q;

    // ------------------------------------------------------------------
    // BRAM port A + syscon registers (single always block: single driver)
    // ------------------------------------------------------------------
    always @(posedge aclk) begin
        if (!aresetn) begin
            bram_we <= 1'b0; bram_addr <= 16'h0;
            bram_wdata <= 32'h0; bram_be <= 4'h0;
        end else begin
            bram_we <= 1'b0;
            if (wr_fire) begin
                if (sel_bram_addr(wr_addr_q)) begin
                    bram_we    <= 1'b1;
                    bram_addr  <= wr_addr_q[17:2];
                    bram_wdata <= wr_data_q;
                    bram_be    <= wr_strb_q;
                end
            end else if (ar_accept) begin
                bram_addr <= s_axil_araddr[17:2];
            end
        end
    end

    // ------------------------------------------------------------------
    // Write FSM (AW/W latching, execute, B response)
    // ------------------------------------------------------------------
    always @(posedge aclk) begin
        if (!aresetn) begin
            aw_got <= 1'b0; w_got <= 1'b0; b_wait <= 1'b0;
            s_axil_bvalid <= 1'b0; s_axil_bresp <= 2'b00;
            core_rst <= 1'b1;         // CPU held in reset after configuration
            scratch  <= 32'h0;
        end else begin
            if (s_axil_awvalid && s_axil_awready) begin
                aw_got    <= 1'b1;
                wr_addr_q <= s_axil_awaddr;
            end
            if (s_axil_wvalid && s_axil_wready) begin
                w_got     <= 1'b1;
                wr_data_q <= s_axil_wdata;
                wr_strb_q <= s_axil_wstrb;
            end

            if (wr_fire) begin
                aw_got <= 1'b0;
                w_got  <= 1'b0;
                b_wait <= 1'b1;
                s_axil_bvalid <= 1'b1;
                if (sel_bram_addr(wr_addr_q)) begin
                    s_axil_bresp <= 2'b00;
                end else if (sel_syscon_addr(wr_addr_q)) begin
                    s_axil_bresp <= 2'b00;
                    if (syscon_widx == 5'd0) begin
                        if (wr_strb_q[0]) core_rst <= wr_data_q[0];
                    end else if (syscon_widx == 5'd3) begin
                        if (wr_strb_q[0]) scratch[7:0] <= wr_data_q[7:0];
                        if (wr_strb_q[1]) scratch[15:8] <= wr_data_q[15:8];
                        if (wr_strb_q[2]) scratch[23:16] <= wr_data_q[23:16];
                        if (wr_strb_q[3]) scratch[31:24] <= wr_data_q[31:24];
                    end
                end else begin
                    s_axil_bresp <= 2'b10;   // SLVERR
                end
            end

            if (b_wait && s_axil_bvalid && s_axil_bready) begin
                b_wait        <= 1'b0;
                s_axil_bvalid <= 1'b0;
            end
        end
    end

    // ------------------------------------------------------------------
    // Read FSM (AR accept -> BRAM read -> R response)
    // ------------------------------------------------------------------
    always @(posedge aclk) begin
        if (!aresetn) begin
            rstate <= 2'd0;
            s_axil_rvalid <= 1'b0;
            s_axil_rresp  <= 2'b00;
            rd_is_sys <= 1'b0;
            rd_sys_q  <= 32'h0;
        end else begin
            case (rstate)
            2'd0: if (ar_accept) begin
                rd_is_sys <= sel_syscon_addr(s_axil_araddr);
                case (syscon_ridx)
                5'd0:  rd_sys_q <= {31'h0, core_rst};                 // CTRL
                5'd1:  rd_sys_q <= {30'h0, core_trap, core_running};   // STATUS
                5'd2:  rd_sys_q <= timer_now;                          // TIME
                5'd3:  rd_sys_q <= scratch;                            // SCRATCH
                5'd4:  rd_sys_q <= CLK_HZ;                             // CLK_HZ
                5'd5:  rd_sys_q <= 32'h534B_454C;                      // MAGIC "SKEL"
                5'd6:  rd_sys_q <= {22'h0, led};                       // LED
                5'd7:  rd_sys_q <= 32'h0002_0000;                    // keyword pulse ABI
                default: rd_sys_q <= 32'h0;
                endcase
                s_axil_rresp <= (sel_bram_addr(s_axil_araddr) ||
                                 sel_syscon_addr(s_axil_araddr)) ? 2'b00 : 2'b10;
                rstate <= 2'd1;
            end
            2'd1: rstate <= 2'd2;
            2'd2: begin
                s_axil_rvalid <= 1'b1;
                s_axil_rdata  <= rd_is_sys ? rd_sys_q : bram_rdata;
                rstate <= 2'd3;
            end
            2'd3: if (s_axil_rready) begin
                s_axil_rvalid <= 1'b0;
                rstate <= 2'd0;
            end
            endcase
        end
    end

endmodule
