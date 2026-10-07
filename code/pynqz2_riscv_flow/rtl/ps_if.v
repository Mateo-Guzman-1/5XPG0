// ps_if.v — the PS-side interface: one AXI4-Lite slave for NUM_CORES
// independent RISC-V SoCs (spike_soc instances).
//
// The Zynq processing system reaches the PL through M_AXI_GP0 -> interconnect
// -> protocol converter at base 0x4000_0000 (8 MB window, offset[22:0]).
// Each core owns a 1 MB slice of that window; offset[22:20] selects the core
// and offset[19:0] is the per-core layout:
//
//   offset[22:20]  core index (0 .. 7; cores >= NUM_CORES answer SLVERR)
//   offset[19:0]   0x0_0000 - 0x3_FFFF   BRAM port A (256 KB program memory)
//                                         (offset[19:18] == 0)
//                  0x4_0000 - 0x4_FFFF   syscon registers
//                                         (offset[19:16] == 4)
//
//   core 0   BRAM 0x4000_0000 - 0x4003_FFFF   syscon 0x4004_0000 - 0x4004_FFFF
//   core 1   BRAM 0x4010_0000 - 0x4013_FFFF   syscon 0x4014_0000 - 0x4014_FFFF
//   core c   BRAM 0x4000_0000 + c*0x10_0000   syscon 0x4004_0000 + c*0x10_0000
//
// Core 0's addresses are the ones of the single-core design.
//
// syscon registers (word offsets from the core's syscon base, e.g. 0x4004_0000):
//   0x00 CTRL    RW  bit0 = core reset (1 = hold CPU in reset). Reset value 1.
//   0x04 STATUS  RO  bit0 = core running, bit1 = core trap
//   0x08 TIME    RO  free-running timer snapshot
//   0x0C SCRATCH RW  test register
//   0x10 CLK_HZ  RO  build constant, core clock in Hz
//   0x14 MAGIC   RO  0x534B_454C ("SKEL")
//   0x18 LED     RO  current board-LED value written by the CPU
//
// Per-core ports are flattened buses: core c uses bit c of the 1-bit signals
// and bits [W*c +: W] of the W-bit ones (e.g. bram_addr[16*c +: 16]).
//
// Note: unlike the AURA version there are no chip GPIO pins here, and the
// LED register is read-only from the PS (the RISC-V firmware owns the LEDs).

module ps_if #(
    parameter [31:0]  CLK_HZ    = 32'd100_000_000,
    parameter integer NUM_CORES = 2,                 // 1 .. 8
    parameter         KX        = 1                  // ABI bit3: kdot_pcpi KX instructions (keep equal to spike_soc's KX)
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

    // BRAM port A of each core (word address) — driven only by this module
    output reg  [NUM_CORES-1:0]    bram_we,
    // dont_touch: one register per bit drives every BRAM tile; replicas could split the
    // address of a cascaded RAMB36 pair (DRC REQP-1962 in the KX build).
    (* dont_touch = "true" *) output reg  [16*NUM_CORES-1:0] bram_addr,
    output reg  [32*NUM_CORES-1:0] bram_wdata,
    output reg  [4*NUM_CORES-1:0]  bram_be,
    input  wire [32*NUM_CORES-1:0] bram_rdata,

    // syscon of each core
    output reg  [NUM_CORES-1:0]    core_rst,     // 1 = hold CPU in reset
    input  wire [NUM_CORES-1:0]    core_running,
    input  wire [NUM_CORES-1:0]    core_trap,
    input  wire [32*NUM_CORES-1:0] timer_now,
    output reg  [32*NUM_CORES-1:0] scratch,
    input  wire [10*NUM_CORES-1:0] led           // LED value owned by each CPU
);

    // ------------------------------------------------------------------
    // Address decode (full AXI address; slave base 0x4000_0000)
    // ------------------------------------------------------------------
    function sel_core_addr;           // offset[22:20] names an implemented core
        input [31:0] a;
        sel_core_addr = ({29'd0, a[22:20]} < NUM_CORES);
    endfunction

    function sel_bram_addr;
        input [31:0] a;
        sel_bram_addr = sel_core_addr(a) && (a[19:18] == 2'b00);
    endfunction

    function sel_syscon_addr;
        input [31:0] a;
        sel_syscon_addr = sel_core_addr(a) && (a[19:16] == 4'b0100);
    endfunction

    wire [4:0] syscon_ridx = s_axil_araddr[6:2];
    wire [2:0] rcore       = s_axil_araddr[22:20];

    // ------------------------------------------------------------------
    // Write channel: latch AW and W independently, fire when both present.
    // ------------------------------------------------------------------
    reg        aw_got, w_got, b_wait;
    reg [1:0]  rstate;        // read FSM: 0 idle, 1 bram reading, 2 capture, 3 present
    reg [31:0] wr_addr_q, wr_data_q;
    reg [3:0]  wr_strb_q;

    wire [4:0] syscon_widx = wr_addr_q[6:2];
    wire [2:0] wcore       = wr_addr_q[22:20];

    assign s_axil_awready = !aw_got && !b_wait;
    assign s_axil_wready  = !w_got  && !b_wait;
    // The shared BRAM address must remain stable until the read is captured.
    // AW/W may queue independently, but cannot execute during that read.
    wire wr_fire = aw_got && w_got && !b_wait && (rstate == 2'd0) && !s_axil_rvalid;

    // Read channel accept (blocked for one cycle while a write fires, so the
    // write and the read never fight over bram_addr in the same cycle)
    assign s_axil_arready = (rstate == 2'd0) && !s_axil_rvalid && !wr_fire;
    wire ar_accept = s_axil_arvalid && s_axil_arready;

    reg        rd_is_sys;
    reg        rd_err;
    reg [2:0]  rd_core;
    reg [31:0] rd_sys_q;

    integer c;

    // ------------------------------------------------------------------
    // BRAM port A of every core (single always block: single driver).
    // Only the addressed core's port moves; the others keep their values.
    // ------------------------------------------------------------------
    always @(posedge aclk) begin
        if (!aresetn) begin
            bram_we <= {NUM_CORES{1'b0}}; bram_addr <= {16*NUM_CORES{1'b0}};
            bram_wdata <= {32*NUM_CORES{1'b0}}; bram_be <= {4*NUM_CORES{1'b0}};
        end else begin
            bram_we <= {NUM_CORES{1'b0}};
            for (c = 0; c < NUM_CORES; c = c + 1) begin
                if (wr_fire) begin
                    if (sel_bram_addr(wr_addr_q) && {29'd0, wcore} == c) begin
                        bram_we[c]            <= 1'b1;
                        bram_addr[16*c +: 16] <= wr_addr_q[17:2];
                        bram_wdata[32*c +: 32] <= wr_data_q;
                        bram_be[4*c +: 4]     <= wr_strb_q;
                    end
                end else if (ar_accept && sel_bram_addr(s_axil_araddr) && {29'd0, rcore} == c) begin
                    bram_addr[16*c +: 16] <= s_axil_araddr[17:2];
                end
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
            core_rst <= {NUM_CORES{1'b1}};    // CPUs held in reset after configuration
            scratch  <= {32*NUM_CORES{1'b0}};
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
                    for (c = 0; c < NUM_CORES; c = c + 1) begin
                        if ({29'd0, wcore} == c) begin
                            if (syscon_widx == 5'd0) begin
                                if (wr_strb_q[0]) core_rst[c] <= wr_data_q[0];
                            end else if (syscon_widx == 5'd3) begin
                                if (wr_strb_q[0]) scratch[32*c +  0 +: 8] <= wr_data_q[7:0];
                                if (wr_strb_q[1]) scratch[32*c +  8 +: 8] <= wr_data_q[15:8];
                                if (wr_strb_q[2]) scratch[32*c + 16 +: 8] <= wr_data_q[23:16];
                                if (wr_strb_q[3]) scratch[32*c + 24 +: 8] <= wr_data_q[31:24];
                            end
                        end
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
            rd_err    <= 1'b0;
            rd_core   <= 3'd0;
            rd_sys_q  <= 32'h0;
        end else begin
            case (rstate)
            2'd0: if (ar_accept) begin
                rd_is_sys <= sel_syscon_addr(s_axil_araddr);
                rd_err    <= !(sel_bram_addr(s_axil_araddr) || sel_syscon_addr(s_axil_araddr));
                rd_core   <= sel_core_addr(s_axil_araddr) ? rcore : 3'd0;
                rd_sys_q  <= 32'h0;
                for (c = 0; c < NUM_CORES; c = c + 1) begin
                    if ({29'd0, rcore} == c) begin
                        case (syscon_ridx)
                        5'd0:  rd_sys_q <= {31'h0, core_rst[c]};                    // CTRL
                        5'd1:  rd_sys_q <= {30'h0, core_trap[c], core_running[c]};  // STATUS
                        5'd2:  rd_sys_q <= timer_now[32*c +: 32];                   // TIME
                        5'd3:  rd_sys_q <= scratch[32*c +: 32];                     // SCRATCH
                        5'd4:  rd_sys_q <= CLK_HZ;                                  // CLK_HZ
                        5'd5:  rd_sys_q <= 32'h534B_454C;                           // MAGIC "SKEL"
                        5'd6:  rd_sys_q <= {22'h0, led[10*c +: 10]};                // LED
                        5'd7:  rd_sys_q <= 32'h0002_0007 | (KX ? 32'h8 : 32'h0); // keyword pulse ABI + bit0 kdot + bit1 neuron engine + bit2 engine layer 1 + bit3 KX
                        default: rd_sys_q <= 32'h0;
                        endcase
                    end
                end
                s_axil_rresp <= (sel_bram_addr(s_axil_araddr) ||
                                 sel_syscon_addr(s_axil_araddr)) ? 2'b00 : 2'b10;
                rstate <= 2'd1;
            end
            2'd1: rstate <= 2'd2;
            2'd2: begin
                s_axil_rvalid <= 1'b1;
                s_axil_rdata  <= rd_err    ? 32'h0 :
                                 rd_is_sys ? rd_sys_q : bram_rdata[32*rd_core +: 32];
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
