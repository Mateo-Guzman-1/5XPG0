// spike_soc.v — minimal RISC-V SoC: PicoRV32 + 256 KB BRAM + peripherals.
//
// This is a stripped-down version of the AURA controller. All the
// neuromorphic-chip GPIO and bias-generator hardware is gone; what is left
// is the minimum needed to run C code on a soft core and collect spikes:
//
//   0x0000_0000  BRAM (256 KB)  program + data + console + spike log
//   0x1000_0000  SYSCTRL        0x00 MAGIC  0x04 VERSION
//                               0x08 SCRATCH 0x0C TRAP
//   0x1000_1000  TIMER          0x00 TIME   (32-bit free-running counter)
//   0x1000_2000  LED            0x00 OUT    ([9:0] PYNQ-Z2 board LEDs)
//   0x1000_3000  POISSON        hardware spike generator (see rtl/poisson.v)
//
// The BRAM has two ports: port A belongs to the PS (via ps_if, used to load
// the program and read the console / spike log), port B belongs to the CPU.
// The CPU fetches and loads/stores from the SAME BRAM (execute-in-place).
//
// Peripheral bit layouts (LED / POISSON) MUST match firmware/board.h.

module spike_soc #(
    parameter integer MEM_WORDS = 65536   // 256 KB at 32 bits/word
)(
    input  wire        clk,
    input  wire        core_rst_n,        // 0 = CPU held in reset

    // BRAM port A (PS side). pa_addr is a WORD address.
    input  wire             pa_we,
    input  wire [15:0]      pa_addr,
    input  wire [31:0]      pa_wdata,
    input  wire [3:0]       pa_be,
    output wire [31:0]      pa_rdata,

    // status / housekeeping
    output wire         core_running,
    output wire         core_trap,
    output reg  [31:0]  timer_lo,          // free-running (snapped to the PS)
    output reg  [9:0]   led_o              // board LEDs, written by the CPU
);

    // ------------------------------------------------------------------
    // CPU memory interface (PicoRV32 holds req signals until mem_ready)
    // ------------------------------------------------------------------
    wire        cpu_mem_valid;
    wire        cpu_mem_instr;
    reg         cpu_mem_ready;
    wire [31:0] cpu_mem_addr;
    wire [31:0] cpu_mem_wdata;
    wire [3:0]  cpu_mem_wstrb;
    reg  [31:0] cpu_mem_rdata;

    wire sel_bram = (cpu_mem_addr[31:28] == 4'h0);   // 0x0000_0000 region
    wire sel_peri = (cpu_mem_addr[31:28] == 4'h1);   // 0x1000_0000 region
    wire req_write = (|cpu_mem_wstrb);

    // Read data phase: BRAM is sync-read, so reads complete a few cycles
    // after accept. Writes complete on the accept cycle itself.
    reg  [3:0] xfer;                 // 0 idle, counting up to READ_WAIT
    localparam [3:0] READ_WAIT = 4'd8;
    reg        xfer_bram;            // 1 = data comes from BRAM
    reg  [31:0] peri_rdata_r;        // peripheral value captured at accept

    wire acc     = cpu_mem_valid && !cpu_mem_ready && (xfer == 4'd0);
    wire acc_wr  = acc && req_write;
    wire acc_rd  = acc && !req_write;
    wire bram_wr_now = acc_wr && sel_bram;

    // ------------------------------------------------------------------
    // Main memory: 256 KB true dual-port block RAM (one always block per
    // port, byte-enable writes, 1-cycle registered read).
    // ------------------------------------------------------------------
    (* ram_style = "block" *) reg [31:0] mem [0:MEM_WORDS-1];

    reg  [31:0] pb_rdata;            // port B read data (CPU)
    reg  [31:0] pa_rdata_i;          // port A read data (PS)
    assign pa_rdata = pa_rdata_i;

    wire [15:0] pb_word_addr = cpu_mem_addr[17:2];

    // Port A (PS)
    always @(posedge clk) begin
        if (pa_we) begin
            if (pa_be[0]) mem[pa_addr][7:0]   <= pa_wdata[7:0];
            if (pa_be[1]) mem[pa_addr][15:8]  <= pa_wdata[15:8];
            if (pa_be[2]) mem[pa_addr][23:16] <= pa_wdata[23:16];
            if (pa_be[3]) mem[pa_addr][31:24] <= pa_wdata[31:24];
        end
        pa_rdata_i <= mem[pa_addr];
    end

    // Port B (CPU)
    always @(posedge clk) begin
        if (bram_wr_now) begin
            if (cpu_mem_wstrb[0]) mem[pb_word_addr][7:0]   <= cpu_mem_wdata[7:0];
            if (cpu_mem_wstrb[1]) mem[pb_word_addr][15:8]  <= cpu_mem_wdata[15:8];
            if (cpu_mem_wstrb[2]) mem[pb_word_addr][23:16] <= cpu_mem_wdata[23:16];
            if (cpu_mem_wstrb[3]) mem[pb_word_addr][31:24] <= cpu_mem_wdata[31:24];
        end
        pb_rdata <= mem[pb_word_addr];
    end

    // ------------------------------------------------------------------
    // Peripherals
    // ------------------------------------------------------------------
    reg [31:0] sys_scratch;

    // Hardware Poisson spike generator.
    wire        po_wr  = acc_wr && sel_peri && (cpu_mem_addr[15:12] == 4'h3);
    wire [7:0]  po_addr = cpu_mem_addr[7:0];
    wire [31:0] po_rdata;
    wire [7:0]  po_pending;      // (unused by the SoC, available for expansion)
    wire [31:0] po_total;

    poisson #(
        .CLK_HZ (32'd100_000_000)
    ) u_poisson (
        .clk     (clk),
        .rst_n   (core_rst_n),
        .wr_en   (po_wr),
        .wr_addr (po_addr),
        .wr_data (cpu_mem_wdata),
        .rd_addr (po_addr),
        .rd_data (po_rdata),
        .pending (po_pending),
        .total   (po_total)
    );

    // Free-running timer (not reset by core_rst_n, so the PS can tell the
    // fabric is alive even while the CPU is held in reset).
    always @(posedge clk)
        timer_lo <= timer_lo + 32'h1;

    // ------------------------------------------------------------------
    // CPU bus FSM + peripheral write/read decode
    // ------------------------------------------------------------------
    always @(posedge clk) begin
        if (!core_rst_n) begin
            cpu_mem_ready <= 1'b0;
            xfer          <= 4'd0;
            sys_scratch   <= 32'h0;
            led_o         <= 10'h0;
        end else begin
            cpu_mem_ready <= 1'b0;

            if (xfer == 4'd0) begin
                if (acc_wr) begin
                    cpu_mem_ready <= 1'b1;
                    if (sel_peri) begin
                        case (cpu_mem_addr[15:12])
                        4'h0: if (cpu_mem_addr[3:2] == 2'd2)        // SYSCTRL SCRATCH
                                  sys_scratch <= cpu_mem_wdata;
                        4'h2: if (cpu_mem_addr[3:2] == 2'd0)        // LED OUT
                                  led_o <= cpu_mem_wdata[9:0];
                        default: ;   // TIMER/POISSON have no CPU writes here
                        endcase
                    end
                    // (BRAM writes happen in the port-B block above)
                end else if (acc_rd) begin
                    xfer      <= 4'd1;
                    xfer_bram <= sel_bram;
                    case (cpu_mem_addr[15:12])
                    4'h0: case (cpu_mem_addr[3:2])                  // SYSCTRL
                          2'd0: peri_rdata_r <= 32'h534B_454C;     // "SKEL"
                          2'd1: peri_rdata_r <= 32'h0001_0000;     // v1.0
                          2'd2: peri_rdata_r <= sys_scratch;
                          2'd3: peri_rdata_r <= {31'b0, core_trap};
                          endcase
                    4'h1: peri_rdata_r <= timer_lo;                 // TIMER
                    4'h2: peri_rdata_r <= {22'b0, led_o};           // LED
                    4'h3: peri_rdata_r <= po_rdata;                 // POISSON
                    default: peri_rdata_r <= 32'hDEAD_BEEF;
                    endcase
                end
            end else if (xfer != READ_WAIT) begin
                xfer <= xfer + 4'd1;
            end else begin
                // read completion
                xfer          <= 4'd0;
                cpu_mem_ready <= 1'b1;
                cpu_mem_rdata <= xfer_bram ? pb_rdata : peri_rdata_r;
            end
        end
    end

    assign core_running = core_rst_n;

    // ------------------------------------------------------------------
    // CPU
    // ------------------------------------------------------------------
    picorv32 #(
        .ENABLE_COUNTERS   (1),
        .ENABLE_COUNTERS64 (0),
        .ENABLE_MUL        (0),
        .ENABLE_FAST_MUL   (1),
        .ENABLE_DIV        (1),
        .ENABLE_IRQ        (0),
        .ENABLE_TRACE      (0),
        .PROGADDR_RESET    (32'h0000_0000),
        .STACKADDR         (32'h0000_0000)
    ) u_cpu (
        .clk         (clk),
        .resetn      (core_rst_n),
        .trap        (core_trap),
        .mem_valid   (cpu_mem_valid),
        .mem_instr   (cpu_mem_instr),
        .mem_ready   (cpu_mem_ready),
        .mem_addr    (cpu_mem_addr),
        .mem_wdata   (cpu_mem_wdata),
        .mem_wstrb   (cpu_mem_wstrb),
        .mem_rdata   (cpu_mem_rdata),
        .pcpi_valid  (), .pcpi_insn (), .pcpi_rs1 (), .pcpi_rs2 (),
        .pcpi_wr     (1'b0), .pcpi_rd (32'h0), .pcpi_wait (1'b0), .pcpi_ready (1'b0),
        .irq         (32'h0),
        .eoi         (),
        .trace_valid (), .trace_data ()
    );

endmodule
