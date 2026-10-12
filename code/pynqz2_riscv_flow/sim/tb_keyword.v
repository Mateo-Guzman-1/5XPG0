// tb_keyword.v — full-SoC simulation of the keyword firmware (xsim).
//
// Plays the role of the PS / keyword_bridge.py: preloads the firmware image
// into the BRAM, then for every frame writes the 128-byte spectrogram to
// INPUT_FRAME_BASE, posts MB_CMD_CLASSIFY through the mailbox and waits for
// the reply. One line per frame:
//   RESULT <index> <decision> <score_not> <score_keyword> <cycles> <layer1_cycles>
// Driven by run_keyword_sim.py, which writes the hex files and checks the
// scores against board_vectors.npz.
//
// Plusargs: +prog=<hex> (word per line, from address 0)
//           +frames=<hex> (32 words per frame)  +nframes=<n>
`timescale 1ns / 1ps

module tb_keyword;
    // firmware/board.h
    localparam integer MAILBOX_W   = 32'h10400 / 4;
    localparam integer TIMING_W    = 32'h10440 / 4;
    localparam integer FRAME_W     = 32'h22000 / 4;
    localparam integer FRAME_WORDS = 32;          // 128 inputs, 1 byte each
    localparam [31:0]  CMD_CLASSIFY = 32'd8;
    localparam [31:0]  TIMING_MAGIC = 32'h4B57544D;
    localparam integer MAX_FRAMES  = 64;
    localparam integer TIMEOUT     = 5_000_000;   // cycles per frame

    reg clk = 1'b0;
    always #5 clk = ~clk;                         // 100 MHz, as on the board

    reg core_rst_n = 1'b0;

    wire        core_running, core_trap;
    wire [31:0] timer_lo;
    wire [9:0]  led_o;
    wire [31:0] pa_rdata;

    spike_soc dut (
        .clk          (clk),
        .core_rst_n   (core_rst_n),
        .pa_we        (1'b0),
        .pa_addr      (16'h0),
        .pa_wdata     (32'h0),
        .pa_be        (4'h0),
        .pa_rdata     (pa_rdata),
        .core_running (core_running),
        .core_trap    (core_trap),
        .timer_lo     (timer_lo),
        .led_o        (led_o)
    );

    reg [31:0] frames [0:MAX_FRAMES*FRAME_WORDS-1];
    reg [1023:0] prog_file, frames_file;
    integer nframes, f, i, waited, seq;

    initial begin
        // The timer has no reset on the board (it just runs from power-up);
        // give it a defined start value in simulation.
        dut.timer_lo = 32'h0;
        for (i = 0; i < 65536; i = i + 1)
            dut.mem[i] = 32'h0;

        if (!$value$plusargs("prog=%s", prog_file)) $fatal(1, "need +prog=");
        if (!$value$plusargs("frames=%s", frames_file)) $fatal(1, "need +frames=");
        if (!$value$plusargs("nframes=%d", nframes)) $fatal(1, "need +nframes=");
        if (nframes > MAX_FRAMES) $fatal(1, "too many frames");
        $readmemh(prog_file, dut.mem);
        $readmemh(frames_file, frames);

        repeat (10) @(posedge clk);
        core_rst_n = 1'b1;

        // main() lights LED 0 once its console / mailbox state is set up.
        waited = 0;
        while (led_o !== 10'h001 && waited < TIMEOUT) begin
            @(posedge clk);
            waited = waited + 1;
        end
        if (led_o !== 10'h001) $fatal(1, "firmware did not start");

        seq = dut.mem[MAILBOX_W + 0];
        for (f = 0; f < nframes; f = f + 1) begin
            @(negedge clk);
            for (i = 0; i < FRAME_WORDS; i = i + 1)
                dut.mem[FRAME_W + i] = frames[f * FRAME_WORDS + i];
            dut.mem[MAILBOX_W + 2] = CMD_CLASSIFY;          // cmd[0]
            dut.mem[MAILBOX_W + 3] = 32'h22000;             // cmd[1] frame address
            dut.mem[MAILBOX_W + 4] = FRAME_WORDS * 4;       // cmd[2] bytes
            dut.mem[MAILBOX_W + 5] = f;                     // cmd[3] sequence
            seq = seq + 1;
            dut.mem[MAILBOX_W + 0] = seq;                   // seq_in

            waited = 0;
            while (dut.mem[MAILBOX_W + 1] !== seq && waited < TIMEOUT
                   && !core_trap) begin
                @(posedge clk);
                waited = waited + 1;
            end
            if (core_trap) $fatal(1, "CPU trapped on frame %0d", f);
            if (dut.mem[MAILBOX_W + 1] !== seq) $fatal(1, "timeout on frame %0d", f);
            if (dut.mem[TIMING_W + 0] !== TIMING_MAGIC || dut.mem[TIMING_W + 1] !== f)
                $fatal(1, "no timing words for frame %0d", f);

            $display("RESULT %0d %0d %0d %0d %0d %0d", f,
                     dut.mem[MAILBOX_W + 6], $signed(dut.mem[MAILBOX_W + 7]),
                     $signed(dut.mem[MAILBOX_W + 8]), dut.mem[MAILBOX_W + 9],
                     dut.mem[TIMING_W + 2]);
        end
        $display("DONE");
        $finish;
    end
endmodule
