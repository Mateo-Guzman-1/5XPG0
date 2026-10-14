`timescale 1ns / 1ps

module tb_kw_mac;

    localparam [6:0] F_LDX = 7'd0, F_SETACC = 7'd1, F_MAC4 = 7'd2, F_MAC8 = 7'd3;
    localparam [6:0] OPCODE_CUSTOM_0 = 7'b0001011;

    // --- clock (100 MHz) and reset ---
    reg clk = 1'b0;
    always #5 clk = ~clk;
    reg rst_n = 1'b0;

    // --- signals to/from the unit ---
    reg         pcpi_valid = 1'b0;
    reg  [31:0] pcpi_insn  = 32'd0;
    reg  [31:0] pcpi_rs1   = 32'd0;
    reg  [31:0] pcpi_rs2   = 32'd0;
    wire        pcpi_wr, pcpi_wait, pcpi_ready;
    wire [31:0] pcpi_rd;

    kw_mac_pcpi dut (
        .clk        (clk),
        .rst_n      (rst_n),
        .pcpi_valid (pcpi_valid),
        .pcpi_insn  (pcpi_insn),
        .pcpi_rs1   (pcpi_rs1),
        .pcpi_rs2   (pcpi_rs2),
        .pcpi_wr    (pcpi_wr),
        .pcpi_rd    (pcpi_rd),
        .pcpi_wait  (pcpi_wait),
        .pcpi_ready (pcpi_ready)
    );

    // --- test data ---
    integer x  [0:127];      // inputs, 0..255
    integer w8 [0:127];      // int8 weights, -128..127
    integer w4 [0:127];      // int4 weights, -8..7
    integer errors = 0;
    integer i, j, expected, cycles;
    reg [31:0] word, got_rd;
    reg        got_wr;

    // --- pretend to be the CPU: send one instruction, wait for the answer ---
    task send(input [6:0] funct7, input [6:0] opcode,
              input [31:0] rs1, input [31:0] rs2);
        begin
            @(negedge clk);
            pcpi_insn  = {funct7, 10'd0, 3'd0, 5'd0, opcode};
            pcpi_rs1   = rs1;
            pcpi_rs2   = rs2;
            pcpi_valid = 1'b1;
            @(posedge clk);
            cycles = 0;
            while (!pcpi_ready && cycles < 16) begin
                @(posedge clk);
                cycles = cycles + 1;
            end
            got_wr = pcpi_wr;
            got_rd = pcpi_rd;
            if (!pcpi_ready) begin
                $display("ERROR: no answer to funct7=%0d", funct7);
                errors = errors + 1;
            end
            #1 pcpi_valid = 1'b0;            // like the CPU: drop valid
        end
    endtask

    task check(input [255:0] what, input integer want);
        begin
            if (got_wr !== 1'b1 || $signed(got_rd) !== want) begin
                $display("ERROR: %0s: got wr=%b rd=%0d, want wr=1 rd=%0d",
                         what, got_wr, $signed(got_rd), want);
                errors = errors + 1;
            end
        end
    endtask

    task check_no_write(input [255:0] what);
        begin
            if (got_wr !== 1'b0) begin
                $display("ERROR: %0s wrote a register", what);
                errors = errors + 1;
            end
        end
    endtask

    // one int8 neuron: SETACC(bias) + 32 x MAC4, checking every running sum
    task neuron_int8(input integer bias);
        begin
            send(F_SETACC, OPCODE_CUSTOM_0, bias, 0);
            check_no_write("SETACC");
            expected = bias;
            for (i = 0; i < 32; i = i + 1) begin
                word = 0;
                for (j = 0; j < 4; j = j + 1) begin
                    word[8*j +: 8] = w8[4*i + j];
                    expected = expected + w8[4*i + j] * x[4*i + j];
                end
                send(F_MAC4, OPCODE_CUSTOM_0, word, 0);
                check("MAC4", expected);
            end
            $display("int8 neuron, bias %0d: sum %0d (expected %0d)",
                     bias, $signed(got_rd), expected);
        end
    endtask

    initial begin
        // inputs: a spread of values, with the extremes 0 and 255
        for (i = 0; i < 128; i = i + 1) begin
            x[i]  = (i * 37 + 11) % 256;
            w8[i] = (i * 53 + 7) % 256 - 128;
            w4[i] = (i * 5 + 3) % 16 - 8;
        end
        x[0] = 255; x[1] = 0;   x[2] = 255; x[3] = 255;
        w8[0] = -128; w8[2] = 127; w4[0] = -8; w4[2] = 7;

        repeat (3) @(posedge clk);
        rst_n = 1'b1;

        // 1. load the 128 inputs, 4 per KW_LDX
        for (i = 0; i < 32; i = i + 1) begin
            word = {x[4*i + 3][7:0], x[4*i + 2][7:0], x[4*i + 1][7:0], x[4*i][7:0]};
            send(F_LDX, OPCODE_CUSTOM_0, 4 * i, word);
            check_no_write("LDX");
        end
        for (i = 0; i < 128; i = i + 1)
            if (dut.input_buffer[i] !== x[i][7:0]) begin
                $display("ERROR: input_buffer[%0d] = %0d, want %0d",
                         i, dut.input_buffer[i], x[i]);
                errors = errors + 1;
            end

        // 2. int8 neuron
        neuron_int8(-1000);

        // 3. int4 neuron: SETACC + 16 x MAC8
        send(F_SETACC, OPCODE_CUSTOM_0, 500, 0);
        check_no_write("SETACC");
        expected = 500;
        for (i = 0; i < 16; i = i + 1) begin
            word = 0;
            for (j = 0; j < 8; j = j + 1) begin
                word[4*j +: 4] = w4[8*i + j];
                expected = expected + w4[8*i + j] * x[8*i + j];
            end
            send(F_MAC8, OPCODE_CUSTOM_0, word, 0);
            check("MAC8", expected);
        end
        $display("int4 neuron, bias 500: sum %0d (expected %0d)",
                 $signed(got_rd), expected);

        // 4. a second int8 neuron: SETACC must start again at input 0
        neuron_int8(12345);

        // 5. a mul (opcode 0110011, funct7 0000001) is not ours
        @(negedge clk);
        pcpi_insn  = {7'b0000001, 10'd0, 3'd0, 5'd0, 7'b0110011};
        pcpi_valid = 1'b1;
        repeat (5) begin
            @(posedge clk);
            if (pcpi_ready) begin
                $display("ERROR: the unit answered a mul instruction");
                errors = errors + 1;
            end
        end
        #1 pcpi_valid = 1'b0;

        if (errors == 0) $display("PASS");
        else             $display("FAIL: %0d errors", errors);
        $finish;
    end

endmodule