module ram_be2_dec #(parameter integer AW = 16) (
    input wire clk, input wire a_re, input wire [3:0] a_be, input wire a_we, input wire [AW-1:0] a_addr, input wire [31:0] a_wd, output reg [31:0] a_rd,
    input wire b_re, input wire [3:0] b_be, input wire [AW-1:0] b_addr, input wire [31:0] b_wd, output reg [31:0] b_rd);
    (* ram_style = "block", ram_decomp = "power" *) reg [31:0] mem [0:(1<<AW)-1];
    wire b_we = |b_be;
    // variant 1 (as spike_soc today): separate write and read enables
    always @(posedge clk) begin
        if (a_we || a_re) begin if (a_we) begin
            if (a_be[0]) mem[a_addr][7:0]   <= a_wd[7:0];
            if (a_be[1]) mem[a_addr][15:8]  <= a_wd[15:8];
            if (a_be[2]) mem[a_addr][23:16] <= a_wd[23:16];
            if (a_be[3]) mem[a_addr][31:24] <= a_wd[31:24];
        end
        a_rd <= mem[a_addr]; end
    end
    always @(posedge clk) begin
        if (b_we || b_re) begin if (b_we) begin
            if (b_be[0]) mem[b_addr][7:0]   <= b_wd[7:0];
            if (b_be[1]) mem[b_addr][15:8]  <= b_wd[15:8];
            if (b_be[2]) mem[b_addr][23:16] <= b_wd[23:16];
            if (b_be[3]) mem[b_addr][31:24] <= b_wd[31:24];
        end
        b_rd <= mem[b_addr]; end
    end
endmodule
