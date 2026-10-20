module ram #(parameter integer AW = 16, parameter DECOMP = 0) (
    input wire clk, input wire a_we, input wire a_re, input wire [AW-1:0] a_addr, input wire [31:0] a_wd, output reg [31:0] a_rd,
    input wire b_we, input wire b_re, input wire [AW-1:0] b_addr, input wire [31:0] b_wd, output reg [31:0] b_rd);
    (* ram_style = "block" *) reg [31:0] mem [0:(1<<AW)-1];
    always @(posedge clk) begin
        if (a_we || a_re) begin
            if (a_we) mem[a_addr] <= a_wd;
            a_rd <= mem[a_addr];
        end
    end
    always @(posedge clk) begin
        if (b_we || b_re) begin
            if (b_we) mem[b_addr] <= b_wd;
            b_rd <= mem[b_addr];
        end
    end
endmodule
