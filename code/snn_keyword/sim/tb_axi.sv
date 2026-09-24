`timescale 1ns/1ps
module tb_axi;
    reg clk=0;
    always #5 clk=~clk;
    reg resetn=0;
    reg [31:0] awaddr=0,wdata=0,araddr=0;
    reg awvalid=0,wvalid=0,bready=0,arvalid=0,rready=0;
    reg [3:0] wstrb=0;
    wire awready,wready,bvalid,arready,rvalid;
    wire [1:0] bresp,rresp;
    wire [31:0] rdata;
    wire we; wire [15:0] addr; wire [31:0] wd; wire [3:0] be;
    reg [31:0] rd;
    reg [31:0] mem[0:65535];
    wire core_rst; wire [31:0] scratch;
    integer i;
    ps_if dut(.aclk(clk),.aresetn(resetn),
      .s_axil_awaddr(awaddr),.s_axil_awvalid(awvalid),.s_axil_awready(awready),
      .s_axil_wdata(wdata),.s_axil_wstrb(wstrb),.s_axil_wvalid(wvalid),.s_axil_wready(wready),
      .s_axil_bresp(bresp),.s_axil_bvalid(bvalid),.s_axil_bready(bready),
      .s_axil_araddr(araddr),.s_axil_arvalid(arvalid),.s_axil_arready(arready),
      .s_axil_rdata(rdata),.s_axil_rresp(rresp),.s_axil_rvalid(rvalid),.s_axil_rready(rready),
      .bram_we(we),.bram_addr(addr),.bram_wdata(wd),.bram_be(be),.bram_rdata(rd),
      .core_rst(core_rst),.core_running(1'b0),.core_trap(1'b0),.timer_now(32'd123),.scratch(scratch),.led(10'd5));
    always @(posedge clk) begin
        if(we) for(integer b=0;b<4;b=b+1) if(be[b]) mem[addr][b*8+:8]<=wd[b*8+:8];
        rd<=mem[addr];
    end
    task aw(input [31:0] a);
        @(negedge clk); awaddr=a;awvalid=1;
        do @(posedge clk); while(!awready);
        @(negedge clk); awvalid=0;
    endtask
    task ww(input [31:0] d,input [3:0] b);
        @(negedge clk); wdata=d;wstrb=b;wvalid=1;
        do @(posedge clk); while(!wready);
        @(negedge clk); wvalid=0;
    endtask
    task response;
        wait(bvalid); repeat(3) @(negedge clk);
        if(!bvalid || bresp!==0) $fatal(1,"Write response lost");
        bready=1; @(negedge clk); bready=0;
    endtask
    task read_check(input [31:0] a,input [31:0] expected,input [1:0] status);
        @(negedge clk); araddr=a; arvalid=1;
        do @(posedge clk); while(!arready);
        @(negedge clk); arvalid=0;
        wait(rvalid); repeat(3) @(negedge clk);
        if(rdata!==expected || rresp!==status || !rvalid)
            $fatal(1,"Read %x got %x expected %x status %x",a,rdata,expected,rresp);
        rready=1; @(negedge clk); rready=0;
    endtask
    initial begin
        for(i=0;i<65536;i=i+1) mem[i]=0;
        repeat(4) @(negedge clk); resetn=1;
        aw(32'h40000000); repeat(3) @(negedge clk); ww(32'h12345678,15); response();
        ww(32'haabbccdd,5); repeat(2) @(negedge clk); aw(32'h40000000); response();
        read_check(32'h40000000,32'h12bb56dd,0);
        aw(32'h40000004); ww(32'hcafef00d,15); response();
        // Read address must not be replaced by a queued write during capture.
        fork
            read_check(32'h40000000,32'h12bb56dd,0);
            begin aw(32'h40000004); ww(32'hdeadbeef,15); response(); end
        join
        read_check(32'h40000004,32'hdeadbeef,0);
        aw(32'h4004000c); ww(32'h11223344,15); response();
        aw(32'h4004000c); ww(32'hff000000,8); response();
        read_check(32'h4004000c,32'hff223344,0);
        read_check(32'h40040014,32'h534b454c,0);
        $display("PASS AXI: independent AW/W, byte lanes, backpressure, concurrent BRAM read/write, syscon");
        $finish;
    end
    initial begin #100000; $fatal(1,"AXI timeout"); end
endmodule
