`timescale 1ns/1ps
module tb_led;
    reg clk=0, rst=0;
    always #5 clk=~clk;
    wire [9:0] led;
    wire [31:0] timer;
    reg valid=0;
    reg [31:0] addr=0,data=0;
    spike_soc dut(.clk(clk),.core_rst_n(rst),.pa_we(1'b0),.pa_addr(16'd0),
      .pa_wdata(32'd0),.pa_be(4'd0),.pa_rdata(),.core_running(),.core_trap(),.timer_lo(timer),.led_o(led));
    task write(input [31:0] a,input [31:0] d);
        @(negedge clk);addr=a;data=d;valid=1;
        do @(negedge clk);while(!dut.cpu_mem_ready);
        valid=0;
        @(negedge clk);
    endtask
    initial begin
        force dut.cpu_mem_valid=valid;
        force dut.cpu_mem_addr=addr;
        force dut.cpu_mem_wdata=data;
        force dut.cpu_mem_wstrb=4'hf;
        repeat(4) @(negedge clk);
        if (^timer === 1'bx) $fatal(1,"Timer starts unknown");
        rst=1;
        write(32'h10002004,20);
        if(led!==1) $fatal(1,"Pulse not started");
        repeat(3) @(negedge clk);
        write(32'h10002004,20);
        if(dut.led_remaining!==19) $fatal(1,"Pulse not retriggered");
        repeat(18) @(negedge clk);
        if(led!==1) $fatal(1,"Pulse ended early");
        @(negedge clk);
        if(led!==0) $fatal(1,"Pulse ended late");
        write(32'h10002004,100);
        write(32'h10002004,0);
        if(led!==0) $fatal(1,"Zero duration did not cancel");
        write(32'h10002004,100);
        rst=0; @(negedge clk);
        if(led!==0) $fatal(1,"Reset did not clear LED");
        rst=1;
        force dut.timer_lo=32'hfffffff8;
        @(negedge clk); release dut.timer_lo;
        write(32'h10002004,20);
        repeat(20) @(negedge clk);
        if(led!==0 || timer>100) $fatal(1,"Timer wrap behavior failed");
        $display("PASS LED: initialization, exact duration, retrigger, cancel, reset, timer wrap");
        $finish;
    end
    initial begin #100000; $fatal(1,"LED test timeout"); end
endmodule
