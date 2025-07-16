`timescale 1ns/1ps

module tb;
    reg clk;
    reg rst;
    reg[3:0] trigger = 4'd0;
    wire [3:0] count_trojan;

    // Instantiate the design
    trojan_counter trusted_inst (
        .clk(clk),
        .rst(rst),.trigger(trigger),
        .count_trojan(count_trojan)
    );

    // Clock generator (10ns period)
    initial clk = 0;
    always #5 clk = ~clk;

    // Stimulus
    initial begin
        rst = 1;
        #10;
        rst = 0;
        #50 trigger = 10;
        #150 trigger =0;
        #500
              $dumpfile("dumptrojanT.vcd");
                $dumpvars(0, tb);
        #400; 
        $finish;
    end
endmodule
