`timescale 1ns/1ps

module tb;
    reg clk;
    reg rst;
    wire [3:0] count_trusted;

    // Instantiate the design
    trusted_counter trusted_inst (
        .clk(clk),
        .rst(rst),
        .count(count_trusted)
    );

    // Clock generator (10ns period)
    initial clk = 0;
    always #5 clk = ~clk;

    // Stimulus
    initial begin
        rst = 1;
        #20;
        rst = 0;
              $dumpfile("dumptrustedA.vcd");
                $dumpvars(0, tb);
        #500;
        $finish;
    end
endmodule
