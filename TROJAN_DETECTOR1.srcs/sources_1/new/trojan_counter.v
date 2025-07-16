`timescale 1ns / 1ps

module trojan_counter (
    input clk,
    input  rst,
    input  [3:0] trigger,
    output reg [3:0] count_trojan
);
    
    always @(posedge clk or posedge rst) begin
        if (rst)
            count_trojan <= 4'b0000;
        else if (trigger == 4'b1010)
        begin
           
            count_trojan <= 4'b1000+ count_trojan +2 ;
            end // Malicious behavior
        else
            count_trojan <= count_trojan + 1; 
    end
endmodule
