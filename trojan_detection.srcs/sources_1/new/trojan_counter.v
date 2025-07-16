// trojan3.v
module trojan_counter (
    input clk,
    input rst,
    input [3:0] trigger,
    output reg [3:0] count_trojan
);
    reg [7:0] delay_count = 0;
    reg active = 0;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            delay_count <= 0;
            active <= 0;
            count_trojan <= 0;
        end else begin
            if (trigger == 4'b1010)
                delay_count <= delay_count + 1;

            if (delay_count == 10)
                active <= 1;

            if (active)
                count_trojan <= count_trojan + 1;
        end
    end
endmodule
