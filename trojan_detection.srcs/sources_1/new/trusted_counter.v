
module trusted_counter (
    input clk,
    input rst,
    output reg [3:0] count
);
always @(posedge clk or posedge rst) begin
        if (rst)
            count <= 4'b0000;  // Reset to all zeros
        else
            count <= {~count[0], count[3:1]};  // Circular shift with inversion
    end
endmodule