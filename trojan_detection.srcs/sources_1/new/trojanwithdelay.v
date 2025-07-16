module trojanwithdelay (
    input clk,
    input rst,
    input [3:0] trigger,
    output reg [3:0] count_trojan
);

    reg [3:0] trigger_count = 0;
    reg activate = 0;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            trigger_count <= 0;
            activate <= 0;
            count_trojan <= 0;
        end else begin
            // Count how many times trigger pattern occurs
            if (trigger == 4'b1010)
                trigger_count <= trigger_count + 1;

            // Activate Trojan behavior after 3 detections
            if (trigger_count == 3)
                activate <= 1;

            // Conditional output increment
            if (activate)
                count_trojan <= count_trojan + $urandom_range(1, 3);  // Trojan behavior
            else
                count_trojan <= count_trojan+ 1;                     // Normal behavior
        end
    end

endmodule
