
module subtinctrojan( input clk, input rst, input [3:0] trigger, output reg [3:0] count_trojan);

reg [1:0] boost = 0;

always @(posedge clk or posedge rst)
begin
 if(rst)
 count_trojan <= 4'b0000;
 else
 begin
  if (trigger == 4'b1010)
    boost <= 2; 
  else
    boost <= 0;
  count_trojan <= count_trojan + 1 + boost;
end
end
endmodule 