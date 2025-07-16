module randombitflip_trojan ( input clk, input rst, input [3:0] trigger, output reg [3:0] count_trojan);

always @(posedge clk) begin

 if(rst)
 count_trojan <= 4'b0000;
 else
 begin
  if (trigger == 4'b1010) 
    count_trojan <= count_trojan^ (1 << $urandom_range(0, 3));  // Random bit flip
  else 
    count_trojan <= count_trojan + 1;
  
end
end
endmodule