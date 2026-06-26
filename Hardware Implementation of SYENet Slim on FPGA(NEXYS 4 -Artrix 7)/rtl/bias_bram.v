// =============================================================================
// bias_bram.v  –  Bias storage BRAM (INT32, one entry per output channel)
// Target  : Xilinx Artix-7 XC7A100T
// Style   : Verilog-2001, $readmemh initialisation
//
// Layout (each entry = 32-bit signed bias, already scaled for INT8 inference):
//   [L0]  head_block1_0.bias  : 12 entries @ 0x00
//   [L1]  head_block1_2.bias  : 12 entries @ 0x0C
//   [L2]  head_block2.bias    : 12 entries @ 0x18
//   [L3]  body_block1.bias    : 12 entries @ 0x24
//   [L4]  body_block2.bias    : 12 entries @ 0x30
//   [L5]  att_1.bias          : 12 entries @ 0x3C
//   [L6]  att_3.bias          : 12 entries @ 0x48
//   [L7]  tail_1.bias         :  3 entries @ 0x54
//   Total: 87 entries → DEPTH = 128
// =============================================================================
`timescale 1ns/1ps
module bias_bram #(
    parameter DEPTH    = 128,
    parameter DW       = 32,
    parameter AW       = 7,
    parameter MEM_FILE = "mem/biases.mem"
) (
    input  wire          clk,
    input  wire          en,
    input  wire [AW-1:0] addr,
    output reg  [DW-1:0] dout
);
    reg [DW-1:0] mem [0:DEPTH-1];

    initial begin
        $readmemh(MEM_FILE, mem);
    end

    always @(posedge clk) begin
        if (en)
            dout <= mem[addr];
    end

endmodule
