// =============================================================================
// image_bram.v  –  Raw image reception buffer (256×256×4 = 262 144 bytes)
// Target  : Xilinx Artix-7 XC7A100T
// Style   : Verilog-2001, Simple Dual-Port inferred BRAM
//
// Port A: UART image loader write
// Port B: Image resize module read
// =============================================================================
`timescale 1ns/1ps
module image_bram #(
    parameter DEPTH = 262144,   // 256*256*4
    parameter DW    = 8,
    parameter AW    = 18        // ceil(log2(262144)) = 18
) (
    input  wire          clk,
    // Port A – write
    input  wire          a_en,
    input  wire          a_we,
    input  wire [AW-1:0] a_addr,
    input  wire [DW-1:0] a_din,
    // Port B – read
    input  wire          b_en,
    input  wire [AW-1:0] b_addr,
    output reg  [DW-1:0] b_dout
);
    reg [DW-1:0] mem [0:DEPTH-1];

    always @(posedge clk) begin
        if (a_en && a_we)
            mem[a_addr] <= a_din;
    end

    always @(posedge clk) begin
        if (b_en)
            b_dout <= mem[b_addr];
    end
endmodule
