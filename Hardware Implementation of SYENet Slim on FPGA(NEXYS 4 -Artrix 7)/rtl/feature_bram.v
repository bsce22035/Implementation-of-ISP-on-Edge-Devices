// =============================================================================
// feature_bram.v  –  Feature-map double-buffer BRAM
// Target  : Xilinx Artix-7 XC7A100T
// Style   : Verilog-2001, True Dual-Port inferred BRAM
//
// Stores two complete 128×128×C_MAX feature maps (ping-pong).
//   Buffer 0: addresses 0x00000 … 0x2FFFF  (C_MAX=12, 128×128×12 = 196 608)
//   Buffer 1: addresses 0x30000 … 0x5FFFF
//   Total depth = 393 216 × 8-bit entries
//   In practice Vivado infers multiple BRAM36 blocks automatically.
//
// Port A: CNN accelerator write (current output buffer)
// Port B: CNN accelerator read  (previous output / current input buffer)
//
// buf_sel: 0 → Port-A writes buffer 0, Port-B reads buffer 1
//          1 → Port-A writes buffer 1, Port-B reads buffer 0
// (Swapped each time a complete feature map is finished.)
//
// Address scheme per buffer:
//   addr = row*IMG_W*C + col*C + ch
//   where IMG_H=IMG_W=128, C=12
// =============================================================================
`timescale 1ns/1ps
module feature_bram #(
    parameter IMG_H  = 128,
    parameter IMG_W  = 128,
    parameter C      = 12,
    parameter DW     = 8,
    parameter DEPTH  = 393216,  // 2 * IMG_H * IMG_W * C
    parameter AW     = 19       // ceil(log2(393216)) = 19
) (
    input  wire          clk,
    input  wire          buf_sel,   // 0 or 1
    // Port A – write
    input  wire          a_en,
    input  wire          a_we,
    input  wire [AW-2:0] a_addr,    // 18-bit offset within one buffer
    input  wire [DW-1:0] a_din,
    // Port B – read
    input  wire          b_en,
    input  wire [AW-2:0] b_addr,
    output reg  [DW-1:0] b_dout
);
    localparam BUF_SIZE = IMG_H * IMG_W * C;  // 196 608

    reg [DW-1:0] mem [0:DEPTH-1];

    // Full addresses (buffer select prepended as MSB)
    wire [AW-1:0] a_full = {buf_sel,        a_addr};
    wire [AW-1:0] b_full = {~buf_sel,       b_addr};

    always @(posedge clk) begin
        if (a_en && a_we)
            mem[a_full] <= a_din;
    end

    always @(posedge clk) begin
        if (b_en)
            b_dout <= mem[b_full];
    end

endmodule
