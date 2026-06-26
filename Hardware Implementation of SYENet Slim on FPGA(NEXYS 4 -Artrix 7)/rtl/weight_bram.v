// =============================================================================
// weight_bram.v  –  Weight storage BRAM (True Dual-Port inferred)
// Target  : Xilinx Artix-7 XC7A100T
// Style   : Verilog-2001, $readmemh initialisation
//
// Stores all INT8 conv weights for every layer, packed as 8-bit entries.
// Layout in memory (byte-addressed):
//   [L0]  head_block1_0.weight  : 4×12×3×3  = 432 entries  @  0x000
//   [L1]  head_block1_2.weight  : 12×12×3×3 = 1296 entries @ 0x1B0
//   [L2]  head_block2.weight    : 12×4×1×1  = 48 entries   @ 0x6C0
//   [L3]  body_block1.weight    : 12×12×3×3 = 1296 entries @ 0x6F0
//   [L4]  body_block2.weight    : 12×12×1×1 = 144 entries  @ 0xC00
//   [L5]  att_1.weight          : 12×12×1×1 = 144 entries  @ 0xC90
//   [L6]  att_3.weight          : 12×12×1×1 = 144 entries  @ 0xD20
//   [L7]  tail_1.weight         : 3×12×1×1  = 36 entries   @ 0xDB0
//   Total: 3540 entries  →  DEPTH ≥ 4096 (2^12)
//
// Port A: write (unused at runtime, for simulation initialisation only)
// Port B: read (CNN accelerator reads sequentially)
// =============================================================================
`timescale 1ns/1ps
module weight_bram #(
    parameter DEPTH    = 8192,
    parameter DW       = 8,
    parameter AW       = 13,
    parameter MEM_FILE = "mem/weights.mem"
) (
    input  wire          clk,
    // Port A (write – simulation/init only)
    input  wire          a_en,
    input  wire          a_we,
    input  wire [AW-1:0] a_addr,
    input  wire [DW-1:0] a_din,
    // Port B (read – CNN accelerator)
    input  wire          b_en,
    input  wire [AW-1:0] b_addr,
    output reg  [DW-1:0] b_dout
);
    reg [DW-1:0] mem [0:DEPTH-1];

    initial begin
        $readmemh(MEM_FILE, mem);
    end

    // Port A
    always @(posedge clk) begin
        if (a_en && a_we)
            mem[a_addr] <= a_din;
    end

    // Port B (read-only at inference time)
    always @(posedge clk) begin
        if (b_en)
            b_dout <= mem[b_addr];
    end

endmodule
