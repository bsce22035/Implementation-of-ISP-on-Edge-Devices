// =============================================================================
// sliding_window.v  –  K×K sliding window generator
// Target  : Xilinx Artix-7 XC7A100T
// Style   : Verilog-2001, synchronous active-low reset
//
// Receives K rows from line_buffer output (flat packed bus) and produces
// a flattened K×K×C window of input pixels every valid cycle.
//
// row_in port is flat packed [K*C*DW-1:0] (Verilog-2001 compatible):
//   row_in[r*C*DW +: C*DW] = row r data
//
// window output:
//   window[(r*K+c)*C*DW +: C*DW] = shift_reg[r][c]
// =============================================================================
`timescale 1ns/1ps
module sliding_window #(
    parameter K  = 3,
    parameter C  = 12,
    parameter DW = 8
) (
    input  wire                        clk,
    input  wire                        rst_n,
    input  wire                        wr_en,
    // K rows from line buffer – flat packed (Verilog-2001 compatible)
    input  wire [K*C*DW-1:0]           row_in,
    // Flattened K*K*C*DW output
    output wire [K*K*C*DW-1:0]         window,
    output reg                          window_valid
);
    // -------------------------------------------------------------------------
    // K shift registers, each K pixels wide, each pixel = C*DW bits
    // 2D reg arrays are legal in Verilog-2001 (multi-dimensional memories)
    // -------------------------------------------------------------------------
    reg [C*DW-1:0] sr [0:K-1][0:K-1];  // sr[row][col]

    integer r, c;

    always @(posedge clk) begin
        if (!rst_n) begin
            window_valid <= 1'b0;
            for (r = 0; r < K; r = r + 1)
                for (c = 0; c < K; c = c + 1)
                    sr[r][c] <= {(C*DW){1'b0}};
        end else if (wr_en) begin
            for (r = 0; r < K; r = r + 1) begin
                // Shift left: col 0 is oldest, col K-1 is newest
                for (c = 0; c < K-1; c = c + 1)
                    sr[r][c] <= sr[r][c+1];
                // Variable part-select: legal in Verilog-2001
                sr[r][K-1] <= row_in[r*C*DW +: C*DW];
            end
            window_valid <= 1'b1;
        end else begin
            window_valid <= 1'b0;
        end
    end

    // -------------------------------------------------------------------------
    // Flatten to output bus: window[row][col] packed
    //   Bit offset for (row r, col c) = (r*K + c) * C*DW
    // -------------------------------------------------------------------------
    genvar gr, gc;
    generate
        for (gr = 0; gr < K; gr = gr + 1) begin : flat_row
            for (gc = 0; gc < K; gc = gc + 1) begin : flat_col
                assign window[(gr*K+gc)*C*DW +: C*DW] = sr[gr][gc];
            end
        end
    endgenerate

endmodule
