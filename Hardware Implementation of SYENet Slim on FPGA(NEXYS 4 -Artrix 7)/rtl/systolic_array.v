// =============================================================================
// systolic_array.v  –  4×4 Weight-Stationary Systolic Array
// Target  : Xilinx Artix-7 XC7A100T
// Style   : Verilog-2001, synchronous active-low reset
//
// Topology (ROWS × COLS = 4 × 4 = 16 DSP48 MACs):
//
//        act_in[0]  act_in[1]  act_in[2]  act_in[3]
//             │          │          │          │
//    row0: PE[0][0]--PE[0][1]--PE[0][2]--PE[0][3]
//    row1: PE[1][0]--PE[1][1]--PE[1][2]--PE[1][3]
//    row2: PE[2][0]--PE[2][1]--PE[2][2]--PE[2][3]
//    row3: PE[3][0]--PE[3][1]--PE[3][2]--PE[3][3]
//             │          │          │          │
//        psum_out[0] psum_out[1] psum_out[2] psum_out[3]
//
// Flattened 2D wire encoding (Verilog-2001 compatible):
//   act_h_flat  [(r*(COLS+1) + c)*DW  +: DW]  r=0..ROWS-1, c=0..COLS
//   wgt_h_flat  [(r*(COLS+1) + c)*DW  +: DW]  r=0..ROWS-1, c=0..COLS
//   psum_v_flat [(r*COLS     + c)*AW  +: AW]  r=0..ROWS,   c=0..COLS-1
// =============================================================================
`timescale 1ns/1ps
module systolic_array #(
    parameter ROWS = 4,   // = C_IN  (input channel parallelism)
    parameter COLS = 4,   // = C_OUT slice (output channel parallelism)
    parameter DW   = 8,   // INT8
    parameter AW   = 32   // INT32 accumulator
) (
    input  wire                        clk,
    input  wire                        rst_n,
    // --- weight load interface ---
    input  wire                        wload,
    input  wire [ROWS*DW-1:0]          wgt_in,   // one weight per row
    // --- compute interface ---
    input  wire                        comp,
    input  wire                        acc_clr,
    input  wire [COLS*DW-1:0]          act_in,   // one activation per column
    // --- output ---
    output wire [COLS*AW-1:0]          psum_out  // partial sum per column
);
    // -------------------------------------------------------------------------
    // Flat 1D encodings of 2D wire arrays (Verilog-2001: no 2D unpacked wires)
    //
    //   act_h_flat [(r*(COLS+1) + c)*DW +: DW]  r in 0..ROWS-1, c in 0..COLS
    //   wgt_h_flat [(r*(COLS+1) + c)*DW +: DW]  r in 0..ROWS-1, c in 0..COLS
    //   psum_v_flat[(r*COLS     + c)*AW +: AW]  r in 0..ROWS,   c in 0..COLS-1
    // -------------------------------------------------------------------------
    localparam ACT_FLAT_W  = ROWS * (COLS+1) * DW;
    localparam PSUM_FLAT_W = (ROWS+1) * COLS * AW;

    wire [ACT_FLAT_W-1:0]  act_h_flat;
    wire [ACT_FLAT_W-1:0]  wgt_h_flat;
    wire [PSUM_FLAT_W-1:0] psum_v_flat;

    genvar r, c;

    // -------------------------------------------------------------------------
    // Boundary conditions
    // -------------------------------------------------------------------------

    // Broadcast activation: column c feeds act_h[r][0] = act_in[c] for ALL r
    // (Note: in this topology act_in is broadcast per column, not per row,
    //  so act_in[c] (col-indexed) drives act_h[r][0] for every row r)
    generate
        for (r = 0; r < ROWS; r = r + 1) begin : act_bc
            // act_h[r][0] = act_in[r*DW +: DW]
            // (columns broadcast: each column c gets the same activation to all rows;
            //  here COLS=ROWS=4, so index is compatible)
            assign act_h_flat[(r*(COLS+1) + 0)*DW +: DW] = act_in[r*DW +: DW];
        end
    endgenerate

    // Top partial sums are zero: psum_v[0][c] = 0
    generate
        for (c = 0; c < COLS; c = c + 1) begin : psum_top
            assign psum_v_flat[(0*COLS + c)*AW +: AW] = {AW{1'b0}};
        end
    endgenerate

    // Left weight inputs: wgt_h[r][0] = wgt_in[r*DW +: DW]
    generate
        for (r = 0; r < ROWS; r = r + 1) begin : wgt_in_conn
            assign wgt_h_flat[(r*(COLS+1) + 0)*DW +: DW] = wgt_in[r*DW +: DW];
        end
    endgenerate

    // -------------------------------------------------------------------------
    // PE array instantiation
    // -------------------------------------------------------------------------
    generate
        for (r = 0; r < ROWS; r = r + 1) begin : row_gen
            for (c = 0; c < COLS; c = c + 1) begin : col_gen
                pe #(
                    .DW (DW),
                    .AW (AW)
                ) u_pe (
                    .clk      (clk),
                    .rst_n    (rst_n),
                    .wload    (wload),
                    .wgt_in   (wgt_h_flat  [(r*(COLS+1) + c    )*DW +: DW]),
                    .wgt_out  (wgt_h_flat  [(r*(COLS+1) + c + 1)*DW +: DW]),
                    .comp     (comp),
                    .acc_clr  (acc_clr),
                    .act_in   (act_h_flat  [(r*(COLS+1) + c    )*DW +: DW]),
                    .act_out  (act_h_flat  [(r*(COLS+1) + c + 1)*DW +: DW]),
                    .psum_out (psum_v_flat [((r+1)*COLS + c    )*AW +: AW])
                );
            end
        end
    endgenerate

    // -------------------------------------------------------------------------
    // Output: sum partial sums from all rows for each column
    // psum_out[c] = sum over r of psum_v[r+1][c]
    // -------------------------------------------------------------------------
    generate
        for (c = 0; c < COLS; c = c + 1) begin : col_sum
            assign psum_out[c*AW +: AW] =
                psum_v_flat[(1*COLS + c)*AW +: AW] +
                psum_v_flat[(2*COLS + c)*AW +: AW] +
                psum_v_flat[(3*COLS + c)*AW +: AW] +
                psum_v_flat[(4*COLS + c)*AW +: AW];
        end
    endgenerate

endmodule
