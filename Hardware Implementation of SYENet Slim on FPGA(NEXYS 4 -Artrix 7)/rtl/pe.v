// =============================================================================
// pe.v  –  Processing Element for 4×4 Systolic Array
// Target  : Xilinx Artix-7 XC7A100T (Nexys-4)
// Style   : Verilog-2001, synchronous reset, DSP48-friendly
//
// Dataflow (weight-stationary):
//   • Weight is loaded once per output-channel group via wload strobe.
//   • During compute, activation is forwarded right (act_out) and the
//     running partial sum is updated every cycle that comp is asserted.
//   • acc_clr resets the accumulator for the first MAC of a new output pixel.
//
// DSP48 mapping hint (Vivado):
//   Stage 1 – register A and B inputs (maps to DSP48 A/B registers)
//   Stage 2 – P <= A_r * B_r          (DSP48 multiplier)
//   Stage 3 – acc <= acc + P           (DSP48 P-reg accumulator cascade)
// =============================================================================
`timescale 1ns/1ps
module pe #(
    parameter DW = 8,    // data width (INT8)
    parameter AW = 32    // accumulator width (INT32)
) (
    input  wire              clk,
    input  wire              rst_n,
    // --- weight load port ---
    input  wire              wload,      // 1 = load weight from wgt_in
    input  wire [DW-1:0]     wgt_in,
    output wire [DW-1:0]     wgt_out,    // daisy-chain weight load
    // --- compute port ---
    input  wire              comp,       // 1 = MAC active this cycle
    input  wire              acc_clr,    // 1 = clear accumulator (new pixel)
    input  wire [DW-1:0]     act_in,
    output reg  [DW-1:0]     act_out,
    output reg  [AW-1:0]     psum_out
);
    // -------------------------------------------------------------------------
    // Stationary weight register
    // -------------------------------------------------------------------------
    reg signed [DW-1:0] weight;
    assign wgt_out = weight;   // expose for daisy-chain inspection / next row

    // -------------------------------------------------------------------------
    // Pipeline registers (three-stage DSP48 friendly)
    //   Stage 1: input registers (A-reg, B-reg in DSP48)
    //   Stage 2: product register (P-reg in DSP48)
    //   Stage 3: accumulator
    // -------------------------------------------------------------------------
    reg signed [DW-1:0]     a_r, b_r;
    reg signed [2*DW-1:0]   p_r;
    reg signed [AW-1:0]     acc;
    reg                      comp_r1, comp_r2;
    reg                      clr_r1,  clr_r2;

    // -------------------------------------------------------------------------
    // Sequential logic
    // -------------------------------------------------------------------------
    always @(posedge clk) begin
        if (!rst_n) begin
            weight    <= {DW{1'b0}};
            a_r       <= {DW{1'b0}};
            b_r       <= {DW{1'b0}};
            p_r       <= {(2*DW){1'b0}};
            acc       <= {AW{1'b0}};
            act_out   <= {DW{1'b0}};
            psum_out  <= {AW{1'b0}};
            comp_r1   <= 1'b0;
            comp_r2   <= 1'b0;
            clr_r1    <= 1'b0;
            clr_r2    <= 1'b0;
        end else begin
            // Weight load (overrides compute this cycle)
            if (wload)
                weight <= $signed(wgt_in);

            // Stage 1 – register
            a_r     <= weight;
            b_r     <= $signed(act_in);
            act_out <= act_in;   // forward horizontally (unregistered option)
            comp_r1 <= comp;
            clr_r1  <= acc_clr;

            // Stage 2 – multiply
            p_r    <= a_r * b_r;
            comp_r2 <= comp_r1;
            clr_r2  <= clr_r1;

            // Stage 3 – accumulate
            if (clr_r2)
                acc <= {{(AW-2*DW){p_r[2*DW-1]}}, p_r};
            else if (comp_r2)
                acc <= acc + {{(AW-2*DW){p_r[2*DW-1]}}, p_r};

            psum_out <= acc;
        end
    end
endmodule
