// =============================================================================
// prelu.v  –  Parametric ReLU (PReLU) activation
// Target  : Xilinx Artix-7 XC7A100T
// Style   : Verilog-2001
//
// Computes:
//   out = (x >= 0) ? x : alpha * x
//
// alpha is Q0.8 unsigned (0..255 maps to 0.0..~1.0).
// Pipeline: 2 cycles latency.
// =============================================================================
`timescale 1ns/1ps
module prelu #(
    parameter AW    = 32,
    parameter OW    = 8,
    parameter SHIFT = 7
) (
    input  wire                   clk,
    input  wire                   rst_n,
    input  wire                   valid_in,
    input  wire signed [AW-1:0]   acc_in,
    input  wire signed [AW-1:0]   bias_in,
    input  wire [OW-1:0]          alpha,
    output reg                    valid_out,
    output reg  [OW-1:0]          data_out
);
    reg signed [AW-1:0]  sum_r;
    reg                   v1;

    // Module-level temporaries (Verilog-2001: no declarations in unnamed blocks)
    reg [OW-1:0]          prelu_abs_x;
    reg [2*OW-1:0]        prelu_scaled;

    always @(posedge clk) begin
        if (!rst_n) begin
            sum_r     <= {AW{1'b0}};
            v1        <= 1'b0;
            valid_out <= 1'b0;
            data_out  <= {OW{1'b0}};
        end else begin
            // Stage 1: add bias
            sum_r <= acc_in + bias_in;
            v1    <= valid_in;

            // Stage 2: PReLU and clamp
            valid_out <= v1;
            if (v1) begin
                if (!sum_r[AW-1]) begin
                    // Positive branch – ReLU-like clamp
                    if (sum_r[AW-1:SHIFT] > {{(AW-SHIFT-OW){1'b0}},{OW{1'b1}}})
                        data_out <= {OW{1'b1}};
                    else
                        data_out <= sum_r[SHIFT+OW-1:SHIFT];
                end else begin
                    // Negative branch: out = -(alpha * |x >> SHIFT|) >> 8
                    prelu_abs_x  = (-sum_r) >> SHIFT;
                    prelu_scaled = alpha * prelu_abs_x;
                    // Saturate to zero if magnitude overflows OW bits
                    data_out <= (prelu_scaled[2*OW-1:OW] != {OW{1'b0}}) ?
                                 {OW{1'b0}} :
                                 (~prelu_scaled[OW-1:0] + 1'b1);
                end
            end
        end
    end
endmodule
