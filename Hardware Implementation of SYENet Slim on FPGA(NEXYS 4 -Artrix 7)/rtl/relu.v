// =============================================================================
// relu.v  –  Pipelined ReLU + bias-add + scale-shift for INT8 output
// Target  : Xilinx Artix-7 XC7A100T
// Style   : Verilog-2001
//
// Computes:
//   out = clamp( (acc + bias) >> SHIFT , 0, 255 )
//
// Parameters:
//   AW    – accumulator width (32)
//   OW    – output width (8)
//   SHIFT – right-shift amount for INT8 re-quantisation (default 7 for Q1.7)
// =============================================================================
`timescale 1ns/1ps
module relu #(
    parameter AW    = 32,
    parameter OW    = 8,
    parameter SHIFT = 7
) (
    input  wire              clk,
    input  wire              rst_n,
    input  wire              valid_in,
    input  wire signed [AW-1:0]  acc_in,
    input  wire signed [AW-1:0]  bias_in,
    output reg               valid_out,
    output reg  [OW-1:0]     data_out
);
    // -------------------------------------------------------------------------
    // Stage 1: add bias
    // Stage 2: shift and clamp
    // -------------------------------------------------------------------------
    reg signed [AW-1:0] sum_r;
    reg                  v_r;

    always @(posedge clk) begin
        if (!rst_n) begin
            sum_r    <= {AW{1'b0}};
            v_r      <= 1'b0;
            valid_out <= 1'b0;
            data_out  <= {OW{1'b0}};
        end else begin
            // Stage 1
            sum_r <= acc_in + bias_in;
            v_r   <= valid_in;
            // Stage 2
            valid_out <= v_r;
            if (v_r) begin
                if (sum_r[AW-1])
                    // Negative → zero (ReLU)
                    data_out <= {OW{1'b0}};
                else begin
                    // Positive: right-shift, saturate to 255
                    if (sum_r[AW-1:SHIFT] > {{(AW-SHIFT-OW){1'b0}},{OW{1'b1}}})
                        data_out <= {OW{1'b1}};   // saturate
                    else
                        data_out <= sum_r[SHIFT+OW-1:SHIFT];
                end
            end
        end
    end
endmodule
