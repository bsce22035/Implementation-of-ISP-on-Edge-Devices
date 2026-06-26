// =============================================================================
// global_avg_pool.v  –  Global Average Pooling
// Target  : Xilinx Artix-7 XC7A100T
// Style   : Verilog-2001
//
// Streams IMG_H × IMG_W × C_IN pixels in (one channel per cycle, channel-last
// order) and outputs a C_IN-element vector of averages.
//
// Division by H×W implemented as right-shift:
//   H=W=128 → H×W=16384 → shift by 14 (exact power of two)
//
// Accumulates INT8 inputs into INT32 accumulators (no overflow for 128×128).
// Outputs INT8 averages after shifting.
// =============================================================================
`timescale 1ns/1ps
module global_avg_pool #(
    parameter IMG_H = 128,
    parameter IMG_W = 128,
    parameter C     = 12,
    parameter DW    = 8,
    parameter AW    = 32
) (
    input  wire              clk,
    input  wire              rst_n,
    input  wire              start,       // pulse: begin accumulating
    input  wire              valid_in,    // one pixel-channel per cycle
    input  wire [DW-1:0]     din,
    input  wire [$clog2(C)-1:0] ch_in,   // which channel this pixel belongs to
    output reg               done,        // pulse: outputs are valid
    output reg  [C*DW-1:0]  avg_out      // C channels packed
);
    // -------------------------------------------------------------------------
    // Accumulators: one per channel
    // -------------------------------------------------------------------------
    localparam TOTAL    = IMG_H * IMG_W;   // 16384
    localparam LOG2_TOT = 14;              // log2(16384)
    localparam CNT_W    = $clog2(TOTAL)+1;

    reg [AW-1:0] acc  [0:C-1];
    reg [CNT_W-1:0] pixel_cnt;
    reg running;

    integer i;

    always @(posedge clk) begin
        if (!rst_n) begin
            for (i = 0; i < C; i = i + 1)
                acc[i] <= {AW{1'b0}};
            pixel_cnt <= {CNT_W{1'b0}};
            running   <= 1'b0;
            done      <= 1'b0;
            avg_out   <= {C*DW{1'b0}};
        end else begin
            done <= 1'b0;

            if (start) begin
                for (i = 0; i < C; i = i + 1)
                    acc[i] <= {AW{1'b0}};
                pixel_cnt <= {CNT_W{1'b0}};
                running   <= 1'b1;
            end

            if (running && valid_in) begin
                // Accumulate (unsigned sum of UINT8 values)
                acc[ch_in] <= acc[ch_in] + {{(AW-DW){1'b0}}, din};
                pixel_cnt  <= pixel_cnt + 1'b1;

                if (pixel_cnt == TOTAL*C - 1) begin
                    running <= 1'b0;
                    done    <= 1'b1;
                    // Shift to produce average per channel
                    for (i = 0; i < C; i = i + 1)
                        avg_out[i*DW +: DW] <= acc[i][LOG2_TOT+DW-1:LOG2_TOT];
                end
            end
        end
    end
endmodule
