// =============================================================================
// scheduler_fsm.v  –  Layer Scheduler FSM for SYENet-Slim
// Target  : Xilinx Artix-7 XC7A100T
// Style   : Verilog-2001
//
// Sequences the following inference pipeline:
//   LAYER 0: Conv2d(4→12, 3×3)   head.block1.0
//   LAYER 1: BN scale             head.block1.1  (fused into weights/biases)
//   LAYER 2: Conv2d(12→12, 3×3)  head.block1.2  (re-param)
//   LAYER 3: Conv2d(4→12, 1×1)   head.block2    (skip)
//   LAYER 4: Add(L2, L3) + PReLU
//   LAYER 5: Conv2d(12→12, 3×3)  body.block1
//   LAYER 6: Conv2d(12→12, 1×1)  body.block2    (skip)
//   LAYER 7: Add(L5, L6) + PReLU
//   LAYER 8: ChannelAttention(12)
//   LAYER 9: GlobalAvgPool(12)
//   LAYER10: FC/Conv1×1(12→3)    tail.1
//   DONE
//
// Each layer fires its start pulse; the scheduler waits for done pulse.
// Buffer swap (buf_sel toggle) happens after each full feature-map layer.
// =============================================================================
`timescale 1ns/1ps
module scheduler_fsm (
    input  wire  clk,
    input  wire  rst_n,
    input  wire  go,              // from system controller: start inference

    // Layer control pulses
    output reg   l0_start,
    input  wire  l0_done,
    output reg   l1_start,        // BN fused – fires immediately after L0
    input  wire  l1_done,
    output reg   l2_start,
    input  wire  l2_done,
    output reg   l3_start,        // skip conv
    input  wire  l3_done,
    output reg   l4_start,        // Add + PReLU
    input  wire  l4_done,
    output reg   l5_start,
    input  wire  l5_done,
    output reg   l6_start,
    input  wire  l6_done,
    output reg   l7_start,
    input  wire  l7_done,
    output reg   l8_start,        // channel attention
    input  wire  l8_done,
    output reg   l9_start,        // global avg pool
    input  wire  l9_done,
    output reg   l10_start,       // classifier FC
    input  wire  l10_done,

    // Buffer swap control
    output reg   buf_swap,

    // Done / status
    output reg   infer_done,
    output reg [3:0] layer_id     // current layer (for debug/LEDs)
);
    localparam S_IDLE  = 5'd0;
    localparam S_L0    = 5'd1;
    localparam S_L1    = 5'd2;
    localparam S_L2    = 5'd3;
    localparam S_L3    = 5'd4;
    localparam S_L4    = 5'd5;
    localparam S_L5    = 5'd6;
    localparam S_L6    = 5'd7;
    localparam S_L7    = 5'd8;
    localparam S_L8    = 5'd9;
    localparam S_L9    = 5'd10;
    localparam S_L10   = 5'd11;
    localparam S_DONE  = 5'd12;

    reg [4:0] state;

    always @(posedge clk) begin
        if (!rst_n) begin
            state      <= S_IDLE;
            l0_start   <= 1'b0;
            l1_start   <= 1'b0;
            l2_start   <= 1'b0;
            l3_start   <= 1'b0;
            l4_start   <= 1'b0;
            l5_start   <= 1'b0;
            l6_start   <= 1'b0;
            l7_start   <= 1'b0;
            l8_start   <= 1'b0;
            l9_start   <= 1'b0;
            l10_start  <= 1'b0;
            buf_swap   <= 1'b0;
            infer_done <= 1'b0;
            layer_id   <= 4'd0;
        end else begin
            l0_start  <= 1'b0; l1_start <= 1'b0; l2_start  <= 1'b0;
            l3_start  <= 1'b0; l4_start <= 1'b0; l5_start  <= 1'b0;
            l6_start  <= 1'b0; l7_start <= 1'b0; l8_start  <= 1'b0;
            l9_start  <= 1'b0; l10_start <= 1'b0;
            buf_swap   <= 1'b0;
            infer_done <= 1'b0;

            case (state)
            S_IDLE: begin
                layer_id <= 4'd0;
                if (go) begin l0_start <= 1'b1; state <= S_L0; end
            end
            S_L0: begin
                layer_id <= 4'd0;
                if (l0_done) begin l1_start <= 1'b1; state <= S_L1; end
            end
            S_L1: begin
                layer_id <= 4'd1;
                if (l1_done) begin
                    l2_start <= 1'b1; l3_start <= 1'b1;  // L2 and L3 in parallel
                    state    <= S_L2;
                end
            end
            S_L2: begin
                layer_id <= 4'd2;
                if (l2_done && l3_done) begin
                    l4_start <= 1'b1; state <= S_L4;
                end
            end
            S_L4: begin
                layer_id <= 4'd4;
                if (l4_done) begin
                    buf_swap  <= 1'b1;
                    l5_start  <= 1'b1; l6_start <= 1'b1;
                    state     <= S_L5;
                end
            end
            S_L5: begin
                layer_id <= 4'd5;
                if (l5_done && l6_done) begin
                    l7_start <= 1'b1; state <= S_L7;
                end
            end
            S_L7: begin
                layer_id <= 4'd7;
                if (l7_done) begin
                    buf_swap  <= 1'b1;
                    l8_start  <= 1'b1; state <= S_L8;
                end
            end
            S_L8: begin
                layer_id <= 4'd8;
                if (l8_done) begin l9_start <= 1'b1; state <= S_L9; end
            end
            S_L9: begin
                layer_id <= 4'd9;
                if (l9_done) begin l10_start <= 1'b1; state <= S_L10; end
            end
            S_L10: begin
                layer_id <= 4'd10;
                if (l10_done) begin state <= S_DONE; end
            end
            S_DONE: begin
                infer_done <= 1'b1;
                state      <= S_IDLE;
            end
            default: state <= S_IDLE;
            endcase
        end
    end
endmodule
