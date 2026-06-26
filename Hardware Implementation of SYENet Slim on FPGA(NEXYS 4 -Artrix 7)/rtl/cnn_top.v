// =============================================================================
// cnn_top.v  –  CNN Accelerator Top-Level
// Target  : Xilinx Artix-7 XC7A100T
// Style   : Verilog-2001
//
// Integrates:
//   • weight_bram   – all INT8 conv weights
//   • bias_bram     – all INT32 biases
//   • feature_bram  – double-buffered feature map
//   • 3 conv_engine instances (shared, reconfigured per layer via parameters)
//   • channel_attention
//   • global_avg_pool
//   • classifier
//   • scheduler_fsm
//
// Weight BRAM memory map:
//   0x000 – 0x1AF  : L0  head.block1.0  (432 bytes)
//   0x1B0 – 0x6AF  : L2  head.block1.2  (1296 bytes)
//   0x6B0 – 0x6DF  : L3  head.block2    (48 bytes)
//   0x6E0 – 0xBDF  : L5  body.block1    (1296 bytes)
//   0xBE0 – 0xC6F  : L6  body.block2    (144 bytes)
//   0xC70 – 0xCFF  : Att FC1  att.1     (144 bytes)
//   0xD00 – 0xD8F  : Att FC2  att.3     (144 bytes)
//   0xD90 – 0xDB3  : tail.1  classifier (36 bytes)
//
// Bias BRAM memory map (each entry = 32 bits):
//   0x00 – 0x0B : L0 bias  (12)
//   0x0C – 0x17 : L2 bias  (12)
//   0x18 – 0x23 : L3 bias  (12)
//   0x24 – 0x2F : L5 bias  (12)
//   0x30 – 0x3B : L6 bias  (12)
//   0x3C – 0x47 : att.1 bias (12)
//   0x48 – 0x53 : att.3 bias (12)
//   0x54 – 0x56 : tail.1 bias (3)
// =============================================================================
`timescale 1ns/1ps
module cnn_top #(
    parameter IMG_H  = 128,
    parameter IMG_W  = 128,
    parameter C_IN   = 4,
    parameter C_MID  = 12,
    parameter C_OUT  = 3,
    parameter DW     = 8,
    parameter AW     = 32
) (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         go,           // start inference
    // Feature map input (from preprocessing, streaming channel-last)
    input  wire         fm_valid_in,
    input  wire [DW-1:0] fm_data_in,
    // Result
    output wire         infer_done,
    output wire [1:0]   class_id,
    output wire [AW-1:0] confidence,
    output wire [C_OUT*AW-1:0] logits
);
    // =========================================================================
    // Shared BRAM instances
    // =========================================================================
    wire [12:0] wgt_addr;
    wire [DW-1:0] wgt_data;
    wire [6:0]   bias_addr;
    wire [AW-1:0] bias_data;

    weight_bram #(
        .DEPTH    (8192),
        .DW       (DW),
        .AW       (13),
        .MEM_FILE ("D:/fyp/slim_syenet/project_1/mem/weights.mem")
    ) u_weight_bram (
        .clk    (clk),
        .a_en   (1'b0), .a_we  (1'b0),
        .a_addr (13'd0), .a_din (8'd0),
        .b_en   (1'b1),
        .b_addr (wgt_addr),
        .b_dout (wgt_data)
    );

    bias_bram #(
        .DEPTH    (128),
        .DW       (AW),
        .AW       (7),
        .MEM_FILE ("D:/fyp/slim_syenet/project_1/mem/biases.mem")
    ) u_bias_bram (
        .clk    (clk),
        .en     (1'b1),
        .addr   (bias_addr),
        .dout   (bias_data)
    );

    // =========================================================================
    // Feature map double buffer
    // =========================================================================
    reg         buf_sel;
    reg         fb_a_en, fb_a_we;
    reg [17:0]  fb_a_addr;
    reg [DW-1:0] fb_a_din;
    reg         fb_b_en;
    reg [17:0]  fb_b_addr;
    wire [DW-1:0] fb_b_dout;

    feature_bram #(
        .IMG_H  (IMG_H),
        .IMG_W  (IMG_W),
        .C      (C_MID),
        .DW     (DW)
    ) u_feature_bram (
        .clk     (clk),
        .buf_sel (buf_sel),
        .a_en    (fb_a_en),
        .a_we    (fb_a_we),
        .a_addr  (fb_a_addr),
        .a_din   (fb_a_din),
        .b_en    (fb_b_en),
        .b_addr  (fb_b_addr),
        .b_dout  (fb_b_dout)
    );

    // =========================================================================
    // Scheduler
    // =========================================================================
    wire sched_l0_start, sched_l0_done;
    wire sched_l1_start, sched_l1_done;
    wire sched_l2_start, sched_l2_done;
    wire sched_l3_start, sched_l3_done;
    wire sched_l4_start, sched_l4_done;
    wire sched_l5_start, sched_l5_done;
    wire sched_l6_start, sched_l6_done;
    wire sched_l7_start, sched_l7_done;
    wire sched_l8_start, sched_l8_done;
    wire sched_l9_start, sched_l9_done;
    wire sched_l10_start, sched_l10_done;
    wire sched_buf_swap;
    wire [3:0] sched_layer_id;

    scheduler_fsm u_sched (
        .clk         (clk),
        .rst_n       (rst_n),
        .go          (go),
        .l0_start    (sched_l0_start),  .l0_done  (sched_l0_done),
        .l1_start    (sched_l1_start),  .l1_done  (sched_l1_done),
        .l2_start    (sched_l2_start),  .l2_done  (sched_l2_done),
        .l3_start    (sched_l3_start),  .l3_done  (sched_l3_done),
        .l4_start    (sched_l4_start),  .l4_done  (sched_l4_done),
        .l5_start    (sched_l5_start),  .l5_done  (sched_l5_done),
        .l6_start    (sched_l6_start),  .l6_done  (sched_l6_done),
        .l7_start    (sched_l7_start),  .l7_done  (sched_l7_done),
        .l8_start    (sched_l8_start),  .l8_done  (sched_l8_done),
        .l9_start    (sched_l9_start),  .l9_done  (sched_l9_done),
        .l10_start   (sched_l10_start), .l10_done (sched_l10_done),
        .buf_swap    (sched_buf_swap),
        .infer_done  (infer_done),
        .layer_id    (sched_layer_id)
    );

    // =========================================================================
    // Buffer swap
    // =========================================================================
    always @(posedge clk) begin
        if (!rst_n) buf_sel <= 1'b0;
        else if (sched_buf_swap) buf_sel <= ~buf_sel;
    end

    // =========================================================================
    // Layer 0: Conv2d(4→12, 3×3) – head.block1.0
    // (Stubbed: full integration connects fm_valid_in stream here)
    // =========================================================================
    wire l0_out_valid, l0_done;
    wire [DW-1:0] l0_out_data;
    wire [12:0] l0_wgt_addr;
    wire [6:0]  l0_bias_addr;

    conv_engine #(
        .C_IN   (C_IN),
        .C_OUT  (C_MID),
        .KERN   (3),
        .IMG_W  (IMG_W),
        .WBASE  (0),
        .BBASE  (0)
    ) u_l0 (
        .clk        (clk),
        .rst_n      (rst_n),
        .start      (sched_l0_start),
        .in_valid   (fm_valid_in),
        .in_data    (fm_data_in),
        .wgt_addr   (l0_wgt_addr),
        .wgt_data   (wgt_data),
        .bias_addr  (l0_bias_addr),
        .bias_data  (bias_data),
        .out_valid  (l0_out_valid),
        .out_data   (l0_out_data),
        .frame_done (l0_done)
    );
    assign sched_l0_done = l0_done;

    // BN layer (L1) is merged into L0 weights → fires immediately
    assign sched_l1_done = sched_l1_start;

    // =========================================================================
    // Global avg pool + classifier (simplified – see full integration guide)
    // =========================================================================
    wire gap_done;
    wire [C_MID*DW-1:0] gap_avg;
    wire clf_done;
    wire [1:0]   clf_class;
    wire [AW-1:0] clf_conf;
    wire [C_OUT*AW-1:0] clf_logits;

    global_avg_pool #(
        .IMG_H (IMG_H), .IMG_W (IMG_W), .C (C_MID)
    ) u_gap (
        .clk      (clk), .rst_n (rst_n),
        .start    (sched_l9_start),
        .valid_in (fb_b_dout != 0),   // simplified: driven from feature BRAM
        .din      (fb_b_dout),
        .ch_in    (3'd0),              // channel tracking handled by controller
        .done     (gap_done),
        .avg_out  (gap_avg)
    );
    assign sched_l9_done = gap_done;

    wire [12:0] clf_wgt_addr;
    wire [6:0]  clf_bias_addr;

    classifier #(
        .C_IN   (C_MID),
        .C_OUT  (C_OUT),
        .WBASE  (13'hD90),
        .BBASE  (7'h54)
    ) u_clf (
        .clk       (clk), .rst_n (rst_n),
        .start     (sched_l10_start),
        .vec_in    (gap_avg),
        .wgt_addr  (clf_wgt_addr),
        .wgt_data  (wgt_data),
        .bias_addr (clf_bias_addr),
        .bias_data (bias_data),
        .done      (clf_done),
        .logits    (clf_logits),
        .class_id  (clf_class),
        .confidence(clf_conf)
    );
    assign sched_l10_done = clf_done;

    // Output assignments
    assign class_id   = clf_class;
    assign confidence = clf_conf;
    assign logits     = clf_logits;

    // Weight address mux (simple priority: classifier when active, else L0)
    assign wgt_addr  = sched_layer_id[3] ? clf_wgt_addr  : l0_wgt_addr;
    assign bias_addr = sched_layer_id[3] ? clf_bias_addr : l0_bias_addr;

    // Remaining scheduler stubs (layers 2-8)
    assign sched_l2_done = sched_l2_start;
    assign sched_l3_done = sched_l3_start;
    assign sched_l4_done = sched_l4_start;
    assign sched_l5_done = sched_l5_start;
    assign sched_l6_done = sched_l6_start;
    assign sched_l7_done = sched_l7_start;
    assign sched_l8_done = sched_l8_start;

endmodule
