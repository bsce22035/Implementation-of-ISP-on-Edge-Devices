// =============================================================================
// classifier.v  –  Final FC classifier (12 → 3 classes)
// Target  : Xilinx Artix-7 XC7A100T
// Style   : Verilog-2001
//
// Computes: logit[c] = SUM(w[c][i] * z[i]) + bias[c]  for c ∈ {0,1,2}
// Takes the C=12 global-average-pool output vector as input.
// Outputs: 3 signed INT32 logits (no softmax; argmax on host PC).
//
// Also computes argmax for on-device class selection.
// Weight layout in weight BRAM at WBASE: [3×12] = 36 INT8 entries
//   row-major: [class0_w0..w11, class1_w0..w11, class2_w0..w11]
// =============================================================================
`timescale 1ns/1ps
module classifier #(
    parameter C_IN   = 12,
    parameter C_OUT  = 3,
    parameter DW     = 8,
    parameter AW     = 32,
    parameter WBASE  = 12'd3504,   // weight BRAM offset for tail.1
    parameter BBASE  = 7'd84,      // bias BRAM offset for tail.1.bias
    parameter SHIFT  = 7
) (
    input  wire              clk,
    input  wire              rst_n,
    input  wire              start,
    // Input: C_IN-element vector from global average pool
    input  wire [C_IN*DW-1:0] vec_in,
    // Weight/bias BRAM
    output reg  [12:0]       wgt_addr,
    input  wire [DW-1:0]     wgt_data,
    output reg  [6:0]        bias_addr,
    input  wire [AW-1:0]     bias_data,
    // Output
    output reg               done,
    output reg  [C_OUT*AW-1:0] logits,      // INT32 logits (signed)
    output reg  [1:0]          class_id,    // argmax
    output reg  [AW-1:0]       confidence   // max logit value
);
    reg [2:0]  state;
    reg [3:0]  oc;    // output class (0..2)
    reg [3:0]  ic;    // input channel (0..11)
    reg [AW-1:0] acc [0:C_OUT-1];
    reg [DW-1:0] w_r;  // weight pipeline register
    reg [DW-1:0] x_r;  // activation pipeline register

    // Temporaries for argmax (must be module-level in Verilog-2001)
    reg signed [AW-1:0] argmax_l0, argmax_l1, argmax_l2, argmax_lmax;
    reg [1:0]            argmax_cid;

    localparam S_IDLE = 3'd0;
    localparam S_MAC  = 3'd1;
    localparam S_BIAS = 3'd2;
    localparam S_MAX  = 3'd3;
    localparam S_DONE = 3'd4;

    integer i;

    always @(posedge clk) begin
        if (!rst_n) begin
            state      <= S_IDLE;
            done       <= 1'b0;
            oc         <= 4'd0;
            ic         <= 4'd0;
            wgt_addr   <= 13'd0;
            bias_addr  <= 7'd0;
            logits     <= {(C_OUT*AW){1'b0}};
            class_id   <= 2'd0;
            confidence <= {AW{1'b0}};
            w_r        <= {DW{1'b0}};
            x_r        <= {DW{1'b0}};
            for (i = 0; i < C_OUT; i = i + 1)
                acc[i] <= {AW{1'b0}};
        end else begin
            done <= 1'b0;

            case (state)
            S_IDLE: begin
                if (start) begin
                    for (i = 0; i < C_OUT; i = i + 1)
                        acc[i] <= {AW{1'b0}};
                    oc <= 4'd0; ic <= 4'd0;
                    wgt_addr <= WBASE;
                    state    <= S_MAC;
                end
            end

            S_MAC: begin
                // Issue read address for next cycle
                wgt_addr <= WBASE + oc*C_IN + ic;
                // Read activation from vec_in (combinatorial)
                x_r <= vec_in[ic*DW +: DW];
                w_r <= wgt_data;

                // Accumulate (registered weight from previous cycle)
                if (ic > 0 || oc > 0) begin
                    acc[oc] <= acc[oc] +
                        ($signed({{(AW-DW){w_r[DW-1]}}, w_r}) *
                         $signed({{(AW-DW){x_r[DW-1]}}, x_r}));
                end

                if (ic == C_IN-1) begin
                    ic <= 4'd0;
                    if (oc == C_OUT-1) begin
                        // Final accumulate
                        acc[oc] <= acc[oc] +
                            ($signed({{(AW-DW){w_r[DW-1]}}, w_r}) *
                             $signed({{(AW-DW){x_r[DW-1]}}, x_r}));
                        oc         <= 4'd0;
                        bias_addr  <= BBASE[6:0];
                        state      <= S_BIAS;
                    end else
                        oc <= oc + 1'b1;
                end else
                    ic <= ic + 1'b1;
            end

            S_BIAS: begin
                bias_addr <= BBASE[6:0] + oc;
                if (oc > 0)
                    logits[(oc-1)*AW +: AW] <= acc[oc-1] + bias_data;
                oc <= oc + 1'b1;
                if (oc == C_OUT) begin
                    logits[(C_OUT-1)*AW +: AW] <= acc[C_OUT-1] + bias_data;
                    state <= S_MAX;
                end
            end

            S_MAX: begin
                // Argmax over 3 logits using module-level temporaries
                argmax_l0 = $signed(logits[0*AW +: AW]);
                argmax_l1 = $signed(logits[1*AW +: AW]);
                argmax_l2 = $signed(logits[2*AW +: AW]);
                if (argmax_l0 >= argmax_l1 && argmax_l0 >= argmax_l2) begin
                    argmax_lmax = argmax_l0; argmax_cid = 2'd0;
                end else if (argmax_l1 >= argmax_l2) begin
                    argmax_lmax = argmax_l1; argmax_cid = 2'd1;
                end else begin
                    argmax_lmax = argmax_l2; argmax_cid = 2'd2;
                end
                class_id   <= argmax_cid;
                confidence <= argmax_lmax;
                state      <= S_DONE;
            end

            S_DONE: begin
                done  <= 1'b1;
                state <= S_IDLE;
            end
            endcase
        end
    end
endmodule
