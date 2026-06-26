// =============================================================================
// channel_attention.v  –  Channel Attention Module (hardware-efficient)
// Target  : Xilinx Artix-7 XC7A100T
// Style   : Verilog-2001  (no local declarations in unnamed blocks)
//
// SYENet channel attention:
//   z  = GlobalAvgPool(x)
//   z1 = ReLU(FC1(z))             att.1 weights
//   z2 = Sigmoid_approx(FC2(z1))  att.3 weights
//   y  = x * z2  (channel-wise scale)
//
// Sigmoid approximation (fixed-point, no division):
//   out = clamp(128 + (x >> 3), 0, 255)   ← linear approximation near 0
// =============================================================================
`timescale 1ns/1ps
module channel_attention #(
    parameter C       = 12,
    parameter DW      = 8,
    parameter AW      = 32,
    parameter SHIFT   = 7,
    parameter W1_BASE = 12'd288,
    parameter W2_BASE = 12'd432
) (
    input  wire                   clk,
    input  wire                   rst_n,
    input  wire                   start,
    // Feature map stream (first pass – for GAP)
    input  wire                   fm_valid,
    input  wire [DW-1:0]          fm_din,
    input  wire [$clog2(C)-1:0]   fm_ch,
    // Weight BRAM
    output reg  [12:0]            wgt_addr,
    input  wire [DW-1:0]          wgt_din,
    // Feature map stream (second pass – for scaling)
    input  wire                   fm_in2_valid,
    input  wire [DW-1:0]          fm_in2,
    input  wire [$clog2(C)-1:0]   fm_in2_ch,
    // Output
    output reg                    fm_out_valid,
    output reg  [DW-1:0]          fm_out,
    output reg                    done
);
    localparam S_IDLE  = 3'd0;
    localparam S_GAP   = 3'd1;
    localparam S_FC1   = 3'd2;
    localparam S_FC2   = 3'd3;
    localparam S_SCALE = 3'd4;

    localparam GAP_TOTAL = 128 * 128;
    localparam LOG2_GAP  = 14;

    reg [2:0]  state;
    reg [AW-1:0]            gap_acc [0:C-1];
    reg [DW-1:0]            z       [0:C-1];
    reg [DW-1:0]            z1      [0:C-1];
    reg [DW-1:0]            z2      [0:C-1];
    reg [$clog2(C)-1:0]     oc, ic;
    reg [AW-1:0]            fc_acc;
    reg [17:0]              gap_cnt;
    reg [DW-1:0]            wgt_r;

    // Module-level temporaries (Verilog-2001 rule: no decls in unnamed blocks)
    reg signed [AW-1:0]     att_s;
    reg signed [7:0]        att_clip;
    reg [8:0]               att_sig;
    reg [AW-1:0]            att_result;
    reg [2*DW-1:0]          att_prod;

    integer i;

    // Registered weight (1-cycle read latency from BRAM)
    always @(posedge clk) begin
        wgt_r <= wgt_din;
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            state        <= S_IDLE;
            done         <= 1'b0;
            fm_out_valid <= 1'b0;
            fm_out       <= {DW{1'b0}};
            gap_cnt      <= 18'd0;
            oc           <= {$clog2(C){1'b0}};
            ic           <= {$clog2(C){1'b0}};
            fc_acc       <= {AW{1'b0}};
            wgt_addr     <= 13'd0;
            for (i = 0; i < C; i = i + 1) begin
                gap_acc[i] <= {AW{1'b0}};
                z[i]       <= {DW{1'b0}};
                z1[i]      <= {DW{1'b0}};
                z2[i]      <= {DW{1'b0}};
            end
        end else begin
            done         <= 1'b0;
            fm_out_valid <= 1'b0;

            case (state)
            // -----------------------------------------------------------------
            S_IDLE: begin
                if (start) begin
                    for (i = 0; i < C; i = i + 1)
                        gap_acc[i] <= {AW{1'b0}};
                    gap_cnt <= 18'd0;
                    state   <= S_GAP;
                end
            end
            // -----------------------------------------------------------------
            S_GAP: begin
                if (fm_valid) begin
                    gap_acc[fm_ch] <= gap_acc[fm_ch] + {{(AW-DW){1'b0}}, fm_din};
                    gap_cnt        <= gap_cnt + 1'b1;
                    if (gap_cnt == GAP_TOTAL * C - 1) begin
                        for (i = 0; i < C; i = i + 1)
                            z[i] <= gap_acc[i][LOG2_GAP+DW-1:LOG2_GAP];
                        oc       <= {$clog2(C){1'b0}};
                        ic       <= {$clog2(C){1'b0}};
                        fc_acc   <= {AW{1'b0}};
                        wgt_addr <= W1_BASE;
                        state    <= S_FC1;
                    end
                end
            end
            // -----------------------------------------------------------------
            S_FC1: begin
                wgt_addr <= W1_BASE + {oc, ic};
                if (ic > 0 || oc > 0)
                    fc_acc <= fc_acc +
                              ($signed({{(AW-DW){wgt_r[DW-1]}}, wgt_r}) *
                               $signed({{(AW-DW){z[ic][DW-1]}}, z[ic]}));

                if (ic == C-1) begin
                    ic <= {$clog2(C){1'b0}};
                    // ReLU + shift (use module-level att_result)
                    att_result = $signed(fc_acc) >>> SHIFT;
                    if (att_result[AW-1])
                        z1[oc] <= {DW{1'b0}};
                    else if (|att_result[AW-1:DW])
                        z1[oc] <= {DW{1'b1}};
                    else
                        z1[oc] <= att_result[DW-1:0];

                    fc_acc <= {AW{1'b0}};
                    if (oc == C-1) begin
                        oc       <= {$clog2(C){1'b0}};
                        fc_acc   <= {AW{1'b0}};
                        wgt_addr <= W2_BASE;
                        state    <= S_FC2;
                    end else
                        oc <= oc + 1'b1;
                end else
                    ic <= ic + 1'b1;
            end
            // -----------------------------------------------------------------
            S_FC2: begin
                wgt_addr <= W2_BASE + {oc, ic};
                if (ic > 0 || oc > 0)
                    fc_acc <= fc_acc +
                              ($signed({{(AW-DW){wgt_r[DW-1]}}, wgt_r}) *
                               $signed({{(AW-DW){z1[ic][DW-1]}}, z1[ic]}));

                if (ic == C-1) begin
                    ic <= {$clog2(C){1'b0}};
                    // Sigmoid approx using module-level temporaries
                    att_s    = $signed(fc_acc) >>> SHIFT;
                    att_clip = (att_s > 32'sd127)  ? 8'sd127  :
                               (att_s < -32'sd128) ? -8'sd128 : att_s[7:0];
                    att_sig  = 9'd128 + {{1{att_clip[7]}}, att_clip[7:3]};
                    z2[oc]  <= att_sig[7:0];

                    fc_acc <= {AW{1'b0}};
                    if (oc == C-1)
                        state <= S_SCALE;
                    else
                        oc <= oc + 1'b1;
                end else
                    ic <= ic + 1'b1;
            end
            // -----------------------------------------------------------------
            S_SCALE: begin
                // y = x * (z2 / 256)  →  (x * z2) >> 8
                if (fm_in2_valid) begin
                    att_prod     = fm_in2 * z2[fm_in2_ch];
                    fm_out       <= att_prod[2*DW-1:DW];
                    fm_out_valid <= 1'b1;
                end
                if (done)
                    state <= S_IDLE;
            end
            endcase
        end
    end
endmodule
