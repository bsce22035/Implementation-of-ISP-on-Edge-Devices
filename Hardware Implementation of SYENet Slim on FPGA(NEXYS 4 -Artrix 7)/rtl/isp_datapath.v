// =============================================================================
// isp_datapath.v  -  Classical Datapath for SYENet-Slim ISP
//
// Receives a control word (ctrl_dp_op + conv-config regs) from isp_control.
// Holds all arithmetic registers: accumulator, loop counters, sigmoid vector,
// QCU latch, tail pipeline register.
// Emits status flags back to the controller and drives all memory/ROM addresses.
//
// KEY INVARIANT: selA is ALWAYS 2'd0.  M0/M1 are read exclusively via portB.
// =============================================================================
`timescale 1ns/1ps
`include "isp_params.vh"
module isp_datapath (
    input  wire        clk,
    input  wire        rst_n,

    // ---- control word from isp_control -----------------------------------
    input  wire [4:0]  ctrl_dp_op,
    input  wire [3:0]  ctrl_opx,
    input  wire [2:0]  ctrl_pwmode,
    input  wire [2:0]  ctrl_cf_k,
    input  wire [4:0]  ctrl_cf_ich,
    input  wire [4:0]  ctrl_cf_och,
    input  wire [2:0]  ctrl_cf_pad,
    input  wire [13:0] ctrl_cf_wbase,
    input  wire [6:0]  ctrl_cf_bbase,
    input  wire [6:0]  ctrl_cf_rqbase,
    input  wire [1:0]  ctrl_cf_ssel,
    input  wire [1:0]  ctrl_cf_dsel,
    input  wire [7:0]  ctrl_cf_iw,
    input  wire [7:0]  ctrl_cf_ih,
    input  wire [7:0]  ctrl_cf_sx,
    input  wire [7:0]  ctrl_cf_dx,

    // ---- ROM read-back (1-cycle registered outputs from isp_roms) --------
    input  wire signed [7:0]  w_dout,
    input  wire signed [31:0] b_dout,
    input  wire        [23:0] rq_dout,
    input  wire signed [7:0]  pr_dout,
    input  wire        [7:0]  sg_dout,
    input  wire signed [31:0] qb_dout,

    // ---- feat_mem read-back (1-cycle registered outputs) -----------------
    input  wire [7:0]  doutA,         // portA (always BUF_IN)
    input  wire [7:0]  doutB,         // portB (M0 or M1)

    // ---- ROM address outputs --------------------------------------------
    output reg  [13:0] w_addr,
    output reg  [6:0]  b_addr,
    output reg  [6:0]  rq_addr,
    output reg  [4:0]  pr_addr,
    output reg  [7:0]  sg_addr,
    output reg  [4:0]  qb_addr,

    // ---- feat_mem address / write outputs --------------------------------
    output reg  [1:0]  selA,          // always 2'd0 (BUF_IN invariant)
    output reg  [17:0] addrA,
    output reg  [1:0]  selB,
    output reg  [17:0] addrB,
    output reg         f_wen,
    output reg  [1:0]  f_wsel,
    output reg  [17:0] f_waddr,
    output reg  [7:0]  f_wdata,

    // ---- RGB output byte stream ------------------------------------------
    output reg         o_valid,
    output reg  [7:0]  o_data,
    input  wire        o_ready,

    // ---- status flags to isp_control ------------------------------------
    output wire        stat_kx_last,
    output wire        stat_ky_last,
    output wire        stat_ic_last,
    output wire        stat_oc_last,
    output wire        stat_ox_last,
    output wire        stat_oy_last,
    output wire        stat_idx_last,
    output wire        stat_gch_last,
    output wire        stat_gpix_last,
    output wire        stat_oc11,
    output wire        stat_tkx_last,
    output wire        stat_tky_last,
    output wire        stat_tic_last,
    output wire        stat_toc_last,
    output wire        stat_tox_last,
    output wire        stat_toy_last,
    output wire        stat_o_busy
);

    // ---- datapath operation codes (must match isp_control.v) ------------
    localparam DP_NOP       = 5'd0;
    localparam DP_DISPATCH  = 5'd1;
    localparam DP_CST       = 5'd2;
    localparam DP_LOAD_BIAS = 5'd3;
    localparam DP_CONV_ADDR = 5'd4;
    localparam DP_CONV_MAC  = 5'd5;
    localparam DP_CONV_WR   = 5'd6;
    localparam DP_PW_ADDR   = 5'd7;
    localparam DP_QH_LATCH  = 5'd8;
    localparam DP_PW_WR     = 5'd9;
    localparam DP_GAP_ADDR  = 5'd10;
    localparam DP_GAP_ACC   = 5'd11;
    localparam DP_SIG_ADDR  = 5'd12;
    localparam DP_SIG_LUT   = 5'd13;
    localparam DP_SIG_LATCH = 5'd14;
    localparam DP_TST       = 5'd15;
    localparam DP_TAIL_ADDR = 5'd16;
    localparam DP_TAIL_MAC  = 5'd17;
    localparam DP_TAIL_RQ   = 5'd18;
    localparam DP_TAIL_OUT  = 5'd19;

    // ---- op index --------------------------------------------------------
    localparam OP_PRELU=1, OP_QHEAD=4, OP_QBODY=7, OP_SCALE=13;
    localparam OP_AG2=10,  OP_GAP=8,   OP_SIG=12,  OP_TAIL=14;
    localparam GX=0, A1X=1, A2X=2, A3X=3;

    // ---- requant helpers -------------------------------------------------
    function signed [7:0] rqms;
        input signed [31:0] a; input [15:0] M; input [7:0] sh;
        reg signed [63:0] p, r;
        begin
            p = $signed(a) * $signed({1'b0, M});
            if (sh != 0) r = (p + ($signed(64'd1) <<< (sh - 1))) >>> sh; else r = p;
            if (r >  127) rqms =  8'sd127;
            else if (r < -127) rqms = -8'sd127;
            else rqms = r[7:0];
        end
    endfunction
    function [7:0] rqu;
        input signed [31:0] a; input [15:0] M; input [7:0] sh;
        reg signed [63:0] p, r;
        begin
            p = $signed(a) * $signed({1'b0, M});
            if (sh != 0) r = (p + ($signed(64'd1) <<< (sh - 1))) >>> sh; else r = p;
            if (r > 255) rqu = 8'd255;
            else if (r < 0) rqu = 8'd0;
            else rqu = r[7:0];
        end
    endfunction

    // ---- signed aliases for feat_mem outputs ----------------------------
    wire signed [7:0] sA = doutA;
    wire signed [7:0] sB = doutB;

    // ---- arithmetic & loop-counter registers ----------------------------
    reg signed [31:0] acc;
    reg signed [31:0] rgbtmp;
    reg signed [7:0]  qh_tmp;
    reg signed [7:0]  gv [0:11];       // sigmoid outputs g[c]
    reg                ib;             // in-bounds flag (registered)

    // conv inner loop
    reg [4:0] oc, ic, ky, kx;
    reg [7:0] oy, ox;

    // pointwise index
    reg [17:0] idx;

    // GAP counters
    reg [4:0]  gch;
    reg [13:0] gpix;

    // tail counters
    reg [8:0]  toy, tox;
    reg [1:0]  toc, tic, tky, tkx;

    // address computation temporaries (combinational, synthesised as wires)
    reg signed [9:0] iy, ix;

    integer i;

    // ---- status flags (combinational from registered counters) ----------
    assign stat_kx_last   = (kx   == ctrl_cf_k   - 1);
    assign stat_ky_last   = (ky   == ctrl_cf_k   - 1);
    assign stat_ic_last   = (ic   == ctrl_cf_ich - 1);
    assign stat_oc_last   = (oc   == ctrl_cf_och - 1);
    assign stat_ox_last   = (ox   == ctrl_cf_iw  - 1);
    assign stat_oy_last   = (oy   == ctrl_cf_ih  - 1);
    assign stat_idx_last  = (idx  == 18'd196607);
    assign stat_gch_last  = (gch  == 5'd11);
    assign stat_gpix_last = (gpix == 14'd16383);
    assign stat_oc11      = (oc   == 5'd11);
    assign stat_tkx_last  = (tkx  == 2'd2);
    assign stat_tky_last  = (tky  == 2'd2);
    assign stat_tic_last  = (tic  == 2'd2);
    assign stat_toc_last  = (toc  == 2'd2);
    assign stat_tox_last  = (tox  == 9'd255);
    assign stat_toy_last  = (toy  == 9'd255);
    assign stat_o_busy    = o_valid && !o_ready;

    // ---- clocked datapath -----------------------------------------------
    always @(posedge clk) begin
        if (!rst_n) begin
            oc<=0; ic<=0; ky<=0; kx<=0; oy<=0; ox<=0;
            idx<=0; gch<=0; gpix<=0; acc<=0; ib<=0;
            toy<=0; tox<=0; toc<=0; tic<=0; tky<=0; tkx<=0;
            f_wen<=0; o_valid<=0; selA<=2'd0;
            for (i=0; i<12; i=i+1) gv[i]<=0;
        end else begin
            f_wen <= 1'b0;
            if (o_valid && o_ready) o_valid <= 1'b0;

            case (ctrl_dp_op)

            // ---- reset loop counters per op type -------------------------
            DP_DISPATCH: begin
                oc<=0; oy<=0; ox<=0; ic<=0; ky<=0; kx<=0;
                case (ctrl_opx)
                OP_PRELU, OP_QHEAD, OP_QBODY, OP_SCALE:
                    begin idx<=0; end
                OP_AG2:
                    begin /* oc already 0 above */ end
                OP_GAP:
                    begin gch<=0; gpix<=0; acc<=0; end
                OP_SIG:
                    begin /* oc already 0 above */ end
                OP_TAIL:
                    begin toy<=0; tox<=0; toc<=0; end
                default: ;
                endcase
            end

            // ---- conv: issue bias + requant ROM addresses ----------------
            DP_CST: begin
                b_addr  <= ctrl_cf_bbase  + oc;
                rq_addr <= ctrl_cf_rqbase + oc;
                ic<=0; ky<=0; kx<=0;
            end

            // ---- load bias into accumulator (conv & tail) ----------------
            DP_LOAD_BIAS: begin
                acc <= b_dout;
            end

            // ---- conv: address generation for one tap --------------------
            DP_CONV_ADDR: begin
                iy = $signed({2'b0, oy}) + $signed({2'b0, ky})
                       - $signed({7'b0, ctrl_cf_pad});
                ix = $signed({2'b0, ox}) + $signed({2'b0, kx})
                       - $signed({7'b0, ctrl_cf_pad});
                ib <= (iy >= 0) && (iy < $signed({2'b0, ctrl_cf_ih}))
                   && (ix >= 0) && (ix < $signed({2'b0, ctrl_cf_iw}));
                if (ctrl_cf_ssel == 2'd0) begin
                    addrA <= ic * 16384 + iy * 18'sd128 + ix + ctrl_cf_sx;
                    selA  <= 2'd0;
                end else begin
                    addrB <= ic * 16384 + iy * 18'sd128 + ix + ctrl_cf_sx;
                    selB  <= ctrl_cf_ssel;
                end
                w_addr <= ctrl_cf_wbase
                        + ((oc * ctrl_cf_ich + ic) * ctrl_cf_k + ky) * ctrl_cf_k + kx;
            end

            // ---- conv: MAC + inner-loop counter advance ------------------
            DP_CONV_MAC: begin
                if (ib)
                    acc <= acc + ((ctrl_cf_ssel == 2'd0) ? sA : sB) * w_dout;
                if (kx == ctrl_cf_k - 1) begin kx <= 0;
                    if (ky == ctrl_cf_k - 1) begin ky <= 0;
                        if (ic == ctrl_cf_ich - 1) ic <= 0;
                        else                        ic <= ic + 1;
                    end else ky <= ky + 1;
                end else kx <= kx + 1;
            end

            // ---- conv: requant write + outer-loop counter advance --------
            DP_CONV_WR: begin
                f_wdata <= rqms(acc, rq_dout[23:8], rq_dout[7:0]);
                f_wsel  <= ctrl_cf_dsel;
                f_waddr <= oc * 16384 + (oy << 7) + ox + ctrl_cf_dx;
                f_wen   <= 1'b1;
                if (ox == ctrl_cf_iw - 1) begin ox <= 0;
                    if (oy == ctrl_cf_ih - 1) begin oy <= 0;
                        if (oc == ctrl_cf_och - 1) oc <= 0;
                        else                        oc <= oc + 1;
                    end else oy <= oy + 1;
                end else ox <= ox + 1;
            end

            // ---- pointwise: issue feature address ------------------------
            DP_PW_ADDR: begin
                if (ctrl_pwmode == 3'd4) begin
                    // AG2 PReLU: per-channel, read M1 x=A1X
                    addrB    <= oc * 16384 + A1X;
                    selB     <= 2'd2;
                    pr_addr  <= 5'd12 + oc;
                end else if (ctrl_pwmode == 3'd1 || ctrl_pwmode == 3'd2) begin
                    // QCU phase-1: read M1 at idx (whole 12-ch map flattened)
                    addrB   <= idx;
                    selB    <= 2'd2;
                    qb_addr <= (ctrl_pwmode == 3'd1) ? idx[17:14]
                                                     : (5'd12 + idx[17:14]);
                end else begin
                    // PReLU (pwmode=0) or SCALE (pwmode=3): read M0 at idx
                    addrB   <= idx;
                    selB    <= 2'd1;
                    pr_addr <= idx[17:14];
                end
            end

            // ---- QCU phase-1 complete: latch M1, re-issue M0 address ----
            DP_QH_LATCH: begin
                qh_tmp  <= sB;
                addrB   <= idx;
                selB    <= 2'd1;
            end

            // ---- pointwise: compute & write result, advance counters -----
            DP_PW_WR: begin
                case (ctrl_pwmode)
                3'd0: f_wdata <= rqms((sB < 0) ? sB * pr_dout : sB * 32'sd128,
                                      `PRELU_HB11_M, `PRELU_HB11_SH);
                3'd1: f_wdata <= rqms(sB * qh_tmp + qb_dout,
                                      `QCU_HEAD_M, `QCU_HEAD_SH);
                3'd2: f_wdata <= rqms(sB * qh_tmp + qb_dout,
                                      `QCU_BODY_M, `QCU_BODY_SH);
                3'd3: f_wdata <= rqms(sB * gv[idx[17:14]],
                                      `ATT_MUL_M, `ATT_MUL_SH);
                3'd4: f_wdata <= rqms((sB < 0) ? sB * pr_dout : sB * 32'sd128,
                                      `PRELU_AG2_M, `PRELU_AG2_SH);
                default: f_wdata <= 8'd0;
                endcase
                if (ctrl_pwmode == 3'd4) begin
                    f_wsel  <= 2'd2;
                    f_waddr <= oc * 16384 + A2X;
                    f_wen   <= 1'b1;
                    if (oc == 5'd11) oc <= 0;
                    else             oc <= oc + 1;
                end else begin
                    f_wsel  <= 2'd1;
                    f_waddr <= idx;
                    f_wen   <= 1'b1;
                    if (idx != 18'd196607) idx <= idx + 1;
                    else                   idx <= 0;
                end
            end

            // ---- GAP: issue M0 read address for one pixel ----------------
            DP_GAP_ADDR: begin
                addrB <= {gch, 14'd0} + gpix;
                selB  <= 2'd1;
            end

            // ---- GAP: accumulate, write when channel complete ------------
            DP_GAP_ACC: begin
                acc <= acc + $signed(sB);
                if (gpix == 14'd16383) begin
                    f_wdata <= rqms(acc + $signed(sB), `GAP_M, `GAP_SH + 8'd14);
                    f_wsel  <= 2'd2;
                    f_waddr <= gch * 16384 + GX;
                    f_wen   <= 1'b1;
                    gpix    <= 0;
                    acc     <= 0;
                    if (gch == 5'd11) gch <= 0;
                    else              gch <= gch + 1;
                end else gpix <= gpix + 1;
            end

            // ---- sigmoid: issue M1 read address (x=A3X per channel) -----
            DP_SIG_ADDR: begin
                addrB <= oc * 16384 + A3X;
                selB  <= 2'd2;
            end

            // ---- sigmoid: issue LUT address (sB is now AG3 output) ------
            DP_SIG_LUT: begin
                sg_addr <= $unsigned(sB + 8'sd128);
            end

            // ---- sigmoid: latch result, advance sigmoid channel ----------
            DP_SIG_LATCH: begin
                gv[oc] <= $signed(sg_dout);
                if (oc == 5'd11) oc <= 0;
                else             oc <= oc + 1;
            end

            // ---- tail conv start: issue bias/rq ROM addresses ------------
            DP_TST: begin
                b_addr  <= `BBASE_TAIL + toc;
                rq_addr <= `RQBASE_TAIL + toc;
                tic<=0; tky<=0; tkx<=0;
            end

            // ---- tail: PixelShuffle address generation -------------------
            DP_TAIL_ADDR: begin
                iy = $signed({1'b0, toy}) + $signed({8'b0, tky}) - 10'sd1;
                ix = $signed({1'b0, tox}) + $signed({8'b0, tkx}) - 10'sd1;
                ib <= (iy >= 0) && (iy < 10'sd256)
                   && (ix >= 0) && (ix < 10'sd256);
                addrB <= ((tic * 4) + (iy[0] ? 2 : 0) + (ix[0] ? 1 : 0)) * 16384
                        + (iy[9:1]) * 8'd128 + ix[9:1];
                selB  <= 2'd1;
                w_addr <= `WBASE_TAIL + ((toc * 3 + tic) * 3 + tky) * 3 + tkx;
            end

            // ---- tail: MAC + inner-loop advance --------------------------
            DP_TAIL_MAC: begin
                if (ib) acc <= acc + sB * w_dout;
                if (tkx == 2'd2) begin tkx <= 0;
                    if (tky == 2'd2) begin tky <= 0;
                        if (tic == 2'd2) tic <= 0;
                        else             tic <= tic + 1;
                    end else tky <= tky + 1;
                end else tkx <= tkx + 1;
            end

            // ---- tail: requantise to signed RGB --------------------------
            DP_TAIL_RQ: begin
                rgbtmp <= rqms(acc, rq_dout[23:8], rq_dout[7:0]);
            end

            // ---- tail: clip to uint8, stream out, advance pixel ----------
            DP_TAIL_OUT: begin
                if (!o_valid || o_ready) begin
                    o_data  <= rqu(rgbtmp, `RGBOUT_M, `RGBOUT_SH);
                    o_valid <= 1'b1;
                    if (toc == 2'd2) begin toc <= 0;
                        if (tox == 9'd255) begin tox <= 0;
                            if (toy == 9'd255) toy <= 0;
                            else               toy <= toy + 1;
                        end else tox <= tox + 1;
                    end else toc <= toc + 1;
                end
            end

            default: ;
            endcase
        end
    end

endmodule
