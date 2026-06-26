// =============================================================================
// isp_control.v  -  Classical Controller for SYENet-Slim ISP
//
// Classical Controller/Datapath partition (industry-standard digital design):
//   CONTROLLER (this file): holds FSM state register, op sequencer, conv-config
//     registers.  Emits a combinational control word to the datapath each cycle.
//   DATAPATH (isp_datapath.v): holds all arithmetic registers (acc, gv, qh_tmp,
//     loop counters) and all address-generation logic.  Returns status flags.
//   WRAPPER  (isp_core.v): connects the two sub-modules to feat_mem + isp_roms.
//
// Control word outputs  -> isp_datapath
// Status flag inputs    <- isp_datapath
// =============================================================================
`timescale 1ns/1ps
`include "isp_params.vh"
module isp_control (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    output reg         done,
    output reg         loading,

    // ---- control word to datapath ----------------------------------------
    output reg  [4:0]  ctrl_dp_op,    // datapath operation this cycle
    output reg  [3:0]  ctrl_opx,      // current op index
    output reg  [2:0]  ctrl_pwmode,   // pointwise mode

    // conv config registers (datapath needs them for address/limit calc)
    output reg  [2:0]  ctrl_cf_k,
    output reg  [4:0]  ctrl_cf_ich,
    output reg  [4:0]  ctrl_cf_och,
    output reg  [2:0]  ctrl_cf_pad,
    output reg  [13:0] ctrl_cf_wbase,
    output reg  [6:0]  ctrl_cf_bbase,
    output reg  [6:0]  ctrl_cf_rqbase,
    output reg  [1:0]  ctrl_cf_ssel,
    output reg  [1:0]  ctrl_cf_dsel,
    output reg  [7:0]  ctrl_cf_iw,
    output reg  [7:0]  ctrl_cf_ih,
    output reg  [7:0]  ctrl_cf_sx,
    output reg  [7:0]  ctrl_cf_dx,

    // ---- status flags from datapath --------------------------------------
    input  wire        stat_kx_last,   // kx == cf_k-1
    input  wire        stat_ky_last,   // ky == cf_k-1
    input  wire        stat_ic_last,   // ic == cf_ich-1
    input  wire        stat_oc_last,   // oc == cf_och-1
    input  wire        stat_ox_last,   // ox == cf_iw-1
    input  wire        stat_oy_last,   // oy == cf_ih-1
    input  wire        stat_idx_last,  // idx == 196607
    input  wire        stat_gch_last,  // gch == 11
    input  wire        stat_gpix_last, // gpix == 16383
    input  wire        stat_oc11,      // oc == 11  (sig / ag2 / scale)
    input  wire        stat_tkx_last,  // tkx == 2
    input  wire        stat_tky_last,  // tky == 2
    input  wire        stat_tic_last,  // tic == 2
    input  wire        stat_toc_last,  // toc == 2
    input  wire        stat_tox_last,  // tox == 255
    input  wire        stat_toy_last,  // toy == 255
    input  wire        stat_o_busy     // o_valid && !o_ready
);

    // ---- datapath operation codes (shared with isp_datapath.v) ----------
    localparam DP_NOP       = 5'd0;
    localparam DP_DISPATCH  = 5'd1;   // S_DISP: reset counters per opx
    localparam DP_CST       = 5'd2;   // S_CST : issue conv bias/rq addresses
    localparam DP_LOAD_BIAS = 5'd3;   // S_CB/S_CB_T: acc <= b_dout
    localparam DP_CONV_ADDR = 5'd4;   // S_CISS: conv address generation
    localparam DP_CONV_MAC  = 5'd5;   // S_CMAC: MAC + advance kx/ky/ic
    localparam DP_CONV_WR   = 5'd6;   // S_CWR : write requant result + advance ox/oy/oc
    localparam DP_PW_ADDR   = 5'd7;   // S_PW  : issue pointwise address
    localparam DP_QH_LATCH  = 5'd8;   // S_QHMID: latch qh_tmp, re-issue M0 addr
    localparam DP_PW_WR     = 5'd9;   // S_PW2 : compute & write PW result
    localparam DP_GAP_ADDR  = 5'd10;  // S_GAP : issue GAP address
    localparam DP_GAP_ACC   = 5'd11;  // S_GAP2: accumulate GAP, write when done
    localparam DP_SIG_ADDR  = 5'd12;  // S_SIG : issue sigmoid read address
    localparam DP_SIG_LUT   = 5'd13;  // S_SIGB: sg_addr <= sB+128
    localparam DP_SIG_LATCH = 5'd14;  // S_SIGC: gv[oc] <= sg_dout
    localparam DP_TST       = 5'd15;  // S_TST : issue tail bias/rq addresses, reset inner counters
    localparam DP_TAIL_ADDR = 5'd16;  // S_TISS: tail address generation
    localparam DP_TAIL_MAC  = 5'd17;  // S_TMAC: tail MAC + advance tkx/tky/tic
    localparam DP_TAIL_RQ   = 5'd18;  // S_TRQ : rgbtmp <= rqms(acc, rq_dout)
    localparam DP_TAIL_OUT  = 5'd19;  // S_TOUT: o_data <= rqu(rgbtmp); o_valid <= 1

    // ---- op index --------------------------------------------------------
    localparam OP_HB10=0,OP_PRELU=1,OP_H1=2,OP_H2=3,OP_QHEAD=4,OP_B1=5,OP_B2=6,
               OP_QBODY=7,OP_GAP=8,OP_AG1=9,OP_AG2=10,OP_AG3=11,OP_SIG=12,
               OP_SCALE=13,OP_TAIL=14,OP_END=15;

    // attention vector x-offsets
    localparam GX=0, A1X=1, A2X=2, A3X=3;

    // ---- FSM states -------------------------------------------------------
    localparam S_IDLE=0,  S_DISP=1,
               S_CST=2,   S_CBWAIT=3,   S_CB=4,
               S_CISS=5,  S_CISS2=6,    S_CISS3=7,   S_CMAC=8,  S_CWR=9,
               S_PW=10,   S_PWWAIT=11,  S_PWWAIT2=12, S_PW2=13,
               S_GAP=14,  S_GAPWAIT=15, S_GAPWAIT2=16, S_GAP2=17,
               S_SIG=18,  S_SIGW=19,    S_SIGWX=20,  S_SIGB=21, S_SIGW2=22, S_SIGC=23,
               S_TST=24,  S_TSWAIT=25,  S_CB_T=26,
               S_TISS=27, S_TISS2=28,   S_TISS3=29,  S_TMAC=30, S_TRQ=31,   S_TOUT=32,
               S_NEXT=33, S_DONE=34,
               S_QHWT1=35, S_QHWT2=36, S_QHMID=37;

    reg [5:0] st;

    // ---- combinational: ctrl_dp_op from current state --------------------
    always @(*) begin
        case (st)
        S_DISP:    ctrl_dp_op = DP_DISPATCH;
        S_CST:     ctrl_dp_op = DP_CST;
        S_CB:      ctrl_dp_op = DP_LOAD_BIAS;
        S_CISS:    ctrl_dp_op = DP_CONV_ADDR;
        S_CMAC:    ctrl_dp_op = DP_CONV_MAC;
        S_CWR:     ctrl_dp_op = DP_CONV_WR;
        S_PW:      ctrl_dp_op = DP_PW_ADDR;
        S_QHMID:   ctrl_dp_op = DP_QH_LATCH;
        S_PW2:     ctrl_dp_op = DP_PW_WR;
        S_GAP:     ctrl_dp_op = DP_GAP_ADDR;
        S_GAP2:    ctrl_dp_op = DP_GAP_ACC;
        S_SIG:     ctrl_dp_op = DP_SIG_ADDR;
        S_SIGB:    ctrl_dp_op = DP_SIG_LUT;
        S_SIGC:    ctrl_dp_op = DP_SIG_LATCH;
        S_TST:     ctrl_dp_op = DP_TST;
        S_CB_T:    ctrl_dp_op = DP_LOAD_BIAS;
        S_TISS:    ctrl_dp_op = DP_TAIL_ADDR;
        S_TMAC:    ctrl_dp_op = DP_TAIL_MAC;
        S_TRQ:     ctrl_dp_op = DP_TAIL_RQ;
        S_TOUT:    ctrl_dp_op = DP_TAIL_OUT;
        default:   ctrl_dp_op = DP_NOP;
        endcase
    end

    // ---- clocked FSM + config register updates ---------------------------
    always @(posedge clk) begin
        if (!rst_n) begin
            st       <= S_IDLE;
            loading  <= 1'b1;
            done     <= 1'b0;
            ctrl_opx <= 4'd0;
            ctrl_pwmode <= 3'd0;
        end else begin
            done <= 1'b0;
            case (st)

            // ------ idle / start ------------------------------------------
            S_IDLE: if (start) begin
                loading  <= 1'b0;
                ctrl_opx <= OP_HB10;
                st       <= S_DISP;
            end

            // ------ op dispatch: load conv config, set pwmode -------------
            S_DISP: begin
                case (ctrl_opx)
                OP_HB10: begin
                    ctrl_cf_k<=3'd5; ctrl_cf_ich<=5'd4;  ctrl_cf_och<=5'd12; ctrl_cf_pad<=3'd2;
                    ctrl_cf_wbase<=`WBASE_HB10; ctrl_cf_bbase<=`BBASE_HB10; ctrl_cf_rqbase<=`RQBASE_HB10;
                    ctrl_cf_ssel<=2'd0; ctrl_cf_dsel<=2'd1;
                    ctrl_cf_iw<=8'd128; ctrl_cf_ih<=8'd128; ctrl_cf_sx<=8'd0; ctrl_cf_dx<=8'd0;
                    st <= S_CST;
                end
                OP_H1: begin
                    ctrl_cf_k<=3'd3; ctrl_cf_ich<=5'd12; ctrl_cf_och<=5'd12; ctrl_cf_pad<=3'd1;
                    ctrl_cf_wbase<=`WBASE_H1; ctrl_cf_bbase<=`BBASE_H1; ctrl_cf_rqbase<=`RQBASE_H1;
                    ctrl_cf_ssel<=2'd1; ctrl_cf_dsel<=2'd2;
                    ctrl_cf_iw<=8'd128; ctrl_cf_ih<=8'd128; ctrl_cf_sx<=8'd0; ctrl_cf_dx<=8'd0;
                    st <= S_CST;
                end
                OP_H2: begin
                    ctrl_cf_k<=3'd5; ctrl_cf_ich<=5'd4;  ctrl_cf_och<=5'd12; ctrl_cf_pad<=3'd2;
                    ctrl_cf_wbase<=`WBASE_H2; ctrl_cf_bbase<=`BBASE_H2; ctrl_cf_rqbase<=`RQBASE_H2;
                    ctrl_cf_ssel<=2'd0; ctrl_cf_dsel<=2'd1;
                    ctrl_cf_iw<=8'd128; ctrl_cf_ih<=8'd128; ctrl_cf_sx<=8'd0; ctrl_cf_dx<=8'd0;
                    st <= S_CST;
                end
                OP_B1: begin
                    ctrl_cf_k<=3'd3; ctrl_cf_ich<=5'd12; ctrl_cf_och<=5'd12; ctrl_cf_pad<=3'd1;
                    ctrl_cf_wbase<=`WBASE_B1; ctrl_cf_bbase<=`BBASE_B1; ctrl_cf_rqbase<=`RQBASE_B1;
                    ctrl_cf_ssel<=2'd1; ctrl_cf_dsel<=2'd2;
                    ctrl_cf_iw<=8'd128; ctrl_cf_ih<=8'd128; ctrl_cf_sx<=8'd0; ctrl_cf_dx<=8'd0;
                    st <= S_CST;
                end
                OP_B2: begin
                    ctrl_cf_k<=3'd1; ctrl_cf_ich<=5'd12; ctrl_cf_och<=5'd12; ctrl_cf_pad<=3'd0;
                    ctrl_cf_wbase<=`WBASE_B2; ctrl_cf_bbase<=`BBASE_B2; ctrl_cf_rqbase<=`RQBASE_B2;
                    ctrl_cf_ssel<=2'd1; ctrl_cf_dsel<=2'd1;
                    ctrl_cf_iw<=8'd128; ctrl_cf_ih<=8'd128; ctrl_cf_sx<=8'd0; ctrl_cf_dx<=8'd0;
                    st <= S_CST;
                end
                OP_AG1: begin
                    ctrl_cf_k<=3'd1; ctrl_cf_ich<=5'd12; ctrl_cf_och<=5'd12; ctrl_cf_pad<=3'd0;
                    ctrl_cf_wbase<=`WBASE_AG1; ctrl_cf_bbase<=`BBASE_AG1; ctrl_cf_rqbase<=`RQBASE_AG1;
                    ctrl_cf_ssel<=2'd2; ctrl_cf_dsel<=2'd2;
                    ctrl_cf_iw<=8'd1; ctrl_cf_ih<=8'd1; ctrl_cf_sx<=GX; ctrl_cf_dx<=A1X;
                    st <= S_CST;
                end
                OP_AG3: begin
                    ctrl_cf_k<=3'd1; ctrl_cf_ich<=5'd12; ctrl_cf_och<=5'd12; ctrl_cf_pad<=3'd0;
                    ctrl_cf_wbase<=`WBASE_AG3; ctrl_cf_bbase<=`BBASE_AG3; ctrl_cf_rqbase<=`RQBASE_AG3;
                    ctrl_cf_ssel<=2'd2; ctrl_cf_dsel<=2'd2;
                    ctrl_cf_iw<=8'd1; ctrl_cf_ih<=8'd1; ctrl_cf_sx<=A2X; ctrl_cf_dx<=A3X;
                    st <= S_CST;
                end
                OP_PRELU: begin ctrl_pwmode<=3'd0; st<=S_PW; end
                OP_QHEAD: begin ctrl_pwmode<=3'd1; st<=S_PW; end
                OP_QBODY: begin ctrl_pwmode<=3'd2; st<=S_PW; end
                OP_SCALE: begin ctrl_pwmode<=3'd3; st<=S_PW; end
                OP_AG2:   begin ctrl_pwmode<=3'd4; st<=S_PW; end
                OP_GAP:   begin st<=S_GAP; end
                OP_SIG:   begin st<=S_SIG; end
                OP_TAIL:  begin st<=S_TST; end
                OP_END:   begin done<=1'b1; st<=S_DONE; end
                default:  begin st<=S_DONE; end
                endcase
            end

            // ------ generic conv ------------------------------------------
            S_CST:    begin st<=S_CBWAIT; end
            S_CBWAIT: begin st<=S_CB;     end
            S_CB:     begin st<=S_CISS;   end
            S_CISS:   begin st<=S_CISS2;  end
            S_CISS2:  begin st<=S_CISS3;  end
            S_CISS3:  begin st<=S_CMAC;   end
            S_CMAC: begin
                if (stat_kx_last) begin
                    if (stat_ky_last) begin
                        if (stat_ic_last) st<=S_CWR;
                        else              st<=S_CISS;
                    end else             st<=S_CISS;
                end else                 st<=S_CISS;
            end
            S_CWR: begin
                if (stat_ox_last) begin
                    if (stat_oy_last) begin
                        if (stat_oc_last) st<=S_NEXT;
                        else              st<=S_CST;
                    end else             st<=S_CST;
                end else                 st<=S_CST;
            end

            // ------ pointwise / QCU / scale / AG2-PReLU -------------------
            S_PW: begin
                if (ctrl_pwmode==3'd1 || ctrl_pwmode==3'd2)
                    st <= S_QHWT1;   // QCU: two-phase M1 then M0
                else
                    st <= S_PWWAIT;  // PReLU / SCALE / AG2
            end
            S_QHWT1:  begin st<=S_QHWT2;  end
            S_QHWT2:  begin st<=S_QHMID;  end
            S_QHMID:  begin st<=S_PWWAIT; end
            S_PWWAIT: begin st<=S_PWWAIT2; end
            S_PWWAIT2:begin st<=S_PW2;    end
            S_PW2: begin
                if (ctrl_pwmode==3'd4) begin          // AG2 PReLU (per-channel)
                    if (stat_oc11) st<=S_NEXT;
                    else           st<=S_PW;
                end else begin                         // all others (per-pixel)
                    if (stat_idx_last) st<=S_NEXT;
                    else               st<=S_PW;
                end
            end

            // ------ GAP ---------------------------------------------------
            S_GAP:      begin st<=S_GAPWAIT;  end
            S_GAPWAIT:  begin st<=S_GAPWAIT2; end
            S_GAPWAIT2: begin st<=S_GAP2;     end
            S_GAP2: begin
                if (stat_gpix_last) begin
                    if (stat_gch_last) st<=S_NEXT;
                    else               st<=S_GAP;
                end else               st<=S_GAP;
            end

            // ------ sigmoid -----------------------------------------------
            S_SIG:  begin st<=S_SIGW;  end
            S_SIGW: begin st<=S_SIGWX; end
            S_SIGWX:begin st<=S_SIGB;  end
            S_SIGB: begin st<=S_SIGW2; end
            S_SIGW2:begin st<=S_SIGC;  end
            S_SIGC: begin
                if (stat_oc11) st<=S_NEXT;
                else           st<=S_SIG;
            end

            // ------ tail conv + PixelShuffle ------------------------------
            S_TST:    begin st<=S_TSWAIT; end
            S_TSWAIT: begin st<=S_CB_T;   end
            S_CB_T:   begin st<=S_TISS;   end
            S_TISS:   begin st<=S_TISS2;  end
            S_TISS2:  begin st<=S_TISS3;  end
            S_TISS3:  begin st<=S_TMAC;   end
            S_TMAC: begin
                if (stat_tkx_last) begin
                    if (stat_tky_last) begin
                        if (stat_tic_last) st<=S_TRQ;
                        else               st<=S_TISS;
                    end else               st<=S_TISS;
                end else                   st<=S_TISS;
            end
            S_TRQ:  begin st<=S_TOUT; end
            S_TOUT: begin
                if (!stat_o_busy) begin
                    if (stat_toc_last) begin
                        if (stat_tox_last) begin
                            if (stat_toy_last) st<=S_NEXT;
                            else               st<=S_TST;
                        end else               st<=S_TST;
                    end else                   st<=S_TST;
                end
            end

            // ------ op advance / done -------------------------------------
            S_NEXT: begin ctrl_opx<=ctrl_opx+1; st<=S_DISP; end
            S_DONE: begin done<=1'b1; st<=S_IDLE; loading<=1'b1; end
            default: st<=S_IDLE;
            endcase
        end
    end

endmodule
