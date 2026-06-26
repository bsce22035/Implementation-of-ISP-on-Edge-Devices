// =============================================================================
// isp_core.v  -  Thin wrapper: classical Controller/Datapath partition
//
// Instantiates:
//   isp_control   - pure FSM / op sequencer / conv-config registers
//   isp_datapath  - all arithmetic, counters, address generation
//   feat_mem      - feature BRAMs (BUF_IN / M0 / M1)
//   isp_roms      - weight / bias / requant / PReLU / sigmoid / QCU ROMs
//
// External interface is identical to the monolithic isp_core.v in Done_fpga,
// so the surrounding RTL (top.v, feat_mem.v, isp_roms.v, etc.) is unchanged.
// =============================================================================
`timescale 1ns/1ps
`include "isp_params.vh"
module isp_core (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire        in_wen,
    input  wire [15:0] in_addr,
    input  wire [7:0]  in_data,
    output wire        o_valid,
    output wire [7:0]  o_data,
    input  wire        o_ready,
    output wire        done
);

    // ---- controller -> datapath wires (control word) --------------------
    wire [4:0]  ctrl_dp_op;
    wire [3:0]  ctrl_opx;
    wire [2:0]  ctrl_pwmode;
    wire [2:0]  ctrl_cf_k;
    wire [4:0]  ctrl_cf_ich, ctrl_cf_och;
    wire [2:0]  ctrl_cf_pad;
    wire [13:0] ctrl_cf_wbase;
    wire [6:0]  ctrl_cf_bbase, ctrl_cf_rqbase;
    wire [1:0]  ctrl_cf_ssel, ctrl_cf_dsel;
    wire [7:0]  ctrl_cf_iw, ctrl_cf_ih, ctrl_cf_sx, ctrl_cf_dx;
    wire        ctrl_loading;

    // ---- datapath -> controller wires (status flags) --------------------
    wire stat_kx_last, stat_ky_last, stat_ic_last, stat_oc_last;
    wire stat_ox_last, stat_oy_last;
    wire stat_idx_last, stat_gch_last, stat_gpix_last, stat_oc11;
    wire stat_tkx_last, stat_tky_last, stat_tic_last;
    wire stat_toc_last, stat_tox_last, stat_toy_last;
    wire stat_o_busy;

    // ---- feat_mem interface wires ---------------------------------------
    wire        dp_f_wen;
    wire [1:0]  dp_f_wsel;
    wire [17:0] dp_f_waddr;
    wire [7:0]  dp_f_wdata;
    wire [1:0]  dp_selA;
    wire [17:0] dp_addrA;
    wire [1:0]  dp_selB;
    wire [17:0] dp_addrB;
    wire [7:0]  doutA, doutB;

    // ---- loading mux: BUF_IN write from UART while loading=1 -----------
    wire        m_wen   = ctrl_loading ? in_wen          : dp_f_wen;
    wire [1:0]  m_wsel  = ctrl_loading ? 2'd0            : dp_f_wsel;
    wire [17:0] m_waddr = ctrl_loading ? {2'd0, in_addr} : dp_f_waddr;
    wire [7:0]  m_wdata = ctrl_loading ? in_data         : dp_f_wdata;

    // ---- ROM interface wires -------------------------------------------
    wire [13:0] w_addr;  wire signed [7:0]  w_dout;
    wire [6:0]  b_addr;  wire signed [31:0] b_dout;
    wire [6:0]  rq_addr; wire [23:0]        rq_dout;
    wire [4:0]  pr_addr; wire signed [7:0]  pr_dout;
    wire [7:0]  sg_addr; wire [7:0]         sg_dout;
    wire [4:0]  qb_addr; wire signed [31:0] qb_dout;

    // ---- sub-module instantiation --------------------------------------

    feat_mem u_mem (
        .clk   (clk),
        .wen   (m_wen),   .wsel  (m_wsel),  .waddr (m_waddr), .wdata (m_wdata),
        .aselA (dp_selA), .addrA (dp_addrA),.doutA (doutA),
        .aselB (dp_selB), .addrB (dp_addrB),.doutB (doutB)
    );

    isp_roms u_rom (
        .clk    (clk),
        .w_addr (w_addr),  .w_dout (w_dout),
        .b_addr (b_addr),  .b_dout (b_dout),
        .rq_addr(rq_addr), .rq_dout(rq_dout),
        .pr_addr(pr_addr), .pr_dout(pr_dout),
        .sg_addr(sg_addr), .sg_dout(sg_dout),
        .qb_addr(qb_addr), .qb_dout(qb_dout)
    );

    isp_control u_ctrl (
        .clk        (clk),
        .rst_n      (rst_n),
        .start      (start),
        .done       (done),
        .loading    (ctrl_loading),
        .ctrl_dp_op     (ctrl_dp_op),
        .ctrl_opx       (ctrl_opx),
        .ctrl_pwmode    (ctrl_pwmode),
        .ctrl_cf_k      (ctrl_cf_k),
        .ctrl_cf_ich    (ctrl_cf_ich),
        .ctrl_cf_och    (ctrl_cf_och),
        .ctrl_cf_pad    (ctrl_cf_pad),
        .ctrl_cf_wbase  (ctrl_cf_wbase),
        .ctrl_cf_bbase  (ctrl_cf_bbase),
        .ctrl_cf_rqbase (ctrl_cf_rqbase),
        .ctrl_cf_ssel   (ctrl_cf_ssel),
        .ctrl_cf_dsel   (ctrl_cf_dsel),
        .ctrl_cf_iw     (ctrl_cf_iw),
        .ctrl_cf_ih     (ctrl_cf_ih),
        .ctrl_cf_sx     (ctrl_cf_sx),
        .ctrl_cf_dx     (ctrl_cf_dx),
        .stat_kx_last   (stat_kx_last),
        .stat_ky_last   (stat_ky_last),
        .stat_ic_last   (stat_ic_last),
        .stat_oc_last   (stat_oc_last),
        .stat_ox_last   (stat_ox_last),
        .stat_oy_last   (stat_oy_last),
        .stat_idx_last  (stat_idx_last),
        .stat_gch_last  (stat_gch_last),
        .stat_gpix_last (stat_gpix_last),
        .stat_oc11      (stat_oc11),
        .stat_tkx_last  (stat_tkx_last),
        .stat_tky_last  (stat_tky_last),
        .stat_tic_last  (stat_tic_last),
        .stat_toc_last  (stat_toc_last),
        .stat_tox_last  (stat_tox_last),
        .stat_toy_last  (stat_toy_last),
        .stat_o_busy    (stat_o_busy)
    );

    isp_datapath u_dp (
        .clk        (clk),
        .rst_n      (rst_n),
        .ctrl_dp_op     (ctrl_dp_op),
        .ctrl_opx       (ctrl_opx),
        .ctrl_pwmode    (ctrl_pwmode),
        .ctrl_cf_k      (ctrl_cf_k),
        .ctrl_cf_ich    (ctrl_cf_ich),
        .ctrl_cf_och    (ctrl_cf_och),
        .ctrl_cf_pad    (ctrl_cf_pad),
        .ctrl_cf_wbase  (ctrl_cf_wbase),
        .ctrl_cf_bbase  (ctrl_cf_bbase),
        .ctrl_cf_rqbase (ctrl_cf_rqbase),
        .ctrl_cf_ssel   (ctrl_cf_ssel),
        .ctrl_cf_dsel   (ctrl_cf_dsel),
        .ctrl_cf_iw     (ctrl_cf_iw),
        .ctrl_cf_ih     (ctrl_cf_ih),
        .ctrl_cf_sx     (ctrl_cf_sx),
        .ctrl_cf_dx     (ctrl_cf_dx),
        .w_dout  (w_dout),  .b_dout  (b_dout),  .rq_dout (rq_dout),
        .pr_dout (pr_dout), .sg_dout (sg_dout), .qb_dout (qb_dout),
        .doutA (doutA), .doutB (doutB),
        .w_addr  (w_addr),  .b_addr  (b_addr),  .rq_addr (rq_addr),
        .pr_addr (pr_addr), .sg_addr (sg_addr), .qb_addr (qb_addr),
        .selA   (dp_selA),  .addrA  (dp_addrA),
        .selB   (dp_selB),  .addrB  (dp_addrB),
        .f_wen  (dp_f_wen), .f_wsel (dp_f_wsel),
        .f_waddr(dp_f_waddr),.f_wdata(dp_f_wdata),
        .o_valid (o_valid), .o_data (o_data), .o_ready (o_ready),
        .stat_kx_last   (stat_kx_last),
        .stat_ky_last   (stat_ky_last),
        .stat_ic_last   (stat_ic_last),
        .stat_oc_last   (stat_oc_last),
        .stat_ox_last   (stat_ox_last),
        .stat_oy_last   (stat_oy_last),
        .stat_idx_last  (stat_idx_last),
        .stat_gch_last  (stat_gch_last),
        .stat_gpix_last (stat_gpix_last),
        .stat_oc11      (stat_oc11),
        .stat_tkx_last  (stat_tkx_last),
        .stat_tky_last  (stat_tky_last),
        .stat_tic_last  (stat_tic_last),
        .stat_toc_last  (stat_toc_last),
        .stat_tox_last  (stat_tox_last),
        .stat_toy_last  (stat_toy_last),
        .stat_o_busy    (stat_o_busy)
    );

endmodule
