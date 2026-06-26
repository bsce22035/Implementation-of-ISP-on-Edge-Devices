// =============================================================================
// isp_roms.v  -  Constant ROMs for SYENet-Slim ISP (loaded via $readmemh)
//   weights  : int8   (W length)
//   bias     : int32  (B length)
//   requant  : 24-bit {M[15:0],shift[7:0]} per output channel (RQ length)
//   prelu    : int8 slope Q0.7  (24 = 12 hb11 + 12 ag2)
//   sigmoid  : uint8 LUT (256)
//   qcubias  : int32 (24 = 12 head + 12 body)
// MEM_DIR must point at rtl_isp/mem with a trailing slash, set by parent via param.
// =============================================================================
`timescale 1ns/1ps
`include "isp_params.vh"
module isp_roms #(
    parameter MEM_DIR = "C:/Users/Ahmad/Downloads/Done_fpga/rtl/mem/"
) (
    input  wire        clk,
    input  wire [13:0] w_addr,   output reg signed [7:0]  w_dout,
    input  wire [6:0]  b_addr,   output reg signed [31:0] b_dout,
    input  wire [6:0]  rq_addr,  output reg [23:0]        rq_dout,
    input  wire [4:0]  pr_addr,  output reg signed [7:0]  pr_dout,
    input  wire [7:0]  sg_addr,  output reg [7:0]         sg_dout,
    input  wire [4:0]  qb_addr,  output reg signed [31:0] qb_dout
);
    reg signed [7:0]  wrom  [0:`WROM_LEN-1];
    reg signed [31:0] brom  [0:`BROM_LEN-1];
    reg [23:0]        rqrom [0:`RQROM_LEN-1];
    reg signed [7:0]  prom  [0:23];
    reg [7:0]         sgrom [0:255];
    reg signed [31:0] qbrom [0:23];

    initial begin
        $readmemh({MEM_DIR,"weights.mem"}, wrom);
        $readmemh({MEM_DIR,"bias.mem"},    brom);
        $readmemh({MEM_DIR,"requant.mem"}, rqrom);
        $readmemh({MEM_DIR,"prelu.mem"},   prom);
        $readmemh({MEM_DIR,"sigmoid.mem"}, sgrom);
        $readmemh({MEM_DIR,"qcubias.mem"}, qbrom);
    end

    always @(posedge clk) begin
        w_dout  <= wrom[w_addr];
        b_dout  <= brom[b_addr];
        rq_dout <= rqrom[rq_addr];
        pr_dout <= prom[pr_addr];
        sg_dout <= sgrom[sg_addr];
        qb_dout <= qbrom[qb_addr];
    end
endmodule
