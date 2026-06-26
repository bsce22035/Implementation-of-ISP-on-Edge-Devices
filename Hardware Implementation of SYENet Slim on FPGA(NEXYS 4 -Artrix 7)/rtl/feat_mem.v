// =============================================================================
// feat_mem.v  -  Feature-map memory for SYENet-Slim ISP
// Three INT8 buffers, channel-major addressing: addr = c*16384 + y*128 + x
//   sel 0 = BUF_IN : 4 ch  (4*16384  = 65536)   (input, packed RAW int8)
//   sel 1 = M0     : 12 ch (12*16384 = 196608)
//   sel 2 = M1     : 12 ch (12*16384 = 196608)
// Two registered read channels (A,B) + one write channel.
// isp_core guarantee: aselA is ALWAYS 0 (portA reads BIN only).
//   This means m0_ra = addrB always, m1_ra = addrB always — no mux ambiguity.
// Read latency = 1 cycle.
// Verilog-2001, BRAM-inferable.
// =============================================================================
`timescale 1ns/1ps
module feat_mem (
    input  wire        clk,
    // write channel
    input  wire        wen,
    input  wire [1:0]  wsel,
    input  wire [17:0] waddr,
    input  wire [7:0]  wdata,
    // read channel A  (isp_core always sets aselA=0, so portA reads BIN at addrA)
    input  wire [1:0]  aselA,
    input  wire [17:0] addrA,
    output reg  [7:0]  doutA,
    // read channel B  (reads M0 or M1 at addrB based on aselB)
    input  wire [1:0]  aselB,
    input  wire [17:0] addrB,
    output reg  [7:0]  doutB
);
    localparam IN_D = 65536, M_D = 196608;
    reg [7:0] bin [0:IN_D-1];
    reg [7:0] m0  [0:M_D-1];
    reg [7:0] m1  [0:M_D-1];

    // per-buffer read address mux.
    // Since aselA is always 0: bin_ra=addrA (constant), m0_ra=addrB (constant),
    // m1_ra=addrB (constant).  No runtime mux needed — Vivado infers clean BRAMs.
    wire [17:0] bin_ra = (aselA==2'd0) ? addrA : addrB;
    wire [17:0] m0_ra  = (aselA==2'd1) ? addrA : addrB;
    wire [17:0] m1_ra  = (aselA==2'd2) ? addrA : addrB;

    reg [7:0] bin_rd, m0_rd, m1_rd;
    always @(posedge clk) begin
        if (wen) begin
            case (wsel)
                2'd0: bin[waddr[16:0]] <= wdata;
                2'd1: m0 [waddr]       <= wdata;
                2'd2: m1 [waddr]       <= wdata;
            endcase
        end
        bin_rd <= bin[bin_ra[16:0]];
        m0_rd  <= m0 [m0_ra];
        m1_rd  <= m1 [m1_ra];
    end

    // sel delayed 1 cycle to match BRAM read latency
    reg [1:0] aselA_d, aselB_d;
    always @(posedge clk) begin aselA_d<=aselA; aselB_d<=aselB; end
    always @(*) begin
        case (aselA_d) 2'd0:doutA=bin_rd; 2'd1:doutA=m0_rd; default:doutA=m1_rd; endcase
        case (aselB_d) 2'd0:doutB=bin_rd; 2'd1:doutB=m0_rd; default:doutB=m1_rd; endcase
    end
endmodule
