// =============================================================================
// line_buffer.v  –  K-line circular line buffer for streaming convolution
// Target  : Xilinx Artix-7 XC7A100T
// Style   : Verilog-2001, inferred block RAM (SDP mode)
//
// Stores K complete horizontal lines of W pixels × C channels.
// Each write advances the write pointer; data is read with a line offset.
// Uses SDP BRAM to achieve 1-cycle read latency with registered output.
//
// Port dout is a flat packed vector [K*C*DW-1:0]:
//   dout[k*C*DW +: C*DW] = pixel W*(k+1) ago in the stream
// =============================================================================
`timescale 1ns/1ps
module line_buffer #(
    parameter K  = 3,    // kernel rows
    parameter W  = 128,  // image width (pixels)
    parameter C  = 12,   // channels
    parameter DW = 8     // bits per channel
) (
    input  wire              clk,
    input  wire              rst_n,
    // write side
    input  wire              wr_en,
    input  wire [C*DW-1:0]  din,
    // read side – K-line window, flat packed (Verilog-2001 compatible)
    // dout[k*C*DW +: C*DW] = line k (0=newest, K-1=oldest)
    output wire [K*C*DW-1:0] dout
);
    // -------------------------------------------------------------------------
    // Circular BRAM: depth = K*W, width = C*DW
    // -------------------------------------------------------------------------
    localparam DEPTH = K * W;
    localparam AW    = $clog2(DEPTH);

    reg [C*DW-1:0] mem [0:DEPTH-1];

    // Write pointer (circular)
    reg [AW-1:0] wptr;

    always @(posedge clk) begin
        if (!rst_n) begin
            wptr <= {AW{1'b0}};
        end else if (wr_en) begin
            mem[wptr] <= din;
            if (wptr == DEPTH - 1)
                wptr <= {AW{1'b0}};
            else
                wptr <= wptr + 1'b1;
        end
    end

    // -------------------------------------------------------------------------
    // Internal registered output array (Verilog-2001: no unpacked port arrays)
    // -------------------------------------------------------------------------
    reg [C*DW-1:0] dout_r [0:K-1];

    // -------------------------------------------------------------------------
    // Read: for line k (0 = newest, K-1 = oldest)
    //   read_addr = (wptr - (k+1)*W + DEPTH) % DEPTH
    // -------------------------------------------------------------------------
    genvar k;
    generate
        for (k = 0; k < K; k = k + 1) begin : rd_line
            wire [AW-1:0] raddr;
            assign raddr = (wptr >= (k+1)*W) ?
                            wptr - (k+1)*W :
                            wptr + DEPTH - (k+1)*W;
            always @(posedge clk) begin
                dout_r[k] <= mem[raddr];
            end
            // Connect internal reg slice to flat output port
            assign dout[k*C*DW +: C*DW] = dout_r[k];
        end
    endgenerate

endmodule
