// =============================================================================
// bias_loader.v  –  Bias prefetcher: reads C biases from BRAM starting at BBASE
// Target  : Xilinx Artix-7 XC7A100T
// Style   : Verilog-2001
//
// bias_out is flat packed [C*DW-1:0] (Verilog-2001: no unpacked port arrays)
//   bias_out[i*DW +: DW] = bias for output channel i
// =============================================================================
`timescale 1ns/1ps
module bias_loader #(
    parameter C     = 12,
    parameter BBASE = 0,
    parameter AW    = 7,
    parameter DW    = 32
) (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          start,
    output reg  [AW-1:0] bias_addr,
    input  wire [DW-1:0] bias_data,
    output reg  [C*DW-1:0] bias_out,   // flat packed; slice [i*DW +: DW] per ch
    output reg           done
);
    reg [3:0] cnt;
    reg       running;
    integer i;

    always @(posedge clk) begin
        if (!rst_n) begin
            cnt       <= 4'd0;
            running   <= 1'b0;
            done      <= 1'b0;
            bias_addr <= {AW{1'b0}};
            bias_out  <= {(C*DW){1'b0}};
        end else begin
            done <= 1'b0;
            if (start) begin
                cnt       <= 4'd0;
                running   <= 1'b1;
                bias_addr <= BBASE[AW-1:0];
            end
            if (running) begin
                if (cnt > 0)
                    bias_out[(cnt-1)*DW +: DW] <= bias_data;
                bias_addr <= BBASE[AW-1:0] + cnt + 1'b1;
                cnt <= cnt + 1'b1;
                if (cnt == C) begin
                    bias_out[(C-1)*DW +: DW] <= bias_data;
                    running <= 1'b0;
                    done    <= 1'b1;
                end
            end
        end
    end
endmodule
