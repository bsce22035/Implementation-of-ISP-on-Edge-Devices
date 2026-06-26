// =============================================================================
// weight_loader.v  –  Sequential weight prefetcher for conv_engine
// Reads C_OUT×C_IN×K×K weights from weight_bram starting at WBASE,
// presenting them one per cycle to the systolic array.
// =============================================================================
`timescale 1ns/1ps
module weight_loader #(
    parameter C_OUT = 12,
    parameter C_IN  = 4,
    parameter K     = 3,
    parameter WBASE = 0,
    parameter AW    = 12,
    parameter DW    = 8
) (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          start,
    output reg  [AW-1:0] wgt_addr,
    input  wire [DW-1:0] wgt_data,
    output reg  [DW-1:0] wgt_out,
    output reg           wgt_valid,
    output reg           done
);
    localparam TOTAL = C_OUT * C_IN * K * K;

    reg [11:0] cnt;
    reg        running;

    always @(posedge clk) begin
        if (!rst_n) begin
            cnt      <= 12'd0;
            running  <= 1'b0;
            wgt_addr <= {AW{1'b0}};
            wgt_valid <= 1'b0;
            done     <= 1'b0;
        end else begin
            done      <= 1'b0;
            wgt_valid <= 1'b0;

            if (start) begin
                cnt     <= 12'd0;
                running <= 1'b1;
                wgt_addr <= WBASE[AW-1:0];
            end

            if (running) begin
                wgt_out   <= wgt_data;
                wgt_valid <= 1'b1;
                wgt_addr  <= WBASE[AW-1:0] + cnt + 1'b1;
                cnt       <= cnt + 1'b1;
                if (cnt == TOTAL - 1) begin
                    running <= 1'b0;
                    done    <= 1'b1;
                end
            end
        end
    end
endmodule
