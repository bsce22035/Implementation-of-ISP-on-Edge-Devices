// =============================================================================
// feature_map_ctrl.v  –  Feature map BRAM read/write address controller
// Generates sequential addresses for channel-last feature map access.
// =============================================================================
`timescale 1ns/1ps
module feature_map_ctrl #(
    parameter IMG_H = 128,
    parameter IMG_W = 128,
    parameter C     = 12,
    parameter AW    = 18
) (
    input  wire           clk,
    input  wire           rst_n,
    input  wire           start_rd,    // start sequential read
    input  wire           start_wr,    // start sequential write
    // Write side
    input  wire           wr_data_valid,
    input  wire [7:0]     wr_data,
    output reg  [AW-1:0]  wr_addr,
    output reg            wr_en,
    output reg  [7:0]     wr_din,
    // Read side
    output reg  [AW-1:0]  rd_addr,
    output reg            rd_en,
    // Channel/position tracking outputs (for consumers)
    output reg  [3:0]     ch_out,
    output reg  [6:0]     col_out,
    output reg  [6:0]     row_out,
    output reg            rd_valid,
    output reg            wr_done,
    output reg            rd_done
);
    localparam TOTAL = IMG_H * IMG_W * C;

    reg [17:0] rd_cnt, wr_cnt;
    reg        rd_run, wr_run;

    always @(posedge clk) begin
        if (!rst_n) begin
            rd_cnt <= 18'd0; wr_cnt <= 18'd0;
            rd_run <= 1'b0;  wr_run <= 1'b0;
            wr_en  <= 1'b0;  rd_en  <= 1'b0;
            rd_valid <= 1'b0; wr_done <= 1'b0; rd_done <= 1'b0;
            wr_addr <= {AW{1'b0}}; rd_addr <= {AW{1'b0}};
            ch_out  <= 4'd0; col_out <= 7'd0; row_out <= 7'd0;
        end else begin
            wr_en    <= 1'b0;
            rd_en    <= 1'b1;
            rd_valid <= 1'b0;
            wr_done  <= 1'b0;
            rd_done  <= 1'b0;

            if (start_rd) begin rd_cnt <= 18'd0; rd_run <= 1'b1; end
            if (start_wr) begin wr_cnt <= 18'd0; wr_run <= 1'b1; end

            // Write side
            if (wr_run && wr_data_valid) begin
                wr_en    <= 1'b1;
                wr_addr  <= wr_cnt[AW-1:0];
                wr_din   <= wr_data;
                wr_cnt   <= wr_cnt + 1'b1;
                if (wr_cnt == TOTAL - 1) begin
                    wr_run  <= 1'b0;
                    wr_done <= 1'b1;
                end
            end

            // Read side
            if (rd_run) begin
                rd_addr  <= rd_cnt[AW-1:0];
                rd_valid <= 1'b1;
                // Decode position
                ch_out  <= rd_cnt % C;
                col_out <= (rd_cnt / C) % IMG_W;
                row_out <= (rd_cnt / C) / IMG_W;
                rd_cnt  <= rd_cnt + 1'b1;
                if (rd_cnt == TOTAL - 1) begin
                    rd_run  <= 1'b0;
                    rd_done <= 1'b1;
                end
            end
        end
    end
endmodule
