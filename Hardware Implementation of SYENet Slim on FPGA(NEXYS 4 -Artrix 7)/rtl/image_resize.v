// =============================================================================
// image_resize.v  –  2× area-average downsample: 256×256×C → 128×128×C
// Target  : Xilinx Artix-7 XC7A100T
// Style   : Verilog-2001
//
// Method: 2×2 average pooling (divide by 4 = right-shift 2).
// Streaming: reads from image_bram (256×256×4 pixels, channel-last order),
// outputs pixel stream for feature_bram write port.
//
// Input pixel order:  row-major, channel-last (one byte per clock when ready)
// Output pixel order: same (one byte per clock)
//
// For each 2×2 block of pixels and each channel:
//   out[r/2][c/2][ch] = (p[r][c][ch] + p[r][c+1][ch] +
//                        p[r+1][c][ch] + p[r+1][c+1][ch]) >> 2
//
// BRAM interface:
//   Read address = row*256*C + col*C + ch
//   Write address = (row/2)*(128*C) + (col/2)*C + ch
// =============================================================================
`timescale 1ns/1ps
module image_resize #(
    parameter IN_H  = 256,
    parameter IN_W  = 256,
    parameter OUT_H = 128,
    parameter OUT_W = 128,
    parameter C     = 4,
    parameter DW    = 8
) (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         start,
    // Input image BRAM read port
    output reg  [16:0]  rd_addr,   // 0..256*256*4-1 = 262143
    input  wire [DW-1:0] rd_data,
    // Output feature map BRAM write port
    output reg  [16:0]  wr_addr,   // 0..128*128*4-1 = 65535
    output reg  [DW-1:0] wr_data,
    output reg          wr_en,
    // Status
    output reg          done
);
    localparam IN_STRIDE  = IN_W  * C;   // 1024
    localparam OUT_STRIDE = OUT_W * C;   // 512

    // Sum buffer: 2 input rows × OUT_W × C
    reg [9:0] sum [0:OUT_W*C-1];  // 10-bit sum (max = 4 × 255 = 1020)

    reg [7:0]  in_row, in_col;
    reg [5:0]  ch;
    reg [6:0]  out_row, out_col;
    reg [2:0]  phase;   // 0=rd_top, 1=rd_bot, 2=wr_out
    reg [16:0] rd_lat;

    localparam PH_IDLE    = 3'd0;
    localparam PH_RD_TOP  = 3'd1;
    localparam PH_RD_BOT  = 3'd2;
    localparam PH_WRITE   = 3'd3;
    localparam PH_DONE    = 3'd4;

    reg [2:0]  state;
    reg [9:0]  pix_cnt;  // within a row
    reg [9:0]  resize_idx; // module-level temp (Verilog-2001)
    integer i;

    always @(posedge clk) begin
        if (!rst_n) begin
            state    <= PH_IDLE;
            rd_addr  <= 17'd0;
            wr_addr  <= 17'd0;
            wr_en    <= 1'b0;
            wr_data  <= {DW{1'b0}};
            done     <= 1'b0;
            in_row   <= 8'd0;
            in_col   <= 8'd0;
            out_row  <= 7'd0;
            out_col  <= 7'd0;
            ch       <= 6'd0;
            pix_cnt  <= 10'd0;
            for (i = 0; i < OUT_W*C; i = i + 1) sum[i] <= 10'd0;
        end else begin
            wr_en <= 1'b0;
            done  <= 1'b0;

            case (state)
            PH_IDLE: begin
                if (start) begin
                    in_row  <= 8'd0; in_col  <= 8'd0; ch <= 6'd0;
                    out_row <= 7'd0; out_col <= 7'd0;
                    pix_cnt <= 10'd0;
                    for (i = 0; i < OUT_W*C; i = i + 1) sum[i] <= 10'd0;
                    state   <= PH_RD_TOP;
                end
            end

            // Phase 1: accumulate even row into sum[]
            PH_RD_TOP: begin
                rd_addr <= in_row * IN_STRIDE + in_col * C + ch;
                // One cycle later read arrives; stagger address vs data by 1
                if (pix_cnt > 0) begin
                    // pix_cnt-1 was the last address; rd_data is for that addr
                    resize_idx = (pix_cnt - 1) % (OUT_W * C);
                    sum[resize_idx] <= sum[resize_idx] + {{2{1'b0}}, rd_data};
                end
                pix_cnt <= pix_cnt + 1'b1;
                // Advance ch/col
                if (ch == C-1) begin
                    ch     <= 6'd0;
                    in_col <= in_col + 1'b1;
                end else
                    ch <= ch + 1'b1;
                if (pix_cnt == IN_W*C) begin
                    pix_cnt <= 10'd0;
                    in_row  <= in_row + 1'b1;
                    in_col  <= 8'd0;
                    ch      <= 6'd0;
                    state   <= PH_RD_BOT;
                end
            end

            // Phase 2: accumulate odd row into sum[]
            PH_RD_BOT: begin
                rd_addr <= in_row * IN_STRIDE + in_col * C + ch;
                if (pix_cnt > 0) begin
                    resize_idx = (pix_cnt - 1) % (OUT_W * C);
                    sum[resize_idx] <= sum[resize_idx] + {{2{1'b0}}, rd_data};
                end
                pix_cnt <= pix_cnt + 1'b1;
                if (ch == C-1) begin ch <= 6'd0; in_col <= in_col + 1'b1; end
                else ch <= ch + 1'b1;
                if (pix_cnt == IN_W*C) begin
                    pix_cnt <= 10'd0;
                    in_row  <= in_row + 1'b1;
                    in_col  <= 8'd0;
                    ch      <= 6'd0;
                    state   <= PH_WRITE;
                end
            end

            // Phase 3: write averaged output row
            PH_WRITE: begin
                wr_en   <= 1'b1;
                wr_data <= sum[pix_cnt][9:2];  // divide by 4
                wr_addr <= out_row * OUT_STRIDE + pix_cnt;
                sum[pix_cnt] <= 10'd0;  // clear for next pair of rows
                pix_cnt <= pix_cnt + 1'b1;
                if (pix_cnt == OUT_W*C-1) begin
                    pix_cnt <= 10'd0;
                    out_row <= out_row + 1'b1;
                    if (out_row == OUT_H-1) begin
                        state <= PH_DONE;
                    end else begin
                        state <= PH_RD_TOP;
                    end
                end
            end

            PH_DONE: begin
                done  <= 1'b1;
                state <= PH_IDLE;
            end
            endcase
        end
    end
endmodule
