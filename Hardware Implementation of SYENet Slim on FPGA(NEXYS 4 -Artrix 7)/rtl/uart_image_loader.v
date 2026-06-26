// =============================================================================
// uart_image_loader.v  –  UART image packet receiver and parser
// Target  : Xilinx Artix-7 XC7A100T (Nexys-4)
// Style   : Verilog-2001
//
// Packet format (little-endian):
//   [0..3]   HEADER  : 0xAA 0x55 0xAA 0x55
//   [4..5]   WIDTH   : 16-bit (expected 0x0080 = 128)
//   [6..7]   HEIGHT  : 16-bit (expected 0x0080 = 128)
//   [8..9]   CHANNELS: 16-bit (expected 0x0004)
//   [10..N-5] PAYLOAD: WIDTH × HEIGHT × CHANNELS bytes
//   [N-4..N-1] CRC32 : 32-bit Ethernet CRC of header+payload
//
// Payload size: 128×128×4 = 65 536 bytes
// Total packet : 65 546 bytes
//
// On successful receive: image_done pulsed, BRAM filled with image.
// On CRC error: error pulsed.
//
// Timeout: 2^24 = ~167 ms at 100 MHz resets FSM if no byte arrives.
// =============================================================================
`timescale 1ns/1ps
module uart_image_loader #(
    parameter IMG_W   = 128,
    parameter IMG_H   = 128,
    parameter CHANNELS = 4,
    parameter TIMEOUT = 24'hFFFFFF
) (
    input  wire       clk,
    input  wire       rst_n,
    // From uart_rx
    input  wire       rx_done,
    input  wire [7:0] rx_data,
    // Image BRAM write port
    output reg        bram_we,
    output reg [16:0] bram_addr,  // 0 .. 128*128*4-1 = 65535
    output reg [7:0]  bram_wdata,
    // Status
    output reg        image_done,
    output reg        crc_error,
    output reg        timeout_err
);
    localparam PAYLOAD_SIZE = IMG_W * IMG_H * CHANNELS;  // 65536

    // CRC-32 (Ethernet polynomial: 0xEDB88320 reflected)
    // Simple byte-at-a-time CRC
    reg [31:0] crc_reg;
    wire [31:0] crc_next;

    // CRC lookup: combinatorial one-byte step
    function [31:0] crc32_byte;
        input [31:0] crc_in;
        input [7:0]  data;
        integer j;
        reg [31:0] c;
        begin
            c = crc_in ^ {24'd0, data};
            for (j = 0; j < 8; j = j + 1)
                c = (c[0]) ? (c >> 1) ^ 32'hEDB88320 : (c >> 1);
            crc32_byte = c;
        end
    endfunction

    assign crc_next = crc32_byte(crc_reg, rx_data);

    // FSM
    localparam S_IDLE    = 4'd0;
    localparam S_HDR0    = 4'd1;  // expecting 0xAA
    localparam S_HDR1    = 4'd2;  // expecting 0x55
    localparam S_HDR2    = 4'd3;  // expecting 0xAA
    localparam S_HDR3    = 4'd4;  // expecting 0x55
    localparam S_WIDTH   = 4'd5;  // 2 bytes
    localparam S_HEIGHT  = 4'd6;
    localparam S_CHAN    = 4'd7;
    localparam S_PAYLOAD = 4'd8;
    localparam S_CRC     = 4'd9;
    localparam S_DONE    = 4'd10;

    reg [3:0]  state;
    reg [1:0]  byte_cnt;   // for multi-byte fields
    reg [31:0] payload_cnt;
    reg [31:0] rx_crc;     // received CRC
    reg [23:0] timeout_cnt;

    always @(posedge clk) begin
        if (!rst_n) begin
            state       <= S_IDLE;
            bram_we     <= 1'b0;
            bram_addr   <= 17'd0;
            bram_wdata  <= 8'd0;
            image_done  <= 1'b0;
            crc_error   <= 1'b0;
            timeout_err <= 1'b0;
            crc_reg     <= 32'hFFFFFFFF;
            rx_crc      <= 32'd0;
            payload_cnt <= 32'd0;
            byte_cnt    <= 2'd0;
            timeout_cnt <= 24'd0;
        end else begin
            bram_we    <= 1'b0;
            image_done <= 1'b0;
            crc_error  <= 1'b0;
            timeout_err <= 1'b0;

            // Timeout counter
            if (state != S_IDLE) begin
                if (rx_done)
                    timeout_cnt <= 24'd0;
                else begin
                    timeout_cnt <= timeout_cnt + 1'b1;
                    if (timeout_cnt == TIMEOUT) begin
                        timeout_err <= 1'b1;
                        state       <= S_IDLE;
                        timeout_cnt <= 24'd0;
                    end
                end
            end

            if (rx_done) begin
                case (state)
                S_IDLE: begin
                    crc_reg     <= 32'hFFFFFFFF;
                    payload_cnt <= 32'd0;
                    byte_cnt    <= 2'd0;
                    bram_addr   <= 17'd0;
                    if (rx_data == 8'hAA) state <= S_HDR1;
                end
                // NOTE: CRC is computed over the PAYLOAD ONLY to match the host
                // (host_uart.py: crc32(payload)). crc_reg therefore stays at
                // 0xFFFFFFFF through the header and is folded only in S_PAYLOAD.
                S_HDR1: if (rx_data == 8'h55) state <= S_HDR2;
                         else state <= S_IDLE;
                S_HDR2: if (rx_data == 8'hAA) state <= S_HDR3;
                         else state <= S_IDLE;
                S_HDR3: if (rx_data == 8'h55) begin state <= S_WIDTH; byte_cnt <= 2'd0; end
                         else state <= S_IDLE;
                S_WIDTH: begin
                    byte_cnt <= byte_cnt + 1'b1;
                    if (byte_cnt == 2'd1) begin byte_cnt <= 2'd0; state <= S_HEIGHT; end
                end
                S_HEIGHT: begin
                    byte_cnt <= byte_cnt + 1'b1;
                    if (byte_cnt == 2'd1) begin byte_cnt <= 2'd0; state <= S_CHAN; end
                end
                S_CHAN: begin
                    byte_cnt <= byte_cnt + 1'b1;
                    if (byte_cnt == 2'd1) begin
                        crc_reg <= 32'hFFFFFFFF;   // (re)init for payload CRC
                        state   <= S_PAYLOAD;
                    end
                end
                S_PAYLOAD: begin
                    crc_reg     <= crc_next;
                    bram_we     <= 1'b1;
                    bram_addr   <= payload_cnt[16:0];
                    bram_wdata  <= rx_data;
                    payload_cnt <= payload_cnt + 1'b1;
                    if (payload_cnt == PAYLOAD_SIZE - 1) begin
                        state    <= S_CRC;
                        byte_cnt <= 2'd0;
                        rx_crc   <= 32'd0;
                    end
                end
                S_CRC: begin
                    // Receive 4-byte CRC (LSB first)
                    rx_crc   <= {rx_data, rx_crc[31:8]};
                    byte_cnt <= byte_cnt + 1'b1;
                    if (byte_cnt == 2'd3) begin
                        state <= S_DONE;
                    end
                end
                default: state <= S_IDLE;
                endcase
            end

            if (state == S_DONE) begin
                // Check CRC: finalise with XOR 0xFFFFFFFF
                if ((crc_reg ^ 32'hFFFFFFFF) == rx_crc)
                    image_done <= 1'b1;
                else
                    crc_error  <= 1'b1;
                state <= S_IDLE;
            end
        end
    end
endmodule
