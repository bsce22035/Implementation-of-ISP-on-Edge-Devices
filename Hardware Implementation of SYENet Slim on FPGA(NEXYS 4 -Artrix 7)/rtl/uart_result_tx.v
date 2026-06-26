// =============================================================================
// uart_result_tx.v  –  UART result packet transmitter
// Target  : Xilinx Artix-7 XC7A100T
// Style   : Verilog-2001
//
// Transmits inference result packet:
//   [0..3]   HEADER     : 0x55 0xAA 0x55 0xAA
//   [4]      CLASS_ID   : 1 byte (0..2)
//   [5..8]   CONFIDENCE : INT32 (max logit, little-endian)
//   [9..20]  LOGITS     : 3 × INT32 (all class logits, little-endian)
//   [21..24] CRC32      : 32-bit CRC of bytes 4..20
//
// Total: 25 bytes
// =============================================================================
`timescale 1ns/1ps
module uart_result_tx (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         send,           // pulse: start transmission
    input  wire [1:0]   class_id,
    input  wire [31:0]  confidence,
    input  wire [95:0]  logits,         // 3 × INT32
    // uart_tx interface
    output reg          tx_start,
    output reg  [7:0]   tx_data,
    input  wire         tx_busy,
    // Status
    output reg          done
);
    // Build packet array: 25 bytes
    reg [7:0]  pkt [0:24];
    reg [4:0]  idx;     // current byte index
    reg [31:0] crc_reg;
    reg        busy;

    // CRC function (same as loader)
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

    localparam S_IDLE  = 2'd0;
    localparam S_BUILD = 2'd1;
    localparam S_TX    = 2'd2;
    localparam S_WAIT  = 2'd3;

    reg [1:0] state;
    integer i;
    reg [31:0] crc_val;

    always @(posedge clk) begin
        if (!rst_n) begin
            state    <= S_IDLE;
            tx_start <= 1'b0;
            tx_data  <= 8'd0;
            done     <= 1'b0;
            idx      <= 5'd0;
            busy     <= 1'b0;
            for (i = 0; i < 25; i = i + 1) pkt[i] <= 8'd0;
        end else begin
            tx_start <= 1'b0;
            done     <= 1'b0;

            case (state)
            S_IDLE: begin
                if (send) begin
                    // --- Build packet ---
                    pkt[0]  <= 8'h55; pkt[1]  <= 8'hAA;
                    pkt[2]  <= 8'h55; pkt[3]  <= 8'hAA;
                    pkt[4]  <= {6'd0, class_id};
                    pkt[5]  <= confidence[7:0];
                    pkt[6]  <= confidence[15:8];
                    pkt[7]  <= confidence[23:16];
                    pkt[8]  <= confidence[31:24];
                    pkt[9]  <= logits[7:0];
                    pkt[10] <= logits[15:8];
                    pkt[11] <= logits[23:16];
                    pkt[12] <= logits[31:24];
                    pkt[13] <= logits[39:32];
                    pkt[14] <= logits[47:40];
                    pkt[15] <= logits[55:48];
                    pkt[16] <= logits[63:56];
                    pkt[17] <= logits[71:64];
                    pkt[18] <= logits[79:72];
                    pkt[19] <= logits[87:80];
                    pkt[20] <= logits[95:88];
                    // CRC placeholder (computed next cycle)
                    pkt[21] <= 8'd0; pkt[22] <= 8'd0;
                    pkt[23] <= 8'd0; pkt[24] <= 8'd0;
                    idx     <= 5'd0;
                    state   <= S_BUILD;
                end
            end

            S_BUILD: begin
                // Compute CRC using module-level crc_val register
                crc_val = 32'hFFFFFFFF;
                crc_val = crc32_byte(crc_val, pkt[4]);  crc_val = crc32_byte(crc_val, pkt[5]);
                crc_val = crc32_byte(crc_val, pkt[6]);  crc_val = crc32_byte(crc_val, pkt[7]);
                crc_val = crc32_byte(crc_val, pkt[8]);  crc_val = crc32_byte(crc_val, pkt[9]);
                crc_val = crc32_byte(crc_val, pkt[10]); crc_val = crc32_byte(crc_val, pkt[11]);
                crc_val = crc32_byte(crc_val, pkt[12]); crc_val = crc32_byte(crc_val, pkt[13]);
                crc_val = crc32_byte(crc_val, pkt[14]); crc_val = crc32_byte(crc_val, pkt[15]);
                crc_val = crc32_byte(crc_val, pkt[16]); crc_val = crc32_byte(crc_val, pkt[17]);
                crc_val = crc32_byte(crc_val, pkt[18]); crc_val = crc32_byte(crc_val, pkt[19]);
                crc_val = crc32_byte(crc_val, pkt[20]);
                crc_val = crc_val ^ 32'hFFFFFFFF;
                pkt[21] <= crc_val[7:0];  pkt[22] <= crc_val[15:8];
                pkt[23] <= crc_val[23:16]; pkt[24] <= crc_val[31:24];
                state   <= S_TX;
            end

            S_TX: begin
                if (!tx_busy) begin
                    tx_data  <= pkt[idx];
                    tx_start <= 1'b1;
                    state    <= S_WAIT;
                end
            end

            S_WAIT: begin
                if (!tx_busy) begin
                    if (idx == 5'd24) begin
                        done  <= 1'b1;
                        state <= S_IDLE;
                    end else begin
                        idx   <= idx + 1'b1;
                        state <= S_TX;
                    end
                end
            end
            endcase
        end
    end
endmodule
