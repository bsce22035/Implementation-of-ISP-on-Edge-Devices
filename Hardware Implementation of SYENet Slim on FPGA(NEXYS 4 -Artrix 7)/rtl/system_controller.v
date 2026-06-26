// =============================================================================
// system_controller.v  –  Top-Level System FSM
// Target  : Xilinx Artix-7 XC7A100T (Nexys-4)
// Style   : Verilog-2001
//
// Master state machine:
//   IDLE     -> wait for image packet  (image_done from uart_image_loader)
//   WAIT_RX  -> wait for image_done
//   PREP     -> pulse resize_start (= BRAM streamer warmup trigger)
//               AND pulse cnn_go simultaneously so CNN starts loading weights
//               while the 512-cycle warmup countdown runs.
//   CNN      -> wait for cnn_done (conv_engine processes all pixels)
//   TX       -> transmit result via uart_result_tx
//   DONE     -> show result on LEDs, return to WAIT_RX
//   ERR      -> all LEDs on, restart
//
// LED indicators:
//   led[0]: IDLE / waiting for image
//   led[1]: RECEIVING image
//   led[2]: PREPROCESSING (warmup + streaming)
//   led[3]: INFERENCE RUNNING
//   led[4]: TRANSMITTING RESULT
//   led[7:5]: class_id[1:0] + confidence MSB after done
// =============================================================================
`timescale 1ns/1ps
module system_controller (
    input  wire        clk,
    input  wire        rst_n,
    // UART RX status
    input  wire        image_done,
    input  wire        crc_error,
    input  wire        timeout_err,
    // Stream trigger (resize_start repurposed: triggers BRAM streamer warmup)
    output reg         resize_start,
    input  wire        resize_done,   // unused in this revision
    // CNN
    output reg         cnn_go,
    input  wire        cnn_done,
    // Result
    input  wire [1:0]  class_id,
    input  wire [31:0] confidence,
    output reg         tx_send,
    input  wire        tx_done,
    // Status LEDs
    output reg  [7:0]  led,
    // Debug output
    output reg  [3:0]  state_out
);
    localparam S_IDLE    = 4'd0;
    localparam S_WAIT_RX = 4'd1;
    localparam S_PREP    = 4'd2;
    localparam S_CNN     = 4'd3;
    localparam S_TX      = 4'd4;
    localparam S_DONE    = 4'd5;
    localparam S_ERR     = 4'd6;

    reg [3:0] state;

    // Watchdog: forces a response if the CNN never asserts cnn_done.
    // 2^25 cycles @ 50 MHz ~= 0.67 s.  The simplified datapath should finish
    // long before this; the watchdog only guards against a hung pipeline so
    // the host always receives a result packet.
    localparam WD_MAX = 27'd33_000_000;
    reg [26:0] wd_cnt;
    reg        wd_fired;   // status: 1 if last inference timed out

    always @(posedge clk) begin
        if (!rst_n) begin
            state        <= S_IDLE;
            resize_start <= 1'b0;
            cnn_go       <= 1'b0;
            tx_send      <= 1'b0;
            led          <= 8'h01;
            state_out    <= 4'd0;
            wd_cnt       <= 27'd0;
            wd_fired     <= 1'b0;
        end else begin
            resize_start <= 1'b0;
            cnn_go       <= 1'b0;
            tx_send      <= 1'b0;
            state_out    <= state;

            case (state)
            S_IDLE: begin
                led   <= 8'h01;
                state <= S_WAIT_RX;
            end

            S_WAIT_RX: begin
                led[1] <= 1'b1;
                led[0] <= 1'b0;
                if (crc_error || timeout_err) begin
                    led   <= 8'hFF;
                    state <= S_ERR;
                end else if (image_done) begin
                    state <= S_PREP;
                end
            end

            // Pulse resize_start (BRAM streamer warmup trigger) and cnn_go
            // simultaneously so CNN loads weights during the 512-cycle warmup.
            S_PREP: begin
                led          <= 8'h04;
                resize_start <= 1'b1;   // starts 512-cycle warmup, then pixel stream
                cnn_go       <= 1'b1;   // CNN starts loading bias+weights now
                wd_cnt       <= 27'd0;  // arm watchdog
                wd_fired     <= 1'b0;
                state        <= S_CNN;
            end

            // Wait for CNN to finish (streamer feeds pixels after warmup).
            // Watchdog forces S_TX if cnn_done never arrives.
            S_CNN: begin
                led    <= 8'h08;
                wd_cnt <= wd_cnt + 27'd1;
                if (cnn_done) begin
                    state <= S_TX;
                end else if (wd_cnt >= WD_MAX) begin
                    wd_fired <= 1'b1;   // mark timeout (visible on led[6] in S_DONE)
                    state    <= S_TX;
                end
            end

            S_TX: begin
                led     <= 8'h10;
                tx_send <= 1'b1;
                if (tx_done) state <= S_DONE;
            end

            S_DONE: begin
                // led[7]=conf sign, led[6]=watchdog fired, led[5:4]=class_id,
                // led[3:0]=0001 (done marker)
                led <= {confidence[31], wd_fired, class_id, 4'b0001};
                state <= S_WAIT_RX;  // ready for next image
            end

            S_ERR: begin
                led   <= 8'hFF;
                state <= S_IDLE;
            end

            default: state <= S_IDLE;
            endcase
        end
    end
endmodule
