// uart_tx.v - 8N1 UART transmitter, Verilog-2001
`timescale 1ns/1ps
module uart_tx #(parameter CLK_HZ=50000000, parameter BAUD=115200) (
    input  wire clk, input wire rst_n,
    input  wire tx_start, input wire [7:0] data_in,
    output reg tx, output reg tx_busy
);
    localparam integer DIV = CLK_HZ/BAUD;
    reg [15:0] cnt; reg [3:0] bit_i; reg [9:0] sh; reg [1:0] st;
    localparam IDLE=0, RUN=1;
    always @(posedge clk) begin
        if (!rst_n) begin st<=IDLE; tx<=1'b1; tx_busy<=0; cnt<=0; bit_i<=0; end
        else begin
            case (st)
            IDLE: begin tx<=1'b1; tx_busy<=0;
                if (tx_start) begin sh<={1'b1,data_in,1'b0}; cnt<=DIV-1; bit_i<=0; tx_busy<=1; st<=RUN; end end
            RUN: begin tx<=sh[0];
                if (cnt==0) begin cnt<=DIV-1; sh<={1'b1,sh[9:1]};
                    if (bit_i==9) begin tx_busy<=0; st<=IDLE; tx<=1'b1; end
                    else bit_i<=bit_i+1; end
                else cnt<=cnt-1; end
            endcase
        end
    end
endmodule
