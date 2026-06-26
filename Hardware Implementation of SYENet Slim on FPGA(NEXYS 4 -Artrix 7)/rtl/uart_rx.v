// uart_rx.v - 8N1 UART receiver, Verilog-2001
`timescale 1ns/1ps
module uart_rx #(parameter CLK_HZ=50000000, parameter BAUD=115200) (
    input  wire clk, input wire rst_n, input wire rx,
    output reg [7:0] data_out, output reg rx_done
);
    localparam integer DIV = CLK_HZ/BAUD;
    reg [1:0] sync; reg [15:0] cnt; reg [3:0] bit_i; reg [2:0] st;
    reg [7:0] sh;
    localparam IDLE=0, START=1, DATA=2, STOP=3;
    always @(posedge clk) begin
        if (!rst_n) begin st<=IDLE; rx_done<=0; sync<=2'b11; cnt<=0; bit_i<=0; end
        else begin
            sync<={sync[0],rx}; rx_done<=0;
            case (st)
            IDLE: if (sync[1]==1'b0) begin cnt<=DIV/2; st<=START; end
            START: if (cnt==0) begin if (sync[1]==1'b0) begin cnt<=DIV-1; bit_i<=0; st<=DATA; end else st<=IDLE; end
                   else cnt<=cnt-1;
            DATA: if (cnt==0) begin sh<={sync[1],sh[7:1]}; cnt<=DIV-1;
                      if (bit_i==7) st<=STOP; else bit_i<=bit_i+1; end
                  else cnt<=cnt-1;
            STOP: if (cnt==0) begin data_out<=sh; rx_done<=1; st<=IDLE; end
                  else cnt<=cnt-1;
            endcase
        end
    end
endmodule
