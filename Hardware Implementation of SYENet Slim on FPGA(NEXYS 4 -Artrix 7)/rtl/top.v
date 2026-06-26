// =============================================================================
// top.v  -  SYENet-Slim ISP FPGA top (Nexys-4 / XC7A100T)
// RX: AA 55 AA 55 | 65536 packed-int8 RAW bytes | CRC32  -> BUF_IN, then run core
// TX: 55 AA 55 AA | 196608 RGB bytes (HWC) | CRC32(rgb)
// 100 MHz crystal -> MMCM -> 50 MHz.
// =============================================================================
`timescale 1ns/1ps
module top (
    input  wire CLK100MHZ, input wire CPU_RESETN,
    input  wire UART_TXD_IN, output wire UART_RXD_OUT,
    output wire [7:0] LED, output wire [7:0] AN, output wire [6:0] SEG, output wire DP
);
    // ---- clock 100->25 MHz (VCO=1000, /40) : relaxed timing for the
    //      single-cycle requant path (~28 ns). 25 MHz = 40 ns gives margin.
    wire clk, clkfb, locked, clk_u;
    MMCME2_BASE #(.CLKIN1_PERIOD(10.0), .CLKFBOUT_MULT_F(10.0),
        .CLKOUT0_DIVIDE_F(40.0), .DIVCLK_DIVIDE(1)) u_mmcm (
        .CLKIN1(CLK100MHZ), .CLKFBIN(clkfb), .CLKFBOUT(clkfb),
        .CLKOUT0(clk_u), .LOCKED(locked), .PWRDWN(1'b0), .RST(1'b0),
        .CLKOUT1(),.CLKOUT2(),.CLKOUT3(),.CLKOUT4(),.CLKOUT5(),.CLKOUT6(),
        .CLKOUT0B(),.CLKOUT1B(),.CLKOUT2B(),.CLKOUT3B(),.CLKFBOUTB());
    BUFG u_bg(.I(clk_u), .O(clk));
    wire rst_n = CPU_RESETN & locked;

    // ---- UART ----
    wire [7:0] rxd; wire rxv;
    uart_rx #(.CLK_HZ(25000000)) urx(.clk(clk),.rst_n(rst_n),.rx(UART_TXD_IN),.data_out(rxd),.rx_done(rxv));
    reg tx_start; reg [7:0] tx_data; wire tx_busy;
    uart_tx #(.CLK_HZ(25000000)) utx(.clk(clk),.rst_n(rst_n),.tx_start(tx_start),.data_in(tx_data),.tx(UART_RXD_OUT),.tx_busy(tx_busy));

    // ---- core ----
    reg core_start; reg in_wen; reg [15:0] in_addr; reg [7:0] in_data;
    wire o_valid; wire [7:0] o_data; reg o_ready; wire core_done;
    isp_core u_core(.clk(clk),.rst_n(rst_n),.start(core_start),
        .in_wen(in_wen),.in_addr(in_addr),.in_data(in_data),
        .o_valid(o_valid),.o_data(o_data),.o_ready(o_ready),.done(core_done));

    // ---- CRC32 ----
    function [31:0] crc_b; input [31:0] c0; input [7:0] d; integer j; reg [31:0] c; begin
        c=c0^{24'd0,d};
        for(j=0;j<8;j=j+1) c=c[0]?(c>>1)^32'hEDB88320:(c>>1);
        crc_b=c; end
    endfunction

    // ---- RX loader FSM ----
    localparam R_H0=0,R_H1=1,R_H2=2,R_H3=3,R_PAY=4,R_CRC=5,R_RUN=6,R_WAIT=7,R_TX=8;
    reg [3:0] rst8; reg [16:0] rcnt; reg [1:0] ccnt;
    // ---- TX FSM ----
    localparam T_IDLE=0,T_HDR=1,T_DATA=2,T_CRC=3,T_DN=4;
    reg [2:0] tst; reg [1:0] thi; reg [17:0] tcnt; reg [31:0] crc; reg [31:0] crcf; reg [1:0] tci;
    reg [7:0] hdr_tx [0:3];

    reg [7:0] led_r;
    assign LED = led_r;

    always @(posedge clk) begin
        if (!rst_n) begin
            rst8<=R_H0; rcnt<=0; ccnt<=0; core_start<=0; in_wen<=0;
            tst<=T_IDLE; tx_start<=0; o_ready<=0; led_r<=8'h01;
            hdr_tx[0]<=8'h55; hdr_tx[1]<=8'hAA; hdr_tx[2]<=8'h55; hdr_tx[3]<=8'hAA;
        end else begin
            in_wen<=0; core_start<=0; tx_start<=0; o_ready<=0;
            // ============ RX / control ============
            case (rst8)
            R_H0: begin led_r<=8'h01; if (rxv) rst8<=(rxd==8'hAA)?R_H1:R_H0; end
            R_H1: if (rxv) rst8<=(rxd==8'h55)?R_H2:R_H0;
            R_H2: if (rxv) rst8<=(rxd==8'hAA)?R_H3:R_H0;
            R_H3: if (rxv) begin rst8<=(rxd==8'h55)?R_PAY:R_H0; rcnt<=0; led_r<=8'h02; end
            R_PAY: if (rxv) begin
                       in_wen<=1; in_addr<=rcnt[15:0]; in_data<=rxd;
                       if (rcnt==17'd65535) begin rcnt<=0; ccnt<=0; rst8<=R_CRC; end
                       else rcnt<=rcnt+1;
                   end
            R_CRC: if (rxv) begin if (ccnt==2'd3) rst8<=R_RUN; else ccnt<=ccnt+1; end
            R_RUN: begin core_start<=1; led_r<=8'h08; tst<=T_IDLE; rst8<=R_WAIT; end
            R_WAIT: begin
                       // start TX header immediately; data forwarded as core produces
                       if (tst==T_IDLE) begin tst<=T_HDR; thi<=0; led_r<=8'h10; end
                       if (core_done) rst8<=R_H0;   // ready for next frame after stream ends
                   end
            default: rst8<=R_H0;
            endcase

            // ============ TX FSM ============
            case (tst)
            T_HDR: if (!tx_busy && !tx_start) begin
                       tx_data<=hdr_tx[thi]; tx_start<=1;
                       if (thi==2'd3) begin tst<=T_DATA; crc<=32'hFFFFFFFF; tcnt<=0; end
                       else thi<=thi+1;
                   end
            T_DATA: if (o_valid && !tx_busy && !tx_start) begin
                       tx_data<=o_data; tx_start<=1; o_ready<=1;       // consume core byte
                       crc<=crc_b(crc,o_data);
                       if (tcnt==18'd196607) begin crcf<=crc_b(crc,o_data)^32'hFFFFFFFF; tci<=0; tst<=T_CRC; end
                       else tcnt<=tcnt+1;
                   end
            T_CRC: if (!tx_busy && !tx_start) begin
                       case (tci) 0:tx_data<=crcf[7:0]; 1:tx_data<=crcf[15:8];
                                  2:tx_data<=crcf[23:16]; 3:tx_data<=crcf[31:24]; endcase
                       tx_start<=1;
                       if (tci==2'd3) tst<=T_DN; else tci<=tci+1;
                   end
            T_DN: begin tst<=T_IDLE; led_r<=8'h20; end
            default: ;
            endcase
        end
    end

    assign AN  = 8'b11111110;
    assign SEG = 7'b1000000;
    assign DP  = 1'b1;
endmodule
